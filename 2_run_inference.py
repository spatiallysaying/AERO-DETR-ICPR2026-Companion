"""
AERO-DETR Runway Extraction — Reproduction Inference CLI
Reproduces ICPR Table 3 (validation metrics per runway configuration) end-to-end.
Pipeline stages:
  1. Coarse runway-area OBB detection (YOLOv8x-OBB)
  2. Horizontal normalisation of candidate strips
  3. Marking/runway detection (RT-DETR) + polygon reconstruction
Then merges predictions, computes per-airport IoU vs ground truth, and writes Table 3.

Usage examples
--------------
# Full Table 3 over all five configs (default)
python 2_run_inference.py

# One config only
python 2_run_inference.py --categories inter

# Multi-seed variability on the hardest config (3 seeds)
python 2_run_inference.py --categories inter --seeds 2498 108 54

# Override paths / output prefix
python 2_run_inference.py --rasters_root rasters --gt ground_truth/all_gt.geojson \\
    --out_prefix reproduce_output --models_dir models
"""

import os
import glob
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO, RTDETR

from geo_utils import *
from extract_runway_markings import *

# ─────────────────────────────────────────────────────────────
# Default config
# ─────────────────────────────────────────────────────────────
CATEGORIES = ['single', 'parallel', 'inter', 'mixed', 'complex']
# Imagery is read from rasters/<category>/ as produced by 1_download_dataset.py.
# Zenodo ships JPEG2000 (*.jp2); local GeoTIFFs (*.tif) are also accepted.
RASTER_EXTS = ('*.jp2', '*.tif')
GLOBAL_CONFIG = {
    'rasters_root': 'rasters',
    'gt'         : 'ground_truth/all_gt.geojson',
    'out_prefix' : 'reproduce_output',
    'obb_model'  : 'models/rwy_obb_v1.pt',
    'detr_model' : 'models/rwy_markings_H_v1.pt',
    'iou_thresh' : 0.8,
    'device'     : 0 if torch.cuda.is_available() else 'cpu',
}


# ─────────────────────────────────────────────────────────────
# Pipeline stages (logic unchanged from the validated notebook)
# ─────────────────────────────────────────────────────────────
def obb_predictions_to_geodf(src_raster, results):
    from osgeo import gdal
    src = gdal.Open(src_raster)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    crs_wkt = src.GetProjection()
    forward_transform = (float(ulx), float(xres), float(xskew),
                         float(uly), float(yskew), float(yres))
    del src
    result = results[0]
    detections_list = []
    if result.obb:
        for cls, conf, xywhr, obb in zip(result.obb.cls, result.obb.conf,
                                         result.obb.xywhr, result.obb.xyxyxyxy):
            pixels_x, pixels_y = map(list, zip(*obb))
            latlong_list = [gdal.ApplyGeoTransform(forward_transform, float(x), float(y))
                            for x, y in zip(pixels_x, pixels_y)]
            poly = geometry.Polygon([[lon, lat] for lon, lat in latlong_list])
            detections_list.append({'class_id': int(cls.item()),
                                    'conf': conf.item(), 'geometry': poly})
    gdf_preds = gpd.GeoDataFrame(pd.DataFrame.from_dict(detections_list))
    gdf_preds.set_crs(crs_wkt, inplace=True)
    return gdf_preds


def clip_by_raster_bounding_box(gdf, input_raster):
    import rasterio
    from shapely.geometry import box
    with rasterio.open(input_raster) as rast_src:
        bbox = rast_src.bounds
        crs = rast_src.crs
    bbox_polygon = box(bbox.left, bbox.bottom, bbox.right, bbox.top)
    gdf = gdf.to_crs(crs)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox_polygon]}, crs=crs)
    return gpd.clip(gdf, bbox_gdf)


def predict_rwy_potential(model_obb, input_raster, out_path, device, gen_int=True):
    import rasterio
    analysis_folder = Path(out_path) / Path(input_raster).stem
    analysis_folder.mkdir(parents=True, exist_ok=True)
    # Drop the 4th (alpha) band so RGBA rasters are handled like RGB
    with rasterio.open(input_raster) as rsrc:
        bands = rsrc.read()                       # (C, H, W)
    rgb = np.transpose(bands[:3], (1, 2, 0))      # (H, W, 3), keep first 3 bands
    rgb = rgb[:, :, ::-1]                          # RGB -> BGR for ultralytics
    pred_src = np.ascontiguousarray(rgb)
    results = model_obb.predict(pred_src, save=True, save_txt=True, save_conf=True,
                                device=device, max_det=10, show_boxes=False)
    obb_gdf = obb_predictions_to_geodf(input_raster, results)
    obb_gdf = retain_larger_and_non_overlapping_polygons(obb_gdf, 25)
    medial_axis_gdf = calculate_medial_axis(obb_gdf)
    extended_gdf = extend_medial_axis_to_bounds(medial_axis_gdf, input_raster)
    buffered_gdf = buffer_extended_gdf(extended_gdf, obb_gdf, input_raster)
    buffered_gdf_clipped = clip_by_raster_bounding_box(buffered_gdf, input_raster)
    out_image, out_meta = mask_raster_with_all_geometries(input_raster, buffered_gdf)
    if gen_int:
        obb_gdf.to_file(analysis_folder / 'obb.geojson', driver='GeoJSON')
        medial_axis_gdf.to_file(analysis_folder / 'medial_line.geojson', driver='GeoJSON')
        extended_gdf.to_file(analysis_folder / 'coarse_rwy_dir.geojson', driver='GeoJSON')
        buffered_gdf.to_file(analysis_folder / 'coarse_rwy_dir_poly.geojson', driver='GeoJSON')
        buffered_gdf_clipped.to_file(analysis_folder / 'coarse_rwy_dir_poly_clipped.geojson', driver='GeoJSON')
        with rasterio.open(analysis_folder / 'runways_mask.png', 'w', **out_meta) as dest:
            dest.write(out_image)
    del obb_gdf, medial_axis_gdf, extended_gdf, buffered_gdf


def make_potential_rwys_horizontal(masks_path, icao):
    from osgeo import gdal
    masks_path = Path(masks_path)
    src_raster = masks_path / icao / 'runways_mask.png'
    src_vector = masks_path / icao / 'coarse_rwy_dir_poly_clipped.geojson'
    src_ds = gdal.Open(str(src_raster))
    gdf = gpd.read_file(src_vector)
    for index, row in gdf.iterrows():
        rotation_angle = abs(int(row['rotation_angle']))
        base_name = src_raster.stem
        rotated_raster_path = masks_path / icao / f"rotated_{base_name}_{index}.tif"
        rotated_vector_path = masks_path / icao / f"horizontal_rwy_cb_{index}.geojson"
        clipped_raster_path = masks_path / icao / f"horizontal_rwy_{index}.tif"
        rotate_image(str(src_raster), str(rotated_raster_path), rotation_angle)
        pivot = get_center(src_ds)
        rotated_gdf = rotate_and_store_gdf(gdf.iloc[[index]], -rotation_angle, pivot)
        rotated_gdf = rotated_gdf.set_crs(gdf.crs)
        rotated_gdf.to_file(rotated_vector_path, driver='GeoJSON')
        gdal_clip_raster_with_vector(str(rotated_raster_path), str(rotated_vector_path), str(clipped_raster_path))
    del src_ds, gdf


def yolov8_bb_predictions_to_geodf(src_raster, results):
    from osgeo import gdal
    src = gdal.Open(src_raster)
    ulx, xres, xskew, uly, yskew, yres = src.GetGeoTransform()
    crs_wkt = src.GetProjection()
    forward_transform = (float(ulx), float(xres), float(xskew),
                         float(uly), float(yskew), float(yres))
    del src
    result = results[0]
    detections_list = []
    names = result.names
    if result.boxes:
        for cls, conf, bbox in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
            x1, y1, x2, y2 = bbox.tolist()
            corners = [gdal.ApplyGeoTransform(forward_transform, float(a), float(b))
                       for a, b in [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]]
            poly = geometry.Polygon([[lon, lat] for lon, lat in corners])
            detections_list.append({'class_id': int(cls.item()),
                                    'class_name': names[int(cls.item())],
                                    'conf': conf.item(), 'geometry': poly})
    gdf_preds = gpd.GeoDataFrame(pd.DataFrame.from_dict(detections_list))
    gdf_preds.set_crs(crs_wkt, inplace=True)
    return gdf_preds


def process_rwythr_polygons(gdf_unrotated):
    gdf_rwythr = gdf_unrotated[gdf_unrotated['class_name'] == 'rwythr']
    gdf_desig = gdf_unrotated[gdf_unrotated['class_name'] == 'desig']
    rc, dc = len(gdf_rwythr), len(gdf_desig)
    if rc == 0 and dc == 0:
        return None
    elif rc == 1 and dc >= 1:
        gdf_combined = gpd.GeoDataFrame(pd.concat([gdf_rwythr, gdf_desig], ignore_index=True))
    elif rc == 0 and dc >= 2:
        gdf_combined = gdf_desig
    elif rc == 2 and dc == 2:
        gdf_combined = gdf_rwythr
    elif rc == 2 and dc == 0:
        gdf_combined = gdf_rwythr
    elif rc >= 2 and dc >= 2:
        gdf_combined = gdf_rwythr
    else:
        return None
    return gpd.GeoDataFrame(geometry=gdf_combined.dissolve().convex_hull, crs=gdf_unrotated.crs)


def calculate_runway_id(polygon):
    import math
    import geomag
    from shapely.geometry import LineString
    bbox_coords = list(polygon.minimum_rotated_rectangle.exterior.coords)
    longest_edge, max_length = None, 0
    for i in range(4):
        edge = LineString([bbox_coords[i], bbox_coords[i + 1]])
        if edge.length > max_length:
            max_length, longest_edge = edge.length, edge
    p1, p2 = longest_edge.coords
    azimuth = math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))
    compass = (azimuth + 360) % 360
    var = geomag.declination((p1[1] + p2[1]) / 2, (p1[0] + p2[0]) / 2)
    mag = (compass + var) % 360
    n1 = round(mag / 10) % 36 or 36
    n2 = (n1 + 18) % 36 or 36
    return f"{n1:02}-{n2:02}"


def predict_rwy_markings_using_rtdetr(model, masks_path, icao, device, gen_int=True):
    import rasterio
    analysis_folder = Path(masks_path) / icao
    rwys = sorted(glob.glob(os.path.join(analysis_folder, 'horizontal_rwy*.tif')))
    cbs = sorted(glob.glob(os.path.join(analysis_folder, 'horizontal_rwy_cb*.geojson')))
    pivot = get_pivot(analysis_folder)
    gdf_list, gdf_list_rwy = [], []
    for image_file, cb_path in zip(rwys, cbs):
        rotation_angle = get_unrotate_angle(cb_path)
        with rasterio.open(image_file) as crsrc:
            crop = crsrc.read()
        crop_rgb = np.ascontiguousarray(np.transpose(crop[:3], (1, 2, 0))[:, :, ::-1])
        results = model(crop_rgb, save=True, save_txt=True, save_conf=True, device=device, show_boxes=True)
        if not results[0].boxes:
            continue
        gdf_h = yolov8_bb_predictions_to_geodf(image_file, results)
        if gdf_h is None or gdf_h.empty:
            continue
        gdf_un = recover_from_rotation(gdf_h, rotation_angle, pivot)
        gdf_list.append(gdf_un)
        rwy = process_rwythr_polygons(gdf_un)
        if rwy is not None:
            gdf_list_rwy.append(rwy)
        if gen_int:
            gdf_h.to_file(analysis_folder / (os.path.basename(image_file).split('.')[0] + '.geojson'), driver='GeoJSON')
    if gdf_list:
        gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True)).to_file(
            analysis_folder / 'predicted_rwy_markings_bb_rtdetr.geojson', driver='GeoJSON')
    if gdf_list_rwy:
        rwy_gdf = gpd.GeoDataFrame(pd.concat(gdf_list_rwy, ignore_index=True))
        rwy_gdf['runway_id'] = rwy_gdf['geometry'].apply(calculate_runway_id)
        rwy_gdf.to_file(analysis_folder / 'predicted_rwy.geojson', driver='GeoJSON')


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
def check_iou(predicted_gdf, ground_truth_gdf, thresh):
    results = []
    for icao in predicted_gdf['icao'].unique():
        preds = predicted_gdf[predicted_gdf['icao'] == icao]
        gts = ground_truth_gdf[ground_truth_gdf['icao'] == icao]
        for ip, pr in preds.iterrows():
            for ig, gr in gts.iterrows():
                inter = pr.geometry.intersection(gr.geometry)
                if not inter.is_empty:
                    iou = inter.area / pr.geometry.union(gr.geometry).area
                    if iou > thresh:
                        results.append({'icao': icao, 'iou': iou})
    return results


def merge_predictions(out_folder):
    gdf = gpd.GeoDataFrame()
    for f in glob.glob(f"{out_folder}/**/predicted_rwy.geojson", recursive=True):
        t = gpd.read_file(f).to_crs(epsg=4326)
        t['icao'] = os.path.basename(os.path.dirname(f))
        gdf = gdf._append(t, ignore_index=True)
    return gdf


# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────
def run_category(model_obb, model_detr, cat, cfg):
    out = f"{cfg['out_prefix']}_{cat}"
    cat_dir = Path(cfg['rasters_root']) / cat
    rasters = sorted(str(p) for ext in RASTER_EXTS for p in cat_dir.glob(ext))
    times = []
    for r in rasters:
        icao = Path(r).stem
        t0 = time.time()
        try:
            predict_rwy_potential(model_obb, r, out, cfg['device'])
            make_potential_rwys_horizontal(out, icao)
            predict_rwy_markings_using_rtdetr(model_detr, out, icao, cfg['device'])
            status = 'ok'
        except Exception as e:
            print(f"  skip {icao}: {e}")
            status = 'failed'
        times.append({'category': cat, 'icao': icao, 'inf_time_s': time.time() - t0, 'status': status})
    return rasters, pd.DataFrame(times), out


def main():
    parser = argparse.ArgumentParser(description='AERO-DETR reproduction inference + Table 3')
    parser.add_argument('--categories', nargs='+', default=None,
                        help='subset of: single parallel inter mixed complex')
    parser.add_argument('--rasters_root', type=str, default=None,
                        help='root folder holding <category>/*.jp2 (or *.tif); default: rasters')
    parser.add_argument('--gt', type=str, default=None)
    parser.add_argument('--out_prefix', type=str, default=None)
    parser.add_argument('--models_dir', type=str, default=None)
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='if set, repeats each config per seed for variability')
    args = parser.parse_args()

    cfg = GLOBAL_CONFIG.copy()
    if args.rasters_root: cfg['rasters_root'] = args.rasters_root
    if args.gt:         cfg['gt'] = args.gt
    if args.out_prefix: cfg['out_prefix'] = args.out_prefix
    if args.models_dir:
        cfg['obb_model'] = str(Path(args.models_dir) / Path(cfg['obb_model']).name)
        cfg['detr_model'] = str(Path(args.models_dir) / Path(cfg['detr_model']).name)
    cats = args.categories or CATEGORIES
    seeds = args.seeds or [None]

    print('=' * 60)
    print('  AERO-DETR Reproduction')
    print(f"  Device     : {cfg['device']}   Categories: {cats}   Seeds: {seeds}")
    print('=' * 60)

    model_obb = YOLO(cfg['obb_model'])
    model_detr = RTDETR(cfg['detr_model'])
    gt = gpd.read_file(cfg['gt'])

    summary, var_rows = [], []
    for seed in seeds:
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            cfg['out_prefix'] = f"reproduce_seed{seed}_output"
        all_times = []
        for cat in cats:
            rasters, times, out = run_category(model_obb, model_detr, cat, cfg)
            all_times.append(times)
            pred = merge_predictions(out)
            df = pd.DataFrame(check_iou(pred, gt[gt['icao'].isin(pred['icao'].unique())], cfg['iou_thresh']))
            apt = df.groupby('icao')['iou'].mean() if not df.empty else pd.Series(dtype=float)
            row = {'Category': cat, 'Num_Airports': len(rasters),
                   'IoU_Mean': apt.mean(), 'Num_Airports_IoU_80': apt.shape[0],
                   'Avg_Inf_Time_s': times['inf_time_s'].mean()}
            (var_rows if seed is not None else summary).append({**row, 'seed': seed})
            print(f"seed={seed} {cat}: IoU={apt.mean():.4f} N>0.8={apt.shape[0]}")
        Path('metrics').mkdir(exist_ok=True)
        pd.concat(all_times).to_csv(f"metrics/{cfg['out_prefix']}_times.csv", index=False)

    Path('metrics').mkdir(exist_ok=True)
    out_df = pd.DataFrame(var_rows if seeds != [None] else summary)
    out_df.to_csv('metrics/reproduce_table3.csv', index=False)
    print(out_df.to_string(index=False))
    print(f"\nAverage IoU: {out_df['IoU_Mean'].mean():.4f}")


if __name__ == '__main__':
    main()
