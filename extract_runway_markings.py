'''
Extracts the runway markings from a  Runway that is Horizontally rotated in previous step
'''

import os
import geopandas as gpd
from pathlib import Path
import glob
import numpy as np
import ntpath
import argparse

import cv2
import re
import rasterio
from shapely.geometry import Polygon
from tqdm import tqdm
from osgeo import gdal

import torch

from geo_utils import *
from geo_utils import get_center


'''
import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo


def get_rwymarkings_horizontal_model(rwymarkings_horizontal_model_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")) 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7  
    cfg.MODEL.WEIGHTS = rwymarkings_horizontal_model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.50
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)
          
    DatasetCatalog.clear()
    MetadataCatalog.clear()

    dataset_name = "airport"
    metadata = MetadataCatalog.get(dataset_name)
    MetadataCatalog.get(dataset_name).thing_classes = ['aim', 'rwythr', 'desig', 'disp', 'tdz', 'chevron', 'arrow']

    class_names = {0: 'aim', 1: 'rwythr', 2: 'desig', 3: 'disp', 4: 'tdz', 5: 'chevron', 6: 'arrow'}
    
    return predictor,class_names

def decode_detectron2_pred(src_file, outputs, class_names):
    boxes = []
    scores = []
    pred_classes = []
    class_ids = []
    class_names_list = []

    for box in outputs["instances"].pred_boxes.tensor.cpu().numpy():
        boxes.append(box.tolist())

    for score in outputs["instances"].scores.to('cpu').numpy():
        scores.append(score.tolist())

    for pred_class in outputs["instances"].pred_classes.to('cpu').numpy():
        pred_classes.append(pred_class.tolist())
        class_ids.append(pred_class)
        class_names_list.append(class_names[pred_class])

    coco_results = {
        'file_name': ntpath.basename(src_file),
        'boxes': boxes,
        'scores': scores,
        'pred_classes': pred_classes,
        'class_ids': class_ids,
        'class_names': class_names_list
    }

    return coco_results

def decode_detectron2_pred_v1(src_file, outputs, class_names):
    boxes = []
    scores = []
    pred_classes = []
    class_ids = []
    class_names_list = []

    for box in outputs["instances"].pred_boxes.tensor.cpu().numpy():
        boxes.append(box.tolist())

    for score in outputs["instances"].scores.to('cpu').numpy():
        scores.append(score.tolist())

    for pred_class in outputs["instances"].pred_classes.to('cpu').numpy():
        pred_classes.append(pred_class.tolist())
        class_ids.append(pred_class)
        class_names_list.append(class_names[pred_class])

    coco_results = {
        'file_name': ntpath.basename(src_file),
        'boxes': boxes,
        'scores': scores,
        'pred_classes': pred_classes,
        'class_ids': class_ids,
        'class_names': class_names_list
    }

    return coco_results

def pred2shp(image_file, predictor, class_names):
    gdf = None
    row_list = []
    im = cv2.imread(image_file)
    height, width, channels = im.shape
    outputs = predictor(im)

    #print(f"Predictions for {image_file}: {outputs}")  # Debug statement

    coco_results = decode_detectron2_pred(image_file, outputs, class_names)
    boxes_list = coco_results.get('boxes', [])
    scores_list = coco_results.get('scores', [])
    class_ids_list = coco_results.get('class_ids', [])
    class_names_list = coco_results.get('class_names', [])

    if boxes_list:
        with rasterio.open(image_file) as src:
            for box, score, class_id, class_name in zip(boxes_list, scores_list, class_ids_list, class_names_list):
                min_x, min_y, max_x, max_y = box
                corners = [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y),
                    (min_x, min_y)  # Closing the polygon
                ]

                lons, lats = rasterio.transform.xy(src.transform, np.array([pt[1] for pt in corners]), np.array([pt[0] for pt in corners]))
                lons, lats = np.array(lons), np.array(lats)

                vertices_list = [(lon, lat) for lon, lat in zip(lons, lats)]
                polygon_geom = Polygon(vertices_list)
                
                geom_dict = {
                    'file_name': coco_results['file_name'],
                    'pred_classes': class_id,
                    'class_name': class_name,
                    'pred_score': score,
                    'geometry': polygon_geom
                }
                row_list.append(geom_dict)

            gdf = gpd.GeoDataFrame(row_list, crs=src.crs)
            #print(f"Generated GeoDataFrame: {gdf}")  # Debug statement

    del im
    return gdf
    
'''    

def recover_from_rotation(gdf_horizontal,rotation_angle,pivot):
    # Rotate the current feature in the vector file and save it
    gdf_unrotated = rotate_and_store_gdf(gdf_horizontal, rotation_angle, pivot)
    
    # Reproject the GeoDataFrame to a simple Proj4 string format before saving
    #gdf_unrotated = gdf_unrotated.to_crs("+proj=longlat +datum=WGS84 +no_defs")
    gdf_unrotated = gdf_unrotated.set_crs(gdf_horizontal.crs)

    return gdf_unrotated

def get_unrotate_angle(cb_path):
    rotation_angle=0
    gdf = gpd.read_file(cb_path)
    rotation_angle = abs(int(gdf.iloc[0]['rotation_angle']))
    del gdf
    return rotation_angle    
    
def get_pivot(analysis_folder):
    src_raster = analysis_folder / f"runways_mask.png"
    # Open the source raster and vector files
    src_ds = gdal.Open(str(src_raster))
    pivot = get_center(src_ds)

    del  src_ds
    return pivot

def predict_rwy_markings(predictor, masks_path, icao, class_names, gen_int=True):
    analysis_folder = Path(masks_path) / icao
    horizontal_rwys_path = glob.glob(os.path.join(analysis_folder, 'horizontal_rwy*.tif'))
    horizontal_rwys_cb_path=glob.glob(os.path.join(analysis_folder, 'horizontal_rwy_cb*.geojson'))

    # Sort the lists of file paths
    horizontal_rwys_path.sort()
    horizontal_rwys_cb_path.sort()

    pivot=get_pivot(analysis_folder)
    
    gdf_list = []  # List to collect individual GeoDataFrames
    
    # Use zip to pair up the sorted file paths
    for image_file, cb_path in zip(horizontal_rwys_path, horizontal_rwys_cb_path):
        print(f"Processing {image_file} and {cb_path}")
        rotation_angle=get_unrotate_angle(cb_path)
        gdf_horizontal = pred2shp(image_file, predictor, class_names)
        if gdf_horizontal is not None and not gdf_horizontal.empty:
            gdf_unrotated=recover_from_rotation(gdf_horizontal,rotation_angle,pivot)
            gdf_list.append(gdf_unrotated)
            if gen_int:
                predicted_geojson = os.path.basename(image_file).split('.')[0] + '.geojson'
                predicted_geojson_path = analysis_folder / predicted_geojson
                gdf_horizontal.to_file(predicted_geojson_path, driver='GeoJSON')
        else:
            print(f"No valid polygons found for {image_file}")

    # Combine all GeoDataFrames in gdf_list into a single GeoDataFrame
    if gdf_list:
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        #print(f"Combined GeoDataFrame: {combined_gdf}")
        combined_geojson_path = analysis_folder / "predcited_rwy_markings_bb.geojson"
        combined_gdf.to_file(combined_geojson_path, driver='GeoJSON')
    else:
        print("No valid GeoDataFrames to combine.")

    
'''
usage: 
python extract_runway_makrings.py input_raster out_path
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts the runway markings from a  Runway that is Horizontally rotated")
    parser.add_argument("masks_folder", help="Processed masks folder")
    parser.add_argument("icao", help="ICAO name")
    args = parser.parse_args()

    print("Parsing arguments")
    rwymarkings_horizontal_model_path = r"rwymarkings_horizontal_best_FasterRCNN_Detectron2.pth"
    model_rwymarkings_horizontal = get_rwymarkings_horizontal_model(rwymarkings_horizontal_model_path)
    print("Loaded Faster RCNN model")

    class_names = {0: 'aim', 1: 'rwythr', 2: 'desig', 3: 'disp', 4: 'tdz', 5: 'chevron', 6: 'arrow'}
        
    print("Started Predicting Bounding Boxes of Runway Markings")
    predict_rwy_markings(model_rwymarkings_horizontal, args.masks_folder, args.icao,class_names)
    print("Completed predicting Runway Marking Bounding Boxes")

    del model_rwymarkings_horizontal