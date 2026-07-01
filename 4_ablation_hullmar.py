"""Offline trial: hull->MAR snap + padding sweep. Does it raise IoU vs base?
Reads base predicted_rwy.geojson (control, untouched); writes metrics/ablation_hullmar.csv.
No GPU re-run. Base = 2_run_inference.py outputs remain the reference."""
import glob, os
import numpy as np
import pandas as pd
import geopandas as gpd

CATS = ['single', 'parallel', 'inter', 'mixed', 'complex']
THRESH = 0.8
PADS_M = [0, -2, -5, 2, 5]                 # metre buffers in EPSG:3857 to sweep
gt = gpd.read_file('ground_truth/all_gt.geojson')

def load_base(cat):
    rows = []
    for f in glob.glob(f'reproduce_output_{cat}/*/predicted_rwy.geojson'):
        g = gpd.read_file(f).to_crs(4326); g['icao'] = os.path.basename(os.path.dirname(f))
        rows.append(g)
    return gpd.GeoDataFrame(pd.concat(rows, ignore_index=True)) if rows else None

def mar_pad(gdf, pad_m):
    g = gdf.to_crs(3857).copy()
    geoms = [p.minimum_rotated_rectangle for p in g.geometry]
    if pad_m: geoms = [p.buffer(pad_m).minimum_rotated_rectangle for p in geoms]
    g['geometry'] = geoms
    return g.to_crs(4326)

def score(pred):
    res = []
    for icao in pred['icao'].unique():
        ps, gs = pred[pred.icao==icao], gt[gt.icao==icao]
        for _, p in ps.iterrows():
            for _, q in gs.iterrows():
                it = p.geometry.intersection(q.geometry)
                if not it.is_empty:
                    iou = it.area / p.geometry.union(q.geometry).area
                    if iou > THRESH: res.append({'icao': icao, 'iou': iou})
    a = pd.DataFrame(res).groupby('icao')['iou'].mean() if res else pd.Series(dtype=float)
    return a.mean(), a.shape[0]

rows = []
for cat in CATS:
    base = load_base(cat)
    if base is None: continue
    bm, bn = score(base)
    rows.append({'cat': cat, 'variant': 'base_hull', 'IoU': round(bm,4), 'N>0.8': bn})
    for pad in PADS_M:
        m, n = score(mar_pad(base, pad))
        rows.append({'cat': cat, 'variant': f'MAR_pad{pad}', 'IoU': round(m,4), 'N>0.8': n})

df = pd.DataFrame(rows)
os.makedirs('metrics', exist_ok=True)
df.to_csv('metrics/ablation_hullmar.csv', index=False)
piv = df.pivot(index='cat', columns='variant', values='IoU')
print(piv.to_string())
print('\nbase avg :', round(df[df.variant=="base_hull"].IoU.mean(),4))
for pad in PADS_M:
    print(f'MAR_pad{pad} avg:', round(df[df.variant==f"MAR_pad{pad}"].IoU.mean(),4))
