"""Offline ablation: do the return-None fallback and 1:1 matching raise IoU?
Reuses existing reproduce_output_inter outputs; no GPU inference."""
import glob, os
import pandas as pd
import geopandas as gpd

CAT = 'inter'
THRESH = 0.8
gt = gpd.read_file('ground_truth/all_gt.geojson')

def airport_dirs():
    return [os.path.dirname(f) for f in glob.glob(f'reproduce_output_{CAT}/*/')]

# Build predicted runways: baseline = predicted_rwy.geojson; fallback = convex hull of all markings if missing
base_rows, fb_rows = [], []
for d in glob.glob(f'reproduce_output_{CAT}/*'):
    icao = os.path.basename(d)
    pr = os.path.join(d, 'predicted_rwy.geojson')
    mk = os.path.join(d, 'predicted_rwy_markings_bb_rtdetr.geojson')
    if os.path.exists(pr):
        g = gpd.read_file(pr).to_crs(4326); g['icao'] = icao
        base_rows.append(g); fb_rows.append(g)
    elif os.path.exists(mk):                       # fallback: emit hull of all markings
        m = gpd.read_file(mk).to_crs(4326)
        hull = gpd.GeoDataFrame(geometry=[m.dissolve().convex_hull.iloc[0]], crs=4326)
        hull['icao'] = icao
        fb_rows.append(hull)

base = gpd.GeoDataFrame(pd.concat(base_rows, ignore_index=True))
fb = gpd.GeoDataFrame(pd.concat(fb_rows, ignore_index=True))

def eval_allpairs(pred):
    res = []
    for icao in pred['icao'].unique():
        ps, gs = pred[pred.icao==icao], gt[gt.icao==icao]
        for _, p in ps.iterrows():
            for _, q in gs.iterrows():
                it = p.geometry.intersection(q.geometry)
                if not it.is_empty:
                    iou = it.area / p.geometry.union(q.geometry).area
                    if iou > THRESH: res.append({'icao':icao,'iou':iou})
    a = pd.DataFrame(res).groupby('icao')['iou'].mean() if res else pd.Series(dtype=float)
    return a.mean(), a.shape[0]

def eval_onetoone(pred):
    res = []
    for icao in pred['icao'].unique():
        ps, gs = pred[pred.icao==icao], gt[gt.icao==icao]
        pairs = []
        for ip,p in ps.iterrows():
            for ig,q in gs.iterrows():
                it = p.geometry.intersection(q.geometry)
                if not it.is_empty:
                    pairs.append((it.area/p.geometry.union(q.geometry).area, ip, ig))
        pairs.sort(reverse=True); up, ug = set(), set()
        for iou,ip,ig in pairs:
            if ip in up or ig in ug: continue
            up.add(ip); ug.add(ig)
            if iou > THRESH: res.append({'icao':icao,'iou':iou})
    a = pd.DataFrame(res).groupby('icao')['iou'].mean() if res else pd.Series(dtype=float)
    return a.mean(), a.shape[0]

bm,bn = eval_allpairs(base)
fm,fn = eval_allpairs(fb)
om,on = eval_onetoone(base)
print(f"baseline (all-pairs):      IoU={bm:.4f}  N>0.8={bn}  airports={base.icao.nunique()}")
print(f"+fallback (all-pairs):     IoU={fm:.4f}  N>0.8={fn}  airports={fb.icao.nunique()}")
print(f"baseline (1:1 matching):   IoU={om:.4f}  N>0.8={on}")
