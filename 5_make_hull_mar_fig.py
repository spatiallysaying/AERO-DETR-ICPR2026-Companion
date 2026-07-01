"""Make a hull-vs-MAR+pad visual from a real prediction. Saves to companion_paper/figures/models/hull_vs_mar.png"""
import glob, os
import geopandas as gpd
import matplotlib.pyplot as plt

THRESH = 0.8
gt = gpd.read_file('ground_truth/all_gt.geojson')
# pick an inter airport with a base hull prediction
icao = 'KBUF'
mk = f'reproduce_output_inter/{icao}/predicted_rwy.geojson'
hull = gpd.read_file(mk).to_crs(3857)
mar = hull.copy(); mar['geometry'] = [p.buffer(2).minimum_rotated_rectangle for p in hull.geometry]
g = gt[gt.icao == icao].to_crs(3857)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for a, poly, title in [(ax[0], hull, 'Base: convex hull'), (ax[1], mar, 'MAR + 2 m pad')]:
    g.boundary.plot(ax=a, color='black', lw=1.5, label='ground truth')
    poly.boundary.plot(ax=a, color='red', lw=1.5, label='prediction')
    a.set_title(title); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=7)
plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/hull_vs_mar.png', dpi=150, bbox_inches='tight')
print('saved', icao)
