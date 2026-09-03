"""One-off: per-chip quadrant stats from gt_seeps.gpkg.

Quadrants split at the chip's median centroid x/y. n_large counts polygons
with area above the historical B-class median (471 cm^2) as a likely-B/C
proxy; n_pregrouped would need the label packs, so large area stands in.
"""
import os

import geopandas as gpd
import numpy as np

PRED_DIR = os.path.join("data", "results", "SWIN", "AE", "20260428-1537_SWINxAE.weights")
CHIPS = ["39.tif", "4.tif", "38.tif", "41.tif", "21.tif", "27.tif", "52.tif", "25.tif"]
LARGE_M2 = 0.0471  # historical B-class median area

gt = gpd.read_file(os.path.join(PRED_DIR, "gt_seeps.gpkg"))

print(f"{'chip':>8} {'quad':>4} {'n':>5} {'n_large':>8}")
for chip in CHIPS:
    g = gt[gt["image"] == chip]
    mx, my = g["centroid_x_m"].median(), g["centroid_y_m"].median()
    for name, m in [
        ("NW", (g.centroid_x_m < mx) & (g.centroid_y_m >= my)),
        ("NE", (g.centroid_x_m >= mx) & (g.centroid_y_m >= my)),
        ("SW", (g.centroid_x_m < mx) & (g.centroid_y_m < my)),
        ("SE", (g.centroid_x_m >= mx) & (g.centroid_y_m < my)),
    ]:
        q = g[m]
        print(f"{chip:>8} {name:>4} {len(q):>5} {(q['area_m2'] > LARGE_M2).sum():>8}")
    print()
