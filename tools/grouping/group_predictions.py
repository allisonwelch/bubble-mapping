"""Run the trained grouper on MODEL-PREDICTED bubbles (not GT labels) to see how
grouping looks at serve time.

Train/serve gap to keep in mind: the grouper is trained on hand-drawn GT bubble
polygons, but here it is applied to the detector's connected-component bubbles
(`bubble_features.csv` / `{stem}_cc.tif`), which differ from GT by
fragmentation, missed/false bubbles, feature shift, and -- critically -- have NO
envelope (is_pregrouped) concept. The grouping LOGIC is a learned distance-join
(dist dominates), which transfers reasonably; the INPUT bubble set is the
F1~0.645 detector's, so garbage-in applies.

Outputs (to the pred_dir):
  * console: per-chip predicted-bubble count, predicted-seep count, multi-bubble
    seeps, hull major-axis distribution + over-cap (should be 0 with the cap).
  * pred_bubbles_grouped.gpkg: predicted bubble polygons (from {stem}_cc.tif)
    with the grouper's seep_group_id + the group_hull_by_class style, so the
    grouping can be eyeballed in QGIS over the imagery. (No class on predictions,
    so hull fill is transparent -- you see the seep OUTLINES.)
"""
import os
import sys
import math
import sqlite3
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from tools.grouping.deploy_grouper import (
    train_model, _pair_features, constrained_cluster, build_qml, inject_style)
from tools.grouping.train_grouper import FEATURES, AGGLOM_CAP_M
from tools.eval.bubble_features import polygonize_labels

PD = "data/results/SWIN/AE/20260428-1537_SWINxAE.weights"
PRED_BUBBLES = PD + "/bubble_features.csv"
OUT_GPKG = PD + "/pred_bubbles_grouped.gpkg"
THR = 0.6


def group_predictions(clf, thr=THR, cap=AGGLOM_CAP_M):
    b = pd.read_csv(PRED_BUBBLES)
    b["seep_group_id"] = b["bubble_id"].astype(np.int64)
    print(f"loaded {len(b)} predicted bubbles across {b['image'].nunique()} chips")
    print(f"\n{'chip':>9} {'bubbles':>8} {'seeps':>6} {'multi':>6} "
          f"{'maj_p50':>8} {'maj_p95':>8} {'maj_max':>8} {'>cap':>5}")
    out = {}
    for im, sub in b.groupby("image"):
        sub = sub.reset_index(drop=True)
        fx = sub["centroid_x_m"].to_numpy(float)
        fy = sub["centroid_y_m"].to_numpy(float)
        fa = sub["area_m2"].to_numpy(float)
        pp, feat = _pair_features(sub, fx, fy, fa)
        sgid = sub["bubble_id"].to_numpy(np.int64).copy()
        if len(pp):
            proba = clf.predict_proba(feat[FEATURES].to_numpy(float))[:, 1]
            keep = proba >= thr
            xy = np.column_stack([fx, fy])
            comp = constrained_cluster(len(sub), pp[keep], proba[keep], xy, cap)
            ids = sub["bubble_id"].to_numpy(np.int64)
            for c in np.unique(comp):
                m = np.where(comp == c)[0]
                sgid[m] = int(ids[m].max())
        sub["seep_group_id"] = sgid
        out[im] = sub
    return out


def chip_geometry(im, sub):
    """Polygonize {stem}_cc.tif, attach seep_group_id by bubble_id."""
    stem = im[:-4] if im.endswith(".tif") else im
    cc_fp = f"{PD}/{stem}_cc.tif"
    if not os.path.exists(cc_fp):
        return None
    with rasterio.open(cc_fp) as ds:
        arr = ds.read(1)
        polys = polygonize_labels(arr, ds.transform, ds.crs)
    polys = polys.rename(columns={"id": "bubble_id"})
    g = polys.merge(sub[["bubble_id", "seep_group_id", "area_m2"]],
                    on="bubble_id", how="left")
    g["image"] = im
    return g


def major_axis_stats(g):
    from shapely.ops import unary_union
    majors = []
    for gid, grp in g.groupby("seep_group_id"):
        if len(grp) < 2:
            continue
        mrr = unary_union(list(grp.geometry.values)).minimum_rotated_rectangle
        xs, ys = mrr.exterior.coords.xy
        edges = [math.dist((xs[i], ys[i]), (xs[i + 1], ys[i + 1])) for i in range(4)]
        majors.append(max(edges))
    return np.array(majors)


def main():
    clf = train_model()
    grouped = group_predictions(clf)
    all_g = []
    for im, sub in grouped.items():
        g = chip_geometry(im, sub)
        n_seeps = sub["seep_group_id"].nunique()
        n_multi = int((sub.groupby("seep_group_id").size() > 1).sum())
        if g is not None and len(g):
            maj = major_axis_stats(g)
            p50 = np.percentile(maj, 50) if len(maj) else 0
            p95 = np.percentile(maj, 95) if len(maj) else 0
            mx = maj.max() if len(maj) else 0
            over = int((maj > AGGLOM_CAP_M).sum()) if len(maj) else 0
            all_g.append(g)
        else:
            p50 = p95 = mx = over = 0
        print(f"{im:>9} {len(sub):>8} {n_seeps:>6} {n_multi:>6} "
              f"{p50:>8.3f} {p95:>8.3f} {mx:>8.3f} {over:>5}")

    if all_g:
        full = gpd.GeoDataFrame(pd.concat(all_g, ignore_index=True),
                                geometry="geometry", crs=all_g[0].crs)
        # add empty class column so the hull-by-class style finds it
        full["class"] = ""
        if os.path.exists(OUT_GPKG):
            os.remove(OUT_GPKG)
        full.to_file(OUT_GPKG, layer="labels", driver="GPKG")
        inject_style(OUT_GPKG, build_qml())
        print(f"\nwrote {OUT_GPKG} ({len(full)} bubble polygons) + hull style")
        print("open in QGIS over the chip imagery; seep outlines = grouper on "
              "PREDICTED bubbles (fill transparent -- predictions have no class)")


if __name__ == "__main__":
    main()