"""ARCHIVED 2026-09-03 -- the seep-level (polygon-to-polygon) IoU matcher.

Lifted verbatim out of tools/seep_level_eval.py when the defunct anchor+lonely
clustering stage was deleted. It is NOT stale code: it is the multi-truth
matcher documented in CLAUDE.md 2026-05-14, and it is the piece a future
seep-level evaluation will need. It is parked here only because, with the
hand-tuned clustering rule gone, nothing produces predicted SEEP polygons yet.

WHY THE OLD SEEP-LEVEL METRIC WAS RETIRED
`cluster_f1 = 0.672` came from running this matcher with gt_grouping_mode="auto",
which grouped the ground-truth AND predicted sides with the SAME hand-tuned
rule. Grouping error therefore cancelled out on both sides and the number was a
detection metric wearing a seep-level label.

HOW TO REVIVE IT HONESTLY (the RF-grouper version)
    GT side   : human-grouped bubbles, dissolved by (image, seep_group_id)
    pred side : {stem}_cc.tif bubbles -> tools/seep_grouper_on_predictions.py
                -> dissolved by the predicted seep group
    match     : this function, pred_id="seep_group_id", gt_id="seep_group_id"
That version compounds detector + grouper error, which is what deployment
actually is. Caveat: human grouping on the eval chips exists only for the
quarter units and chip 39 (gt_seeps_label_eval_chips_grouped.gpkg is
model-proposed, 0 rows classed), so it is a labeled-SUBSET metric, not a
full-test-set one. State that limitation wherever the number is quoted.
"""
import numpy as np

try:
    import geopandas as gpd  # noqa: F401
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False


# 3b. Match pred CLUSTER polygons to GT polygons by shapely IoU. Multi-truth:
#     one pred cluster can match MULTIPLE GT polygons. Required because the
#     pred-side anchor-conditional clustering sometimes bridges across GT
#     polygon boundaries -- one-to-one matching converted those into
#     artificial FNs once GT was de-merged from rasterization. Each GT scores
#     its own TP off the best-IoU pred; a pred is FP only when no GT picked
#     it. Caveat for downstream Stage-3 classification: a pred cluster that
#     spans multiple GT seeps still represents a real over-merging problem
#     even though it counts as multiple TPs here -- flag in Limitations.
def match_polygons(pred_gdf, gt_gdf, pred_id="cluster_id", gt_id="seep_id",
                   iou_thresh=0.1):
    matches = []  # list of (pred_id, gt_id); pred_id may repeat
    matched_gt = set()
    matched_pred = set()
    if pred_gdf is None or gt_gdf is None or pred_gdf.empty or gt_gdf.empty:
        pass
    else:
        sindex = pred_gdf.sindex
        for _, g_row in gt_gdf.iterrows():
            gid = int(g_row[gt_id])
            gpoly = g_row.geometry
            if gpoly is None or gpoly.is_empty:
                continue
            best, best_iou = None, 0.0
            for idx in sindex.query(gpoly):
                p_row = pred_gdf.iloc[int(idx)]
                pid = int(p_row[pred_id])
                ppoly = p_row.geometry
                if ppoly is None or ppoly.is_empty:
                    continue
                inter = ppoly.intersection(gpoly).area
                if inter <= 0:
                    continue
                union = ppoly.area + gpoly.area - inter
                if union <= 0:
                    continue
                iou = inter / union
                if iou > best_iou:
                    best_iou, best = iou, pid
            if best is not None and best_iou >= iou_thresh:
                matches.append((best, gid))
                matched_gt.add(gid)
                matched_pred.add(best)
    fn = []
    fp = []
    if gt_gdf is not None and not gt_gdf.empty:
        fn = [int(g[gt_id]) for _, g in gt_gdf.iterrows()
              if int(g[gt_id]) not in matched_gt]
    if pred_gdf is not None and not pred_gdf.empty:
        fp = [int(p[pred_id]) for _, p in pred_gdf.iterrows()
              if int(p[pred_id]) not in matched_pred]
    return matches, fn, fp
