# tools/seep_level_eval.py
import os, glob
import numpy as np
import rasterio
from skimage.measure import label, regionprops
import pandas as pd

# 1. Locate paired (prediction, ground-truth) test images.
#    Predictions are binary .tif files written by evaluation.py into
#    config.results_dir. Ground truth is the LAST BAND of the matching
#    preprocessed chip in config.preprocessed_dir (this mirrors how
#    evaluation.py reads frame.annotations — see evaluation.py:1066-1071).

def load_pair(pred_tif, chip_tif):
    with rasterio.open(pred_tif) as src:
        pred = src.read(1)              # binary 0/1
        transform = src.transform        # for physical sizes
    with rasterio.open(chip_tif) as src:
        gt = src.read(src.count)         # last band = annotation
    if gt.max() > 1.5:
        gt = (gt / 255.0)
    gt_bin = (gt >= 0.5).astype(np.uint8)
    return pred, gt_bin, transform

# 2. Connected components and centroids.
def cc_with_props(binary_mask):
    lab = label(binary_mask, connectivity=2)   # 8-connectivity
    props = regionprops(lab)
    return lab, props

# 3. Match predicted CCs to GT CCs.
#    Greedy by IoU; ties broken by centroid distance.
def match_components(pred_lab, pred_props, gt_lab, gt_props,
                     iou_thresh=0.1, dist_thresh_px=None):
    matches = []           # list of (pred_id, gt_id)
    used_pred, used_gt = set(), set()
    # Build IoU matrix only over bounding-box-overlapping pairs (scales)
    for gp in gt_props:
        best, best_iou = None, 0.0
        gt_mask = (gt_lab == gp.label)
        for pp in pred_props:
            if pp.label in used_pred:
                continue
            pred_mask = (pred_lab == pp.label)
            inter = np.logical_and(gt_mask, pred_mask).sum()
            if inter == 0:
                continue
            union = np.logical_or(gt_mask, pred_mask).sum()
            iou = inter / union
            if iou > best_iou:
                best_iou, best = iou, pp.label
        if best is not None and best_iou >= iou_thresh:
            matches.append((best, gp.label))
            used_pred.add(best); used_gt.add(gp.label)
    fn = [g.label for g in gt_props if g.label not in used_gt]
    fp = [p.label for p in pred_props if p.label not in used_pred]
    return matches, fn, fp

# 4. Aggregate over all test images, compute precision/recall/F1.
def main(pred_dir, chip_dir):
    rows = []
    for pred_fp in sorted(glob.glob(os.path.join(pred_dir, "*.tif"))):
        if pred_fp.endswith("_prob.tif"):
            continue   # skip probability rasters
        chip_fp = os.path.join(chip_dir, os.path.basename(pred_fp))
        if not os.path.exists(chip_fp):
            continue
        pred, gt, _ = load_pair(pred_fp, chip_fp)
        pl, pp = cc_with_props(pred)
        gl, gp = cc_with_props(gt)
        matches, fn_ids, fp_ids = match_components(pl, pp, gl, gp)
        rows.append({
            "image": os.path.basename(pred_fp),
            "n_gt": len(gp),
            "n_pred": len(pp),
            "tp": len(matches),
            "fn": len(fn_ids),
            "fp": len(fp_ids),
        })
    df = pd.DataFrame(rows)
    tp = df["tp"].sum(); fn = df["fn"].sum(); fp = df["fp"].sum()
    recall    = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1        = 2 * precision * recall / max(1e-6, precision + recall)
    print(df)
    print(f"\nSEEP-LEVEL: precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    df.to_csv("seep_level_per_image.csv", index=False)
    return precision, recall, f1