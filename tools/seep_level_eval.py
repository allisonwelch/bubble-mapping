# tools/seep_level_eval.py
import os, glob
import numpy as np
import rasterio
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, binary_opening, disk
import pandas as pd
from tqdm import tqdm

# 1. Locate paired (prediction, ground-truth) test images.
#    Predictions are binary .tif files written by evaluation.py into
#    config.results_dir. Ground truth is the LAST BAND of the matching
#    preprocessed chip in config.preprocessed_dir (this mirrors how
#    evaluation.py reads frame.annotations — see evaluation.py:1066-1071).

def _smooth_pred(mask, radius=1):
    '''Closing+opening to remove boundary noise from per-pixel predictions.
    Applied before connected-component (CC) analysis'''
    m = mask.astype(bool)
    m = binary_closing(m, disk(radius))
    m = binary_opening(m, disk(radius))
    return m.astype(np.uint8)

def load_pair(pred_tif, chip_tif):
    with rasterio.open(pred_tif) as src:
        pred = _smooth_pred(src.read(1))
        transform = src.transform        # for physical sizes
    with rasterio.open(chip_tif) as src:
        n = src.count
        gt = src.read(n)                 # last band = annotation
        if n > 1:
            # source image = bands 1..n-1, transposed to (H, W, C)
            image = np.transpose(src.read(list(range(1, n))), (1, 2, 0))
        else:
            image = gt
    if gt.max() > 1.5:
        gt = (gt / 255.0)
    gt_bin = (gt >= 0.5).astype(np.uint8)
    return pred, gt_bin, image, transform

# 2. Connected components and centroids.
def cc_with_props(binary_mask):
    lab = label(binary_mask, connectivity=2)   # 8-connectivity
    props = regionprops(lab)
    return lab, props

def _bbox_overlap(b1, b2):
    # bbox = (min_row, min_col, max_row, max_col); max is exclusive
    r0 = max(b1[0], b2[0]); c0 = max(b1[1], b2[1])
    r1 = min(b1[2], b2[2]); c1 = min(b1[3], b2[3])
    if r1 <= r0 or c1 <= c0:
        return None
    return r0, c0, r1, c1

# 3. Match predicted CCs to GT CCs.
#    Greedy by IoU; ties broken by centroid distance.
def match_components(pred_lab, pred_props, gt_lab, gt_props,
                     iou_thresh=0.1, dist_thresh_px=None):
    matches = []           # list of (pred_id, gt_id)
    used_pred, used_gt = set(), set()
    # Build IoU matrix only over bounding-box-overlapping pairs (scales)
    for gp in gt_props:
        best, best_iou = None, 0.0
        gb = gp.bbox
        gt_area = gp.area # pixel count
        for pp in pred_props:
            if pp.label in used_pred:
                continue
            ub = _bbox_overlap(gb, pp.bbox)
            if ub is None:
                continue # no overlap, IoU = 0
            r0, c0, r1, c1 = ub
            gt_crop = (gt_lab[r0:r1, c0:c1] == gp.label)
            pr_crop = (pred_lab[r0:r1, c0:c1] == pp.label)
            inter = np.logical_and(gt_crop, pr_crop).sum()
            if inter == 0:
                continue
            union = gt_area + pp.area - inter # cheaper than a second logical_or
            iou = inter / union
            if iou > best_iou:
                best_iou, best = iou, pp.label
        if best is not None and best_iou >= iou_thresh:
            matches.append((best, gp.label))
            used_pred.add(best); used_gt.add(gp.label)
    fn = [g.label for g in gt_props if g.label not in used_gt]
    fp = [p.label for p in pred_props if p.label not in used_pred]
    return matches, fn, fp

def feature_table(image, lab, props, transform):
    pix_m = abs(transform.a)        # meters per pixel (assumes square pixels)
    rows = []
    for p in props:
        ar_m2  = p.area * (pix_m ** 2)
        per_m  = p.perimeter * pix_m
        circ   = 4 * np.pi * ar_m2 / max(1e-9, per_m ** 2)
        ys, xs = np.where(lab == p.label)
        mean_i = image[ys, xs].mean(axis=0) if image.ndim == 3 else image[ys, xs].mean()
        rows.append({"id": p.label, "area_m2": ar_m2, "perim_m": per_m,
                     "circularity": circ, "mean_intensity": mean_i})
    return pd.DataFrame(rows)


# After match_components, build paired feature rows:
def paired_feature_correlation(matches, pred_feats, gt_feats):
    pairs = pd.DataFrame([
        {"feat_pred": pred_feats.set_index("id").loc[pid].to_dict(),
         "feat_gt":   gt_feats.set_index("id").loc[gid].to_dict()}
        for pid, gid in matches
    ])
    out = {}
    for k in ["area_m2", "perim_m", "circularity"]:
        a = np.array([row["feat_pred"][k] for _, row in pairs.iterrows()])
        b = np.array([row["feat_gt"][k]   for _, row in pairs.iterrows()])
        if len(a) < 5:
            out[k] = np.nan
        else:
            out[k] = float(np.corrcoef(a, b)[0, 1])
    return out

# 4. Aggregate over all test images, compute precision/recall/F1.
def main(pred_dir, chip_dir):
    rows = []
    pair_rows = []
    pred_fps = [
        fp for fp in sorted(glob.glob(os.path.join(pred_dir, "*.tif")))
        if not fp.endswith(("_prob.tif", "_epistemic.tif", "_aleatoric.tif"))
    ]
    for pred_fp in tqdm(pred_fps, desc="Seep-level eval"):
        chip_fp = os.path.join(chip_dir, os.path.basename(pred_fp))
        if not os.path.exists(chip_fp):
            continue
        pred, gt, image, transform = load_pair(pred_fp, chip_fp)
        pl, pp = cc_with_props(pred)
        gl, gp = cc_with_props(gt)
        matches, fn_ids, fp_ids = match_components(pl, pp, gl, gp)

        pred_feats = feature_table(image, pl, pp, transform)
        gt_feats   = feature_table(image, gl, gp, transform)
        corrs = paired_feature_correlation(matches, pred_feats, gt_feats)

        rows.append({
            "image": os.path.basename(pred_fp),
            "n_gt": len(gp),
            "n_pred": len(pp),
            "tp": len(matches),
            "fn": len(fn_ids),
            "fp": len(fp_ids),
            "r_area_m2":     corrs.get("area_m2", np.nan),
            "r_perim_m":     corrs.get("perim_m", np.nan),
            "r_circularity": corrs.get("circularity", np.nan),
        })

        if matches:
            pf_idx = pred_feats.set_index("id")
            gf_idx = gt_feats.set_index("id")
            for pid, gid in matches:
                pf, gf = pf_idx.loc[pid], gf_idx.loc[gid]
                pair_rows.append({
                    "image": os.path.basename(pred_fp),
                    "pred_id": pid, "gt_id": gid,
                    "area_m2_pred":     pf["area_m2"],     "area_m2_gt":     gf["area_m2"],
                    "perim_m_pred":     pf["perim_m"],     "perim_m_gt":     gf["perim_m"],
                    "circularity_pred": pf["circularity"], "circularity_gt": gf["circularity"],
                })

    df = pd.DataFrame(rows)
    pairs_df = pd.DataFrame(pair_rows)
    tp = df["tp"].sum(); fn = df["fn"].sum(); fp = df["fp"].sum()
    recall    = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1        = 2 * precision * recall / max(1e-6, precision + recall)

    # Global Pearson r across all matched pairs (more stable than per-image r's).
    global_r = {}
    for k in ["area_m2", "perim_m", "circularity"]:
        if len(pairs_df) >= 5:
            global_r[k] = float(np.corrcoef(pairs_df[f"{k}_pred"].values,
                                            pairs_df[f"{k}_gt"].values)[0, 1])
        else:
            global_r[k] = float("nan")

    print(df)
    print(f"\nSEEP-LEVEL: precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    print(f"FEATURE r (global, n={len(pairs_df)}): "
          f"area={global_r['area_m2']:.3f} perim={global_r['perim_m']:.3f} "
          f"circ={global_r['circularity']:.3f}")

    df.to_csv(os.path.join(pred_dir, "seep_level_per_image.csv"), index=False)
    pairs_df.to_csv(os.path.join(pred_dir, "seep_level_pairs.csv"), index=False)
    pd.DataFrame([{
        "n_images": len(df), "n_pairs": len(pairs_df),
        "tp": int(tp), "fn": int(fn), "fp": int(fp),
        "precision": precision, "recall": recall, "f1": f1,
        "r_area_m2": global_r["area_m2"],
        "r_perim_m": global_r["perim_m"],
        "r_circularity": global_r["circularity"],
    }]).to_csv(os.path.join(pred_dir, "seep_level_summary.csv"), index=False)

    return precision, recall, f1