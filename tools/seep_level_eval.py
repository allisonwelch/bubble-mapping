# tools/seep_level_eval.py
"""BUBBLE-LEVEL evaluation of the detector: predicted connected components
matched one-to-one against ground-truth connected components.

This is the canonical detection metric. A "bubble" is one connected component
of the smoothed prediction (or of the GT label band); nothing here is a *seep*,
because bubbles become seeps only after the learned pairwise grouper runs.
See the naming convention in CLAUDE.md.

Pipeline position:
    pixel  -- evaluation.py       Dice / IoU / F1 straight out of training
    bubble -- THIS FILE           precision / recall / F1 after smoothing + CC
    seep   -- not evaluated yet   needs the RF grouper; see
                                  tools/archive/polygon_matcher.py

Outputs (to out_dir, default pred_dir):
    seep_level_summary.csv    one wide row: P/R/F1 + matched-pair feature r
    seep_level_per_image.csv  per-chip counts and r
    seep_level_pairs.csv      one row per matched (pred CC, GT CC) pair
    seep_features_per_bubble.csv
    {stem}_smoothed.tif, {stem}_cc.tif, optional {stem}_snow.tif

REMOVED 2026-09-03 -- the cluster-level half. `cluster_f1 = 0.672` was produced
by grouping the GT and predicted sides with the SAME hand-tuned anchor+lonely
rule, so grouping error cancelled on both sides and the number was a detection
metric wearing a seep-level label. The rule is gone; the matcher is preserved
at tools/archive/polygon_matcher.py together with the recipe for an honest
RF-grouper version. Canonical detection metric is now bubble F1 = 0.645.

Ground-truth bubble POLYGONS are no longer written here -- that is
tools/gt_seeps_export.py's job, so there is exactly one producer of that file.
"""
import os, sys, glob
import numpy as np
import pandas as pd
import rasterio
from skimage.measure import label, regionprops
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from write_seep_rasters import (
    smooth_pred,
    snow_mask_hsv,
    write_rasters_for_pred,
    write_snow_raster,
)
from seep_feature_table import (
    process_pred as _bubble_feat_process,
    write_feature_csvs as _bubble_feat_write_csvs,
)

def _drop_snow_ccs(pred, snow, drop_frac):
    """Drop CCs in `pred` whose pixel-overlap with `snow` exceeds `drop_frac`.

    Pixel-level masking on a smoothed prediction can carve holes inside CCs,
    fragmenting one snow-FP into several smaller FPs. This filter labels the
    prediction, measures the snow fraction inside each CC, and zeros the
    entire CC (whole-or-nothing) when its snow fraction strictly exceeds
    `drop_frac`. Background (label 0) is ignored.

    Returns (filtered_pred, n_dropped).
    """
    pred_b = pred.astype(bool)
    lab = label(pred_b, connectivity=2)
    n_cc = int(lab.max())
    if n_cc == 0:
        return pred, 0
    snow_b = snow.astype(bool)
    flat = lab.ravel()
    areas = np.bincount(flat, minlength=n_cc + 1)
    snow_per_cc = np.bincount(flat,
                              weights=snow_b.ravel().astype(np.int64),
                              minlength=n_cc + 1)
    fracs = np.zeros(n_cc + 1, dtype=np.float64)
    nz = areas > 0
    fracs[nz] = snow_per_cc[nz] / areas[nz]
    drop_ids = np.where(fracs > float(drop_frac))[0]
    drop_ids = drop_ids[drop_ids > 0]  # never drop background
    if drop_ids.size == 0:
        return pred, 0
    drop_mask = np.isin(lab, drop_ids)
    out = pred.copy()
    out[drop_mask] = 0
    return out, int(drop_ids.size)


def load_pair(pred_tif, chip_tif,
              snow_v_thresh=None, snow_s_thresh=None,
              snow_close_px=0, snow_dilate_px=0,
              snow_cc_drop_frac=0.5):
    """Read prediction + chip, smooth pred, and (optionally) apply the
    CC-level snow filter: drop entire predicted CCs whose pixel-overlap with
    the HSV snow mask exceeds `snow_cc_drop_frac`. The snow raster is still
    computed and returned for QGIS overlay even when no CCs end up dropped.

    Returns (pred, gt_bin, image, transform, profile, snow, n_dropped).
    `snow` is None unless both snow_v_thresh and snow_s_thresh are set.
    `n_dropped` is the number of CCs zeroed by the filter (0 if disabled).
    """
    with rasterio.open(pred_tif) as src:
        pred = smooth_pred(src.read(1))
        transform = src.transform        # for physical sizes
        pred_profile = src.profile.copy()
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

    snow = None
    n_dropped = 0
    if snow_v_thresh is not None and snow_s_thresh is not None:
        snow = snow_mask_hsv(image, v_thresh=snow_v_thresh,
                             s_thresh=snow_s_thresh,
                             close_px=snow_close_px,
                             dilate_px=snow_dilate_px)
        if (snow is not None and snow_cc_drop_frac is not None
                and float(snow_cc_drop_frac) > 0):
            pred, n_dropped = _drop_snow_ccs(pred, snow, snow_cc_drop_frac)
    return pred, gt_bin, image, transform, pred_profile, snow, n_dropped


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


def main(pred_dir, chip_dir,
         snow_mask_enabled=False,
         snow_v_thresh=0.85,
         snow_s_thresh=0.15,
         snow_close_px=0,
         snow_dilate_px=0,
         snow_cc_drop_frac=0.5,
         write_snow_rasters=True,
         out_dir=None):
    """Run bubble-level detection evaluation over every (prediction, chip) pair.

    `out_dir`: where this run's rasters and CSVs are written. Defaults to
    `pred_dir`. Point it at a subdirectory to keep an A/B run's outputs
    separate from the canonical artifacts.
    """
    if out_dir is None:
        out_dir = pred_dir
    os.makedirs(out_dir, exist_ok=True)

    rows, pair_rows, bubble_dfs = [], [], []
    snow_px_total = img_px_total = snow_ccs_dropped_total = 0

    pred_fps = [
        fp for fp in sorted(glob.glob(os.path.join(pred_dir, "*.tif")))
        if not fp.endswith(("_prob.tif", "_epistemic.tif", "_aleatoric.tif",
                            "_smoothed.tif", "_cc.tif", "_seep_cluster.tif",
                            "_snow.tif"))
        and "_r125_r45_r10_lonely_seep_cluster" not in fp
    ]
    sv = snow_v_thresh if snow_mask_enabled else None
    ss = snow_s_thresh if snow_mask_enabled else None

    for pred_fp in tqdm(pred_fps, desc="Bubble-level eval"):
        chip_fp = os.path.join(chip_dir, os.path.basename(pred_fp))
        if not os.path.exists(chip_fp):
            continue
        pred, gt, image, transform, pred_profile, snow, n_dropped = load_pair(
            pred_fp, chip_fp,
            snow_v_thresh=sv, snow_s_thresh=ss,
            snow_close_px=snow_close_px,
            snow_dilate_px=snow_dilate_px,
            snow_cc_drop_frac=snow_cc_drop_frac if snow_mask_enabled else 0,
        )
        snow_ccs_dropped_total += n_dropped
        pl, pp = cc_with_props(pred)
        gl, gp = cc_with_props(gt)
        matches, fn_ids, fp_ids = match_components(pl, pp, gl, gp)

        write_rasters_for_pred(pred_fp, smoothed=pred, cc=pl,
                               profile=pred_profile, out_dir=out_dir)
        if snow is not None:
            snow_px_total += int(snow.sum())
            img_px_total += int(snow.size)
            if write_snow_rasters:
                write_snow_raster(pred_fp, snow, pred_profile, out_dir=out_dir)

        bubble_dfs.append(_bubble_feat_process(
            pred_fp, chip_fp, cc=pl, image=image,
            transform=transform, profile=pred_profile))

        pred_feats = feature_table(image, pl, pp, transform)
        gt_feats = feature_table(image, gl, gp, transform)
        corrs = paired_feature_correlation(matches, pred_feats, gt_feats)

        rows.append({
            "image": os.path.basename(pred_fp),
            "n_gt": len(gp), "n_pred": len(pp),
            "tp": len(matches), "fn": len(fn_ids), "fp": len(fp_ids),
            "r_area_m2": corrs.get("area_m2", np.nan),
            "r_perim_m": corrs.get("perim_m", np.nan),
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
                    "area_m2_pred": pf["area_m2"], "area_m2_gt": gf["area_m2"],
                    "perim_m_pred": pf["perim_m"], "perim_m_gt": gf["perim_m"],
                    "circularity_pred": pf["circularity"],
                    "circularity_gt": gf["circularity"],
                })

    if not rows:
        raise RuntimeError(
            "No (prediction, chip) pairs were processed.\n"
            f"  pred_dir = {pred_dir}\n"
            f"  chip_dir = {chip_dir}\n"
            "Check that pred_dir contains the per-checkpoint subdir written by "
            "evaluation.py (e.g. {results_dir}/{checkpoint_basename}/), and "
            "that chip filenames in chip_dir match the prediction filenames.")

    df = pd.DataFrame(rows)
    pairs_df = pd.DataFrame(pair_rows)
    tp, fn, fp = df["tp"].sum(), df["fn"].sum(), df["fp"].sum()
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    global_r = {}
    for k in ["area_m2", "perim_m", "circularity"]:
        global_r[k] = (float(np.corrcoef(pairs_df[f"{k}_pred"].values,
                                         pairs_df[f"{k}_gt"].values)[0, 1])
                       if len(pairs_df) >= 5 else float("nan"))

    snow_pct = (100.0 * snow_px_total / img_px_total) if img_px_total else 0.0

    print(df)
    if snow_mask_enabled:
        print(f"\nSNOW MASK: v>={snow_v_thresh:.2f} s<={snow_s_thresh:.2f} "
              f"close={snow_close_px}px dilate={snow_dilate_px}px  -> "
              f"{snow_pct:.2f}% of pixels masked")
        print(f"  CC-level filter: drop_frac>{snow_cc_drop_frac:.2f}  -> "
              f"{snow_ccs_dropped_total} CCs dropped across {len(df)} chips")
    print("\nBUBBLE-LEVEL (pred CC vs GT CC):")
    print(f"  precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    print(f"  FEATURE r (n={len(pairs_df)}): "
          f"area={global_r['area_m2']:.3f} perim={global_r['perim_m']:.3f} "
          f"circ={global_r['circularity']:.3f}")

    df.to_csv(os.path.join(out_dir, "seep_level_per_image.csv"), index=False)
    pairs_df.to_csv(os.path.join(out_dir, "seep_level_pairs.csv"), index=False)
    pd.DataFrame([{
        "n_images": len(df), "n_pairs": len(pairs_df),
        "snow_mask_enabled": bool(snow_mask_enabled),
        "snow_v_thresh": snow_v_thresh if snow_mask_enabled else float("nan"),
        "snow_s_thresh": snow_s_thresh if snow_mask_enabled else float("nan"),
        "snow_close_px": int(snow_close_px) if snow_mask_enabled else 0,
        "snow_dilate_px": int(snow_dilate_px) if snow_mask_enabled else 0,
        "snow_pct_masked": snow_pct if snow_mask_enabled else float("nan"),
        "snow_cc_drop_frac": (float(snow_cc_drop_frac) if snow_mask_enabled
                              else float("nan")),
        "snow_ccs_dropped": (int(snow_ccs_dropped_total) if snow_mask_enabled
                             else 0),
        "tp": int(tp), "fn": int(fn), "fp": int(fp),
        "precision": precision, "recall": recall, "f1": f1,
        "r_area_m2": global_r["area_m2"],
        "r_perim_m": global_r["perim_m"],
        "r_circularity": global_r["circularity"],
    }]).to_csv(os.path.join(out_dir, "seep_level_summary.csv"), index=False)

    if bubble_dfs:
        _bubble_feat_write_csvs(out_dir, pd.concat(bubble_dfs, ignore_index=True))

    return precision, recall, f1


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import configSwinUnet
    config = configSwinUnet.Configuration().validate()
    ckpt_pred_dir = os.path.join(config.results_dir, "20260428-1537_SWINxAE.weights")
    sub = getattr(config, "seep_eval_out_subdir", None)
    main(
        pred_dir=ckpt_pred_dir,
        chip_dir=config.preprocessed_dir,
        out_dir=os.path.join(ckpt_pred_dir, sub) if sub else None,
        snow_mask_enabled=getattr(config, "snow_mask_enabled", False),
        snow_v_thresh=getattr(config, "snow_v_thresh", 0.85),
        snow_s_thresh=getattr(config, "snow_s_thresh", 0.15),
        snow_close_px=getattr(config, "snow_mask_close_px", 0),
        snow_dilate_px=getattr(config, "snow_mask_dilate_px", 0),
        snow_cc_drop_frac=getattr(config, "snow_cc_drop_frac", 0.5),
        write_snow_rasters=getattr(config, "write_snow_rasters", True),
    )
