# tools/seep_level_eval.py
import os, sys, glob
import numpy as np
import rasterio
from skimage.measure import label, regionprops
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from write_seep_rasters import (
    smooth_pred,
    snow_mask_hsv,
    write_rasters_for_pred,
    write_snow_raster,
)
from seep_feature_table import (
    process_pred as _seep_feat_process,
    write_feature_csvs as _seep_feat_write_csvs,
    build_gt_seeps_from_source,
    write_seeps_gpkg,
    anchor_cluster,
    lonely_cluster,
    DEFAULT_ANCHOR_AREA_M2,
    DEFAULT_CLUSTER_RADIUS_M,
    DEFAULT_SATELLITE_MAX_AREA_M2,
    DEFAULT_LONELY_CLUSTER_RADIUS_M,
    DEFAULT_LONELY_HALO_RADIUS_M,
    DEFAULT_LONELY_MAX_HALO_NEIGHBORS,
)

try:
    import geopandas as gpd  # noqa: F401
    from shapely.ops import unary_union
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False


def _auto_group_gt_seeps(gt_bubbles, anchor_area_m2, cluster_radius_m,
                         satellite_max_area_m2, lonely_kwargs, image_name):
    """Auto-group full per-chip GT bubble polygons into seeps with the SAME
    two-phase rule used on predictions, then dissolve + recompute seep-level
    features. Used by gt_grouping_mode='auto' so the headline cluster_f1 covers
    every GT seep on the test chips (not just the manually-grouped sample) and
    is an apples-to-apples seep-level DETECTION metric (both sides grouped by
    the identical rule; grouping fidelity vs humans is measured separately by
    the Phase-6 cross-validation).

    `gt_bubbles`: per-polygon GeoDataFrame from build_gt_seeps_from_source
    (columns seep_id, area_m2, perim_m, circularity, solidity, mean_R/G/B,
    centroid_x_m, centroid_y_m, geometry).
    Returns a dissolved GeoDataFrame keyed by seep_id (= rule cluster_id).
    """
    if gt_bubbles is None or gt_bubbles.empty:
        return gt_bubbles
    bub = pd.DataFrame({
        "bubble_id": gt_bubbles["seep_id"].to_numpy(dtype=np.int64),
        "area_m2": gt_bubbles["area_m2"].to_numpy(dtype=float),
        "centroid_x_m": gt_bubbles["centroid_x_m"].to_numpy(dtype=float),
        "centroid_y_m": gt_bubbles["centroid_y_m"].to_numpy(dtype=float),
    })
    bub = anchor_cluster(bub, anchor_area_m2, cluster_radius_m,
                         satellite_max_area_m2)
    bub = lonely_cluster(bub, **lonely_kwargs)
    cid = dict(zip(bub["bubble_id"], bub["cluster_id"]))
    g = gt_bubbles.copy()
    g["_cluster_id"] = g["seep_id"].map(cid).astype(np.int64)

    rows, geoms = [], []
    for gid, grp in g.groupby("_cluster_id", sort=True):
        union = unary_union(list(grp.geometry.values))
        area = float(union.area)
        perim = max(float(union.length), 1e-9)
        hull = float(union.convex_hull.area)
        w = grp["area_m2"].to_numpy(dtype=float)
        wsum = w.sum() if w.sum() > 0 else 1.0
        rows.append({
            "image": image_name,
            "seep_id": int(gid),
            "area_m2": area, "perim_m": perim,
            "circularity": 4 * np.pi * area / (perim ** 2),
            "solidity": (area / hull) if hull > 0 else 0.0,
            "mean_R": float((grp["mean_R"].to_numpy() * w).sum() / wsum),
            "mean_G": float((grp["mean_G"].to_numpy() * w).sum() / wsum),
            "mean_B": float((grp["mean_B"].to_numpy() * w).sum() / wsum),
            "n_polygons_in_group": int(len(grp)),
        })
        geoms.append(union)
    return gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=gt_bubbles.crs)

# 1. Locate paired (prediction, ground-truth) test images.
#    Predictions are binary .tif files written by evaluation.py into
#    config.results_dir. Ground truth is the LAST BAND of the matching
#    preprocessed chip in config.preprocessed_dir (this mirrors how
#    evaluation.py reads frame.annotations — see evaluation.py:1066-1071).

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


_CLUSTER_FEATURE_KEYS = [
    "area_m2", "perim_m", "circularity",
    "solidity", "eccentricity",
    "mean_R", "mean_G", "mean_B",
]


def _cluster_pair_corr(pairs_df):
    """Global Pearson r per feature across all cluster-level matched pairs."""
    out = {}
    for k in _CLUSTER_FEATURE_KEYS:
        pcol, gcol = f"{k}_pred", f"{k}_gt"
        if pcol not in pairs_df.columns or gcol not in pairs_df.columns:
            out[k] = float("nan")
            continue
        a = pairs_df[pcol].values
        b = pairs_df[gcol].values
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 5:
            out[k] = float("nan")
        else:
            out[k] = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return out

# 4. Aggregate over all test images, compute precision/recall/F1.
def main(pred_dir, chip_dir,
         anchor_area_m2=DEFAULT_ANCHOR_AREA_M2,
         cluster_radius_m=DEFAULT_CLUSTER_RADIUS_M,
         satellite_max_area_m2=DEFAULT_SATELLITE_MAX_AREA_M2,
         lonely_cluster_radius_m=DEFAULT_LONELY_CLUSTER_RADIUS_M,
         lonely_halo_radius_m=DEFAULT_LONELY_HALO_RADIUS_M,
         lonely_max_halo_neighbors=DEFAULT_LONELY_MAX_HALO_NEIGHBORS,
         snow_mask_enabled=False,
         snow_v_thresh=0.85,
         snow_s_thresh=0.15,
         snow_close_px=0,
         snow_dilate_px=0,
         snow_cc_drop_frac=0.5,
         write_snow_rasters=True,
         write_seep_cluster_rasters=True,
         out_dir=None,
         source_polygons_fp=None,
         labeled_seeps_fp=None,
         gt_grouping_mode="auto"):
    """Run cluster-level seep evaluation.

    `out_dir`: where per-chip rasters, CSVs and GPKGs from this run are
    written. Defaults to `pred_dir` (writes alongside the predictions).
    Pass a subdirectory path to keep this run's outputs separate from the
    canonical artifacts in `pred_dir` (useful for A/B comparison runs).
    Directory is created if missing.

    `source_polygons_fp`: path to the original drawn-polygon GPKG used to
    build gt_seeps.gpkg with preserved per-polygon shapes. Used for the
    cluster-level seep matcher and the GT GPKG written at the end of the run
    when no grouped GT is supplied. If None, gt_seeps.gpkg is NOT written and
    the cluster-level metrics fall back to empty.

    `gt_grouping_mode`: how the GT side is grouped into seeps for the
    cluster-level matcher. THREE-WAY behavior:
      * "auto"   (default, headline metric): auto-group the FULL per-chip GT
        bubbles with the same fitted two-phase rule used on predictions, then
        match. Covers every GT seep on the test chips (not just the 750-seep
        labeling sample, which is a deliberately non-representative stratified
        sample per 2026-05-13 and is the WRONG denominator for a headline
        cluster_f1). Both sides grouped by the identical rule => this is a
        seep-level DETECTION metric; grouping fidelity vs humans is reported
        separately by the Phase-6 cross-validation. Requires `source_polygons_fp`.
      * "manual" (sample sanity-check): use the dissolved, labeler-grouped GT
        seeps in `labeled_seeps_fp` (gt_seeps_labeled.gpkg) as the GT side.
        Honest human-vs-pred seep matching, but only over the labeled sample.
      * falls back to per-polygon GT (pre-2026-05-28 behavior) if neither a
        source-polygon path (auto) nor a labeled file (manual) is available.

    `labeled_seeps_fp`: path to gt_seeps_labeled.gpkg (used only in "manual"
    mode). `source_polygons_fp`: original drawn-polygon GPKG (used in "auto"
    mode and the per-polygon fallback).
    """
    if out_dir is None:
        out_dir = pred_dir
    os.makedirs(out_dir, exist_ok=True)

    # Resolve the effective GT grouping mode given what's actually available.
    gt_mode = gt_grouping_mode
    labeled_gdf = None
    if gt_mode == "manual":
        if _HAS_GPD and labeled_seeps_fp is not None and os.path.exists(labeled_seeps_fp):
            print(f"GT grouping = manual; loading grouped GT seeps: {labeled_seeps_fp}")
            labeled_gdf = gpd.read_file(labeled_seeps_fp)
            if "image" not in labeled_gdf.columns or "seep_id" not in labeled_gdf.columns:
                raise RuntimeError(
                    f"{labeled_seeps_fp} must have 'image' and 'seep_id' columns "
                    f"(seep_id = the dissolved seep_group_id).")
            print(f"  -> {len(labeled_gdf)} grouped GT seeps across "
                  f"{labeled_gdf['image'].nunique()} chips (crs={labeled_gdf.crs})")
        else:
            print("GT grouping = manual requested but no labeled file found; "
                  "falling back to per-polygon GT.")
            gt_mode = "fallback"

    # Per-polygon source: needed for "auto" (full-GT auto-grouping) and the
    # per-polygon fallback.
    src_gdf = None
    if labeled_gdf is None and source_polygons_fp is not None and _HAS_GPD:
        print(f"loading source polygons: {source_polygons_fp}")
        src_gdf = gpd.read_file(source_polygons_fp)
        print(f"  -> {len(src_gdf)} source polygons (crs={src_gdf.crs})")
    if gt_mode == "auto":
        if src_gdf is None:
            print("GT grouping = auto requested but no source polygons; "
                  "cluster-level GT will be empty.")
        else:
            print("GT grouping = auto; full per-chip GT will be rule-grouped "
                  "into seeps with the fitted clustering params.")
    lonely_kwargs = dict(
        lonely_cluster_radius_m=lonely_cluster_radius_m,
        lonely_halo_radius_m=lonely_halo_radius_m,
        lonely_max_halo_neighbors=lonely_max_halo_neighbors,
    )
    rows = []
    pair_rows = []
    bubble_dfs, cluster_dfs = [], []
    pred_seeps_gdfs, gt_seeps_gdfs = [], []
    cluster_rows = []
    cluster_pair_rows = []
    snow_px_total = 0
    img_px_total = 0
    snow_ccs_dropped_total = 0
    pred_fps = [
        fp for fp in sorted(glob.glob(os.path.join(pred_dir, "*.tif")))
        if not fp.endswith(("_prob.tif", "_epistemic.tif", "_aleatoric.tif",
                            "_smoothed.tif", "_cc.tif", "_seep_cluster.tif",
                            "_snow.tif"))
        and "_r125_r45_r10_lonely_seep_cluster" not in fp
    ]
    sv = snow_v_thresh if snow_mask_enabled else None
    ss = snow_s_thresh if snow_mask_enabled else None
    for pred_fp in tqdm(pred_fps, desc="Seep-level eval"):
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
        crs = pred_profile.get("crs")
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

        b_df, c_df, pred_seeps_gdf = _seep_feat_process(
            pred_fp, chip_fp,
            cc=pl, image=image, transform=transform, profile=pred_profile,
            anchor_area_m2=anchor_area_m2,
            cluster_radius_m=cluster_radius_m,
            satellite_max_area_m2=satellite_max_area_m2,
            lonely_cluster_radius_m=lonely_cluster_radius_m,
            lonely_halo_radius_m=lonely_halo_radius_m,
            lonely_max_halo_neighbors=lonely_max_halo_neighbors,
            write_raster=write_seep_cluster_rasters,
            polygonize=_HAS_GPD and crs is not None,
            out_dir=out_dir,
        )
        bubble_dfs.append(b_df)
        cluster_dfs.append(c_df)
        if pred_seeps_gdf is not None:
            pred_seeps_gdfs.append(pred_seeps_gdf)

        # Build GT seep polygons. Three paths, selected by gt_grouping_mode:
        #  - "manual": this chip's rows from the dissolved labeler-grouped seeps
        #    (seep_id = dissolved seep_group_id). Human-vs-pred over the sample.
        #  - "auto": rebuild the full per-chip GT polygons from the ORIGINAL
        #    drawn shapes, then rule-group them into seeps (same params as pred)
        #    and dissolve. Full-test-set seep-level detection metric.
        #  - "fallback": per-polygon GT (pre-2026-05-28). We avoid
        #    rasterize -> CC -> repolygonize (it merges adjacent polygons,
        #    destroying per-seep labels and distorting features).
        gt_seeps_gdf = None
        if _HAS_GPD and crs is not None and gt_mode == "manual" and labeled_gdf is not None:
            sub = labeled_gdf[labeled_gdf["image"] == os.path.basename(pred_fp)].copy()
            if not sub.empty:
                if sub.crs is not None and sub.crs != crs:
                    sub = sub.to_crs(crs)
                gt_seeps_gdf = sub
                gt_seeps_gdfs.append(gt_seeps_gdf)
        elif _HAS_GPD and crs is not None and src_gdf is not None:
            gt_bubbles = build_gt_seeps_from_source(
                chip_fp, src_gdf, id_name="seep_id"
            )
            if gt_bubbles is not None and not gt_bubbles.empty:
                if gt_mode == "auto":
                    gt_seeps_gdf = _auto_group_gt_seeps(
                        gt_bubbles, anchor_area_m2, cluster_radius_m,
                        satellite_max_area_m2, lonely_kwargs,
                        os.path.basename(pred_fp))
                else:  # per-polygon fallback
                    gt_seeps_gdf = gt_bubbles
                    gt_seeps_gdf.insert(0, "image", os.path.basename(pred_fp))
                if gt_seeps_gdf is not None and not gt_seeps_gdf.empty:
                    gt_seeps_gdfs.append(gt_seeps_gdf)

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

        # 3b. Cluster-level (seep-level) matching: pred cluster polygons vs
        #     GT polygons. This is the headline seep-level F1 going forward.
        #     Skipped only when geopandas/CRS is unavailable (we still need
        #     to count chips with GT-but-no-pred-clusters and vice versa to
        #     keep precision/recall honest, so empties are kept).
        if _HAS_GPD and crs is not None:
            n_pred_c = 0 if pred_seeps_gdf is None else len(pred_seeps_gdf)
            n_gt_c = 0 if gt_seeps_gdf is None else len(gt_seeps_gdf)
            c_matches, c_fn, c_fp = match_polygons(
                pred_seeps_gdf, gt_seeps_gdf,
                pred_id="cluster_id", gt_id="seep_id",
            )
            cluster_rows.append({
                "image": os.path.basename(pred_fp),
                "n_gt": n_gt_c,
                "n_pred_clusters": n_pred_c,
                "tp": len(c_matches),
                "fn": len(c_fn),
                "fp": len(c_fp),
            })
            if c_matches:
                pf_idx = pred_seeps_gdf.set_index("cluster_id")
                gf_idx = gt_seeps_gdf.set_index("seep_id")
                # Multi-truth matcher: a pred can appear in multiple match
                # rows. n_pred_matches lets reviewers spot over-spanning preds
                # when interpreting per-feature r (esp. r_area / r_perim,
                # which deflate when one pred covers several GT polygons).
                from collections import Counter
                pred_share = Counter(pid for pid, _ in c_matches)
                for pid, gid in c_matches:
                    pf, gf = pf_idx.loc[pid], gf_idx.loc[gid]
                    row = {"image": os.path.basename(pred_fp),
                           "cluster_id": pid, "seep_id": gid,
                           "n_pred_matches": int(pred_share[pid])}
                    for k in _CLUSTER_FEATURE_KEYS:
                        row[f"{k}_pred"] = float(pf[k]) if k in pf else np.nan
                        row[f"{k}_gt"] = float(gf[k]) if k in gf else np.nan
                    cluster_pair_rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No (prediction, chip) pairs were processed.\n"
            f"  pred_dir = {pred_dir}\n"
            f"  chip_dir = {chip_dir}\n"
            f"Check that pred_dir contains the per-checkpoint subdir written by "
            f"evaluation.py (e.g. {{results_dir}}/{{checkpoint_basename}}/), and "
            f"that chip filenames in chip_dir match the prediction filenames."
        )
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

    snow_pct = (100.0 * snow_px_total / img_px_total) if img_px_total else 0.0

    print(df)
    if snow_mask_enabled:
        print(f"\nSNOW MASK: v>={snow_v_thresh:.2f} s<={snow_s_thresh:.2f} "
              f"close={snow_close_px}px dilate={snow_dilate_px}px  → "
              f"{snow_pct:.2f}% of pixels masked")
        print(f"  CC-level filter: drop_frac>{snow_cc_drop_frac:.2f}  → "
              f"{snow_ccs_dropped_total} CCs dropped across {len(df)} chips")
    print(f"\nBUBBLE-LEVEL (CC↔CC, fragmentation diagnostic):")
    print(f"  precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    print(f"  FEATURE r (n={len(pairs_df)}): "
          f"area={global_r['area_m2']:.3f} perim={global_r['perim_m']:.3f} "
          f"circ={global_r['circularity']:.3f}")

    df.to_csv(os.path.join(out_dir, "seep_level_per_image.csv"), index=False)
    pairs_df.to_csv(os.path.join(out_dir, "seep_level_pairs.csv"), index=False)

    # Cluster-level (canonical seep) metrics.
    c_precision = c_recall = c_f1 = float("nan")
    c_tp = c_fn = c_fp = 0
    cluster_global_r = {k: float("nan") for k in _CLUSTER_FEATURE_KEYS}
    n_cluster_pairs = 0
    if cluster_rows:
        cdf = pd.DataFrame(cluster_rows)
        c_pairs_df = pd.DataFrame(cluster_pair_rows)
        c_tp = int(cdf["tp"].sum())
        c_fn = int(cdf["fn"].sum())
        c_fp = int(cdf["fp"].sum())
        c_recall    = c_tp / max(1, c_tp + c_fn)
        c_precision = c_tp / max(1, c_tp + c_fp)
        c_f1        = 2 * c_precision * c_recall / max(1e-6, c_precision + c_recall)
        cluster_global_r = _cluster_pair_corr(c_pairs_df)
        n_cluster_pairs = len(c_pairs_df)
        cdf.to_csv(os.path.join(out_dir, "seep_level_per_image_cluster.csv"),
                   index=False)
        c_pairs_df.to_csv(os.path.join(out_dir, "seep_level_pairs_cluster.csv"),
                          index=False)
        print(f"\nCLUSTER-LEVEL (pred cluster ↔ GT polygon, canonical seep F1):")
        print(f"  precision={c_precision:.3f} recall={c_recall:.3f} F1={c_f1:.3f}")
        print(f"  FEATURE r (n={n_cluster_pairs}): "
              f"area={cluster_global_r['area_m2']:.3f} "
              f"perim={cluster_global_r['perim_m']:.3f} "
              f"circ={cluster_global_r['circularity']:.3f} "
              f"sol={cluster_global_r['solidity']:.3f} "
              f"ecc={cluster_global_r['eccentricity']:.3f} "
              f"R={cluster_global_r['mean_R']:.3f} "
              f"G={cluster_global_r['mean_G']:.3f} "
              f"B={cluster_global_r['mean_B']:.3f}")

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
        "cluster_tp": c_tp, "cluster_fn": c_fn, "cluster_fp": c_fp,
        "cluster_precision": c_precision,
        "cluster_recall": c_recall,
        "cluster_f1": c_f1,
        "cluster_n_pairs": n_cluster_pairs,
        "cluster_r_area_m2":     cluster_global_r["area_m2"],
        "cluster_r_perim_m":     cluster_global_r["perim_m"],
        "cluster_r_circularity": cluster_global_r["circularity"],
        "cluster_r_solidity":    cluster_global_r["solidity"],
        "cluster_r_eccentricity": cluster_global_r["eccentricity"],
        "cluster_r_mean_R":      cluster_global_r["mean_R"],
        "cluster_r_mean_G":      cluster_global_r["mean_G"],
        "cluster_r_mean_B":      cluster_global_r["mean_B"],
    }]).to_csv(os.path.join(out_dir, "seep_level_summary.csv"), index=False)

    if bubble_dfs:
        bubbles_all = pd.concat(bubble_dfs, ignore_index=True)
        clusters_all = pd.concat(cluster_dfs, ignore_index=True)
        _seep_feat_write_csvs(out_dir, bubbles_all, clusters_all,
                              anchor_area_m2=anchor_area_m2,
                              cluster_radius_m=cluster_radius_m,
                              satellite_max_area_m2=satellite_max_area_m2,
                              lonely_cluster_radius_m=lonely_cluster_radius_m,
                              lonely_halo_radius_m=lonely_halo_radius_m,
                              lonely_max_halo_neighbors=lonely_max_halo_neighbors)

    if pred_seeps_gdfs:
        out_fp = os.path.join(out_dir, "pred_seeps.gpkg")
        write_seeps_gpkg(out_fp, pred_seeps_gdfs, class_column=False)
        print(f"  wrote {out_fp}")
    if gt_seeps_gdfs and gt_mode == "fallback":
        # Only the per-polygon fallback writes gt_seeps.gpkg with an empty
        # 'class' column for the QGIS labeling exercise. The grouped paths
        # ("manual" uses the supplied gt_seeps_labeled.gpkg; "auto" produces
        # rule-grouped seeps that are NOT a labeling artifact) do not write it.
        out_fp = os.path.join(out_dir, "gt_seeps.gpkg")
        write_seeps_gpkg(out_fp, gt_seeps_gdfs, class_column=True)
        print(f"  wrote {out_fp} (empty 'class' column for QGIS labeling)")
    elif gt_seeps_gdfs:
        print(f"  GT grouping = {gt_mode}; gt_seeps.gpkg not rewritten")

    return precision, recall, f1


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import configSwinUnet
    config = configSwinUnet.Configuration().validate()
    ckpt_pred_dir = os.path.join(config.results_dir, "20260428-1537_SWINxAE.weights")
    # Optional subdir under pred_dir that this run's outputs (rasters + CSVs +
    # GPKGs) get written into. Set in config to keep A/B run artifacts
    # separate from the canonical ones written directly into pred_dir.
    _subdir = getattr(config, "seep_eval_out_subdir", None)
    _out_dir = os.path.join(ckpt_pred_dir, _subdir) if _subdir else ckpt_pred_dir
    main(
        pred_dir=ckpt_pred_dir,
        chip_dir=config.preprocessed_dir,
        anchor_area_m2=getattr(config, "seep_anchor_area_m2",
                               DEFAULT_ANCHOR_AREA_M2),
        cluster_radius_m=getattr(config, "seep_cluster_radius_m",
                                 DEFAULT_CLUSTER_RADIUS_M),
        satellite_max_area_m2=getattr(config, "seep_satellite_max_area_m2",
                                      DEFAULT_SATELLITE_MAX_AREA_M2),
        lonely_cluster_radius_m=getattr(config, "seep_lonely_cluster_radius_m",
                                        DEFAULT_LONELY_CLUSTER_RADIUS_M),
        lonely_halo_radius_m=getattr(config, "seep_lonely_halo_radius_m",
                                     DEFAULT_LONELY_HALO_RADIUS_M),
        lonely_max_halo_neighbors=getattr(config, "seep_lonely_max_halo_neighbors",
                                          DEFAULT_LONELY_MAX_HALO_NEIGHBORS),
        snow_mask_enabled=getattr(config, "snow_mask_enabled", False),
        snow_v_thresh=getattr(config, "snow_v_thresh", 0.85),
        snow_s_thresh=getattr(config, "snow_s_thresh", 0.15),
        snow_close_px=getattr(config, "snow_mask_close_px", 0),
        snow_dilate_px=getattr(config, "snow_mask_dilate_px", 0),
        snow_cc_drop_frac=getattr(config, "snow_cc_drop_frac", 0.5),
        write_snow_rasters=getattr(config, "write_snow_rasters", True),
        write_seep_cluster_rasters=getattr(config, "write_seep_cluster_rasters", True),
        out_dir=_out_dir,
        source_polygons_fp=os.path.join(
            config.training_data_dir, config.training_polygon_fn
        ),
        # GT grouping for the cluster-level matcher. "auto" (default) rule-groups
        # the FULL per-chip GT into seeps for the headline metric; "manual" uses
        # the labeler-grouped sample in gt_labeled_seeps_path for a sample
        # sanity-check. See the 2026-05-29 CLAUDE.md note before choosing.
        gt_grouping_mode=getattr(config, "gt_grouping_mode", "auto"),
        labeled_seeps_fp=getattr(
            config, "gt_labeled_seeps_path",
            os.path.join(ckpt_pred_dir, "gt_seeps_labeled.gpkg"),
        ),
    )