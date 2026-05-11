# tools/seep_feature_table.py
"""
Per-bubble and per-seep-cluster feature tables for predicted seeps.

Inputs (already on disk after evaluation + write_seep_rasters):
  - {stem}_cc.tif      per-bubble connected components (uint16/uint32)
  - {basename}.tif     matching chip in chip_dir (RGB bands for brightness)

Outputs (written to pred_dir):
  - seep_features_per_bubble.csv   one row per detected bubble
  - seep_features_per_cluster.csv  one row per anchor-conditioned cluster
  - {stem}_seep_cluster.tif        pixel value = cluster_id (0 = background)

Anchor-conditional clustering: any bubble with area_m2 >= ANCHOR_AREA_M2 is an
anchor and always a cluster head. Non-anchor bubbles within CLUSTER_RADIUS_M of
an anchor centroid join that anchor's cluster (nearest anchor wins on ties).
Non-anchors with no anchor in range form singleton clusters. A group of
similarly-sized small bubbles never merges because no anchor is present —
which is the point: protect common small-A's from being clustered together
while still capturing the rare "big bubble + nearby satellites" B pattern.
"""
import os
import glob
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy as rio_xy
from skimage.measure import regionprops
from tqdm import tqdm


# Defaults (overridable per call or via config). Anchor area = pi * (25 cm)^2.
DEFAULT_ANCHOR_AREA_M2 = float(np.pi * (0.25 ** 2))
DEFAULT_CLUSTER_RADIUS_M = 0.5
# Satellite area cap: a non-anchor bubble must be at most this large to be
# eligible as a satellite. None disables the cap (every non-anchor is eligible).
# When set, non-anchors above the cap are forced to be singletons even if an
# anchor is in range — guards against medium-sized bubbles getting absorbed.
DEFAULT_SATELLITE_MAX_AREA_M2 = None

_AUX_SUFFIXES = ("_prob.tif", "_epistemic.tif", "_aleatoric.tif",
                 "_smoothed.tif", "_cc.tif", "_seep_cluster.tif")


def _aux_path(pred_fp, suffix):
    stem = os.path.splitext(pred_fp)[0]
    return f"{stem}{suffix}"


def compute_bubble_features(cc_lab, image, transform):
    """One row per CC. Morphological + RGB-brightness features."""
    pix_m = abs(transform.a)
    has_rgb = image.ndim == 3 and image.shape[2] >= 3
    rows = []
    for p in regionprops(cc_lab):
        ar_m2 = p.area * (pix_m ** 2)
        per_m = max(p.perimeter * pix_m, 1e-9)
        circ = 4 * np.pi * ar_m2 / (per_m ** 2)
        cy_px, cx_px = p.centroid
        cx_m, cy_m = rio_xy(transform, cy_px, cx_px)
        ys, xs = np.where(cc_lab == p.label)
        if has_rgb:
            mR = float(image[ys, xs, 0].mean())
            mG = float(image[ys, xs, 1].mean())
            mB = float(image[ys, xs, 2].mean())
        else:
            v = float(image[ys, xs].mean()) if image.ndim == 2 \
                else float(image[ys, xs, 0].mean())
            mR = mG = mB = v
        rows.append({
            "bubble_id": int(p.label),
            "centroid_x_m": float(cx_m),
            "centroid_y_m": float(cy_m),
            "area_m2": float(ar_m2),
            "perim_m": float(per_m),
            "circularity": float(circ),
            "mean_R": mR, "mean_G": mG, "mean_B": mB,
        })
    return pd.DataFrame(rows, columns=[
        "bubble_id", "centroid_x_m", "centroid_y_m",
        "area_m2", "perim_m", "circularity",
        "mean_R", "mean_G", "mean_B",
    ])


def anchor_cluster(bubble_df, anchor_area_m2, cluster_radius_m,
                   satellite_max_area_m2=DEFAULT_SATELLITE_MAX_AREA_M2):
    """Add `is_anchor`, `is_satellite_eligible`, and `cluster_id` columns.

    Bubbles with `area_m2 >= anchor_area_m2` are anchors (cluster heads).
    Bubbles with `area_m2 < anchor_area_m2` AND (cap is None OR
    `area_m2 <= satellite_max_area_m2`) are satellite-eligible. Eligible
    bubbles within `cluster_radius_m` of an anchor join that anchor's
    cluster (nearest anchor wins). Anything else becomes its own singleton.
    """
    df = bubble_df.copy()
    df["is_anchor"] = df["area_m2"] >= anchor_area_m2
    cap = satellite_max_area_m2
    if cap is None or not np.isfinite(cap):
        df["is_satellite_eligible"] = ~df["is_anchor"]
    else:
        df["is_satellite_eligible"] = (~df["is_anchor"]) & (df["area_m2"] <= cap)
    df["cluster_id"] = df["bubble_id"].astype(np.int64).values
    if not df["is_anchor"].any() or not df["is_satellite_eligible"].any():
        return df

    anc = df[df["is_anchor"]]
    non = df[df["is_satellite_eligible"]]
    n_xy = non[["centroid_x_m", "centroid_y_m"]].values   # (N, 2)
    a_xy = anc[["centroid_x_m", "centroid_y_m"]].values   # (A, 2)
    d = np.sqrt(((n_xy[:, None, :] - a_xy[None, :, :]) ** 2).sum(axis=2))
    min_i = d.argmin(axis=1)
    min_d = d[np.arange(len(non)), min_i]
    in_range = min_d <= cluster_radius_m
    assigned = anc["bubble_id"].values[min_i].astype(np.int64)
    fallback = non["bubble_id"].values.astype(np.int64)
    cluster_ids = np.where(in_range, assigned, fallback)
    df.loc[non.index, "cluster_id"] = cluster_ids
    return df


def aggregate_clusters(bubble_df):
    """One row per cluster. Pixel-weighted aggregates use area as the weight."""
    rows = []
    for cid, grp in bubble_df.groupby("cluster_id", sort=True):
        n = len(grp)
        has_anchor = bool(grp["is_anchor"].any())
        anc = grp[grp["is_anchor"]]
        anchor_area = float(anc["area_m2"].iloc[0]) if has_anchor else np.nan

        total_area = float(grp["area_m2"].sum())
        max_area = float(grp["area_m2"].max())
        mean_area = float(grp["area_m2"].mean())
        std_area = float(grp["area_m2"].std()) if n > 1 else 0.0
        ratio = max_area / mean_area if mean_area > 0 else np.nan

        if has_anchor and n > 1:
            ax = float(anc["centroid_x_m"].iloc[0])
            ay = float(anc["centroid_y_m"].iloc[0])
            cr = float(np.hypot(grp["centroid_x_m"].values - ax,
                                grp["centroid_y_m"].values - ay).max())
        else:
            cr = 0.0

        w = grp["area_m2"].values
        wsum = float(w.sum()) if w.sum() > 0 else 1.0
        cx = float((grp["centroid_x_m"].values * w).sum() / wsum)
        cy = float((grp["centroid_y_m"].values * w).sum() / wsum)
        mR = float((grp["mean_R"].values * w).sum() / wsum)
        mG = float((grp["mean_G"].values * w).sum() / wsum)
        mB = float((grp["mean_B"].values * w).sum() / wsum)
        mcirc = float((grp["circularity"].values * w).sum() / wsum)

        rows.append({
            "cluster_id": int(cid),
            "n_bubbles": int(n),
            "has_anchor": has_anchor,
            "anchor_area_m2": anchor_area,
            "total_area_m2": total_area,
            "max_area_m2": max_area,
            "mean_area_m2": mean_area,
            "std_area_m2": std_area,
            "area_ratio_max_to_mean": ratio,
            "cluster_radius_m": cr,
            "centroid_x_m": cx, "centroid_y_m": cy,
            "mean_R": mR, "mean_G": mG, "mean_B": mB,
            "mean_circularity_weighted": mcirc,
        })
    return pd.DataFrame(rows)


def build_cluster_raster(cc_lab, bubble_df):
    """Remap CC labels to cluster IDs. Background (label 0) stays 0."""
    max_id = int(cc_lab.max()) if cc_lab.size > 0 else 0
    lut = np.zeros(max_id + 1, dtype=np.uint32)
    for bid, cid in zip(bubble_df["bubble_id"].values.astype(int),
                        bubble_df["cluster_id"].values.astype(int)):
        lut[bid] = cid
    return lut[cc_lab]


def _write_cluster_raster(pred_fp, raster, profile):
    out = _aux_path(pred_fp, "r135_seep_cluster.tif")
    prof = profile.copy()
    max_val = int(raster.max()) if raster.size > 0 else 0
    dtype = "uint16" if max_val <= 65535 else "uint32"
    prof.update(driver="GTiff", count=1, dtype=dtype, compress="lzw", nodata=0)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(raster.astype(dtype), 1)


def _load_inputs(pred_fp, chip_fp):
    """Standalone-mode loader: read _cc.tif (must exist) and the chip image."""
    cc_fp = _aux_path(pred_fp, "_cc.tif")
    if not os.path.exists(cc_fp):
        raise FileNotFoundError(
            f"_cc.tif missing for {os.path.basename(pred_fp)}: {cc_fp}\n"
            f"Run tools/write_seep_rasters.py first to generate per-bubble CCs."
        )
    with rasterio.open(cc_fp) as src:
        cc = src.read(1).astype(np.int64)
        transform = src.transform
        profile = src.profile.copy()
    with rasterio.open(chip_fp) as src:
        n = src.count
        if n > 1:
            image = np.transpose(src.read(list(range(1, n))), (1, 2, 0))
        else:
            image = src.read(1)
    return cc, image, transform, profile


def process_pred(pred_fp, chip_fp,
                 cc=None, image=None, transform=None, profile=None,
                 anchor_area_m2=DEFAULT_ANCHOR_AREA_M2,
                 cluster_radius_m=DEFAULT_CLUSTER_RADIUS_M,
                 satellite_max_area_m2=DEFAULT_SATELLITE_MAX_AREA_M2,
                 write_raster=True):
    """
    Per-image extraction.

    Pass pre-computed cc/image/transform/profile to skip disk reads (used by
    seep_level_eval.py during its main loop). Otherwise reads _cc.tif and the
    chip from disk. Returns (bubbles_df, clusters_df), both prefixed with an
    'image' column.
    """
    if cc is None or image is None or transform is None or profile is None:
        _cc, _image, _transform, _profile = _load_inputs(pred_fp, chip_fp)
        cc = _cc if cc is None else cc
        image = _image if image is None else image
        transform = _transform if transform is None else transform
        profile = _profile if profile is None else profile

    bubbles = compute_bubble_features(cc, image, transform)
    bubbles = anchor_cluster(bubbles, anchor_area_m2, cluster_radius_m,
                             satellite_max_area_m2=satellite_max_area_m2)
    clusters = aggregate_clusters(bubbles)

    name = os.path.basename(pred_fp)
    bubbles.insert(0, "image", name)
    clusters.insert(0, "image", name)

    if write_raster and len(bubbles) > 0:
        raster = build_cluster_raster(cc, bubbles)
        _write_cluster_raster(pred_fp, raster, profile)
    return bubbles, clusters


def write_feature_csvs(pred_dir, bubbles, clusters,
                       anchor_area_m2=DEFAULT_ANCHOR_AREA_M2,
                       cluster_radius_m=DEFAULT_CLUSTER_RADIUS_M,
                       satellite_max_area_m2=DEFAULT_SATELLITE_MAX_AREA_M2):
    bubbles.to_csv(os.path.join(pred_dir, "seep_features_per_bubble.csv"),
                   index=False)
    clusters.to_csv(os.path.join(pred_dir, "seep_features_per_cluster.csv"),
                    index=False)
    _print_summary(bubbles, clusters, anchor_area_m2, cluster_radius_m,
                   satellite_max_area_m2)


def _print_summary(bubbles, clusters, anchor_area_m2, cluster_radius_m,
                   satellite_max_area_m2):
    n_bub = len(bubbles)
    n_anc = int(bubbles["is_anchor"].sum())
    n_clu = len(clusters)
    n_sat = int(((~bubbles["is_anchor"]) &
                 (bubbles["cluster_id"] != bubbles["bubble_id"])).sum())
    n_singleton = int(clusters["n_bubbles"].eq(1).sum())
    # "Medium" bubbles: non-anchor, area above the satellite cap (forced singletons).
    if "is_satellite_eligible" in bubbles.columns:
        n_medium = int(((~bubbles["is_anchor"]) &
                        (~bubbles["is_satellite_eligible"])).sum())
    else:
        n_medium = 0
    cap_str = (f"{satellite_max_area_m2:.4f}"
               if satellite_max_area_m2 is not None
               and np.isfinite(satellite_max_area_m2) else "off")
    print(f"\nSEEP-FEATURE: anchor_area_m2={anchor_area_m2:.4f}  "
          f"cluster_radius_m={cluster_radius_m:.2f}  "
          f"satellite_max_area_m2={cap_str}")
    print(f"  n_bubbles={n_bub}  n_anchors={n_anc}  n_medium={n_medium}  "
          f"n_clusters={n_clu}  n_satellites={n_sat}  "
          f"n_singletons={n_singleton}")
    hist = clusters["n_bubbles"].value_counts().sort_index()
    print("  cluster-size histogram (n_bubbles -> n_clusters):")
    for size, count in hist.items():
        print(f"    {size:>4} -> {count}")


def process_dir(pred_dir, chip_dir,
                anchor_area_m2=DEFAULT_ANCHOR_AREA_M2,
                cluster_radius_m=DEFAULT_CLUSTER_RADIUS_M,
                satellite_max_area_m2=DEFAULT_SATELLITE_MAX_AREA_M2,
                write_rasters=True):
    pred_fps = [
        fp for fp in sorted(glob.glob(os.path.join(pred_dir, "*.tif")))
        if not fp.endswith(_AUX_SUFFIXES)
    ]
    bubble_dfs, cluster_dfs = [], []
    for pred_fp in tqdm(pred_fps, desc="Seep features"):
        chip_fp = os.path.join(chip_dir, os.path.basename(pred_fp))
        if not os.path.exists(chip_fp):
            continue
        b, c = process_pred(
            pred_fp, chip_fp,
            anchor_area_m2=anchor_area_m2,
            cluster_radius_m=cluster_radius_m,
            satellite_max_area_m2=satellite_max_area_m2,
            write_raster=write_rasters,
        )
        bubble_dfs.append(b)
        cluster_dfs.append(c)
    if not bubble_dfs:
        raise RuntimeError(
            f"No predictions processed.\n  pred_dir={pred_dir}\n  chip_dir={chip_dir}"
        )
    bubbles = pd.concat(bubble_dfs, ignore_index=True)
    clusters = pd.concat(cluster_dfs, ignore_index=True)
    write_feature_csvs(pred_dir, bubbles, clusters,
                       anchor_area_m2=anchor_area_m2,
                       cluster_radius_m=cluster_radius_m,
                       satellite_max_area_m2=satellite_max_area_m2)
    return bubbles, clusters


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import configSwinUnet
    config = configSwinUnet.Configuration().validate()
    ckpt_pred_dir = os.path.join(config.results_dir, "20260428-1537_SWINxAE.weights")
    process_dir(
        pred_dir=ckpt_pred_dir,
        chip_dir=config.preprocessed_dir,
        anchor_area_m2=getattr(config, "seep_anchor_area_m2",
                               DEFAULT_ANCHOR_AREA_M2),
        cluster_radius_m=getattr(config, "seep_cluster_radius_m",
                                 DEFAULT_CLUSTER_RADIUS_M),
        satellite_max_area_m2=getattr(config, "seep_satellite_max_area_m2",
                                      DEFAULT_SATELLITE_MAX_AREA_M2),
        write_rasters=getattr(config, "write_seewp_cluster_rasters", True),
    )