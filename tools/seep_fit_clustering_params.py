"""Phase 6 (2026-05-28 plan): fit clustering parameters from labeler groupings.

Replaces the PROV_* placeholder constants
(seep_anchor_area_m2 / seep_cluster_radius_m / seep_satellite_max_area_m2) with
values derived from how labelers actually grouped bubbles into seeps.

Input: gt_seeps_grouped.gpkg (PRE-dissolve, from tools/seep_labels_consolidate.py).
Each row is one original polygon tagged with the consensus (image, seep_group_id)
and is_pregrouped.

For every multi-polygon group (n >= 2) that is NOT pre-grouped:
  * anchor area        = max member area_m2
  * intra-group radius = max member-centroid -> anchor-centroid distance
  * max satellite area = max non-anchor member area_m2
Pre-grouped polygons are singletons by construction and are excluded anyway.

The recommended thresholds are conservative percentiles of these distributions:
  * seep_anchor_area_m2      <- low  percentile of anchor areas   (admit most real anchors)
  * seep_cluster_radius_m    <- high percentile of intra-group distances
  * seep_satellite_max_area_m2 <- high percentile of satellite areas

NOTE ON THE RADIUS PERCENTILE. The 2026-05-28 plan suggested the 95th percentile
for the radius. A high radius maximizes grouping *recall* but over-reaches in
dense multi-anchor regions (over-merge). Because fragmentation is recoverable
downstream and over-merge is not, this script SWEEPS the radius percentile and
reports, for each, the cross-validated grouping kappa AND the mean cluster span,
so you pick the operating point rather than defaulting to 95.

Cross-validation: re-run the real two-phase clustering (anchor_cluster +
lonely_cluster from tools/seep_feature_table.py) on the GT polygons per chip
using the candidate parameters, then compare the resulting partition to the
labeler-consensus partition via pairwise Cohen's kappa and Adjusted Rand Index.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seep_feature_table import (  # noqa: E402
    anchor_cluster, lonely_cluster,
    DEFAULT_LONELY_CLUSTER_RADIUS_M, DEFAULT_LONELY_HALO_RADIUS_M,
    DEFAULT_LONELY_MAX_HALO_NEIGHBORS,
)

try:
    from sklearn.metrics import adjusted_rand_score
    _HAS_SK = True
except Exception:
    _HAS_SK = False

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
PRED_DIR = os.path.join(REPO_PATH, "data", "results", "SWIN", "AE",
                        "20260428-1537_SWINxAE.weights")
GROUPED_IN = os.path.join(PRED_DIR, "gt_seeps_grouped.gpkg")

PCTS = [5, 10, 25, 50, 75, 90, 95]


# ---------------------------------------------------------------------------
# Measure labeler groups
# ---------------------------------------------------------------------------
def measure_groups(grouped):
    """Return DataFrame, one row per multi-polygon non-pregrouped group, with
    anchor_area_m2, intra_radius_m, max_satellite_area_m2.
    """
    recs = []
    for (image, gid), g in grouped.groupby(["image", "seep_group_id"]):
        if len(g) < 2:
            continue
        if int(round(g["is_pregrouped"].mean())) == 1:
            continue
        areas = g["area_m2"].to_numpy(dtype=float)
        ai = int(np.argmax(areas))
        ax, ay = (float(g["centroid_x_m"].iloc[ai]),
                  float(g["centroid_y_m"].iloc[ai]))
        d = np.hypot(g["centroid_x_m"].to_numpy() - ax,
                     g["centroid_y_m"].to_numpy() - ay)
        sat = np.delete(areas, ai)
        recs.append({
            "image": image, "seep_group_id": int(gid), "n": len(g),
            "anchor_area_m2": float(areas[ai]),
            "intra_radius_m": float(d.max()),
            "max_satellite_area_m2": float(sat.max()) if len(sat) else 0.0,
        })
    return pd.DataFrame(recs)


def _print_pcts(name, arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        print(f"  {name}: (no data)")
        return {}
    qs = np.percentile(arr, PCTS)
    print(f"  {name} (n={len(arr)}): " +
          "  ".join(f"p{p}={q:.4f}" for p, q in zip(PCTS, qs)))
    return dict(zip(PCTS, qs))


# ---------------------------------------------------------------------------
# Partition-comparison metrics (rule clustering vs labeler consensus)
# ---------------------------------------------------------------------------
def _binary_cohen_kappa(a_same, b_same):
    """Cohen's kappa for two raters on a binary label, given paired 0/1 arrays."""
    a = np.asarray(a_same, dtype=int)
    b = np.asarray(b_same, dtype=int)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    if np.isclose(pe, 1.0):
        return 1.0
    return (po - pe) / (1 - pe)


def partition_agreement(part_a, part_b):
    """Compare two partitions of the same items (dicts item->label, same keys).

    Returns (pairwise_cohen_kappa, adjusted_rand_index). Pairwise arrays are
    pooled across whatever keys are shared.
    """
    keys = sorted(set(part_a) & set(part_b))
    if len(keys) < 2:
        return float("nan"), float("nan")
    # Factorize the (image, label) tuples into 1D integer codes.
    la = pd.factorize(pd.Series([str(part_a[k]) for k in keys]))[0]
    lb = pd.factorize(pd.Series([str(part_b[k]) for k in keys]))[0]
    # pairwise same/different
    n = len(keys)
    iu = np.triu_indices(n, k=1)
    a_same = (la[iu[0]] == la[iu[1]]).astype(int)
    b_same = (lb[iu[0]] == lb[iu[1]]).astype(int)
    kappa = _binary_cohen_kappa(a_same, b_same)
    ari = adjusted_rand_score(la, lb) if _HAS_SK else float("nan")
    return kappa, ari


def rule_cluster_partition(grouped, anchor_area_m2, cluster_radius_m,
                           satellite_max_area_m2, lonely_kwargs):
    """Run anchor+lonely clustering per image on the GT polygons; return a
    dict (image, seep_id) -> rule cluster label, plus the consensus partition.
    """
    rule_part = {}
    cons_part = {}
    spans = []
    for image, g in grouped.groupby("image"):
        bub = pd.DataFrame({
            "bubble_id": g["seep_id"].to_numpy(dtype=np.int64),
            "area_m2": g["area_m2"].to_numpy(dtype=float),
            "centroid_x_m": g["centroid_x_m"].to_numpy(dtype=float),
            "centroid_y_m": g["centroid_y_m"].to_numpy(dtype=float),
        })
        clustered = anchor_cluster(bub, anchor_area_m2, cluster_radius_m,
                                   satellite_max_area_m2)
        clustered = lonely_cluster(clustered, **lonely_kwargs)
        for sid, cid in zip(clustered["bubble_id"], clustered["cluster_id"]):
            rule_part[(image, int(sid))] = (image, int(cid))
        for sid, gid in zip(g["seep_id"], g["seep_group_id"]):
            cons_part[(image, int(sid))] = (image, int(gid))
        # mean cluster span (members per rule cluster) for the over-merge meter
        vc = clustered["cluster_id"].value_counts()
        spans.extend(vc.tolist())
    mean_span = float(np.mean(spans)) if spans else float("nan")
    return rule_part, cons_part, mean_span


def main():
    ap = argparse.ArgumentParser(description="Fit clustering params from labeler groups.")
    ap.add_argument("--grouped", default=GROUPED_IN)
    ap.add_argument("--anchor-pct", type=float, default=5.0)
    ap.add_argument("--radius-pct", type=float, default=90.0,
                    help="recommended radius percentile (default 90, not 95 -- "
                         "see module docstring on over-merge)")
    ap.add_argument("--satellite-pct", type=float, default=95.0)
    ap.add_argument("--plot", action="store_true", help="save histograms")
    args = ap.parse_args()

    if not os.path.exists(args.grouped):
        raise SystemExit(f"not found: {args.grouped}\n"
                         f"Run tools/seep_labels_consolidate.py first.")
    grouped = gpd.read_file(args.grouped)
    print(f"loaded {len(grouped)} polygons, "
          f"{grouped.drop_duplicates(['image','seep_group_id']).shape[0]} consensus seeps")

    meas = measure_groups(grouped)
    if meas.empty:
        raise SystemExit(
            "No multi-polygon non-pregrouped groups found. Either labelers have "
            "not grouped anything yet, or every group is pre-grouped/singleton. "
            "Cannot fit clustering parameters until grouping exists.")
    print(f"\n{len(meas)} multi-polygon seeps drive the fit "
          f"(mean {meas['n'].mean():.1f} polygons/seep, max {meas['n'].max()})")

    print("\npercentile spreads:")
    qa = _print_pcts("anchor_area_m2       ", meas["anchor_area_m2"])
    qr = _print_pcts("intra_radius_m       ", meas["intra_radius_m"])
    qs = _print_pcts("max_satellite_area_m2", meas["max_satellite_area_m2"])

    anchor = float(np.percentile(meas["anchor_area_m2"], args.anchor_pct))
    radius = float(np.percentile(meas["intra_radius_m"], args.radius_pct))
    sat = float(np.percentile(meas["max_satellite_area_m2"], args.satellite_pct))

    lonely_kwargs = dict(
        lonely_cluster_radius_m=DEFAULT_LONELY_CLUSTER_RADIUS_M,
        lonely_halo_radius_m=DEFAULT_LONELY_HALO_RADIUS_M,
        lonely_max_halo_neighbors=DEFAULT_LONELY_MAX_HALO_NEIGHBORS,
    )

    # Sweep the radius percentile: kappa vs over-merge tradeoff.
    print("\ncross-validation sweep (anchor & satellite fixed at recommended pct):")
    print(f"  {'radius_pct':>10} {'radius_m':>9} {'cohen_k':>8} {'ARI':>7} {'mean_span':>9}")
    for rp in [50, 75, 90, 95]:
        r = float(np.percentile(meas["intra_radius_m"], rp))
        rule_part, cons_part, mean_span = rule_cluster_partition(
            grouped, anchor, r, sat, lonely_kwargs)
        kappa, ari = partition_agreement(rule_part, cons_part)
        flag = "  <-- recommended" if rp == args.radius_pct else ""
        print(f"  {rp:>10.0f} {r:>9.4f} {kappa:>8.3f} {ari:>7.3f} "
              f"{mean_span:>9.3f}{flag}")

    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG VALUES (config/configSwinUnet.py):")
    print(f"  seep_anchor_area_m2        = {anchor:.6f}   # p{args.anchor_pct:g} of anchor areas")
    print(f"  seep_cluster_radius_m      = {radius:.6f}   # p{args.radius_pct:g} of intra-group dist")
    print(f"  seep_satellite_max_area_m2 = {sat:.6f}   # p{args.satellite_pct:g} of satellite areas")
    print("=" * 60)
    print("Review the sweep above before committing the radius: a higher kappa at "
          "a smaller radius is preferable, since over-merge is irreversible.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
            for ax, col, ttl in zip(
                    axs, ["anchor_area_m2", "intra_radius_m", "max_satellite_area_m2"],
                    ["anchor area (m^2)", "intra-group radius (m)", "max satellite area (m^2)"]):
                ax.hist(meas[col], bins=30)
                ax.set_title(ttl)
            fig.tight_layout()
            out = os.path.join(PRED_DIR, "clustering_param_distributions.png")
            fig.savefig(out, dpi=120)
            print(f"\nsaved histograms -> {os.path.basename(out)}")
        except Exception as e:
            print(f"(plot skipped: {e})")

    return anchor, radius, sat


if __name__ == "__main__":
    main()
