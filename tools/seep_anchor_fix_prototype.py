"""PROTOTYPE: local-max-anchor clustering, scored on per-class seep COUNT error.

Tests the anchor-mechanism fix discussed 2026-06-01 against the current
anchor+lonely rule on chip 39. Two ideas:
  * ANCHOR = local maximum. A bubble is an anchor only if it is the largest-area
    bubble within `nms_radius_m` (ties -> larger bubble_id). Two comparable big
    bubbles near each other are BOTH anchors -> stay separate seep heads, which
    preserves the over-merge guard (the thing pure density clustering loses).
  * REACH scales with anchor size: reach = radius_scale * sqrt(anchor_area/pi)
    (the anchor's own effective radius x scale). Big seeps reach far, small seeps
    stay tight -> addresses the single-global-radius limitation.
No lonely phase here: chip 39 is too dense to contain/tune lonely clusters.

Metric = per-class seep COUNT error (the count-based-flux-relevant one), with
fragmentation (human seep split) and over-merge (rule cluster bridges >1 human
seep) reported underneath. Each rule cluster is assigned the dominant human class
of its labeled members, so this isolates GROUPING's effect on per-class counts
(no classifier noise).
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seep_fit_clustering_params import measure_groups  # noqa: E402
from seep_feature_table import anchor_cluster, lonely_cluster  # noqa: E402

LAB = "data/results/SWIN/AE/20260428-1537_SWINxAE.weights/labeling/"
FULL = LAB + "gt_seeps_label_chip39.gpkg"
CLS = LAB + "gt_seeps_label_chip39_classified.gpkg"


def local_max_anchor_cluster(df, nms_radius_m, radius_scale):
    """Anchor = local area-max within nms_radius_m; reach scales with anchor size."""
    df = df.copy()
    xy = df[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
    area = df["area_m2"].to_numpy(float)
    ids = df["bubble_id"].to_numpy(np.int64)
    n = len(df)
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(2))
    within = (D <= nms_radius_m)
    np.fill_diagonal(within, False)
    # i is suppressed if a neighbor has larger area (ties broken by larger id)
    bigger = (area[None, :] > area[:, None]) | (
        (area[None, :] == area[:, None]) & (ids[None, :] > ids[:, None]))
    suppressed = (within & bigger).any(axis=1)
    is_anchor = ~suppressed
    df["is_anchor"] = is_anchor
    df["cluster_id"] = ids
    anc = np.where(is_anchor)[0]
    non = np.where(~is_anchor)[0]
    if len(anc) == 0 or len(non) == 0:
        return df
    reach = radius_scale * np.sqrt(area[anc] / np.pi)         # per-anchor reach
    Dna = D[np.ix_(non, anc)]
    masked = np.where(Dna <= reach[None, :], Dna, np.inf)
    best = masked.argmin(axis=1)
    bestd = masked[np.arange(len(non)), best]
    new = np.where(np.isfinite(bestd), ids[anc][best], ids[non])
    cid = df["cluster_id"].to_numpy(np.int64).copy()
    cid[non] = new
    df["cluster_id"] = cid
    return df


def score(part):
    """part: DataFrame with columns human, rule, hclass (labeled bubbles only)."""
    N_h = part["human"].nunique()
    N_r = part["rule"].nunique()
    frag = int((part.groupby("human")["rule"].nunique() > 1).sum())
    bridge = int((part.groupby("rule")["human"].nunique() > 1).sum())
    dom = part.groupby("rule")["hclass"].agg(lambda s: s.value_counts().index[0])
    rc = dom.value_counts().to_dict()
    hc = part.drop_duplicates("human").set_index("human")["hclass"].value_counts().to_dict()
    return N_h, N_r, frag, bridge, hc, rc


def make_part(full, cluster_df, labeled):
    cid = dict(zip(cluster_df["bubble_id"], cluster_df["cluster_id"]))
    rows = []
    for _, r in labeled.iterrows():
        rows.append({"human": int(r["seep_group_id"]),
                     "rule": int(cid[int(r["seep_id"])]),
                     "hclass": r["hclass"]})
    return pd.DataFrame(rows)


def main():
    full = gpd.read_file(FULL)
    cls = gpd.read_file(CLS)
    cls = cls.copy()
    cls["hclass"] = cls["class"].fillna("").astype(str).str.strip().str.upper()
    gmap = {(im, s): g for im, s, g in zip(cls["image"], cls["seep_id"], cls["seep_group_id"])}
    pmap = {(im, s): p for im, s, p in zip(cls["image"], cls["seep_id"], cls["is_pregrouped"])}
    full["seep_group_id"] = [gmap.get((im, s), s) for im, s in zip(full["image"], full["seep_id"])]
    full["is_pregrouped"] = [int(pmap.get((im, s), 0)) for im, s in zip(full["image"], full["seep_id"])]

    bub = pd.DataFrame({
        "bubble_id": full["seep_id"].to_numpy(np.int64),
        "area_m2": full["area_m2"].to_numpy(float),
        "centroid_x_m": full["centroid_x_m"].to_numpy(float),
        "centroid_y_m": full["centroid_y_m"].to_numpy(float),
    })
    labeled = cls[["image", "seep_id", "seep_group_id", "hclass"]]

    meas = measure_groups(full)
    a5 = float(np.percentile(meas["anchor_area_m2"], 5))
    r90 = float(np.percentile(meas["intra_radius_m"], 90))
    s95 = float(np.percentile(meas["max_satellite_area_m2"], 95))

    # ground truth
    N_h = labeled["seep_group_id"].nunique()
    hc = labeled.drop_duplicates("seep_group_id")["hclass"].value_counts().to_dict()
    print(f"HUMAN: {N_h} seeps among {len(labeled)} labeled bubbles | class counts {hc}")
    print(f"\n{'method':>34} {'N_seep':>6} {'bias':>5} {'frag':>5} {'omrg':>5} "
          f"{'A':>4} {'B':>4} {'C':>4}")

    def report(tag, cdf):
        part = make_part(full, cdf, labeled)
        nh, nr, frag, bridge, _hc, rc = score(part)
        print(f"{tag:>34} {nr:>6} {nr-nh:>+5} {frag:>5} {bridge:>5} "
              f"{rc.get('A',0):>4} {rc.get('B',0):>4} {rc.get('C',0):>4}")

    # baseline: current rule at fitted p90
    base = anchor_cluster(bub, a5, r90, s95)
    base = lonely_cluster(base)
    report("CURRENT anchor+lonely (fit p90)", base)

    # new: local-max anchor sweep
    for nms in [0.10, 0.15, 0.20, 0.30]:
        for scale in [2.0, 3.0, 4.0, 6.0]:
            cdf = local_max_anchor_cluster(bub, nms, scale)
            report(f"localmax nms={nms:.2f} scale={scale:.0f}", cdf)

    print(f"\n(target: N_seep ~ {N_h}, bias ~ 0, low omrg; A/B/C ~ {hc})")


if __name__ == "__main__":
    main()
