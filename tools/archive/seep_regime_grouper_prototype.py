"""PROTOTYPE Option B: explicit two-regime grouper.

The chip-39 decomposition showed the pairwise model's residual error is one
mechanism: large, spread-out B/C seeps FRAGMENT under any global edge threshold
(a connected-components cut has no notion of "this whole thing is one anchored
seep"). Option A (anchor-context features) barely moved B fragmentation.

Option B handles the two regimes with two mechanisms instead of one threshold:
  Phase 1  ANCHOR-GATHER (for B/C): each local-area-maximum bubble is an anchor;
           it gathers every bubble out to a size-scaled reach into ONE seep, so a
           big seep stays whole. Two comparable big bubbles are both anchors ->
           they stay separate (the over-merge guard). [reused from
           seep_anchor_fix_prototype.local_max_anchor_cluster]
  Phase 2  DENSITY-MERGE (for A): the leftover singletons that sit in a DENSE
           neighborhood get merged by proximity -- this is the dense small-bubble
           field that must become one seep, not many singleton A's. (Opposite gate
           to the retired lonely-cluster, which only merged in SPARSE halos.)

Scored on per-class seep COUNT vs the chip-39 human grouping, with the same
B-frag / A+B-overmerge decomposition used to judge Options A/0. Grouping runs on
the full 1585-bubble field; scoring is on the 406 labeled bubbles.
"""
import os
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seep_pairwise_grouper_prototype import report, decompose, overcap  # noqa: E402

LAB = "data/results/SWIN/AE/20260428-1537_SWINxAE.weights/labeling/"
FULL = LAB + "gt_seeps_label_chip39.gpkg"
CLS = LAB + "gt_seeps_label_chip39_classified.gpkg"
CAP_M = 1.0          # historical p99 major axis (over-merge canary)
DENS_RADIUS = 0.25   # neighborhood radius for the "is this a dense field?" test


def absolute_anchor_cluster(df, anchor_thresh, reach_scale, cap):
    """Phase 1: only genuinely LARGE bubbles (area >= anchor_thresh) are anchors.

    Each anchor gathers every bubble out to reach = reach_scale * anchor_radius
    (capped at cap/2) into its seep, so a big seep is held whole. Small bubbles
    out of any anchor's reach stay singletons -> handed to the density phase.
    Rare anchors (vs local-max) means the dense field is NOT shattered into
    per-bump fragments.
    """
    df = df.copy()
    xy = df[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
    area = df["area_m2"].to_numpy(float)
    ids = df["bubble_id"].to_numpy(np.int64)
    cid = ids.copy()
    anc = np.where(area >= anchor_thresh)[0]
    non = np.where(area < anchor_thresh)[0]
    if len(anc) and len(non):
        reach = np.minimum(reach_scale * np.sqrt(area[anc] / np.pi), cap / 2.0)
        D = np.sqrt(((xy[non][:, None, :] - xy[anc][None, :, :]) ** 2).sum(2))
        masked = np.where(D <= reach[None, :], D, np.inf)
        best = masked.argmin(1)
        bd = masked[np.arange(len(non)), best]
        ok = np.isfinite(bd)
        cid[non[ok]] = ids[anc][best[ok]]
    df["cluster_id"] = cid
    return df


def density_merge(df, full_tree, r_density, min_dense):
    """Merge Phase-1 residual singletons that are close AND in a dense field.

    A residual singleton joins its neighbors only where the local full-field
    density (bubbles within DENS_RADIUS) is at least min_dense -- so dense small
    fields collapse into seeps while isolated lone bubbles stay singletons.
    """
    df = df.copy()
    cid = df["cluster_id"].to_numpy(np.int64).copy()
    xy = df[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
    ids = df["bubble_id"].to_numpy(np.int64)

    vc = pd.Series(cid).value_counts()
    singletons = set(vc[vc == 1].index)
    res = np.array([i for i, c in enumerate(cid) if c in singletons])
    if len(res) < 2:
        df["cluster_id"] = cid
        return df

    dens = np.array([len(full_tree.query_ball_point(xy[k], r_density)) - 1
                     for k in res])
    dense_res = res[dens >= min_dense]
    if len(dense_res) < 2:
        df["cluster_id"] = cid
        return df

    sub = xy[dense_res]
    pairs = cKDTree(sub).query_pairs(r=r_density * 2, output_type="ndarray")
    if len(pairs):
        n = len(dense_res)
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        g = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        for c in np.unique(comp):
            members = dense_res[comp == c]
            if len(members) >= 2:
                cid[members] = int(ids[members].min())
    df["cluster_id"] = cid
    return df


def main():
    full = gpd.read_file(FULL)
    cls = gpd.read_file(CLS).copy()
    cls["hclass"] = cls["class"].fillna("").astype(str).str.strip().str.upper()
    gmap = {(im, s): g for im, s, g in
            zip(cls["image"], cls["seep_id"], cls["seep_group_id"])}
    full["seep_group_id"] = [gmap.get((im, s), s)
                             for im, s in zip(full["image"], full["seep_id"])]

    bub = pd.DataFrame({
        "bubble_id": full["seep_id"].to_numpy(np.int64),
        "area_m2": full["area_m2"].to_numpy(float),
        "centroid_x_m": full["centroid_x_m"].to_numpy(float),
        "centroid_y_m": full["centroid_y_m"].to_numpy(float),
    })
    full_tree = cKDTree(bub[["centroid_x_m", "centroid_y_m"]].to_numpy(float))

    labeled = cls[["seep_id", "seep_group_id", "hclass",
                   "centroid_x_m", "centroid_y_m"]].copy()
    lid = labeled["seep_id"].to_numpy(np.int64)
    lgroup = labeled["seep_group_id"].to_numpy(np.int64)
    lclass = labeled["hclass"].to_numpy(object)
    lxy = labeled[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
    N_h = int(pd.Series(lgroup).nunique())
    hc = labeled.drop_duplicates("seep_group_id")["hclass"].value_counts().to_dict()
    print(f"HUMAN: {N_h} seeps among {len(labeled)} labeled bubbles | class {hc}")
    print(f"\n{'method':>34} {'N_seep':>6} {'bias':>5} {'frag':>5} {'omrg':>5} "
          f"{'A':>4} {'B':>4} {'C':>4} {'>cap':>6}")

    def run(tag, cluster_df, decomp=False):
        cid = dict(zip(cluster_df["bubble_id"], cluster_df["cluster_id"]))
        rule = np.array([int(cid[int(s)]) for s in lid])
        part = pd.DataFrame({"human": lgroup, "rule": rule, "hclass": lclass})
        report(tag, part, N_h, overcap(rule, lxy, CAP_M))
        if decomp:
            decompose(rule, lgroup, lclass, tag)

    # human reference
    run("HUMAN grouping", pd.DataFrame({"bubble_id": lid, "cluster_id": lgroup}))

    # Option B sweep: absolute-anchor threshold x reach x density gate
    area_all = bub["area_m2"].to_numpy(float)
    best = None
    for ap in [90, 95, 98]:
        athr = float(np.percentile(area_all, ap))
        for scale in [3.0, 5.0]:
            anchored = absolute_anchor_cluster(bub, athr, scale, CAP_M)
            for min_dense in [2, 3]:
                merged = density_merge(anchored, full_tree, DENS_RADIUS, min_dense)
                cid = dict(zip(merged["bubble_id"], merged["cluster_id"]))
                rule = np.array([int(cid[int(s)]) for s in lid])
                nr = len(set(rule))
                bdf = pd.DataFrame({"human": lgroup, "rule": rule, "hclass": lclass})
                dom = bdf.groupby("rule")["hclass"].agg(
                    lambda s: s.value_counts().index[0])
                nb = int((dom == "B").sum())
                tag = f"B aThr=p{ap} sc={scale:.0f} d>={min_dense}"
                run(tag, merged)
                key = (abs(nb - hc.get("B", 0)), abs(nr - N_h))
                if best is None or key < best[0]:
                    best = (key, tag, merged)

    print(f"\nbest-B run -> {best[1]}; decomposition:")
    run(best[1], best[2], decomp=True)
    print(f"\n(target N_seep={N_h}, low B-frag, B~{hc.get('B',0)}, A~{hc.get('A',0)})")


if __name__ == "__main__":
    main()
