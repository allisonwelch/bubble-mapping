"""PROTOTYPE: learned pairwise 'same-seep?' grouper with a local-density feature.

Reframes grouping as a binary classification over PAIRS OF BUBBLES:
  * one training row = one pair (i, j) of LABELED bubbles within a candidate radius
  * target y = 1 if i and j share a seep_group_id (same seep), else 0
  * features describe the RELATIONSHIP between i and j, including LOCAL DENSITY
    (how crowded the neighborhood is) so the model can group an isolated clump of
    equal-size bubbles but NOT identical-looking bubbles embedded in a busy field.

Inference = predict P(same) per candidate pair (out-of-fold here), keep edges
above a threshold, take connected components -> seeps. Scored on per-class seep
COUNT error (the count-based-flux-relevant metric), same as the anchor prototype.

Local density is computed against the FULL 1585-bubble field; same/different
labels and the grouping are over the 406 labeled bubbles only (partial labels).
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seep_feature_table import anchor_cluster, lonely_cluster  # noqa: E402
from seep_fit_clustering_params import measure_groups  # noqa: E402

LAB = "data/results/SWIN/AE/20260428-1537_SWINxAE.weights/labeling/"
FULL = LAB + "gt_seeps_label_chip39.gpkg"
CLS = LAB + "gt_seeps_label_chip39_classified.gpkg"

CAND_RADIUS = 0.5        # p95 major axis (historical priors): max plausible intra-seep gap
AGGLOM_CAP_M = 1.0       # p99 major axis (historical priors): a cluster wider than this bridges >1 seep

# Real (historical-size) classifier: convex-hull footprint cuts from
# tools/historical_seep_size_priors.py (A|B=137 cm2, B|C=520 cm2).
HIST_AB_M2 = 137.0 / 1e4
HIST_BC_M2 = 520.0 / 1e4
# Per-class flux rates. B/A ~8x stated by Allison; C is a placeholder. NOTE: the
# size-only classifier MASSIVELY over-predicts C (50 vs 2 truth on chip 39), so
# the flux error IS sensitive to the C rate -- the spurious C/B promotions are
# what blow up the flux. The lesson is the classifier, not the rate.
FLUX_RATE = {"A": 1.0, "B": 8.0, "C": 25.0}


def size_class(hull_area_m2):
    if hull_area_m2 < HIST_AB_M2:
        return "A"
    if hull_area_m2 < HIST_BC_M2:
        return "B"
    return "C"
FEATURES = ["dist", "size_ratio", "max_area", "min_area",
            "bright_diff", "dens_025", "dens_050",
            "circ_mean", "ecc_mean", "sol_mean", "circ_diff",
            "loc_max_area", "anchor_ratio"]   # Option A: anchor context


def score(part):
    N_h = part["human"].nunique()
    N_r = part["rule"].nunique()
    frag = int((part.groupby("human")["rule"].nunique() > 1).sum())
    bridge = int((part.groupby("rule")["human"].nunique() > 1).sum())
    dom = part.groupby("rule")["hclass"].agg(lambda s: s.value_counts().index[0])
    rc = dom.value_counts().to_dict()
    return N_h, N_r, frag, bridge, rc


def report(tag, part, N_h, over=None):
    nh, nr, frag, bridge, rc = score(part)
    extra = f" {over:>6}" if over is not None else ""
    print(f"{tag:>30} {nr:>6} {nr-N_h:>+5} {frag:>5} {bridge:>5} "
          f"{rc.get('A',0):>4} {rc.get('B',0):>4} {rc.get('C',0):>4}{extra}")


def decompose(comp, lgroup, lclass, thr):
    """Attribute the per-class count error to fragmentation vs over-merge."""
    df = pd.DataFrame({"human": lgroup, "rule": comp, "hclass": lclass})
    hcls = df.drop_duplicates("human").set_index("human")["hclass"]
    frag = df.groupby("human")["rule"].nunique()
    fragged = frag[frag > 1]
    print(f"\n  -- decomposition at thr={thr} --")
    print(f"  human seeps by class: {dict(hcls.value_counts())}")
    print(f"  fragmented human seeps (split into >1 cluster) by class: "
          f"{hcls.loc[fragged.index].value_counts().to_dict()}  "
          f"(extra clusters: {int((fragged - 1).sum())})")
    rg = df.groupby("rule")["human"].nunique()
    bridged = rg[rg > 1].index
    within = cross = 0
    cross_pairs = {}
    for r in bridged:
        humans = df[df["rule"] == r]["human"].unique()
        classes = sorted(set(hcls.loc[humans]))
        if len(classes) > 1:
            cross += 1
            key = "+".join(classes)
            cross_pairs[key] = cross_pairs.get(key, 0) + 1
        else:
            within += 1
    print(f"  over-merge clusters: total={len(bridged)}  within-class={within}  "
          f"cross-class={cross} {cross_pairs}")


def overcap(comp, xy, cap):
    """# of clusters whose max pairwise member-centroid span exceeds cap (m).

    Centroid span underestimates the true envelope major axis (it ignores the
    bubbles' own radii), so this is a conservative over-merge canary against the
    historical p99 cap -- step 4 will turn it into an actual split constraint.
    """
    over = 0
    for c in np.unique(comp):
        idx = np.where(comp == c)[0]
        if len(idx) < 2:
            continue
        pts = xy[idx]
        d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        if d.max() > cap:
            over += 1
    return over


def main():
    full = gpd.read_file(FULL)
    cls = gpd.read_file(CLS)
    cls = cls.copy()
    cls["hclass"] = cls["class"].fillna("").astype(str).str.strip().str.upper()
    gmap = {(im, s): g for im, s, g in zip(cls["image"], cls["seep_id"], cls["seep_group_id"])}
    full["seep_group_id"] = [gmap.get((im, s), s) for im, s in zip(full["image"], full["seep_id"])]

    # per-bubble arrays (full field for density + anchor context)
    fxy = full[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
    farea = full["area_m2"].to_numpy(float)
    full_tree = cKDTree(fxy)

    lab = cls.copy()
    lab["bright"] = (lab["mean_R"] + lab["mean_G"] + lab["mean_B"]) / 3.0
    L = lab.reset_index(drop=True)
    for col in ["circularity", "eccentricity", "solidity"]:
        L[col] = L[col].fillna(L[col].median())   # RF can't take NaN
    lxy = L[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
    larea = L["area_m2"].to_numpy(float)
    lbright = L["bright"].to_numpy(float)
    lcirc = L["circularity"].to_numpy(float)
    lecc = L["eccentricity"].to_numpy(float)
    lsol = L["solidity"].to_numpy(float)
    lgroup = L["seep_group_id"].to_numpy(np.int64)
    lid = L["seep_id"].to_numpy(np.int64)
    lclass = L["hclass"].to_numpy(object)
    n = len(L)

    # candidate pairs among labeled bubbles within CAND_RADIUS
    ltree = cKDTree(lxy)
    pairs = ltree.query_pairs(r=CAND_RADIUS, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]
    mid = (lxy[i] + lxy[j]) / 2.0
    # local density: full-field bubbles within radius of the midpoint, minus the 2
    dens_025 = np.array([len(full_tree.query_ball_point(m, 0.25)) for m in mid]) - 2
    dens_050 = np.array([len(full_tree.query_ball_point(m, 0.50)) for m in mid]) - 2
    # anchor context: biggest bubble in the pair's 0.5 m neighborhood (full field).
    nbr = [full_tree.query_ball_point(m, 0.50) for m in mid]
    loc_max_area = np.array([farea[ix].max() if ix else 0.0 for ix in nbr])
    pair_max = np.maximum(larea[i], larea[j])
    anchor_ratio = loc_max_area / np.clip(pair_max, 1e-9, None)
    feat = pd.DataFrame({
        "dist": np.hypot(lxy[i, 0] - lxy[j, 0], lxy[i, 1] - lxy[j, 1]),
        "size_ratio": np.minimum(larea[i], larea[j]) / np.maximum(larea[i], larea[j]),
        "max_area": np.maximum(larea[i], larea[j]),
        "min_area": np.minimum(larea[i], larea[j]),
        "bright_diff": np.abs(lbright[i] - lbright[j]),
        "dens_025": np.clip(dens_025, 0, None),
        "dens_050": np.clip(dens_050, 0, None),
        "circ_mean": (lcirc[i] + lcirc[j]) / 2.0,
        "ecc_mean": (lecc[i] + lecc[j]) / 2.0,
        "sol_mean": (lsol[i] + lsol[j]) / 2.0,
        "circ_diff": np.abs(lcirc[i] - lcirc[j]),
        "loc_max_area": loc_max_area,
        "anchor_ratio": anchor_ratio,
    })
    y = (lgroup[i] == lgroup[j]).astype(int)
    print(f"pairs: {len(y)} candidate (within {CAND_RADIUS} m) | "
          f"same-seep={int(y.sum())}  different={int((1-y).sum())}")

    X = feat[FEATURES].to_numpy(float)
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
    print(f"5-fold pair ROC-AUC = {auc:.3f}")
    clf.fit(X, y)
    print("feature importances: " +
          ", ".join(f"{f}={imp:.2f}" for f, imp in
                    sorted(zip(FEATURES, clf.feature_importances_), key=lambda t: -t[1])))

    # out-of-fold P(same) so the grouping eval isn't trained on itself
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]

    N_h = int(pd.Series(lgroup).nunique())
    hc = L.drop_duplicates("seep_group_id")["hclass"].value_counts().to_dict()
    print(f"\nHUMAN: {N_h} seeps among {n} labeled bubbles | class {hc}")
    print(f"\n{'method':>30} {'N_seep':>6} {'bias':>5} {'frag':>5} {'omrg':>5} "
          f"{'A':>4} {'B':>4} {'C':>4} {'>cap':>6}")

    # baseline: current rule at fitted p90 (restricted to labeled)
    meas = measure_groups(full)
    bub = pd.DataFrame({"bubble_id": full["seep_id"].to_numpy(np.int64),
                        "area_m2": full["area_m2"].to_numpy(float),
                        "centroid_x_m": full["centroid_x_m"].to_numpy(float),
                        "centroid_y_m": full["centroid_y_m"].to_numpy(float)})
    base = lonely_cluster(anchor_cluster(
        bub, float(np.percentile(meas["anchor_area_m2"], 5)),
        float(np.percentile(meas["intra_radius_m"], 90)),
        float(np.percentile(meas["max_satellite_area_m2"], 95))))
    bcid = dict(zip(base["bubble_id"], base["cluster_id"]))
    base_rule = np.array([int(bcid[int(s)]) for s in lid])
    base_part = pd.DataFrame({"human": lgroup, "rule": base_rule, "hclass": lclass})
    report("CURRENT rule (fit p90)", base_part, N_h, overcap(base_rule, lxy, AGGLOM_CAP_M))
    # human grouping itself, as an over-cap reference (should be ~0)
    report("HUMAN grouping", pd.DataFrame({"human": lgroup, "rule": lgroup,
           "hclass": lclass}), N_h, overcap(lgroup, lxy, AGGLOM_CAP_M))

    # pairwise grouper at a few thresholds
    for thr in [0.5, 0.6, 0.7]:
        keep = proba >= thr
        rows = np.concatenate([i[keep], j[keep]])
        cols = np.concatenate([j[keep], i[keep]])
        g = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
        _, comp = connected_components(g, directed=False)
        part = pd.DataFrame({"human": lgroup, "rule": comp, "hclass": lclass})
        report(f"pairwise thr={thr:.1f}", part, N_h, overcap(comp, lxy, AGGLOM_CAP_M))
        if abs(thr - 0.7) < 1e-9:
            decompose(comp, lgroup, lclass, thr)
            comp07 = comp

    # ---- flux-weighted re-score with the REAL classifiers ----
    # Per grouped seep, compute the hull footprint + area-weighted brightness of
    # its member bubbles, then classify two ways: (1) size-only historical cuts;
    # (2) a depth-3 tree on [hull_area, mean_R/G/B] -- the seep_fit_class_tree
    # features. Compare both to the oracle (human-class) flux truth.
    geom = L.geometry.reset_index(drop=True)
    larea_lab = L["area_m2"].to_numpy(float)
    lR, lG, lB = (L["mean_R"].to_numpy(float), L["mean_G"].to_numpy(float),
                  L["mean_B"].to_numpy(float))
    TFEAT = ["hull_area_m2", "mean_R", "mean_G", "mean_B"]

    def cluster_features(labels):
        """One row per cluster: hull footprint + area-weighted mean R/G/B."""
        recs = []
        s = pd.Series(labels)
        for cid_, idx in s.groupby(s).groups.items():
            ii = list(idx)
            ha = unary_union(list(geom.iloc[ii].values)).convex_hull.area
            w = larea_lab[ii]
            w = w if w.sum() > 0 else None
            recs.append({"cluster": cid_, "hull_area_m2": ha,
                         "mean_R": np.average(lR[ii], weights=w),
                         "mean_G": np.average(lG[ii], weights=w),
                         "mean_B": np.average(lB[ii], weights=w)})
        return pd.DataFrame(recs)

    def flux(c):
        return sum(c[k] * FLUX_RATE[k] for k in ("A", "B", "C"))

    def counts(arr):
        c = Counter(arr)
        return {k: int(c.get(k, 0)) for k in ("A", "B", "C")}

    def size_counts(labels):
        c = {"A": 0, "B": 0, "C": 0}
        for ha in cluster_features(labels)["hull_area_m2"]:
            c[size_class(ha)] += 1
        return c

    # train the brightness+area tree on the HUMAN seeps (out-of-fold for the
    # classifier-only row so it isn't scored on its own training data).
    hf = cluster_features(lgroup)
    hcls_map = pd.Series(lclass, index=lgroup).groupby(level=0).first()
    hf["cls"] = hf["cluster"].map(hcls_map)
    Xh, yh = hf[TFEAT].to_numpy(float), hf["cls"].to_numpy()
    tree = DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                  random_state=42)
    cvn = max(2, min(5, min(Counter(yh).values())))
    oof = cross_val_predict(tree, Xh, yh, cv=cvn)
    tree.fit(Xh, yh)
    pf = cluster_features(comp07)

    truth = {k: int(hc.get(k, 0)) for k in ("A", "B", "C")}
    f_truth = flux(truth)
    rows = [
        ("TRUTH (human grp+class)", truth),
        ("human grp + size-only", size_counts(lgroup)),
        (f"human grp + tree (OOF cv{cvn})", counts(oof)),
        ("pairwise grp + size-only", size_counts(comp07)),
        ("pairwise grp + tree", counts(tree.predict(pf[TFEAT].to_numpy(float)))),
    ]
    print(f"\n=== flux-weighted re-score (rates A=1 B=8 C=25*) ===")
    print(f"  {'config':>30} {'A':>4} {'B':>4} {'C':>4} {'flux':>8} {'err%':>7}")
    for tag, c in rows:
        f = flux(c)
        err = 100.0 * (f - f_truth) / f_truth if f_truth else float("nan")
        print(f"  {tag:>30} {c['A']:>4} {c['B']:>4} {c['C']:>4} {f:>8.0f} {err:>+6.1f}%")
    print("  (* C rate placeholder; size-only over-promotes, tree uses brightness)")

    print(f"\n(target N_seep={N_h}, bias 0, low omrg, A/B/C ~ {hc})")


if __name__ == "__main__":
    main()
