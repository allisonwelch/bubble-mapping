"""Learned pairwise 'same-seep?' grouper trained on the COMBINED multi-image
quarter labels (Katey) + the chip-39 grouped labels (same labeler).

This generalises ``seep_pairwise_grouper_prototype.py`` (chip-39 only) to a set
of images, each with its OWN ``seep_group_id`` namespace. Grouping is framed as
binary classification over pairs of labeled bubbles:

  * one training row = one pair (i, j) of labeled bubbles WITHIN THE SAME IMAGE
    and within a candidate radius (historical prior, 0.5 m);
  * target y = 1 iff i and j share the same (image, seep_group_id) -- i.e. the
    same physical seep;
  * features describe the relationship between i and j, including LOCAL DENSITY /
    anchor context computed against that image's FULL bubble field.

Inference = predict P(same) per candidate pair (out-of-fold), keep edges above a
threshold, take connected components -> seeps. Scored on per-class seep COUNT
error (the count-based-flux metric), same as the chip-39 prototype.

Two models are fit and compared per the request: an interpretable
``DecisionTreeClassifier`` and a ``RandomForestClassifier``.

ID-NAMESPACE SAFETY: groups are keyed on the (image, seep_group_id) TUPLE and
candidate pairs are generated within each image only, so the cross-image id
reuse (4.tif and 38.tif both starting seep_group_id at 1, 2, ...) can never
merge bubbles across chips. See the module docstring of the companion plan.

DATA SOURCES
  * gt_seeps_label_chip39_classified.gpkg -- 39.tif, 406 bubbles / 143 groups
    (the authority for 39.tif; per-bubble, NOT dissolved).
  * gt_seeps_label_quarters_katey_kwa.gpkg -- 4/38/21/39.tif quarter labels.
    Its 39.tif rows are DROPPED (chip-39 file is the authority) to avoid a
    seep_id / seep_group_id collision on that image.
  * gt_seeps_label_chip39.gpkg -- full 1585-bubble 39.tif field (density only).

ROW FILTERING (target/training)
  * processed only: ``class`` non-empty (ungrouped rows have blank class).
  * is_overgrouped == 1: EXCLUDED (a single polygon spanning >1 seep cannot be
    split here; flagged for relabel). 1 row.
  * is_context == 1: kept in the per-image density field and as candidate-pair
    NEIGHBOURS, but truncated context groups are NOT scored as targets (their
    members live partly outside the quarter). Pairs with a context endpoint are
    dropped from train/eval.
  * is_pregrouped == 1: a polygon that already encloses a whole seep. Kept as a
    FIXED SINGLETON seep (counts as one seep) but excluded from pair generation
    -- it never merges with a neighbour.
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
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_predict, cross_val_score, GroupKFold
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from collections import Counter
import warnings
from sklearn.exceptions import UndefinedMetricWarning

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seep_feature_table import anchor_cluster, lonely_cluster  # noqa: E402
from seep_fit_clustering_params import measure_groups  # noqa: E402

LAB = "data/results/SWIN/AE/20260428-1537_SWINxAE.weights/labeling/"
# The grouper's training inputs were archived into backup_pre_classified_20260624/
# on 2026-06-24, and katey_kwa was later renamed to katey_grouped. Resolve each
# against its known aliases so the tool keeps working wherever they now live --
# contents are byte-identical, so this is a path fix, not a retrain.
_BACKUP = LAB + "backup_pre_classified_20260624/"


def _resolve(*candidates):
    for c in candidates:
        if os.path.exists(c):
            return c
    raise SystemExit("[fail] none of these grouper training inputs exist:\n  "
                     + "\n  ".join(candidates))


KWA = _resolve(LAB + "gt_seeps_label_quarters_katey_kwa.gpkg",
               LAB + "gt_seeps_label_quarters_katey_grouped.gpkg",
               _BACKUP + "gt_seeps_label_quarters_katey_kwa.gpkg")
C39_CLS = _resolve(LAB + "gt_seeps_label_chip39_classified.gpkg",
                   _BACKUP + "gt_seeps_label_chip39_classified.gpkg")
# full field for density on 39.tif
C39_FULL = _resolve(LAB + "gt_seeps_label_chip39.gpkg",
                    _BACKUP + "gt_seeps_label_chip39.gpkg")

# historical_seep_size_priors.json -> "grouping" block.
CAND_RADIUS = 0.5        # candidate_radius_m (p95 major axis): max plausible gap
AGGLOM_CAP_M = 1.0       # agglomeration_major_cap_m (p99): wider clusters bridge >1 seep

FLUX_RATE = {"A": 1.0, "B": 8.0, "C": 25.0}   # B/A ~8x (Allison); C placeholder

FEATURES = ["dist", "size_ratio", "max_area", "min_area",
            "bright_diff", "dens_025", "dens_050",
            "circ_mean", "ecc_mean", "sol_mean", "circ_diff",
            "loc_max_area", "anchor_ratio"]


def _norm_class(s):
    return s.fillna("").astype(str).str.strip().str.upper()


def load_labeled():
    """Return (L, fields) where L is the combined labeled-bubble frame and
    fields maps image -> full per-image density field (x, y, area arrays)."""
    kwa = gpd.read_file(KWA, layer="labels")
    kwa["class"] = _norm_class(kwa["class"])
    c39 = gpd.read_file(C39_CLS)
    c39["class"] = _norm_class(c39["class"])
    for col in ("is_context",):           # chip-39 file predates is_context
        if col not in c39.columns:
            c39[col] = 0

    # 39.tif authority = chip-39 file; drop kwa's overlapping 39.tif rows.
    kwa = kwa[kwa["image"] != "39.tif"].copy()

    keep = ["image", "seep_id", "class", "seep_group_id", "is_pregrouped",
            "is_overgrouped", "is_context", "centroid_x_m", "centroid_y_m",
            "area_m2", "circularity", "eccentricity", "solidity",
            "mean_R", "mean_G", "mean_B", "geometry"]
    combined = pd.concat([c39[keep], kwa[keep]], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry")

    # processed (grouped) rows only
    proc = combined[combined["class"] != ""].copy()
    proc = proc[proc["is_overgrouped"] != 1].copy()      # cannot split -> drop

    # ----- per-image FULL density fields (all bubbles, regardless of class) ---
    fields = {}
    for im in proc["image"].unique():
        if im == "39.tif":
            ff = gpd.read_file(C39_FULL)
            fields[im] = (ff["centroid_x_m"].to_numpy(float),
                          ff["centroid_y_m"].to_numpy(float),
                          ff["area_m2"].to_numpy(float))
        else:
            ff = kwa[kwa["image"] == im]
            fields[im] = (ff["centroid_x_m"].to_numpy(float),
                          ff["centroid_y_m"].to_numpy(float),
                          ff["area_m2"].to_numpy(float))
    return proc.reset_index(drop=True), fields


def build_pairs(L, fields, pregrouped_anchors=True):
    """Within-image candidate pairs among labeled bubbles + their features.

    Returns a dict of arrays. Pair endpoints exclude only CONTEXT bubbles
    (truncated neighbour-quarter groups, not scored). is_pregrouped envelopes
    ARE eligible endpoints: in truth they anchor nearby satellite bubbles (16/83
    chip-39 multi-bubble groups mix an envelope with satellites, incl. the
    13-member one), so they must be able to merge. A single envelope polygon can
    only ever GAIN neighbours, never split, so this is one-directional and safe.
    DENSITY is measured against the full per-image field."""
    L = L.copy()
    L["bright"] = (L["mean_R"] + L["mean_G"] + L["mean_B"]) / 3.0
    for col in ("circularity", "eccentricity", "solidity"):
        L[col] = L[col].fillna(L[col].median())
    # global key for human grouping (collision-safe across images)
    L["gkey"] = L["image"].astype(str) + "::" + L["seep_group_id"].astype(str)
    L["g_int"] = pd.factorize(L["gkey"])[0]

    rec = {k: [] for k in ("i", "j", "y", "img")}
    feat_rows = []
    for im, sub in L.groupby("image"):
        idx = sub.index.to_numpy()
        # eligible-to-pair endpoints: owned (not context). Pregrouped envelopes
        # are KEPT as eligible anchors (they group with satellites in truth)
        # unless the ablation toggle excludes them.
        mask = sub["is_context"] != 1
        if not pregrouped_anchors:
            mask &= sub["is_pregrouped"] != 1
        elig = sub[mask]
        ei = elig.index.to_numpy()
        if len(ei) < 2:
            continue
        xy = elig[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
        tree = cKDTree(xy)
        pp = tree.query_pairs(r=CAND_RADIUS, output_type="ndarray")
        if len(pp) == 0:
            continue
        fx, fy, fa = fields[im]
        ftree = cKDTree(np.column_stack([fx, fy]))
        a = elig["area_m2"].to_numpy(float)
        b = elig["bright"].to_numpy(float)
        c = elig["circularity"].to_numpy(float)
        e = elig["eccentricity"].to_numpy(float)
        s = elig["solidity"].to_numpy(float)
        g = elig["g_int"].to_numpy(np.int64)
        glob = ei
        for p, q in pp:
            mid = (xy[p] + xy[q]) / 2.0
            d025 = max(0, len(ftree.query_ball_point(mid, 0.25)) - 2)
            d050 = max(0, len(ftree.query_ball_point(mid, 0.50)) - 2)
            nbr = ftree.query_ball_point(mid, 0.50)
            loc_max = float(fa[nbr].max()) if len(nbr) else 0.0
            pair_max = max(a[p], a[q])
            feat_rows.append({
                "dist": float(np.hypot(xy[p, 0] - xy[q, 0], xy[p, 1] - xy[q, 1])),
                "size_ratio": min(a[p], a[q]) / max(a[p], a[q]),
                "max_area": pair_max, "min_area": min(a[p], a[q]),
                "bright_diff": abs(b[p] - b[q]),
                "dens_025": d025, "dens_050": d050,
                "circ_mean": (c[p] + c[q]) / 2.0,
                "ecc_mean": (e[p] + e[q]) / 2.0,
                "sol_mean": (s[p] + s[q]) / 2.0,
                "circ_diff": abs(c[p] - c[q]),
                "loc_max_area": loc_max,
                "anchor_ratio": loc_max / max(pair_max, 1e-9),
            })
            rec["i"].append(int(glob[p]))
            rec["j"].append(int(glob[q]))
            rec["y"].append(int(g[p] == g[q]))
            rec["img"].append(im)
    feat = pd.DataFrame(feat_rows)
    return (L, feat, np.array(rec["i"]), np.array(rec["j"]),
            np.array(rec["y"]), np.array(rec["img"]))


# ---------------------------------------------------------------------------
# scoring helpers (per-class seep COUNT error)
# ---------------------------------------------------------------------------
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
    print(f"{tag:>26} {nr:>6} {nr - N_h:>+5} {frag:>5} {bridge:>5} "
          f"{rc.get('A', 0):>4} {rc.get('B', 0):>4} {rc.get('C', 0):>4}{extra}")


def overcap(comp, xy, cap):
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


def components_from_edges(n, i, j, keep):
    rows = np.concatenate([i[keep], j[keep]])
    cols = np.concatenate([j[keep], i[keep]])
    g = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    return comp


def components_capped(n, i, j, proba, xy, keep, cap):
    """Diameter-capped agglomeration: merge kept edges in descending P(same),
    rejecting any merge whose cluster centroid-span would exceed `cap`. Mirrors
    seep_grouper_deploy.constrained_cluster but over global node indices."""
    parent = list(range(n))
    members = {k: [k] for k in range(n)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    idx = np.where(keep)[0]
    for k in idx[np.argsort(-proba[idx])]:
        a, b = int(i[k]), int(j[k])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        union = members[ra] + members[rb]
        pts = xy[union]
        if np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)).max() <= cap:
            parent[rb] = ra
            members[ra] = union
            del members[rb]
    roots = np.array([find(k) for k in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def main():
    L, fields = load_labeled()
    n_img = L["image"].nunique()
    print(f"loaded {len(L)} labeled bubbles across {n_img} images: "
          f"{L['image'].value_counts().to_dict()}")
    print(f"  pregrouped (eligible anchors): {(L['is_pregrouped']==1).sum()}  "
          f"context (features only): {(L['is_context']==1).sum()}")

    pregrouped_anchors = os.environ.get("PREGROUP_ANCHORS", "1") != "0"
    print(f"  [config] pregrouped_anchors = {pregrouped_anchors}")
    L, feat, ii, jj, y, img = build_pairs(L, fields, pregrouped_anchors)
    print(f"candidate pairs (within {CAND_RADIUS} m, owned non-pregrouped "
          f"endpoints): {len(y)} | same-seep={int(y.sum())}  "
          f"different={int((1-y).sum())}")
    if y.sum() < 3:
        print("WARNING: <3 positive pairs -- grouping signal is extremely thin.")

    X = feat[FEATURES].to_numpy(float)
    n = len(L)

    # ----- the SCORED population: owned, non-overgrouped seeps (pregrouped are
    #       fixed singletons & counted; context excluded). -----
    scored_mask = (L["is_context"] != 1).to_numpy()
    hg = L["g_int"].to_numpy(np.int64)
    hcls = L["class"].to_numpy(object)
    sxy = L[["centroid_x_m", "centroid_y_m"]].to_numpy(float)

    def restrict(comp):
        part = pd.DataFrame({"human": hg[scored_mask], "rule": comp[scored_mask],
                             "hclass": hcls[scored_mask]})
        return part

    N_h = int(pd.Series(hg[scored_mask]).nunique())
    hc = (pd.DataFrame({"g": hg[scored_mask], "c": hcls[scored_mask]})
          .drop_duplicates("g")["c"].value_counts().to_dict())
    print(f"\nHUMAN (scored): {N_h} seeps | class {hc}")

    models = {
        "DecisionTree(d4)": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", min_samples_leaf=5,
            random_state=42),
        "RandomForest(400)": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=42,
            n_jobs=-1),
    }

    gkf_groups = pd.factorize(img)[0]
    n_groups = len(np.unique(gkf_groups))

    def loio_auc(clf):
        """Leave-one-image-out AUC, averaged over non-degenerate folds only
        (a held-out image with all-same or all-different pairs is skipped)."""
        gkf = GroupKFold(n_splits=min(n_groups, 4))
        aucs = []
        for tr, te in gkf.split(X, y, groups=gkf_groups):
            if len(np.unique(y[te])) < 2:
                continue
            m = clone(clf).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        return float(np.mean(aucs)) if aucs else float("nan")

    print(f"\n{'model':>20} {'5f AUC':>7} {'LOIO AUC':>9}")
    oof = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for name, clf in models.items():
            auc5 = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
            aucg = loio_auc(clf)
            print(f"{name:>20} {auc5:>7.3f} {aucg:>9.3f}")
            oof[name] = cross_val_predict(clf, X, y, cv=5,
                                          method="predict_proba")[:, 1]

    # decision-tree interpretable rules
    dt = models["DecisionTree(d4)"].fit(X, y)
    print("\n--- DecisionTree(d4) learned rules ---")
    print(export_text(dt, feature_names=FEATURES, max_depth=4))
    print("feature importances: " + ", ".join(
        f"{f}={imp:.2f}" for f, imp in
        sorted(zip(FEATURES, dt.feature_importances_), key=lambda t: -t[1])
        if imp > 0))
    rf = models["RandomForest(400)"].fit(X, y)
    print("\nRandomForest importances: " + ", ".join(
        f"{f}={imp:.2f}" for f, imp in
        sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1])[:6]))

    # ----- baseline: current anchor+lonely rule, per image, fitted percentiles -
    grouped_for_fit = L[(L["is_context"] != 1)].copy()
    grouped_for_fit["seep_group_id"] = grouped_for_fit["g_int"]
    meas = measure_groups(grouped_for_fit)
    base_comp = np.arange(n)               # default: each bubble its own cluster
    if len(meas) >= 1:
        aa = float(np.percentile(meas["anchor_area_m2"], 5))
        rr = float(np.percentile(meas["intra_radius_m"], 90))
        ss = float(np.percentile(meas["max_satellite_area_m2"], 95))
        next_id = 0
        for im, sub in L.groupby("image"):
            bub = pd.DataFrame({
                "bubble_id": np.arange(len(sub)),
                "area_m2": sub["area_m2"].to_numpy(float),
                "centroid_x_m": sub["centroid_x_m"].to_numpy(float),
                "centroid_y_m": sub["centroid_y_m"].to_numpy(float)})
            cl = lonely_cluster(anchor_cluster(bub, aa, rr, ss))
            local = cl["cluster_id"].to_numpy()
            remap = {v: next_id + k for k, v in enumerate(pd.unique(local))}
            base_comp[sub.index.to_numpy()] = [remap[v] for v in local]
            next_id += len(remap)

    print(f"\n{'method':>26} {'N_seep':>6} {'bias':>5} {'frag':>5} {'omrg':>5} "
          f"{'A':>4} {'B':>4} {'C':>4} {'>cap':>6}")
    report("CURRENT rule (fit p90)", restrict(base_comp), N_h,
           overcap(base_comp[scored_mask], sxy[scored_mask], AGGLOM_CAP_M))
    report("HUMAN grouping", pd.DataFrame(
        {"human": hg[scored_mask], "rule": hg[scored_mask],
         "hclass": hcls[scored_mask]}), N_h,
        overcap(hg[scored_mask], sxy[scored_mask], AGGLOM_CAP_M))

    # diameter cap (historical agglomeration major-axis cap). ENFORCE_CAP=0
    # reverts to plain connected-components for the capped-vs-uncapped A/B.
    enforce_cap = os.environ.get("ENFORCE_CAP", "1") != "0"
    print(f"  [config] enforce_cap = {enforce_cap} (cap={AGGLOM_CAP_M} m)")

    # for each model, keep the threshold whose seep COUNT is closest to human
    best = {}        # name -> (thr, comp)
    for name in models:
        cand = []
        for thr in (0.5, 0.6, 0.7):
            keep = oof[name] >= thr
            if enforce_cap:
                comp = components_capped(n, ii, jj, oof[name], sxy, keep,
                                         AGGLOM_CAP_M)
            else:
                comp = components_from_edges(n, ii, jj, keep)
            nr = pd.Series(comp[scored_mask]).nunique()
            report(f"{name} thr={thr:.1f}", restrict(comp), N_h,
                   overcap(comp[scored_mask], sxy[scored_mask], AGGLOM_CAP_M))
            cand.append((abs(nr - N_h), thr, comp))
        cand.sort(key=lambda t: t[0])
        best[name] = (cand[0][1], cand[0][2])
        print()

    # ----- flux-weighted re-score (oracle class) -----
    print("=== flux-weighted re-score (oracle human class; rates A=1 B=8 C=25*) ===")
    print(f"  {'config':>26} {'A':>4} {'B':>4} {'C':>4} {'flux':>8} {'err%':>7}")

    def class_counts_of(comp):
        df = pd.DataFrame({"rule": comp[scored_mask], "hclass": hcls[scored_mask]})
        dom = df.groupby("rule")["hclass"].agg(lambda s: s.value_counts().index[0])
        c = Counter(dom)
        return {k: int(c.get(k, 0)) for k in ("A", "B", "C")}

    def flux(c):
        return sum(c[k] * FLUX_RATE[k] for k in ("A", "B", "C"))

    truth = {k: int(hc.get(k, 0)) for k in ("A", "B", "C")}
    f_truth = flux(truth)
    rows = [("TRUTH (human grouping)", truth)]
    for name in models:
        thr, comp = best[name]
        rows.append((f"{name} thr={thr:.1f}", class_counts_of(comp)))
    rows.append(("CURRENT rule", class_counts_of(base_comp)))
    for tag, c in rows:
        f = flux(c)
        err = 100.0 * (f - f_truth) / f_truth if f_truth else float("nan")
        print(f"  {tag:>26} {c['A']:>4} {c['B']:>4} {c['C']:>4} {f:>8.0f} {err:>+6.1f}%")
    print("  (* C rate placeholder; class is the oracle human label here -- this")
    print("     isolates GROUPING error from classifier error.)")
    print(f"\n(target N_seep={N_h}, bias 0, low omrg, A/B/C ~ {hc})")


if __name__ == "__main__":
    main()
