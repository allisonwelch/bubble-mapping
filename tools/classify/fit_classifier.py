"""Fit + validate an A/B/C seep classifier on the three returned labeler packs,
and report inter-labeler agreement (Cohen's kappa).

This is the multi-labeler successor to `seep_fit_class_tree.py`, which expects a
single pre-dissolved `gt_seeps_labeled.gpkg`. Here the inputs are the three
per-bubble quarter packs (prajna / allison / katey), so the script does its own
dissolve-to-seeps first.

Feature set is the one locked on 2026-06-01 and re-confirmed 2026-06-10:
    hull_area_m2, mean_R, mean_G, mean_B
`hull_area_m2` (convex hull of the grouped bubbles) is the SIZE axis because it
is the only size measure defined consistently across pregrouped envelopes,
labeler-grouped unions, and pred clusters. Solidity/eccentricity are excluded as
hand-drawn-annotation artifacts (2026-05-14).

THE LEAKAGE PROBLEM THIS SCRIPT SOLVES
The three labelers overlap on a shared calibration set (units 21-NE, 38-SW,
4-SE), so the SAME physical seep appears up to three times in the pooled data.
Naive k-fold would put labeler A's copy in train and labeler B's copy in test
and report a badly inflated score. Every seep is therefore assigned a
`phys_id` -- connected components over "shares at least one member (image,
bubble_id)" -- and all cross-validation is GROUPED on `phys_id`, so every copy of
a seep lands in the same fold. Leave-one-image-out is reported alongside as the
harder cross-chip generalization test.

Rows excluded from training (not from the kappa): seeps touching `is_context=1`
(neighbour-quarter groups are truncated, so their hull footprint is wrong) and
`is_overgrouped=1` (one polygon spanning several seeps).
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    raise SystemExit("geopandas is required; run inside venv-bubble")

from shapely.ops import unary_union
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix)
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     cross_val_predict)
from sklearn.tree import DecisionTreeClassifier, export_text

FEATURES = ["hull_area_m2", "mean_R", "mean_G", "mean_B"]
CLASSES = ["A", "B", "C"]
CLASS_RANK = {"A": 1, "B": 2, "C": 3}
# Per-seep flux rates, mg CH4/day averaged over the year (Allison, 2026-07-29).
# These are the REAL field rates and supersede the old 1/8/25 placeholder --
# the true ratio is 1 : 8.2 : 60.7, so the placeholder understated C by 2.4x.
# C is only ~4% of seeps but ~52% of total flux, which makes C recall (not the
# A/B boundary) the dominant lever on the headline number.
FLUX_RATE = {"A": 16.0, "B": 131.0, "C": 971.0}
# 1-sigma uncertainty on each rate. Propagated by class count, these alone put
# a +/-14% floor under any flux estimate, regardless of classifier quality.
FLUX_SIGMA = {"A": 10.0, "B": 30.0, "C": 142.0}
LABELERS = ["prajna", "allison", "katey"]


# --------------------------------------------------------------------------- #
# loading + dissolve
# --------------------------------------------------------------------------- #
def load_pack(path: str, labeler: str) -> gpd.GeoDataFrame:
    g = gpd.read_file(path, layer="labels")
    g["class"] = g["class"].fillna("").astype(str).str.strip()
    for c in ("is_context", "is_pregrouped", "is_overgrouped"):
        g[c] = g[c].fillna(0).astype(int)
    g["labeler"] = labeler
    return g


def dissolve_to_seeps(g: gpd.GeoDataFrame) -> pd.DataFrame:
    """One row per (image, seep_group_id), with hull + area-weighted brightness.

    Class of a seep is the LARGEST class among its members (C>B>A) -- the same
    rule used to reconcile the returned packs. Groups are already single-class
    on all three packs except one apiece, which this reports.
    """
    rows = []
    mixed = 0
    for (img, gid), sub in g.groupby(["image", "seep_group_id"], sort=True):
        cls = sub.loc[sub["class"] != "", "class"]
        if cls.empty:
            continue  # unlabeled group (katey's model-grouped remainder)
        if cls.nunique() > 1:
            mixed += 1
        klass = max(cls, key=CLASS_RANK.get)

        geom = unary_union(sub.geometry.values)
        hull = geom.convex_hull
        w = sub["area_m2"].to_numpy(dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        rows.append({
            "labeler": sub["labeler"].iloc[0],
            "image": img,
            "seep_group_id": int(gid),
            "unit": sub["unit"].dropna().iloc[0] if sub["unit"].notna().any() else "",
            "class": klass,
            "n_bubbles": len(sub),
            "area_m2": float(geom.area),
            "hull_area_m2": float(hull.area),
            "perim_m": float(geom.length),
            "mean_R": float(np.dot(w, sub["mean_R"].to_numpy(dtype=float))),
            "mean_G": float(np.dot(w, sub["mean_G"].to_numpy(dtype=float))),
            "mean_B": float(np.dot(w, sub["mean_B"].to_numpy(dtype=float))),
            "is_context": int(sub["is_context"].max()),
            "is_pregrouped": int(sub["is_pregrouped"].max()),
            "is_overgrouped": int(sub["is_overgrouped"].max()),
            "members": frozenset(zip(sub["image"], sub["bubble_id"].astype(int))),
        })
    if mixed:
        print(f"    [note] {mixed} mixed-class group(s) resolved by C>B>A")
    return pd.DataFrame(rows)


def assign_phys_id(seeps: pd.DataFrame) -> pd.Series:
    """Union-find: seeps sharing any member (image, bubble_id) are one physical seep.

    This is what makes the cross-validation honest across the shared calibration
    units -- all three labelers' copies of a seep get the same id and therefore
    never straddle a train/test split.
    """
    parent = list(range(len(seeps)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    seen: dict[tuple, int] = {}
    for idx, members in enumerate(seeps["members"]):
        for m in members:
            if m in seen:
                union(idx, seen[m])
            else:
                seen[m] = idx
    return pd.Series([find(i) for i in range(len(seeps))], index=seeps.index)


# --------------------------------------------------------------------------- #
# inter-labeler agreement
# --------------------------------------------------------------------------- #
def kappa_report(packs: dict[str, gpd.GeoDataFrame]) -> None:
    """Cohen's kappa on the polygons two labelers BOTH classified.

    Agreement is measured per-POLYGON, not per-seep: the labelers grouped
    differently, so there is no shared seep unit to compare on, but every pack
    is built from the same underlying bubble polygons keyed by (image, bubble_id).
    Both unweighted and linearly-weighted kappa are given -- weighted is the
    honest one for an ordered A<B<C scale, since an A/C disagreement is worse
    than an A/B one.
    """
    print("\n" + "=" * 72)
    print("INTER-LABELER AGREEMENT (Cohen's kappa, per-polygon)")
    print("=" * 72)

    for a, b in itertools.combinations(LABELERS, 2):
        A, B = packs[a], packs[b]
        m = A[["image", "bubble_id", "unit", "class"]].merge(
            B[["image", "bubble_id", "class"]], on=["image", "bubble_id"],
            suffixes=("_a", "_b"))
        both = m[(m["class_a"] != "") & (m["class_b"] != "")]
        shared = sorted(set(A["unit"].dropna()) & set(B["unit"].dropna()))
        print(f"\n{a} vs {b}   shared units: {', '.join(shared) or 'none'}")
        if both.empty:
            print("  no co-classified polygons")
            continue
        k = cohen_kappa_score(both["class_a"], both["class_b"], labels=CLASSES)
        kw = cohen_kappa_score(both["class_a"], both["class_b"], labels=CLASSES,
                               weights="linear")
        agree = (both["class_a"] == both["class_b"]).mean()
        print(f"  n={len(both)}  raw agreement={agree:.3f}  "
              f"kappa={k:.3f}  linear-weighted kappa={kw:.3f}  [{interpret(k)}]")
        ct = pd.crosstab(both["class_a"], both["class_b"])
        ct = ct.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
        print(f"  rows={a}, cols={b}")
        print("    " + ct.to_string().replace("\n", "\n    "))

    # three-way subset -> Fleiss
    keys = None
    for who in LABELERS:
        k = set(map(tuple, packs[who].loc[packs[who]["class"] != "",
                                          ["image", "bubble_id"]].values))
        keys = k if keys is None else keys & k
    print(f"\nPolygons classified by all three: n={len(keys)}")
    if keys:
        idx = pd.MultiIndex.from_tuples(sorted(keys), names=["image", "bubble_id"])
        mat = pd.DataFrame(index=idx)
        for who in LABELERS:
            s = packs[who].set_index(["image", "bubble_id"])["class"]
            mat[who] = s.reindex(idx).values
        counts = np.stack([(mat == c).sum(axis=1).to_numpy() for c in CLASSES], 1)
        print(f"  Fleiss' kappa = {fleiss_kappa(counts):.3f}  "
              f"[{interpret(fleiss_kappa(counts))}]")
        unan = (counts.max(axis=1) == 3).mean()
        print(f"  unanimous on {unan:.1%} of them")


def fleiss_kappa(counts: np.ndarray) -> float:
    n_items, n_raters = counts.shape[0], counts.sum(axis=1)[0]
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_e = np.sum((counts.sum(axis=0) / (n_items * n_raters)) ** 2)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0


def interpret(k: float) -> str:
    for thr, word in ((0.81, "almost perfect"), (0.61, "substantial"),
                      (0.41, "moderate"), (0.21, "fair"), (0.0, "slight")):
        if k >= thr:
            return word
    return "poor"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def evaluate(name, clf, X, y, groups, cv) -> dict:
    """Grouped-CV out-of-fold predictions -> the metrics we actually report."""
    pred = cross_val_predict(clf, X, y, groups=groups, cv=cv)
    acc = (pred == y).mean()
    rep = classification_report(y, pred, labels=CLASSES, output_dict=True,
                                zero_division=0)
    print(f"\n--- {name} ---")
    print(f"accuracy = {acc:.3f}   macro-F1 = {rep['macro avg']['f1-score']:.3f}"
          f"   weighted-F1 = {rep['weighted avg']['f1-score']:.3f}")
    print(classification_report(y, pred, labels=CLASSES, zero_division=0,
                                digits=3))
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES),
                      index=[f"true_{c}" for c in CLASSES],
                      columns=[f"pred_{c}" for c in CLASSES])
    print("confusion matrix (rows=truth):")
    print("  " + cm.to_string().replace("\n", "\n  "))

    # The project's real objective: count-based flux = sum(count x rate).
    ft = sum(FLUX_RATE[c] * (y == c).sum() for c in CLASSES)
    fp = sum(FLUX_RATE[c] * (pred == c).sum() for c in CLASSES)
    err = 100 * (fp - ft) / ft
    # Rate uncertainty propagated by class count -- the floor no classifier
    # can beat. Report it next to the model error so the two are comparable.
    rate_unc = 100 * np.sqrt(sum((FLUX_SIGMA[c] * (y == c).sum()) ** 2
                                 for c in CLASSES)) / ft
    print(f"count-based flux (mg CH4/day, annual mean): truth={ft:,.0f}  "
          f"pred={fp:,.0f}")
    print(f"  model error={err:+.1f}%   rate uncertainty=+/-{rate_unc:.1f}%   "
          f"combined=+/-{np.hypot(abs(err), rate_unc):.1f}%")
    # C carries ~half the flux on ~4% of seeps, so its recall is reported
    # separately -- a small C miscount moves the total more than all A/B error.
    nc_t, nc_p = int((y == "C").sum()), int((pred == "C").sum())
    print(f"  C seeps: {nc_p} predicted vs {nc_t} true "
          f"({100 * FLUX_RATE['C'] * (nc_p - nc_t) / ft:+.1f}% of total flux)")
    return {"name": name, "accuracy": acc,
            "macro_f1": rep["macro avg"]["f1-score"],
            "flux_err_pct": err, "rate_unc_pct": rate_unc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeling-dir", default=os.path.join(
        "data", "results", "SWIN", "AE", "20260428-1537_SWINxAE.weights",
        "labeling", "final_labeler_packs"),
        help="directory holding the three *_grouped.gpkg labeler packs. "
             "Defaults to final_labeler_packs/, where all three now live; "
             "the parent labeling/ dir holds only katey's, so a run there "
             "used to silently find 1 of 3 packs.")
    ap.add_argument("--max-depth", type=int, default=0,
                    help="0 = sweep 2..6 and pick best grouped-CV macro-F1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default="", help="optional seep table dump")
    args = ap.parse_args()

    packs, seeps = {}, []
    print("=" * 72)
    print("LOADING PACKS")
    print("=" * 72)
    for who in LABELERS:
        fp = os.path.join(args.labeling_dir,
                          f"gt_seeps_label_quarters_{who}_grouped.gpkg")
        if not os.path.exists(fp):
            raise SystemExit(
                f"missing labeler pack for {who!r}: {fp}\n"
                f"All three of {LABELERS} must be present -- fitting on a "
                f"subset silently changes the class balance and the kappa "
                f"report. Pass --labeling-dir if they live elsewhere.")
        g = load_pack(fp, who)
        packs[who] = g
        s = dissolve_to_seeps(g)
        n_cls = int((g["class"] != "").sum())
        print(f"  {who:8s} {len(g):5d} polygons ({n_cls} classified) "
              f"-> {len(s):4d} labeled seeps")
        seeps.append(s)

    kappa_report(packs)

    seeps = pd.concat(seeps, ignore_index=True)
    seeps["phys_id"] = assign_phys_id(seeps)

    print("\n" + "=" * 72)
    print("SEEP-LEVEL TRAINING SET")
    print("=" * 72)
    keep = (seeps["is_context"] == 0) & (seeps["is_overgrouped"] == 0)
    print(f"  {len(seeps)} labeled seeps; dropped "
          f"{int((seeps['is_context'] == 1).sum())} context-touching, "
          f"{int((seeps['is_overgrouped'] == 1).sum())} overgrouped")
    df = seeps[keep].dropna(subset=FEATURES).reset_index(drop=True)
    n_dup = len(df) - df["phys_id"].nunique()
    print(f"  -> {len(df)} training seeps over {df['phys_id'].nunique()} "
          f"distinct physical seeps ({n_dup} are duplicate labelings)")
    print(f"  class balance: {df['class'].value_counts().to_dict()}")
    print(f"  by labeler:\n"
          + pd.crosstab(df['labeler'], df['class']).to_string()
          .replace("\n", "\n    ").rjust(4))

    X = df[FEATURES].to_numpy(dtype=float)
    y = df["class"].to_numpy()
    groups = df["phys_id"].to_numpy()

    print("\n" + "=" * 72)
    print("MODEL SELECTION (grouped 5-fold, grouped on physical seep)")
    print("=" * 72)
    cv = GroupKFold(n_splits=5)
    if args.max_depth:
        depths, best = [args.max_depth], args.max_depth
    else:
        depths = [2, 3, 4, 5, 6]
        scores = {}
        for d in depths:
            clf = DecisionTreeClassifier(max_depth=d, class_weight="balanced",
                                         random_state=args.seed)
            p = cross_val_predict(clf, X, y, groups=groups, cv=cv)
            r = classification_report(y, p, labels=CLASSES, output_dict=True,
                                      zero_division=0)
            scores[d] = (r["macro avg"]["f1-score"], (p == y).mean())
            print(f"  depth {d}: macro-F1={scores[d][0]:.3f}  "
                  f"accuracy={scores[d][1]:.3f}")
        best = max(scores, key=lambda d: scores[d][0])
        print(f"  -> selected max_depth={best}")

    clf = DecisionTreeClassifier(max_depth=best, class_weight="balanced",
                                 random_state=args.seed)

    print("\n" + "=" * 72)
    print(f"PERFORMANCE (DecisionTree depth {best}, class_weight=balanced)")
    print("=" * 72)
    evaluate("Grouped 5-fold CV", clf, X, y, groups, cv)

    n_img = df["image"].nunique()
    if n_img > 1:
        evaluate(f"Leave-one-image-out ({n_img} chips) -- cross-chip transfer",
                 clf, X, y, df["image"].to_numpy(), LeaveOneGroupOut())

    print("\n" + "=" * 72)
    print("FINAL MODEL (fit on all data)")
    print("=" * 72)
    clf.fit(X, y)
    print("feature importances:")
    for f, imp in sorted(zip(FEATURES, clf.feature_importances_),
                         key=lambda t: -t[1]):
        print(f"  {f:16s} {imp:.3f}")
    print("\nlearned thresholds:")
    print(export_text(clf, feature_names=FEATURES, decimals=4))

    if args.out_csv:
        df.drop(columns="members").to_csv(args.out_csv, index=False)
        print(f"[out] seep table -> {args.out_csv}")


if __name__ == "__main__":
    main()