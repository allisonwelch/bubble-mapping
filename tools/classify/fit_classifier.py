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

Everything printed below is also written to a workbook under
`<results>/<CANONICAL_PRED_SUBDIR>/classify/`, one sheet per table, so the
numbers can be pasted without re-reading console scrollback.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import itertools
import os

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    raise SystemExit("geopandas is required; run inside venv-bubble")

from shapely.ops import unary_union
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, cohen_kappa_score,
                             confusion_matrix)
from sklearn.model_selection import (GroupKFold, LeaveOneGroupOut,
                                     cross_val_predict)
from sklearn.tree import DecisionTreeClassifier, export_text

from tools.flux.rates import FLUX_RATE_ANNUAL as FLUX_RATE
from tools.flux.rates import lake_total
from tools.paths import CANONICAL_PRED_SUBDIR

FEATURES = ["hull_area_m2", "mean_R", "mean_G", "mean_B"]
CLASSES = ["A", "B", "C"]
CLASS_RANK = {"A": 1, "B": 2, "C": 3}
# Flux rates now live in tools/flux/rates.py, which owns them for the whole
# pipeline; they used to be duplicated here. That module also carries summer
# and winter rates, but this script stays on the ANNUAL ones -- the classifier
# diagnostic below is about class counts, and the season only rescales it.
LABELERS = ["prajna", "allison", "katey"]

# The 2026-07-29 deploy-point forest, the same one `c_augment_eval` scores. It
# runs here on the locked 4 FEATURES, not on model_comparison's RICH 13 -- the
# feature sweep is that script's job, and importing its `enrich` would be a
# circular import (it imports this module).
RF_KWARGS = dict(n_estimators=500, min_samples_leaf=2,
                 class_weight="balanced", n_jobs=-1)

# Outputs land beside the checkpoint's other artifacts, not in the labeling dir
# (which the labelers sync) and not in the CWD.
CLASSIFY_OUT_DIR = os.path.join("data", "results", "SWIN", "AE",
                                CANONICAL_PRED_SUBDIR, "classify")


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
def kappa_report(packs: dict[str, gpd.GeoDataFrame]) -> dict[str, pd.DataFrame]:
    """Cohen's kappa on the polygons two labelers BOTH classified.

    Agreement is measured per-POLYGON, not per-seep: the labelers grouped
    differently, so there is no shared seep unit to compare on, but every pack
    is built from the same underlying bubble polygons keyed by (image, bubble_id).
    Both unweighted and linearly-weighted kappa are given -- weighted is the
    honest one for an ordered A<B<C scale, since an A/C disagreement is worse
    than an A/B one.

    Prints as before and returns the same numbers as three tidy tables
    (pairwise kappas, the pairwise crosstabs in long form, Fleiss) for the
    workbook.
    """
    print("\n" + "=" * 72)
    print("INTER-LABELER AGREEMENT (Cohen's kappa, per-polygon)")
    print("=" * 72)

    pairwise, crosstabs = [], []
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
            pairwise.append({"labeler_a": a, "labeler_b": b,
                             "shared_units": ", ".join(shared),
                             "n_co_classified": 0})
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
        pairwise.append({"labeler_a": a, "labeler_b": b,
                         "shared_units": ", ".join(shared),
                         "n_co_classified": int(len(both)),
                         "raw_agreement": float(agree),
                         "kappa": float(k),
                         "kappa_linear_weighted": float(kw),
                         "interpretation": interpret(k)})
        for ca in CLASSES:
            for cb in CLASSES:
                crosstabs.append({"labeler_a": a, "labeler_b": b,
                                  "class_a": ca, "class_b": cb,
                                  "n": int(ct.loc[ca, cb])})

    # three-way subset -> Fleiss
    keys = None
    for who in LABELERS:
        k = set(map(tuple, packs[who].loc[packs[who]["class"] != "",
                                          ["image", "bubble_id"]].values))
        keys = k if keys is None else keys & k
    print(f"\nPolygons classified by all three: n={len(keys)}")
    fleiss = {"n_polygons_all_three": int(len(keys))}
    if keys:
        idx = pd.MultiIndex.from_tuples(sorted(keys), names=["image", "bubble_id"])
        mat = pd.DataFrame(index=idx)
        for who in LABELERS:
            s = packs[who].set_index(["image", "bubble_id"])["class"]
            mat[who] = s.reindex(idx).values
        counts = np.stack([(mat == c).sum(axis=1).to_numpy() for c in CLASSES], 1)
        fk = fleiss_kappa(counts)
        print(f"  Fleiss' kappa = {fk:.3f}  [{interpret(fk)}]")
        unan = (counts.max(axis=1) == 3).mean()
        print(f"  unanimous on {unan:.1%} of them")
        fleiss.update({"fleiss_kappa": float(fk),
                       "interpretation": interpret(fk),
                       "unanimous_fraction": float(unan)})
    return {"kappa_pairwise": pd.DataFrame(pairwise),
            "kappa_crosstabs": pd.DataFrame(crosstabs),
            "kappa_fleiss": pd.DataFrame([fleiss])}


def default_labeling_dir() -> str:
    """Where the three completed labeler packs live."""
    return os.path.join("data", "results", "SWIN", "AE", CANONICAL_PRED_SUBDIR,
                        "labeling", "final_labeler_packs")


def build_trainable_table(labeling_dir: str):
    """Load the three packs and build the seep table the classifier trains on.

    Returns (packs, seeps, df, pack_rows):
      packs      labeler -> per-bubble GeoDataFrame (needed for the kappa report)
      seeps      every dissolved seep, including the rows training drops
      df         the TRAINABLE rows: is_context == 0, is_overgrouped == 0, and
                 no NaN in FEATURES, with `phys_id` attached
      pack_rows  one dict per pack for the workbook

    Shared by `main()` (which reports on it) and `fit_deploy_model()` (which
    fits the deployed forest on it), so the two can never drift apart on which
    rows count as trainable -- that filter changes the class balance, and the
    class balance is the flux.

    `assign_phys_id` runs BEFORE the trainable-row filter, deliberately: it
    union-finds seeps sharing a member (image, bubble_id), and those links can
    run THROUGH a row the filter drops. Filtering first silently splits such
    groups and weakens the leakage guard (CLAUDE.md).
    """
    packs, per_pack, pack_rows = {}, [], []
    for who in LABELERS:
        fp = os.path.join(labeling_dir,
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
        per_pack.append(s)
        pack_rows.append({"labeler": who, "pack": os.path.basename(fp),
                          "n_polygons": int(len(g)), "n_classified": n_cls,
                          "n_labeled_seeps": int(len(s))})

    seeps = pd.concat(per_pack, ignore_index=True)
    seeps["phys_id"] = assign_phys_id(seeps)

    keep = (seeps["is_context"] == 0) & (seeps["is_overgrouped"] == 0)
    df = seeps[keep].dropna(subset=FEATURES).reset_index(drop=True)
    return packs, seeps, df, pack_rows


def fit_deploy_model(labeling_dir: str | None = None, seed: int = 42):
    """Fit the DEPLOY-POINT classifier on every trainable labeled seep.

    This is the model the whole-lake runner applies (tools/deploy/), so it is
    fit on ALL the data -- no held-out fold. Its cross-chip skill is not
    measured here; quote the leave-one-image-out row from `main()`'s workbook
    for that, because LOIO is the regime deployment actually runs in (an unseen
    chip, or in the runner's case an unseen lake).

    Returns (rf, info) where `info` records what it was fit on, for the runner's
    run_info.json -- a flux number is only interpretable next to the class
    balance behind it.
    """
    labeling_dir = labeling_dir or default_labeling_dir()
    _, _, df, _ = build_trainable_table(labeling_dir)
    rf = RandomForestClassifier(random_state=seed, **RF_KWARGS)
    rf.fit(df[FEATURES].to_numpy(dtype=float), df["class"].to_numpy())
    info = {
        "labeling_dir": os.path.abspath(labeling_dir),
        "features": list(FEATURES),
        "rf_kwargs": dict(RF_KWARGS),
        "seed": seed,
        "n_training_seeps": int(len(df)),
        "n_physical_seeps": int(df["phys_id"].nunique()),
        "n_chips": int(df["image"].nunique()),
        "class_balance": {c: int((df["class"] == c).sum()) for c in CLASSES},
    }
    return rf, info


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
def evaluate(model_name, cv_name, clf, X, y, groups, cv) -> dict:
    """Grouped-CV out-of-fold predictions -> the metrics we actually report.

    Returns the printed numbers as three workbook-ready pieces: a one-row
    `summary`, a per-class `per_class` table, and the `confusion` matrix.
    """
    pred = cross_val_predict(clf, X, y, groups=groups, cv=cv)
    acc = (pred == y).mean()
    rep = classification_report(y, pred, labels=CLASSES, output_dict=True,
                                zero_division=0)
    name = f"{model_name} | {cv_name}"
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
    n_true = {c: int((y == c).sum()) for c in CLASSES}
    n_pred = {c: int((pred == c).sum()) for c in CLASSES}
    ft, ft_std_err = lake_total(n_true)
    fp, _ = lake_total(n_pred)
    err = 100 * (fp - ft) / ft
    # Standard error on the published rates, propagated by class count -- the
    # floor no classifier can beat. It does not shrink with more seeps (the SE
    # is on each class mean, so it is shared by every seep of that class), which
    # is why it is a floor. Report it next to the model error to compare the two.
    rate_unc = 100 * ft_std_err / ft
    print(f"count-based flux (mg CH4/day, annual mean): truth={ft:,.0f}  "
          f"pred={fp:,.0f}")
    print(f"  model error={err:+.1f}%   rate uncertainty=+/-{rate_unc:.1f}%   "
          f"combined=+/-{np.hypot(abs(err), rate_unc):.1f}%")
    # C is the rarest class but carries the largest share of the flux, so its
    # recall is reported separately -- a small C miscount moves the total more
    # than all A/B error.
    nc_t, nc_p = int((y == "C").sum()), int((pred == "C").sum())
    c_term = 100 * FLUX_RATE["C"] * (nc_p - nc_t) / ft
    print(f"  C seeps: {nc_p} predicted vs {nc_t} true "
          f"({c_term:+.1f}% of total flux)")

    summary = {"model": model_name, "cv": cv_name, "n_seeps": int(len(y)),
               "accuracy": float(acc),
               "macro_f1": rep["macro avg"]["f1-score"],
               "weighted_f1": rep["weighted avg"]["f1-score"]}
    for c in CLASSES:
        summary[f"{c}_precision"] = rep[c]["precision"]
        summary[f"{c}_recall"] = rep[c]["recall"]
        summary[f"{c}_f1"] = rep[c]["f1-score"]
    for c in CLASSES:
        summary[f"n_true_{c}"] = n_true[c]
        summary[f"n_pred_{c}"] = n_pred[c]
    summary.update({"flux_truth_mg_per_day": ft, "flux_pred_mg_per_day": fp,
                    "flux_err_pct": err, "rate_unc_pct": rate_unc,
                    "combined_unc_pct": float(np.hypot(abs(err), rate_unc)),
                    "C_flux_term_pct": c_term})

    per_class = pd.DataFrame(
        [{"model": model_name, "cv": cv_name, "class": c,
          "precision": rep[c]["precision"], "recall": rep[c]["recall"],
          "f1": rep[c]["f1-score"], "support": int(rep[c]["support"])}
         for c in CLASSES])
    conf = cm.reset_index(names="truth")
    conf.insert(0, "cv", cv_name)
    conf.insert(0, "model", model_name)
    return {"name": name, "summary": summary, "per_class": per_class,
            "confusion": conf, "pred": pred}


# --------------------------------------------------------------------------- #
# workbook
# --------------------------------------------------------------------------- #
def write_workbook(sheets: dict[str, pd.DataFrame], path: str) -> list[str]:
    """One sheet per table. Falls back to a CSV per table without openpyxl."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            for name, d in sheets.items():
                # Excel caps sheet names at 31 chars; every name below is
                # shorter, but truncate rather than raise if one grows.
                d.to_excel(xw, sheet_name=name[:31], index=False)
        return [path]
    except ImportError:
        stem = os.path.splitext(path)[0]
        print("[warn] openpyxl not installed -- writing one CSV per sheet")
        out = []
        for name, d in sheets.items():
            p = f"{stem}_{name}.csv"
            d.to_csv(p, index=False)
            out.append(p)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeling-dir", default=os.path.join(
        "data", "results", "SWIN", "AE", CANONICAL_PRED_SUBDIR,
        "labeling", "final_labeler_packs"),
        help="directory holding the three *_grouped.gpkg labeler packs. "
             "Defaults to final_labeler_packs/, where all three now live; "
             "the parent labeling/ dir holds only katey's, so a run there "
             "used to silently find 1 of 3 packs.")
    ap.add_argument("--max-depth", type=int, default=0,
                    help="0 = sweep 2..6 and pick best grouped-CV macro-F1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-csv", default="", help="optional seep table dump")
    ap.add_argument("--out-dir", default=CLASSIFY_OUT_DIR,
                    help="where the results workbook is written "
                         f"(default {CLASSIFY_OUT_DIR})")
    ap.add_argument("--out-xlsx", default="classifier_results.xlsx",
                    help="workbook filename inside --out-dir")
    args = ap.parse_args()

    print("=" * 72)
    print("LOADING PACKS")
    print("=" * 72)
    packs, seeps, df, pack_rows = build_trainable_table(args.labeling_dir)

    kappa_sheets = kappa_report(packs)

    print("\n" + "=" * 72)
    print("SEEP-LEVEL TRAINING SET")
    print("=" * 72)
    print(f"  {len(seeps)} labeled seeps; dropped "
          f"{int((seeps['is_context'] == 1).sum())} context-touching, "
          f"{int((seeps['is_overgrouped'] == 1).sum())} overgrouped")
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
    sweep_rows = []
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
        sweep_rows = [{"max_depth": d, "macro_f1": scores[d][0],
                       "accuracy": float(scores[d][1]),
                       "selected": d == best} for d in depths]

    clf = DecisionTreeClassifier(max_depth=best, class_weight="balanced",
                                 random_state=args.seed)
    rf = RandomForestClassifier(random_state=args.seed, **RF_KWARGS)
    dt_name = f"DecisionTree d{best}"
    rf_name = f"RandomForest {RF_KWARGS['n_estimators']}"

    n_img = df["image"].nunique()
    # Both CVs for both models. LOIO is the one to quote: its folds are the
    # chips, so it is the cross-chip test and it is insensitive to the group
    # label ordering that moves grouped 5-fold by +/-0.035 macro-F1.
    runs = [("Grouped 5-fold CV", groups, cv)]
    if n_img > 1:
        runs.append((f"Leave-one-image-out ({n_img} chips)",
                     df["image"].to_numpy(), LeaveOneGroupOut()))

    results = []
    for model_name, model in ((dt_name, clf), (rf_name, rf)):
        print("\n" + "=" * 72)
        print(f"PERFORMANCE ({model_name}, class_weight=balanced)")
        print("=" * 72)
        for cv_name, g_, cv_ in runs:
            results.append(evaluate(model_name, cv_name, model, X, y, g_, cv_))

    print("\n" + "=" * 72)
    print("FINAL MODELS (fit on all data)")
    print("=" * 72)
    clf.fit(X, y)
    rf.fit(X, y)
    imp_rows = []
    for model_name, model in ((dt_name, clf), (rf_name, rf)):
        print(f"{model_name} feature importances:")
        for f, imp in sorted(zip(FEATURES, model.feature_importances_),
                             key=lambda t: -t[1]):
            print(f"  {f:16s} {imp:.3f}")
            imp_rows.append({"model": model_name, "feature": f,
                             "importance": float(imp)})
    rules = export_text(clf, feature_names=FEATURES, decimals=4)
    print(f"\n{dt_name} learned thresholds:")
    print(rules)

    if args.out_csv:
        df.drop(columns="members").to_csv(args.out_csv, index=False)
        print(f"[out] seep table -> {args.out_csv}")

    # ------------------------------------------------------------- workbook
    summary = pd.DataFrame([r["summary"] for r in results])
    run_info = pd.DataFrame([
        {"key": "run_utc", "value": _dt.datetime.now(_dt.timezone.utc)
            .strftime("%Y-%m-%d %H:%M:%S UTC")},
        {"key": "labeling_dir", "value": os.path.abspath(args.labeling_dir)},
        {"key": "features", "value": ", ".join(FEATURES)},
        {"key": "seed", "value": args.seed},
        {"key": "decision_tree_max_depth", "value": best},
        {"key": "random_forest", "value": str(RF_KWARGS)},
        {"key": "n_labeled_seeps", "value": int(len(seeps))},
        {"key": "n_training_seeps", "value": int(len(df))},
        {"key": "n_physical_seeps", "value": int(df["phys_id"].nunique())},
        {"key": "n_duplicate_labelings", "value": int(n_dup)},
        {"key": "n_dropped_context", "value":
            int((seeps["is_context"] == 1).sum())},
        {"key": "n_dropped_overgrouped", "value":
            int((seeps["is_overgrouped"] == 1).sum())},
        {"key": "n_chips", "value": int(n_img)},
    ] + [{"key": f"n_class_{c}", "value": int((df["class"] == c).sum())}
         for c in CLASSES])

    sheets = {
        "run_info": run_info,
        "summary": summary,
        "per_class": pd.concat([r["per_class"] for r in results],
                               ignore_index=True),
        "confusion": pd.concat([r["confusion"] for r in results],
                               ignore_index=True),
        "kappa_pairwise": kappa_sheets["kappa_pairwise"],
        "kappa_crosstabs": kappa_sheets["kappa_crosstabs"],
        "kappa_fleiss": kappa_sheets["kappa_fleiss"],
        "packs": pd.DataFrame(pack_rows),
        "class_by_labeler": pd.crosstab(df["labeler"], df["class"])
            .reindex(columns=CLASSES, fill_value=0).reset_index(),
        "feature_importance": pd.DataFrame(imp_rows),
        "tree_rules": pd.DataFrame({"rule": rules.splitlines()}),
        "seeps": df.drop(columns="members"),
    }
    if sweep_rows:
        sheets["depth_sweep"] = pd.DataFrame(sweep_rows)

    written = write_workbook(sheets,
                             os.path.join(args.out_dir, args.out_xlsx))
    for p in written:
        print(f"[out] {p}")

    print("\n" + "=" * 72)
    print("HEADLINE (quote the leave-one-image-out row -- it is the "
          "cross-chip test)")
    print("=" * 72)
    print(summary[["model", "cv", "accuracy", "macro_f1", "C_recall", "C_f1",
                   "flux_err_pct", "rate_unc_pct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()