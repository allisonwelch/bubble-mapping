"""Can we beat the depth-5 DecisionTree on the 3-labeler seep set?

Three levers are tested independently so their contributions are separable:

  1. FEATURES  -- the locked 4 (hull_area_m2 + mean RGB) vs. a richer set that
     adds seep-shape and internal-structure terms the 4 cannot express:
     oriented-bbox major/minor axis (the L x W the historical field workbooks
     actually measured), fill ratio (bubble area / hull area = how much of the
     footprint is gas vs. interstitial ice), bubble-count and size spread, and
     local bubble density. Per-polygon solidity/eccentricity stay excluded
     (hand-drawing artifacts, 2026-05-14) but fill_ratio is computed from the
     GROUPING, not from drawn vertices, so it is not the same quantity.

  2. MODEL FAMILY -- single tree vs. ensembles vs. a linear baseline.

  3. OBJECTIVE -- 'balanced' class weights vs. FLUX-weighted sample weights.
     The deployed objective is count-based flux, where calling an A a B costs
     7 flux units and the reverse refunds 7. Weighting training samples by
     their flux rate tells the model that B and C errors are the expensive
     ones. This is the lever most aligned with what the paper reports.

All scoring is grouped-CV on physical seep (see fit_classifier)
so the shared calibration units cannot leak between train and test.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd


from tools.classify.fit_classifier import (
    CLASSES, FLUX_RATE, LABELERS, assign_phys_id, dissolve_to_seeps, load_pack)

import geopandas as gpd
from shapely.ops import unary_union
from sklearn.ensemble import (ExtraTreesClassifier,
                              GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

BASE = ["hull_area_m2", "mean_R", "mean_G", "mean_B"]
RICH = BASE + ["major_axis_m", "minor_axis_m", "aspect", "fill_ratio",
               "n_bubbles", "max_bubble_area_m2", "bubble_area_cv",
               "dens_050", "bright_range"]


def enrich(g: gpd.GeoDataFrame, seeps: pd.DataFrame) -> pd.DataFrame:
    """Add the shape / internal-structure / context features to a seep table."""
    # Local density is measured against EVERY bubble in the image, labeled or
    # not -- a seep in a crowded field is a different object from an isolated
    # one, and that context was the #2 signal in the pairwise grouper.
    pts = {img: np.c_[s["centroid_x_m"].to_numpy(), s["centroid_y_m"].to_numpy()]
           for img, s in g.groupby("image")}

    rows = []
    for _, r in seeps.iterrows():
        sub = g[(g["image"] == r["image"]) &
                (g["seep_group_id"] == r["seep_group_id"])]
        geom = unary_union(sub.geometry.values)
        rect = geom.minimum_rotated_rectangle
        xs, ys = rect.exterior.coords.xy
        e = [np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i]) for i in range(4)]
        major, minor = max(e[0], e[1]), min(e[0], e[1])

        ba = sub["area_m2"].to_numpy(dtype=float)
        cx = sub["centroid_x_m"].mean()
        cy = sub["centroid_y_m"].mean()
        P = pts[r["image"]]
        d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
        bright = (sub["mean_R"] + sub["mean_G"] + sub["mean_B"]).to_numpy() / 3

        rows.append({
            "major_axis_m": major,
            "minor_axis_m": minor,
            "aspect": major / minor if minor > 0 else 1.0,
            "fill_ratio": r["area_m2"] / r["hull_area_m2"]
            if r["hull_area_m2"] > 0 else 1.0,
            "max_bubble_area_m2": float(ba.max()),
            "bubble_area_cv": float(ba.std() / ba.mean()) if ba.mean() > 0 else 0.0,
            "dens_050": int((d < 0.5).sum()) - len(sub),
            "bright_range": float(bright.max() - bright.min()),
        })
    return pd.concat([seeps.reset_index(drop=True),
                      pd.DataFrame(rows)], axis=1)


def score(name, clf, X, y, groups, cv, sw=None) -> dict:
    fp = {"sample_weight": sw} if sw is not None else None
    pred = cross_val_predict(clf, X, y, groups=groups, cv=cv, params=fp)
    rep = classification_report(y, pred, labels=CLASSES, output_dict=True,
                                zero_division=0)
    ft = sum(FLUX_RATE[c] * (y == c).sum() for c in CLASSES)
    fpx = sum(FLUX_RATE[c] * (pred == c).sum() for c in CLASSES)
    return {"model": name, "acc": (pred == y).mean(),
            "macroF1": rep["macro avg"]["f1-score"],
            "A_f1": rep["A"]["f1-score"],
            "B_prec": rep["B"]["precision"], "B_rec": rep["B"]["recall"],
            "C_f1": rep["C"]["f1-score"],
            "flux_err%": 100 * (fpx - ft) / ft,
            "_pred": pred}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeling-dir", default=os.path.join(
        "data", "results", "SWIN", "AE", "20260428-1537_SWINxAE.weights",
        "labeling"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    S, G = [], []
    for w in LABELERS:
        g = load_pack(os.path.join(
            args.labeling_dir,
            f"gt_seeps_label_quarters_{w}_grouped.gpkg"), w)
        G.append(g)
        S.append(enrich(g, dissolve_to_seeps(g)))
    seeps = pd.concat(S, ignore_index=True)
    seeps["phys_id"] = assign_phys_id(seeps)
    d = seeps[(seeps["is_context"] == 0) &
              (seeps["is_overgrouped"] == 0)].dropna(subset=RICH)
    d = d.reset_index(drop=True)
    y = d["class"].to_numpy()
    groups = d["phys_id"].to_numpy()
    cv = GroupKFold(n_splits=5)
    flux_w = np.array([FLUX_RATE[c] for c in y])
    print(f"{len(d)} seeps / {d['phys_id'].nunique()} physical  "
          f"{pd.Series(y).value_counts().to_dict()}\n")

    sd = args.seed
    rows = []

    # ---- lever 1: features, model held fixed ------------------------------
    tree = lambda: DecisionTreeClassifier(max_depth=5, class_weight="balanced",
                                          random_state=sd)
    for tag, feats in (("BASE 4 feat", BASE), ("RICH 13 feat", RICH)):
        rows.append(score(f"DecisionTree d5 | {tag}", tree(),
                          d[feats].to_numpy(float), y, groups, cv))

    # ---- lever 2: model family, features held at RICH ----------------------
    Xr = d[RICH].to_numpy(float)
    fams = {
        "RandomForest 500": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=sd, n_jobs=-1),
        "ExtraTrees 500": ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=sd, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=sd),
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=sd)),
    }
    for n, m in fams.items():
        rows.append(score(f"{n} | RICH", m, Xr, y, groups, cv))

    # ---- lever 3: flux-weighted objective ---------------------------------
    for n, m in (("DecisionTree d5", tree()),
                 ("RandomForest 500", RandomForestClassifier(
                     n_estimators=500, min_samples_leaf=2, random_state=sd,
                     n_jobs=-1))):
        rows.append(score(f"{n} | RICH | FLUX-weighted", m, Xr, y, groups, cv,
                          sw=flux_w))

    res = pd.DataFrame(rows)
    best = res.loc[res["macroF1"].idxmax()]
    show = res.drop(columns="_pred").copy()
    for c in show.columns[1:]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    print("=" * 100)
    print("GROUPED 5-FOLD CV  (grouped on physical seep)")
    print("=" * 100)
    print(show.to_string(index=False))

    print(f"\nbest macro-F1: {best['model']}")
    cm = pd.DataFrame(confusion_matrix(y, best["_pred"], labels=CLASSES),
                      index=[f"true_{c}" for c in CLASSES],
                      columns=[f"pred_{c}" for c in CLASSES])
    print(cm.to_string())

    # what the added features are actually worth
    rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                class_weight="balanced", random_state=sd,
                                n_jobs=-1).fit(Xr, y)
    print("\nRandomForest feature importances (RICH):")
    for f, i in sorted(zip(RICH, rf.feature_importances_), key=lambda t: -t[1]):
        print(f"  {f:20s} {i:.3f}")


if __name__ == "__main__":
    main()