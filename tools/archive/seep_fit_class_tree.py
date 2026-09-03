"""Phase 7 (2026-05-28 plan): discover A/B/C classification thresholds.

Fits a shallow decision tree on the labeled, dissolved GT seeps to turn Walter's
A/B/C class judgments into human-readable threshold rules, then applies those
rules to the predicted seeps and (separately) to the pre-grouped GT seeps.

Input: gt_seeps_labeled.gpkg (from tools/seep_labels_consolidate.py).

Key decisions baked in (from CLAUDE.md 2026-05-14 / 2026-05-28 / 2026-06-01):
  * Features = hull_area_m2, mean_R, mean_G, mean_B.
    - hull_area_m2 (convex-hull footprint) is the SIZE axis. It replaces
      bubble-union area_m2 / perim_m because hull area is defined consistently
      across pregrouped envelopes, labeler-grouped unions, AND pred clusters,
      whereas bubble-union area_m2 means different things for an envelope (a
      footprint) vs a union (summed bubble area) and so does not transfer.
    - solidity / eccentricity / circularity stay EXCLUDED -- annotation
      artifacts (hand-drawn polygon smoothing), not physical seep morphology.
  * Train on ALL labeled seeps, INCLUDING is_pregrouped == True (changed
    2026-06-01). Earlier we excluded pregrouped envelopes because their
    bubble-union area/perim/brightness were envelope-contaminated. With the
    size axis now carried by the consistent hull_area_m2, and the chip-39
    pilot showing the envelope brightness offset is small (<=12 of a ~30-pt A/B
    gap) and benign, pregrouped seeps can stay in -- which matters because the
    large-hull B regime lives almost entirely in the pregrouped population.
  * Shallow tree (max_depth=3, class_weight='balanced') so the splits read as
    simple thresholds and the tree does not overfit the few hundred labels.

Sanity check (step 5): on seep_level_pairs_cluster.csv restricted to clean 1:1
matches (n_pred_matches == 1), apply the tree to the pred-side feature columns
and compare to the matched GT seep's labeled class -- this is the GT->pred
threshold-portability check.
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd

try:
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.model_selection import cross_val_score
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
LABELED_IN = os.path.join(PRED_DIR, "gt_seeps_labeled.gpkg")
PRED_IN = os.path.join(PRED_DIR, "pred_seeps.gpkg")
PAIRS_IN = os.path.join(PRED_DIR, "seep_level_pairs_cluster.csv")

FEATURES = ["hull_area_m2", "mean_R", "mean_G", "mean_B"]


def _clean_class(s):
    return s.fillna("").astype(str).str.strip()


def fit_tree(labeled, max_depth=3, seed=42):
    """Fit on all labeled seeps (any is_pregrouped) with a non-empty class.

    Pregrouped envelopes are kept in training now that the size feature is the
    consistent hull_area_m2 rather than the envelope-contaminated bubble-union
    area_m2 (see module docstring, 2026-06-01).
    """
    df = labeled.copy()
    df["class"] = _clean_class(df["class"])
    df = df[df["class"] != ""]
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise SystemExit(f"labeled file missing feature columns: {missing}")
    df = df.dropna(subset=FEATURES)
    if len(df) < 10:
        raise SystemExit(f"only {len(df)} usable labeled seeps -- need labeling "
                         f"to be done (non-empty class on grouped seeps) first.")
    X = df[FEATURES].to_numpy(dtype=float)
    y = df["class"].to_numpy()
    print(f"training on {len(df)} seeps; class counts: "
          f"{dict(pd.Series(y).value_counts())}")
    clf = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced",
                                 random_state=seed)
    clf.fit(X, y)
    train_acc = clf.score(X, y)
    print(f"train accuracy = {train_acc:.3f}")
    n_classes = len(np.unique(y))
    if len(df) >= 5 * n_classes:
        cv = min(5, int(pd.Series(y).value_counts().min()))
        if cv >= 2:
            scores = cross_val_score(clf, X, y, cv=cv)
            print(f"{cv}-fold CV accuracy = {scores.mean():.3f} +/- {scores.std():.3f}")
    print("\nfeature importances:")
    for f, imp in sorted(zip(FEATURES, clf.feature_importances_),
                         key=lambda t: -t[1]):
        print(f"  {f:10s} {imp:.3f}")
    print("\n=== decision rules (export_text) ===")
    print(export_text(clf, feature_names=FEATURES))
    return clf


def apply_tree(clf, gdf, out_path, label="rows"):
    df = gdf.copy()
    ok = df[FEATURES].notna().all(axis=1)
    df["predicted_class"] = ""
    if ok.any():
        df.loc[ok, "predicted_class"] = clf.predict(
            df.loc[ok, FEATURES].to_numpy(dtype=float))
    if os.path.exists(out_path):
        os.remove(out_path)
    df.to_file(out_path, driver="GPKG")
    vc = dict(pd.Series(df.loc[ok, "predicted_class"]).value_counts())
    print(f"  classified {int(ok.sum())}/{len(df)} {label} -> "
          f"{os.path.basename(out_path)}  {vc}")
    return df


def sanity_check_pairs(clf, pairs_path, labeled):
    """GT->pred threshold portability on clean 1:1 matches."""
    if not os.path.exists(pairs_path):
        print(f"\n(sanity check skipped: {os.path.basename(pairs_path)} not found)")
        return
    pairs = pd.read_csv(pairs_path)
    if "n_pred_matches" not in pairs.columns:
        print("\n(sanity check skipped: no n_pred_matches column)")
        return
    clean = pairs[pairs["n_pred_matches"] == 1].copy()
    pred_cols = [f"{f}_pred" for f in FEATURES]
    if not all(c in clean.columns for c in pred_cols):
        print("\n(sanity check skipped: pred feature columns absent)")
        return
    clean = clean.dropna(subset=pred_cols)
    if clean.empty:
        print("\n(sanity check skipped: no clean 1:1 rows with features)")
        return

    # Build GT class lookup from the labeled file, keyed (image, seep_id).
    lab = labeled.copy()
    lab["class"] = _clean_class(lab["class"])
    key_cols = [c for c in ("image", "seep_id") if c in lab.columns and c in clean.columns]
    pred_class = clf.predict(clean[pred_cols].to_numpy(dtype=float))
    clean["pred_from_pred_features"] = pred_class

    if "image" in key_cols and "seep_id" in key_cols:
        gt_lut = lab.set_index(["image", "seep_id"])["class"].to_dict()
        clean["gt_class"] = [gt_lut.get((im, sid), "")
                             for im, sid in zip(clean["image"], clean["seep_id"])]
    elif "seep_id" in clean.columns:
        gt_lut = lab.set_index("seep_id")["class"].to_dict()
        clean["gt_class"] = clean["seep_id"].map(gt_lut).fillna("")
    else:
        print("\n(sanity check skipped: cannot join GT class)")
        return

    have = clean[clean["gt_class"] != ""]
    if have.empty:
        print("\n(sanity check: no matched GT classes available yet)")
        return
    agree = (have["pred_from_pred_features"] == have["gt_class"]).mean()
    print(f"\nGT->pred threshold portability (clean 1:1, n={len(have)}): "
          f"class-agreement = {agree:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Fit A/B/C class tree on labeled seeps.")
    ap.add_argument("--labeled", default=LABELED_IN)
    ap.add_argument("--pred", default=PRED_IN)
    ap.add_argument("--pairs", default=PAIRS_IN)
    ap.add_argument("--max-depth", type=int, default=3)
    args = ap.parse_args()

    if not _HAS_SK:
        raise SystemExit("scikit-learn required (pip install scikit-learn)")
    if not os.path.exists(args.labeled):
        raise SystemExit(f"not found: {args.labeled}\n"
                         f"Run tools/seep_labels_consolidate.py first.")

    labeled = gpd.read_file(args.labeled)
    clf = fit_tree(labeled, max_depth=args.max_depth)

    print("\napplying tree:")
    # Pre-grouped GT seeps written as a convenience subset. They are now part of
    # training too (2026-06-01); this just emits their predicted_class for review.
    if "is_pregrouped" in labeled.columns:
        pre = labeled[labeled["is_pregrouped"].fillna(0).astype(int) == 1]
        if len(pre):
            apply_tree(clf, pre,
                       os.path.join(PRED_DIR, "gt_seeps_pregrouped_classified.gpkg"),
                       label="pre-grouped GT seeps")
    # Predicted seeps.
    if os.path.exists(args.pred):
        pred = gpd.read_file(args.pred)
        apply_tree(clf, pred,
                   os.path.join(PRED_DIR, "pred_seeps_classified.gpkg"),
                   label="pred seeps")
    else:
        print(f"  (pred file {os.path.basename(args.pred)} not found, skipping)")

    sanity_check_pairs(clf, args.pairs, labeled)


if __name__ == "__main__":
    main()
