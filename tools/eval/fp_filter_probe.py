# tools/eval/fp_filter_probe.py
"""Can a post-hoc OBJECT-LEVEL filter reject false-positive bubble detections?

THE QUESTION
The detector's false positives concentrate where seeps do not. On the canonical
run the three shore chips (47/50/51) hold no ground truth at all yet emit 320
FPs -- 22% of the total, at zero recall cost -- and the two snow chips (21/52)
emit another 398 from 87 GT bubbles, while the densest chip (39) emits 151 from
1,316. Shore is removable geometrically by cropping to the lake polygon. Snow on
the ice is NOT: it speckles the surface rather than covering it, so there is no
clean polygon to cut. This probe asks whether those detections can instead be
rejected downstream, as a learned stage between connected-components and the
grouper -- the same post-hoc architecture the grouper and classifier already use.

WHY THIS IS NOT THE 2026-05-12 SNOW MASK AGAIN
That experiment thresholded HSV on PIXELS, unsupervised, and failed because snow
and bubble-rich ice are spectrally confusable. This is a SUPERVISED classifier on
OBJECTS, and the signal it leans on is size, not colour: FP components are 2-19x
smaller than true bubbles in every chip, while their brightness percentile sits
mid-pack (0.31-0.54), i.e. FPs are speckle, not bright blobs. Different mechanism,
different failure modes.

RESULT (2026-09-08, canonical run, LOIO by chip): ABSOLUTE FEATURES WIN
The module was built expecting within-image percentile ranks to win, by analogy
with the 2026-07-29 finding that absolute brightness is a per-chip exposure
confound. The ablation refuted that, and the default is `abs` because of it:

    features   pooled AUC   21.tif   52.tif   F1 @ keep-95% of TP
    abs             0.775    0.945    0.850   0.7055
    both            0.705    0.909    0.642   --
    rel             0.678    0.827    0.517   0.6814   (= no better than none)

TWO REASONS THE BRIGHTNESS LESSON DOES NOT TRANSFER TO AREA.
  1. A percentile rank is taken against the population being filtered, so it
     silently encodes that chip's own FP rate -- the thing you do not know at
     deploy time and which varies enormously (52.tif is 92% FP, so its true
     bubbles sit in the top 8% of the area rank; 39.tif is 85% TP, so its true
     bubbles span nearly the whole range). The same rank value means opposite
     things on the two chips. For the seep classifier the reference population
     is fixed regardless of the label being predicted, so no such feedback
     exists.
  2. Bubble area has an absolute physical meaning that "bright" does not. A
     20 cm2 blob is small on any lake; brightness is only interpretable
     relative to exposure. The confound argument is specific to radiometry.

A single GLOBAL area threshold remains a trap for the reason that motivated the
percentile idea -- true bubble size varies ~10x across chips (mean TP area
0.0124 m2 on 41.tif vs 0.1203 on 52.tif) and a 0.006 m2 floor takes global
bubble F1 from 0.645 to 0.527. The forest handles this by conditioning area on
shape and neighbourhood context rather than by rescaling it.

ALSO TESTED AND REJECTED: an unsupervised per-chip Otsu cut on log10(area),
motivated by the very high WITHIN-chip AUC of raw area (0.985 on 52.tif, 0.962
on 27.tif). It is excellent exactly where the area distribution is bimodal --
52.tif FPs 199 -> 19 with all 18 TPs kept -- and catastrophic where it is not,
splitting the unimodal dense chips through the middle (39.tif loses 572 of 879
TPs). Pooled F1 0.528, worse than doing nothing. Revisit only behind a
bimodality test that decides per chip whether the cut applies.

LABEL CAVEAT -- READ BEFORE TRUSTING A HIGH SCORE
"FP" here means "matched no GT component", NOT "is not a bubble". Where the GT is
incomplete, a real detection is labelled FP and the filter is being taught to
reject real bubbles. That is safe on 47/50/51 (no seeps exist there) but a live
risk on 21.tif, whose GT count swings 54 -> 40 between the full and certain
chippings. The script writes --review-out for QGIS spot-checking: the FPs the
filter is most confident are real, which is where mislabelled truth will sit.

VALIDATION
Leave-one-image-out by chip, always. A filter that memorises per-chip size
distributions is worthless on a new lake; 21 and 52 held out are the test that
matters, since they are the snow chips this is meant to fix. Shore chips are
excluded from train and eval (they are handled by cropping, and being 100% FP
they would inflate every score) and reported separately as a transfer check.

Baselines are single features needing no fitting, so they are honest controls: if
raw area alone matches the forest, use a threshold and skip the model.

Usage:
  python -m tools.eval.fp_filter_probe
  python -m tools.eval.fp_filter_probe --features rel     # abs | rel | both
  python -m tools.eval.fp_filter_probe --pred-dir PATH --scores-out s.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from tools.paths import CANONICAL_PRED_DIR, CANONICAL_PRED_RELDIR

# Chips whose GT band is empty. Every detection on them is an FP by construction,
# so they cannot inform a TP-vs-FP boundary; they are a pure transfer check.
SHORE_CHIPS = ("47.tif", "50.tif", "51.tif")
# The two chips with on-lake snow. Not special-cased anywhere in the fitting --
# named only so the report can call them out when they are the held-out fold.
SNOW_CHIPS = ("21.tif", "52.tif")

# Number of GT components missed entirely on the 9 GT-bearing chips (fn in
# bubble_level_per_image.csv). Recall must be scored against every GT bubble,
# including those no detection reached, or the operating-point table flatters
# the filter by shrinking its own denominator.
FN_UNDETECTED = 676

# Shape features straight off the connected component. Unlike GT-polygon
# solidity/eccentricity (excluded as hand-drawing artifacts, CLAUDE.md
# 2026-05-14) these are pixel-derived from the prediction, so they carry
# detector behaviour rather than annotation style and are fair to use here.
ABS_SHAPE = ["area_m2", "perim_m", "circularity", "solidity", "eccentricity"]
ABS_COLOR = ["mean_R", "mean_G", "mean_B"]
REL_FEATURES = [
    "area_pct", "perim_pct", "bright_pct",      # rank within the chip
    "log_area_rel", "bright_rel",               # ratio / offset vs chip median
    "circularity", "solidity", "eccentricity",  # scale-free already
]
# Neighbourhood context is scale-free in metres and identical at train and serve
# time. Density dominated the grouper, and isolated speckle should look different
# from a real satellite bubble sitting in a seep.
CONTEXT = ["n_025", "n_050", "n_100", "d_nn", "d_nn5"]


def load_labeled_bubbles(pred_dir):
    """Every predicted bubble, labelled TP/FP by the eval's matched pairs."""
    feats = pd.read_csv(os.path.join(pred_dir, "bubble_features.csv"))
    pairs = pd.read_csv(os.path.join(pred_dir, "bubble_level_pairs.csv"))
    matched = set(zip(pairs.image, pairs.pred_id))
    feats["is_tp"] = [(i, b) in matched for i, b in zip(feats.image, feats.bubble_id)]
    return feats


def add_derived_features(df):
    """Within-image ranks plus neighbourhood context, computed per chip.

    Everything here is derivable from predictions alone -- no ground truth, no
    cross-chip statistics -- so the deploy-time value equals the training value.
    """
    df = df.copy()
    df["bright"] = df[ABS_COLOR].mean(axis=1)
    out = []
    for _, g in df.groupby("image", sort=False):
        g = g.copy()
        for col, name in (("area_m2", "area_pct"), ("perim_m", "perim_pct"),
                          ("bright", "bright_pct")):
            # pct=True gives the rank in [0,1]; a single-detection chip yields
            # 1.0, which is harmless because such a chip carries no signal anyway.
            g[name] = g[col].rank(pct=True)
        med_area = g["area_m2"].median()
        g["log_area_rel"] = (np.log10(g["area_m2"] / med_area)
                             if med_area > 0 else 0.0)
        g["bright_rel"] = g["bright"] - g["bright"].median()

        xy = g[["centroid_x_m", "centroid_y_m"]].to_numpy()
        tree = cKDTree(xy)
        for r, name in ((0.25, "n_025"), (0.50, "n_050"), (1.00, "n_100")):
            # -1 drops the point's own match; a bare count would be r-independent.
            g[name] = tree.query_ball_point(xy, r, return_length=True) - 1
        # k=1 is the point itself, so the nearest neighbour is column 1.
        k = min(6, len(g))
        d, _ = tree.query(xy, k=k)
        d = np.atleast_2d(d)
        g["d_nn"] = d[:, 1] if k > 1 else np.inf
        g["d_nn5"] = d[:, min(5, k - 1)] if k > 1 else np.inf
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    # A lone detection has no neighbour; a large finite value keeps the split
    # meaningful ("isolated") where inf would poison the forest.
    for c in ("d_nn", "d_nn5"):
        df[c] = df[c].replace(np.inf, 99.0)
    return df


def feature_set(kind):
    if kind == "abs":
        cols = ABS_SHAPE + ABS_COLOR + CONTEXT
    elif kind == "rel":
        cols = REL_FEATURES + CONTEXT
    elif kind == "both":
        cols = ABS_SHAPE + ABS_COLOR + REL_FEATURES + CONTEXT
    else:
        raise SystemExit("--features must be abs|rel|both, got %r" % kind)
    # circularity/solidity/eccentricity are scale-free, so they sit in both the
    # absolute and the relative list; "both" would otherwise pass sklearn a
    # frame with duplicate column names.
    return list(dict.fromkeys(cols))


def loio_scores(df, cols, seed=0):
    """Out-of-fold P(is_tp): each chip scored by a model that never saw it."""
    scores = pd.Series(np.nan, index=df.index, dtype=float)
    for chip in sorted(df.image.unique()):
        tr, te = df[df.image != chip], df[df.image == chip]
        if tr.is_tp.nunique() < 2:
            continue
        rf = RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed)
        rf.fit(tr[cols], tr.is_tp)
        scores.loc[te.index] = rf.predict_proba(te[cols])[:, 1]
    return scores


def auc_or_none(y, s):
    """AUC is undefined on a fold that is all-TP or all-FP (25.tif has 0 FP)."""
    y = np.asarray(y)
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else None


def fmt_auc(a):
    return "   n/a" if a is None else "%6.3f" % a


def report_baselines(df):
    """Single features need no fitting, so their LOIO AUC is just their AUC."""
    print("\nBASELINE single features (no model; higher = separates TP from FP)")
    print("  %-16s %7s  per-chip AUC" % ("feature", "pooled"))
    for feat, sign in (("area_m2", 1), ("area_pct", 1), ("bright", 1),
                       ("bright_pct", 1), ("d_nn", -1)):
        pooled = auc_or_none(df.is_tp, sign * df[feat])
        per = []
        for chip in sorted(df.image.unique()):
            g = df[df.image == chip]
            per.append("%s=%s" % (chip.split(".")[0],
                                  fmt_auc(auc_or_none(g.is_tp, sign * g[feat])).strip()))
        print("  %-16s %7s  %s" % (feat, fmt_auc(pooled), " ".join(per)))


def operating_points(df, n_gt):
    """What the filter costs and buys at recall-preserving thresholds.

    Dropping a TP does not delete the ground truth -- it becomes an FN -- so
    recall falls as well as precision rising. That is why F1 is reported rather
    than precision alone.
    """
    tp_scores = df.loc[df.is_tp, "score"].to_numpy()
    print("\nOPERATING POINTS (bubble-level, n_gt=%d)" % n_gt)
    print("  %8s %7s | %5s %5s %5s | %6s %6s %6s | %10s"
          % ("keep TP", "thresh", "tp", "fn", "fp", "P", "R", "F1", "FP removed"))
    base_fp = int((~df.is_tp).sum())
    for keep in (1.00, 0.99, 0.98, 0.95, 0.90, 0.85):
        thr = -np.inf if keep >= 1.0 else np.quantile(tp_scores, 1 - keep)
        k = df[df.score >= thr]
        tp = int(k.is_tp.sum())
        fp = len(k) - tp
        fn = n_gt - tp
        P = tp / (tp + fp) if tp + fp else 0.0
        R = tp / n_gt
        f1 = 2 * P * R / (P + R) if P + R else 0.0
        pct = 100 * (base_fp - fp) / base_fp if base_fp else 0.0
        print("  %7.0f%% %7.3f | %5d %5d %5d | %6.4f %6.4f %6.4f | %9.0f%%"
              % (keep * 100, thr, tp, fn, fp, P, R, f1, pct))


def per_chip_at(df, keep):
    """The snow chips are the point of the exercise -- show them individually."""
    thr = np.quantile(df.loc[df.is_tp, "score"].to_numpy(), 1 - keep)
    print("\nPER-CHIP at the keep-%.0f%%-of-TP threshold (%.3f), "
          "each chip scored while held out" % (keep * 100, thr))
    print("  %8s %13s %15s" % ("chip", "tp kept", "fp kept"))
    for chip in sorted(df.image.unique()):
        g = df[df.image == chip]
        k = g[g.score >= thr]
        t0, t1 = int(g.is_tp.sum()), int(k.is_tp.sum())
        f0, f1_ = int((~g.is_tp).sum()), int((~k.is_tp).sum())
        drop = "-%.0f%%" % (100 * (f0 - f1_) / f0) if f0 else "  --"
        tag = "  <-SNOW" if chip in SNOW_CHIPS else ""
        print("  %8s %6d/%-6d %6d/%-6d %6s%s" % (chip, t1, t0, f1_, f0, drop, tag))


def shore_transfer_check(model_df, shore, cols, keep, seed=0):
    """Would the filter also have caught the shore FPs it never trained on?

    Redundancy, not a substitute for cropping -- but if it transfers here it is
    learning something about detection quality rather than about these 9 chips.
    """
    if shore.empty:
        return
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced",
        n_jobs=-1, random_state=seed)
    rf.fit(model_df[cols], model_df.is_tp)
    s = rf.predict_proba(shore[cols])[:, 1]
    thr = np.quantile(model_df.loc[model_df.is_tp, "score"].to_numpy(), 1 - keep)
    print("\nSHORE TRANSFER (never trained on; all %d are FP by construction, "
          "threshold %.3f)" % (len(shore), thr))
    for chip in sorted(shore.image.unique()):
        m = shore.image.to_numpy() == chip
        print("  %8s %4d/%-4d survive the filter"
              % (chip, int((s[m] >= thr).sum()), int(m.sum())))
    print("  %8s %4d/%-4d (%.0f%% of shore FPs rejected)"
          % ("TOTAL", int((s >= thr).sum()), len(shore), 100 * (s < thr).mean()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    default_dir = (CANONICAL_PRED_RELDIR if os.path.isdir(CANONICAL_PRED_RELDIR)
                   else CANONICAL_PRED_DIR)
    ap.add_argument("--pred-dir", default=default_dir)
    ap.add_argument("--features", default="abs", choices=["abs", "rel", "both"],
                    help="abs = raw shape/colour/context (default; measured "
                         "best, pooled AUC 0.775). rel = within-image "
                         "percentile ranks (0.678 -- worse; see the RESULT "
                         "block in the module docstring for why). both = 0.705.")
    ap.add_argument("--keep-tp", type=float, default=0.95,
                    help="TP fraction retained at the reported operating point")
    ap.add_argument("--scores-out", default=None,
                    help="CSV of per-bubble out-of-fold scores, for QGIS")
    ap.add_argument("--review-out", default=None,
                    help="CSV of the FPs the filter is most confident are real "
                         "-- where incomplete ground truth would show up")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not os.path.isdir(args.pred_dir):
        print("ERROR: pred dir not found: %s" % args.pred_dir, file=sys.stderr)
        return 1

    raw = load_labeled_bubbles(args.pred_dir)
    df = add_derived_features(raw)
    shore = df[df.image.isin(SHORE_CHIPS)].reset_index(drop=True)
    model_df = df[~df.image.isin(SHORE_CHIPS)].reset_index(drop=True)

    n_gt = int(model_df.is_tp.sum()) + FN_UNDETECTED
    print("pred dir : %s" % args.pred_dir)
    print("features : %s" % args.features)
    print("modelled : %d detections on %d GT-bearing chips (%d TP / %d FP)"
          % (len(model_df), model_df.image.nunique(),
             int(model_df.is_tp.sum()), int((~model_df.is_tp).sum())))
    print("excluded : %d detections on %d shore chips"
          % (len(shore), len(SHORE_CHIPS)))

    report_baselines(model_df)

    cols = feature_set(args.features)
    model_df["score"] = loio_scores(model_df, cols, seed=args.seed)

    print("\nRANDOM FOREST, leave-one-image-out, %d features" % len(cols))
    print("  %14s %6s %7s" % ("held-out chip", "n", "AUC"))
    for chip in sorted(model_df.image.unique()):
        g = model_df[model_df.image == chip]
        tag = "  <-SNOW" if chip in SNOW_CHIPS else ""
        print("  %14s %6d %s%s"
              % (chip, len(g), fmt_auc(auc_or_none(g.is_tp, g.score)), tag))
    print("  %14s %6d %s" % ("POOLED", len(model_df),
                             fmt_auc(auc_or_none(model_df.is_tp, model_df.score))))

    operating_points(model_df, n_gt)
    per_chip_at(model_df, args.keep_tp)
    shore_transfer_check(model_df, shore, cols, args.keep_tp, seed=args.seed)

    if args.scores_out:
        keep = ["image", "bubble_id", "centroid_x_m", "centroid_y_m",
                "area_m2", "is_tp", "score"]
        model_df[keep].to_csv(args.scores_out, index=False)
        print("\nwrote %s" % args.scores_out)
    if args.review_out:
        sus = (model_df[~model_df.is_tp]
               .nlargest(200, "score")
               [["image", "bubble_id", "centroid_x_m", "centroid_y_m",
                 "area_m2", "score"]])
        sus.to_csv(args.review_out, index=False)
        print("wrote %s -- %d 'FPs' the filter thinks are real; spot-check "
              "these in QGIS before trusting the labels" % (args.review_out, len(sus)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
