"""Does adding hand-hunted C seeps improve the A/B/C classifier?

Tests the 2026-07-29 top-priority labeling action ("have labelers hunt C seeps
specifically") by scoring a C-only enrichment pack against the 3-labeler
baseline. C is ~4% of seeps but ~52% of the methane, and cross-chip C recall was
the weak point (LOIO C F1 0.475, recall 0.55), so C examples are the highest-
value labels available.

TWO SEPARATE QUESTIONS, TWO SEPARATE EXPERIMENTS
  1. PROBE -- how good is the CURRENT model at finding C on unseen chips?
     The enrichment seeps are all truth-C on chips the model never saw, so
     predicting them measures cross-chip C RECALL directly. No retraining, no
     class-prior assumptions, nothing to argue about.
  2. AUGMENT -- does TRAINING on them improve C on the original set?
     Baseline vs baseline+enrichment, both scored on the ORIGINAL seeps only, so
     the test distribution (and therefore the flux truth) is identical between
     the two arms and the delta is attributable to the added training rows.

WHY THE ENRICHMENT SET IS NOT A TEST SET FOR ANYTHING ELSE
The labeler hunted C and labeled ONLY C, so those chips have no A/B truth. The
sample is deliberately enriched, which means:
  * it CAN be trained on (adding minority-class examples is the point);
  * it CANNOT be used to measure accuracy, prevalence, class balance, or total
    flux -- a set that is 100% C by construction makes every one of those
    meaningless. All headline metrics below are computed on the ORIGINAL seeps.
The enrichment chips are disjoint from the original 8, which the script asserts.
That disjointness is what makes leave-one-image-out clean: holding out an
original chip leaves every enrichment row in train, by construction.

Model is the 2026-07-29 deploy point: class_weight='balanced' RandomForest(500,
min_samples_leaf=2) on the RICH 13 features. Grouping/CV discipline (phys_id
union-find) is inherited from seep_fit_class_tree_labelers.

Usage:
  python tools/seep_c_augment_eval.py                       # uses the default pack
  python tools/seep_c_augment_eval.py --augment-pack PACK.gpkg
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import geopandas as gpd  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from seep_fit_class_tree_labelers import (  # noqa: E402
    CLASSES, FLUX_RATE, FLUX_SIGMA, LABELERS, assign_phys_id,
    dissolve_to_seeps, load_pack)
from seep_class_model_comparison import RICH, enrich  # noqa: E402

warnings.filterwarnings("ignore")

LAB_DIR = os.path.join("data", "results", "SWIN", "AE",
                       "20260428-1537_SWINxAE.weights", "labeling")
DEFAULT_AUGMENT = os.path.join(
    LAB_DIR, "gt_seeps_label_all_chips_grouped.pre_hulls_20260729-130809.gpkg")
SEED = 42


def rf():
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight="balanced", random_state=SEED,
                                  n_jobs=-1)


# --------------------------------------------------------------------------- #
# per-image brightness normalization
# --------------------------------------------------------------------------- #
# WHY (established 2026-07-29, Allison): absolute mean_R/G/B is NOT comparable
# across chips -- the chips differ in exposure, so a whole image is uniformly
# darker or brighter and A/B/C all shift with it. What identifies a C seep is
# being bright RELATIVE TO ITS OWN IMAGE, not bright on the 0-255 scale.
# Measured on the labeled sets: original C sits at the 95.4th within-image
# brightness percentile and the hunted C at the 96.8th -- nearly identical --
# while their ABSOLUTE brightness differs by 41 units (222 vs 181) purely
# because the hunted C's chips are ~22 units darker. Absolute brightness
# therefore encodes chip exposure as if it were seep class, which is exactly the
# kind of feature that survives in-distribution CV and fails cross-chip.
#
# The reference population is every DETECTED bubble in the image, so this is
# computable at deploy time with no labels -- train/serve consistent.
REL_CHANNELS = ["mean_R", "mean_G", "mean_B"]


def image_reference(ref: gpd.GeoDataFrame) -> dict:
    """image -> sorted per-channel values + brightness spread, for ranking."""
    out = {}
    for img, s in ref.groupby("image"):
        bright = (s["mean_R"] + s["mean_G"] + s["mean_B"]).to_numpy(float) / 3
        out[img] = {
            **{c: np.sort(s[c].to_numpy(dtype=float)) for c in REL_CHANNELS},
            "_std": float(bright.std()) or 1.0,
        }
    return out


def relativize(df: pd.DataFrame, ref: dict, mode: str) -> pd.DataFrame:
    """Replace absolute brightness with a within-image relative measure.

    mode 'pct' = percentile rank of the seep among the image's bubbles (bounded,
    robust to the image's brightness distribution shape); 'abs' = leave alone.
    bright_range (a within-seep spread) is scaled by the image's own brightness
    std under 'pct', so it too stops carrying chip exposure.
    """
    if mode == "abs":
        return df
    d = df.copy()
    for c in REL_CHANNELS:
        vals = np.empty(len(d))
        for i, (img, v) in enumerate(zip(d["image"], d[c].to_numpy(float))):
            arr = ref[img][c]
            vals[i] = np.searchsorted(arr, v) / max(len(arr), 1)
        d[c] = vals
    if "bright_range" in d.columns:
        d["bright_range"] = [r / ref[img]["_std"]
                             for img, r in zip(d["image"], d["bright_range"])]
    return d


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build(pack_paths: dict[str, str]) -> pd.DataFrame:
    """packs -> enriched seep table (one row per labeled (image, group))."""
    out = []
    for who, fp in pack_paths.items():
        g = load_pack(fp, who)
        s = dissolve_to_seeps(g)
        if s.empty:
            print(f"  {who:10s} NO LABELED SEEPS in {os.path.basename(fp)}")
            continue
        s = enrich(g, s)
        n_cls = int((g["class"] != "").sum())
        print(f"  {who:10s} {len(g):5d} polygons ({n_cls:4d} classified) "
              f"-> {len(s):4d} labeled seeps  {s['class'].value_counts().to_dict()}")
        out.append(s)
    return pd.concat(out, ignore_index=True)


def trainable(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Drop the rows that are excluded from FITTING (not from kappa).

    context-touching = truncated group = wrong hull footprint; overgrouped = one
    polygon spanning several seeps.
    """
    keep = (df["is_context"] == 0) & (df["is_overgrouped"] == 0)
    dropped = (~keep).sum()
    d = df[keep].dropna(subset=RICH).reset_index(drop=True)
    print(f"  [{tag}] {len(df)} labeled -> {len(d)} trainable "
          f"(dropped {int(dropped)} context/overgrouped, "
          f"{len(df) - int(dropped) - len(d)} missing features)")
    return d


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def oof(folds, Xo, yo, Xa=None, ya=None):
    """Out-of-fold predictions on the ORIGINAL rows.

    `folds` index into the original arrays only; the augment rows (if given) are
    appended to EVERY training fold and never tested on, so both arms of the A/B
    are scored on identical test sets.
    """
    pred = np.empty(len(yo), dtype=object)
    for tr, te in folds:
        X, y = Xo[tr], yo[tr]
        if Xa is not None and len(Xa):
            X, y = np.vstack([X, Xa]), np.concatenate([y, ya])
        pred[te] = rf().fit(X, y).predict(Xo[te])
    return pred


def flux(y) -> float:
    return float(sum(FLUX_RATE[c] * (y == c).sum() for c in CLASSES))


def report(name, y, pred) -> dict:
    acc = float((pred == y).mean())
    rep = classification_report(y, pred, labels=CLASSES, output_dict=True,
                                zero_division=0)
    ft, fp = flux(y), flux(pred)
    err = 100 * (fp - ft) / ft
    rate_unc = 100 * np.sqrt(sum((FLUX_SIGMA[c] * (y == c).sum()) ** 2
                                for c in CLASSES)) / ft
    n_ct, n_cp = int((y == "C").sum()), int((pred == "C").sum())
    # C carries ~52% of the flux on ~4% of seeps, so its own contribution to the
    # error is reported separately -- it dominates the total.
    c_term = 100 * FLUX_RATE["C"] * (n_cp - n_ct) / ft
    print(f"\n--- {name} ---")
    print(f"  accuracy={acc:.3f}  macro-F1={rep['macro avg']['f1-score']:.3f}")
    for c in CLASSES:
        r = rep[c]
        print(f"    {c}: precision={r['precision']:.3f} recall={r['recall']:.3f} "
              f"F1={r['f1-score']:.3f}  (n={int(r['support'])})")
    cm = pd.DataFrame(confusion_matrix(y, pred, labels=CLASSES),
                      index=[f"true_{c}" for c in CLASSES],
                      columns=[f"pred_{c}" for c in CLASSES])
    print("  " + cm.to_string().replace("\n", "\n  "))
    print(f"  flux: truth={ft:,.0f}  pred={fp:,.0f}  model err={err:+.1f}%  "
          f"rate unc=+/-{rate_unc:.1f}%")
    print(f"  C seeps: {n_cp} predicted vs {n_ct} true -> C term {c_term:+.1f}% "
          f"of total flux")
    return {"name": name, "acc": acc,
            "macro_f1": rep["macro avg"]["f1-score"],
            "C_recall": rep["C"]["recall"], "C_prec": rep["C"]["precision"],
            "C_f1": rep["C"]["f1-score"], "flux_err": err, "C_term": c_term}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labeling-dir", default=LAB_DIR)
    ap.add_argument("--augment-pack", default=DEFAULT_AUGMENT)
    ap.add_argument("--brightness", choices=("abs", "pct"), default="pct",
                    help="brightness features: 'abs' = raw mean_R/G/B (the old "
                         "locked form), 'pct' = within-image percentile rank "
                         "(corrects for per-chip exposure)")
    args = ap.parse_args()

    print("=" * 74)
    print("BASELINE: the three labeler quarter packs")
    print("=" * 74)
    base = build({who: os.path.join(
        args.labeling_dir, f"gt_seeps_label_quarters_{who}_grouped.gpkg")
        for who in LABELERS})

    print("\n" + "=" * 74)
    print(f"ENRICHMENT: {os.path.basename(args.augment_pack)}")
    print("=" * 74)
    aug = build({"C-hunt": args.augment_pack})

    # The enrichment pack is C-only BY CONSTRUCTION. Say so loudly -- every
    # accuracy/prevalence/flux statement below is on the original set for exactly
    # this reason.
    print(f"\n  enrichment class balance: {aug['class'].value_counts().to_dict()}")
    if set(aug["class"]) == {"C"}:
        print("  -> C-ONLY (targeted hunt): usable as training rows and as a "
              "C-recall probe;\n     NOT usable for accuracy, prevalence, or "
              "total flux.")

    base_chips = sorted(base["image"].unique())
    aug_chips = sorted(aug["image"].unique())
    overlap = sorted(set(base_chips) & set(aug_chips))
    print(f"\n  original chips ({len(base_chips)}): {base_chips}")
    print(f"  enrichment chips ({len(aug_chips)}): {aug_chips}")
    print(f"  overlap: {overlap or 'NONE -- leave-one-image-out is clean'}")
    if overlap:
        raise SystemExit(
            "enrichment shares chips with the baseline; the LOIO design below "
            "assumes disjoint chips (otherwise held-out chips leak via the "
            "enrichment rows). Re-check before trusting any number.")

    # phys_id MUST be computed on the UNFILTERED pooled set, then filtered --
    # never the other way round. Union-find links seeps that share a member
    # (image, seep_id), and those links can run THROUGH a row that the
    # context/overgrouped filter drops: if excluded seep X shares members with
    # both Y and Z, only the unfiltered pass puts Y and Z in one group. Filtering
    # first silently splits such groups and weakens the leakage guard -- measured
    # cost on this data: grouped-CV macro-F1 reads 0.730 instead of 0.765 and
    # flux +14.3% instead of +9.2%, purely from the changed fold assignment.
    # (LOIO is immune -- its folds are the chips, not phys_id.)
    pooled = pd.concat([base, aug], ignore_index=True)
    pooled["phys_id"] = assign_phys_id(pooled)
    db = trainable(pooled.iloc[:len(base)], "baseline")
    da = trainable(pooled.iloc[len(base):], "enrichment")
    print(f"  baseline class balance: {db['class'].value_counts().to_dict()}")
    n_dup = len(db) - db["phys_id"].nunique()
    print(f"  baseline: {len(db)} seeps over {db['phys_id'].nunique()} physical "
          f"({n_dup} duplicate labelings across the shared calibration units)")

    # Per-image brightness reference: every detected bubble in every chip. Taken
    # from the whole-chip pack so the ORIGINAL quarter-pack seeps and the hunted
    # seeps are ranked against the same population for their image.
    ref = image_reference(gpd.read_file(args.augment_pack, layer="labels"))
    missing = (set(db["image"]) | set(da["image"])) - set(ref)
    if missing:
        raise SystemExit(f"no per-image brightness reference for {sorted(missing)}; "
                         f"--augment-pack must be the whole-chip pack")

    yo = db["class"].to_numpy()
    ya = da["class"].to_numpy()
    mats = {m: (relativize(db, ref, m)[RICH].to_numpy(float),
                relativize(da, ref, m)[RICH].to_numpy(float))
            for m in ("abs", "pct")}

    loio_ = [(np.flatnonzero(db["image"].to_numpy() != c),
              np.flatnonzero(db["image"].to_numpy() == c)) for c in base_chips]
    gkf_ = list(GroupKFold(n_splits=5).split(np.zeros((len(db), 1)), yo,
                                            groups=db["phys_id"]))

    print("\n" + "=" * 74)
    print("HEAD-TO-HEAD: absolute vs per-image-relative brightness")
    print("=" * 74)
    print("'abs' = mean_R/G/B as stored. 'pct' = within-image percentile rank of\n"
          "the seep among that chip's bubbles. All scored on the ORIGINAL seeps.")
    h2h = []
    for mode, (Xo_, Xa_) in mats.items():
        for cvn, folds in (("LOIO", loio_), ("groupedCV", gkf_)):
            for arm, xa, yaa in (("baseline", None, None), ("+81C", Xa_, ya)):
                p = oof(folds, Xo_, yo, xa, yaa)
                r = classification_report(yo, p, labels=CLASSES,
                                          output_dict=True, zero_division=0)
                ft, fp = flux(yo), flux(p)
                h2h.append({"bright": mode, "cv": cvn, "arm": arm,
                            "acc": (p == yo).mean(),
                            "macro_f1": r["macro avg"]["f1-score"],
                            "C_prec": r["C"]["precision"],
                            "C_recall": r["C"]["recall"],
                            "C_f1": r["C"]["f1-score"],
                            "flux_err": 100 * (fp - ft) / ft})
    h = pd.DataFrame(h2h)
    print(h.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n  effect of per-image normalization (pct minus abs):")
    for cvn in ("LOIO", "groupedCV"):
        for arm in ("baseline", "+81C"):
            a = h[(h.bright == "abs") & (h.cv == cvn) & (h.arm == arm)].iloc[0]
            b = h[(h.bright == "pct") & (h.cv == cvn) & (h.arm == arm)].iloc[0]
            print(f"    {cvn:9s} {arm:8s} C_f1 {a.C_f1:.3f}->{b.C_f1:.3f} "
                  f"({b.C_f1 - a.C_f1:+.3f})   C_recall {a.C_recall:.3f}->"
                  f"{b.C_recall:.3f}   flux {a.flux_err:+.1f}%->{b.flux_err:+.1f}%")

    Xo, Xa = mats[args.brightness]
    print(f"\n[detail below uses brightness mode '{args.brightness}']")

    # ---------------------------------------------------------------- probe
    print("\n" + "=" * 74)
    print("EXPERIMENT 1 -- PROBE: current model vs the hand-hunted C seeps")
    print("=" * 74)
    print("Model trained on the baseline only, predicting seeps on 7 chips it "
          "has never seen.\nEvery truth label is C, so this is cross-chip C "
          "RECALL with nothing else mixed in.")
    m = rf().fit(Xo, yo)
    pa = m.predict(Xa)
    hit = int((pa == "C").sum())
    print(f"\n  found {hit}/{len(pa)} = {hit / len(pa):.1%} C recall on unseen chips")
    print(f"  predictions: {pd.Series(pa).value_counts().to_dict()}")
    per_chip = (pd.DataFrame({"image": da["image"], "ok": pa == "C"})
                .groupby("image")["ok"].agg(["sum", "count"]))
    per_chip["recall"] = per_chip["sum"] / per_chip["count"]
    print("\n  by chip:")
    print("    " + per_chip.to_string().replace("\n", "\n    "))
    missed = da.loc[pa != "C", ["image", "seep_group_id", "hull_area_m2",
                                "major_axis_m", "n_bubbles", "mean_R"]]
    if len(missed):
        print(f"\n  the {len(missed)} missed C seeps (called {sorted(set(pa[pa != 'C']))}):")
        print("    " + missed.to_string(index=False).replace("\n", "\n    "))
        print(f"\n  missed vs found hull_area_m2: "
              f"{missed['hull_area_m2'].median():.4f} vs "
              f"{da.loc[pa == 'C', 'hull_area_m2'].median():.4f} (median)")

    # -------------------------------------------------------------- augment
    print("\n" + "=" * 74)
    print("EXPERIMENT 2 -- AUGMENT: does training on them help the original set?")
    print("=" * 74)
    print("Both arms scored on the ORIGINAL seeps only (identical test sets, so\n"
          "the flux truth is identical and the delta is the added training rows).")

    loio, gkf = loio_, gkf_

    rows = []
    for cv_name, folds in (("Leave-one-image-out (cross-chip)", loio),
                           ("Grouped 5-fold CV (in-distribution)", gkf)):
        print("\n" + "-" * 74)
        print(cv_name)
        print("-" * 74)
        b = report(f"{cv_name} | BASELINE", yo, oof(folds, Xo, yo))
        a = report(f"{cv_name} | + {len(da)} hunted C", yo,
                   oof(folds, Xo, yo, Xa, ya))
        b["cv"], a["cv"] = cv_name, cv_name
        b["arm"], a["arm"] = "baseline", f"+{len(da)}C"
        rows += [b, a]

    # ------------------------------------------------- where do they sit?
    # If the hunted C seeps occupy the feature region the original labelers
    # called B, then training on them necessarily converts original B -> C, and
    # no amount of reweighting fixes it -- it is a labeling-consistency problem,
    # not a class-balance one. So compare the populations directly.
    print("\n" + "=" * 74)
    print("DIAGNOSTIC -- where the hunted C seeps sit in feature space")
    print("=" * 74)
    key = ["hull_area_m2", "major_axis_m", "n_bubbles", "mean_R", "fill_ratio"]
    pops = {"original A": db[db["class"] == "A"], "original B": db[db["class"] == "B"],
            "original C": db[db["class"] == "C"], "hunted C": da}
    med = pd.DataFrame({k: v[key].median() for k, v in pops.items()}).T
    med.insert(0, "n", [len(v) for v in pops.values()])
    print(med.to_string(float_format=lambda v: f"{v:.4f}"))
    oc, hc = db.loc[db["class"] == "C", "hull_area_m2"], da["hull_area_m2"]
    print(f"\n  hull_area_m2: hunted-C median {hc.median():.4f} vs original-C "
          f"{oc.median():.4f}  (ratio {hc.median() / oc.median():.2f}x)")
    ob = db.loc[db["class"] == "B", "hull_area_m2"]
    print(f"  fraction of hunted C smaller than the original-B median "
          f"({ob.median():.4f}): {(hc < ob.median()).mean():.1%}")
    print(f"  fraction of hunted C inside the original-B interquartile range: "
          f"{((hc >= ob.quantile(.25)) & (hc <= ob.quantile(.75))).mean():.1%}")

    # ------------------------------------------------------- weight sweep
    # The enrichment is a TARGETED sample, so pooling it at full weight distorts
    # the C prior the model sees. Sweeping its sample weight traces the
    # recall-vs-flux trade and shows whether an operating point exists that keeps
    # the recall gain without over-calling C.
    print("\n" + "=" * 74)
    print("WEIGHT SWEEP -- enrichment rows down-weighted (targeted sample)")
    print("=" * 74)
    print("w=0 is the baseline; w=1 is naive pooling. Scored LOIO on the "
          "original seeps.")
    sweep = []
    for w in (0.0, 0.1, 0.25, 0.5, 1.0):
        pred = np.empty(len(yo), dtype=object)
        for tr, te in loio:
            X, y = Xo[tr], yo[tr]
            sw = np.ones(len(tr))
            if w > 0:
                X, y = np.vstack([X, Xa]), np.concatenate([y, ya])
                sw = np.concatenate([sw, np.full(len(Xa), w)])
            pred[te] = rf().fit(X, y, sample_weight=sw).predict(Xo[te])
        r = classification_report(yo, pred, labels=CLASSES, output_dict=True,
                                  zero_division=0)
        ft, fp = flux(yo), flux(pred)
        sweep.append({"w": w, "acc": (pred == yo).mean(),
                      "macro_f1": r["macro avg"]["f1-score"],
                      "C_prec": r["C"]["precision"], "C_recall": r["C"]["recall"],
                      "C_f1": r["C"]["f1-score"],
                      "n_C_pred": int((pred == "C").sum()),
                      "flux_err": 100 * (fp - ft) / ft})
    sw_t = pd.DataFrame(sweep)
    print(sw_t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = sw_t.iloc[sw_t["flux_err"].abs().values.argmin()]
    print(f"\n  |flux error| minimised at w={best['w']}: {best['flux_err']:+.1f}% "
          f"(C recall {best['C_recall']:.3f}, C F1 {best['C_f1']:.3f})")
    print(f"  best C F1 at w={sw_t.iloc[sw_t['C_f1'].values.argmax()]['w']}")

    # --------------------------------------------------- who flips, and why
    # The flux blow-up is driven by original B seeps being reclassified C. Those
    # flips are either model error OR original-truth error -- the hunted C set
    # covers a large-but-dark region the original 65 C seeps missed, so some
    # original B may genuinely be C. That cannot be settled from the data; it
    # needs re-adjudication, so dump the exact seeps to review.
    print("\n" + "=" * 74)
    print("WHICH ORIGINAL SEEPS FLIP, AND ARE THEY PLAUSIBLY C?")
    print("=" * 74)
    pb = oof(loio, Xo, yo)
    pa2 = oof(loio, Xo, yo, Xa, ya)
    flip = db[(yo == "B") & (pb != "C") & (pa2 == "C")].copy()
    print(f"  original B seeps newly called C: {len(flip)}")
    if len(flip):
        print(f"  by labeler: {flip['labeler'].value_counts().to_dict()}")
        print(f"  by chip: {flip['image'].value_counts().to_dict()}")
        print(f"\n  hull_area_m2 median -- flipped B {flip['hull_area_m2'].median():.4f}"
              f" | all original B {db.loc[yo == 'B', 'hull_area_m2'].median():.4f}"
              f" | original C {db.loc[yo == 'C', 'hull_area_m2'].median():.4f}"
              f" | hunted C {da['hull_area_m2'].median():.4f}")
        big = (flip["hull_area_m2"] > db.loc[yo == "C", "hull_area_m2"].median()).mean()
        print(f"  flipped B larger than the median original C: {big:.1%}")
        print("  -> the larger this is, the more likely these are TRUTH errors "
              "(really C)\n     rather than model errors, and the more the "
              "baseline -17.3% is an artifact.")
        out = os.path.join(args.labeling_dir, "review_B_to_C_flips.csv")
        flip[["labeler", "image", "seep_group_id", "class", "n_bubbles",
              "hull_area_m2", "major_axis_m", "minor_axis_m", "mean_R",
              "mean_G", "mean_B", "fill_ratio"]].sort_values(
            "hull_area_m2", ascending=False).to_csv(out, index=False)
        print(f"\n  [out] review list -> {out}")

    print("\n" + "=" * 74)
    print("SUMMARY (all metrics on the ORIGINAL seeps)")
    print("=" * 74)
    t = pd.DataFrame(rows)[["cv", "arm", "acc", "macro_f1", "C_prec",
                            "C_recall", "C_f1", "flux_err", "C_term"]]
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\ndeltas (+C minus baseline):")
    for cv_name in t["cv"].unique():
        s = t[t["cv"] == cv_name]
        d = s.iloc[1][["acc", "macro_f1", "C_prec", "C_recall", "C_f1",
                       "flux_err", "C_term"]] - s.iloc[0][
            ["acc", "macro_f1", "C_prec", "C_recall", "C_f1", "flux_err",
             "C_term"]]
        print(f"  {cv_name}")
        print("    " + "  ".join(f"{k}={v:+.3f}" for k, v in d.items()))


if __name__ == "__main__":
    main()
