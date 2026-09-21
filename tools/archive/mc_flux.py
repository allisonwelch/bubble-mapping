# tools/archive/mc_flux.py
"""ARCHIVED 2026-09-21. Superseded by tools/deploy/uncertainty_{labels,proba}.py.

Kept for its lognormal rate draw and its shard/summarize plumbing, both of
which were ported forward. Do not run it.

TWO REASONS IT WAS RETIRED.

1. THE ANCHOR IS NOT A CONFIDENCE INTERVAL. `summarize` took the relative
   spread from the sampled ensemble and applied it to a point estimate produced
   by the thresholded chain -- `anchored_lo = point * (p2.5 / mc_median)`. That
   is an interval for neither estimator: nothing sampled the uncertainty AROUND
   the deploy estimator, and the interval was translated off the ensemble's own
   centre. The docstring conceded it ("It does not correct the shift").

2. THREE OF THE FOUR TERMS SAMPLED A DIFFERENT ESTIMATOR, NOT A PERTURBATION OF
   THE DEPLOYED ONE. Measured on the 2026-09-15 Octopus run at 500 draws, the
   ensemble median sat at 57,806 seeps against the point estimate's 92,465
   (-37.5%), class A -42.7%, class C +92.3%.

     detector    thinning by (1 - precision) is a BIAS correction -- the count
                 is systematically high -- applied one-sided, from a fixed
                 scalar the loop never sampled. Most of the class-A deflation.
     grouper     firing each edge at its own P(same) instead of thresholding.
                 Merging is monotone under union-find, so it merges more every
                 time, by construction.
     classifier  posterior sampling reproduces the model's marginal; argmax
                 deliberately does not. It also DISCARDED `decision_rule` and
                 `overcall_penalty`, so under `conservative` the Monte Carlo
                 perturbed a classifier the runner never deployed.
     rate        correct, and carried forward unchanged.

   A third consequence: `rate_only_sd` was the median of a per-draw column
   computed on the SAMPLED counts, so the reported published-rate floor
   described an ensemble the detector artifact had shrunk rather than the
   published estimate.

The rule the replacements follow: every term is either an uncertainty or a
bias, never both. An uncertainty is centred on the deploy estimate and sets the
width. A bias moves the centre, or is stated as a bound, and never appears as
width.

Superseded, too, by the fact that `build_seeps` and `point_estimate` here were
a second and third copy of the chain that `postproc.run` also implemented. That
duplication is what let the 2026-09-15 crack-screen change reach one copy and
not the others, which surfaced as an unexplained 6% class-C gap. The chain now
lives once, in tools/deploy/chain.py.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import pandas as pd
from shapely import affinity

from tools.classify.fit_classifier import CLASSES
from tools.deploy import postproc
from tools.deploy.postproc import (DEFAULT_DECISION_RULE,
                                   DEFAULT_OVERCALL_PENALTY)
from tools.flux import rates as flux_rates
from tools.grouping.train_grouper import AGGLOM_CAP_M

TERMS = ("detector", "grouper", "classifier", "rate")
DEFAULT_FN_JITTER_M = 5.0


# --------------------------------------------------------------------------- #
# the detector term
# --------------------------------------------------------------------------- #
def resample_detections(bubbles, rng, precision=None, recall=None,
                        fn_jitter_m=DEFAULT_FN_JITTER_M):
    """Perturb the detected bubble set to its measured precision and recall.

    False positives are well defined: a fraction (1 - precision) of the
    detections are not bubbles, so drop each detection independently with that
    probability. Dropping happens BEFORE grouping, which is the point -- a
    spurious bubble is not only a bad seep of its own, it is also a pairing
    candidate that can bridge two real seeps into one.

    False negatives need an assumption and are off unless `recall` is given.
    A missed bubble has no geometry on disk, so injecting one means inventing
    one: this draws the count from NegativeBinomial(n_observed, recall) -- the
    posterior on how many were missed, given that each true bubble was detected
    independently with probability `recall` -- then copies that many detections
    and displaces each by `fn_jitter_m`. The jitter default sits well outside
    the agglomeration cap so injected bubbles mostly form their own seeps
    rather than inflating existing ones, which would bias the count the other
    way. Treat this term as a sensitivity test, not a measurement.

    Quote precision and recall from the LAKE-ICE domain of
    `bubble_level_summary.csv`, not from all eval chips. Deployment crops to
    the lake polygon, so the snow-covered shoreline chips with `n_gt = 0` are
    never seen and must not inflate the false-positive rate.
    """
    b = bubbles
    if precision is not None and precision < 1.0:
        b = b.loc[rng.random(len(b)) < precision]
    if recall is not None and recall < 1.0:
        n_obs = len(b)
        if n_obs == 0:
            return b.reset_index(drop=True)
        n_missed = int(rng.negative_binomial(n_obs, recall))
        if n_missed > 0:
            take = rng.integers(0, n_obs, n_missed)
            extra = b.iloc[take].copy()
            ang = rng.uniform(0, 2 * np.pi, n_missed)
            dx = fn_jitter_m * np.cos(ang)
            dy = fn_jitter_m * np.sin(ang)
            extra = extra.set_geometry(
                [affinity.translate(g, xoff=x, yoff=y)
                 for g, x, y in zip(extra.geometry.values, dx, dy)])
            extra["centroid_x_m"] = extra["centroid_x_m"].to_numpy(float) + dx
            extra["centroid_y_m"] = extra["centroid_y_m"].to_numpy(float) + dy
            # bubble_id keys the anchor convention and must stay unique within
            # an image, so injected bubbles take fresh ids above the maximum.
            nxt = int(b["bubble_id"].max()) + 1
            extra["bubble_id"] = np.arange(nxt, nxt + n_missed, dtype=np.int64)
            b = pd.concat([b, extra], ignore_index=True)
    return b.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# the classifier term
# --------------------------------------------------------------------------- #
def sample_classes(seeps, rng):
    """Draw each seep's class from its posterior, rather than taking argmax.

    `classify_seeps` already writes p_A / p_B / p_C, so this needs no second
    forest call. A class the forest never saw has no column and reads as zero
    probability.

    This samples the model's INTERNAL confidence, which is not the same as its
    cross-chip error -- a forest can be confidently wrong on a chip unlike any
    it trained on. The honest width for C comes from the leave-one-image-out C
    recall. Sampling a LOIO confusion matrix in place of the posterior is the
    upgrade to this function, and changes nothing else in the loop.
    """
    p = np.zeros((len(seeps), len(CLASSES)), dtype=float)
    for j, c in enumerate(CLASSES):
        col = f"p_{c}"
        if col in seeps.columns:
            p[:, j] = seeps[col].to_numpy(dtype=float)
    tot = p.sum(axis=1, keepdims=True)
    p = np.divide(p, tot, out=np.full_like(p, 1.0 / len(CLASSES)), where=tot > 0)
    # One uniform per seep against the posterior's cumulative sum: the
    # vectorised equivalent of a per-row rng.choice, and far faster at 90k rows.
    draw = (p.cumsum(axis=1) < rng.random((len(seeps), 1))).sum(axis=1)
    return np.asarray(CLASSES, dtype=object)[np.clip(draw, 0, len(CLASSES) - 1)]


# --------------------------------------------------------------------------- #
# one draw
# --------------------------------------------------------------------------- #
def _counts(labels):
    vals, cnt = np.unique(np.asarray(labels, dtype=object), return_counts=True)
    return {str(v): int(n) for v, n in zip(vals, cnt)}


def build_seeps(bubbles, grouper, classifier, rng, terms, *, thr, cap,
                precision, recall, fn_jitter_m, brightness_mode,
                brightness_cell_m, decision_rule, overcall_penalty):
    """Detections -> classified seeps, with the upstream terms sampled.

    This is the expensive half of a draw: resampling detections or the grouping
    forces a regroup and a re-dissolve. `draw_rows` handles the cheap half.
    """
    b = bubbles
    if "detector" in terms:
        b = resample_detections(b, rng, precision=precision, recall=recall,
                                fn_jitter_m=fn_jitter_m)
    b = b.reset_index(drop=True)
    b["seep_group_id"] = postproc.group_bubbles(
        grouper, b, thr=thr, cap=cap, progress=False,
        edge_rng=rng if "grouper" in terms else None)

    seeps = postproc.dissolve_bubbles_to_seeps(b, progress=False)
    if brightness_mode != "abs":
        seeps = postproc._brightness.relativize(
            seeps, b, mode=brightness_mode, cell_m=brightness_cell_m)
    seeps = postproc.classify_seeps(seeps, classifier,
                                    decision_rule=decision_rule,
                                    overcall_penalty=overcall_penalty)
    return seeps, int(len(b))


def draw_rows(seeps, n_bubbles, rng, terms, seasons, inner=1):
    """The cheap half of a draw: sample the labels and the rates.

    Yields `inner` x `len(seasons)` summary dicts. `inner > 1` buys extra
    samples of the two cheap terms almost free -- at the price that those draws
    SHARE a grouping, so the effective sample size for the detector and grouper
    terms stays at the number of outer draws.
    """
    for k in range(inner):
        labels = (sample_classes(seeps, rng) if "classifier" in terms
                  else seeps["class"].to_numpy(dtype=object))
        counts = _counts(labels)
        for season in seasons:
            total, sigma = flux_rates.lake_total(
                counts, season=season, rng=rng if "rate" in terms else None)
            yield {
                "inner": k,
                "season": season,
                "n_bubbles": n_bubbles,
                "n_seeps": int(len(seeps)),
                **{f"n_{c}": int(counts.get(c, 0)) for c in CLASSES},
                "total_mg_CH4_per_day": total,
                "rate_std_err_mg_CH4_per_day": sigma,
            }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def precision_from_summary(path):
    """Read `precision_lake_ice` from a `bubble_level_summary.csv`.

    Reading it beats typing it: the file carries both a whole-eval-set
    `precision` and the lake-ice one, and the wrong column silently widens
    every interval. Deployment crops to the lake polygon, so the three
    snow-shoreline chips with `n_gt = 0` -- where every detection is a false
    positive by construction -- never appear at deploy and must not count here.
    """
    df = pd.read_csv(path)
    col = "precision_lake_ice"
    if col not in df.columns:
        raise SystemExit(
            f"{path} has no '{col}' column, so it predates "
            "lake_ice_domain_metrics. Re-run `python -m "
            "tools.eval.bubble_level_eval` against the canonical checkpoint "
            "and chip dir to emit it.")
    return float(df[col].iloc[0])


def point_estimate(bubbles_fp, out_dir, thr=None, cap=AGGLOM_CAP_M,
                   seasons=("annual",), artifacts_dir=None,
                   decision_rule=DEFAULT_DECISION_RULE,
                   overcall_penalty=DEFAULT_OVERCALL_PENALTY):
    """The unsampled chain, written to `mc_point.json` for the anchor method.

    Every term off: the deploy threshold applies, labels come from argmax, and
    the rates are the published point values. This reproduces what
    `postproc.run` reports, and `summarize` anchors the interval on it.

    Run it through this module rather than reading the deploy run's
    `run_info_postproc.json`, so the point estimate and the draws share one
    screening pass, one artifact load and one operating point. Anchoring on a
    number produced by a different code path is how a silent mismatch gets in.
    """
    os.makedirs(out_dir, exist_ok=True)
    bubbles, upstream = postproc.load_bubbles(bubbles_fp)
    grouper, classifier, model_prov = postproc.load_models(
        artifacts_dir=artifacts_dir)
    if thr is None:
        thr = postproc._grouper_threshold(model_prov)
    brightness_mode = postproc._classifier_brightness(model_prov)

    bubbles = bubbles.reset_index(drop=True)
    bubbles, _ = postproc.screen_bubbles(bubbles, progress=False)
    bubbles = bubbles.reset_index(drop=True)

    seeps, n_b = build_seeps(
        bubbles, grouper, classifier, None, (), thr=thr, cap=cap,
        precision=None, recall=None, fn_jitter_m=DEFAULT_FN_JITTER_M,
        brightness_mode=brightness_mode,
        brightness_cell_m=(None if brightness_mode == "abs"
                           else postproc._default_brightness_cell_m(upstream)),
        decision_rule=decision_rule, overcall_penalty=overcall_penalty)

    point = {r["season"]: r
             for r in draw_rows(seeps, n_b, None, (), seasons, inner=1)}
    out = {"seasons": point, "group_threshold": thr, "agglom_cap_m": cap,
           "decision_rule": decision_rule, "brightness": brightness_mode,
           "surveyed_area_m2": upstream.get("surveyed_area_m2"),
           "bubbles": os.path.abspath(bubbles_fp)}
    fp = os.path.join(out_dir, "mc_point.json")
    with open(fp, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[mc] point estimate -> {fp}")
    for s, r in point.items():
        mix = ", ".join("{}={}".format(c, r["n_{}".format(c)]) for c in CLASSES)
        print(f"[mc]   {s}: {r['n_seeps']} seeps ({mix})")
    return out


def run(bubbles_fp, out_dir, draws=20, offset=0, base_seed=20260917,
        terms=TERMS, precision=None, recall=None,
        fn_jitter_m=DEFAULT_FN_JITTER_M, thr=None, cap=AGGLOM_CAP_M,
        seasons=("annual",), artifacts_dir=None,
        decision_rule=DEFAULT_DECISION_RULE,
        overcall_penalty=DEFAULT_OVERCALL_PENALTY, inner=1, part=None):
    """Run `draws` realizations and write one part file of summary rows."""
    terms = tuple(terms)
    unknown = set(terms) - set(TERMS)
    if unknown:
        raise ValueError(f"unknown terms {sorted(unknown)}; "
                         f"expected a subset of {list(TERMS)}")
    if "detector" in terms and precision is None and recall is None:
        raise ValueError(
            "the detector term needs --precision (and optionally --recall) "
            "from the lake-ice domain of bubble_level_summary.csv. Refusing to "
            "guess: a made-up rate would propagate into the reported interval.")

    os.makedirs(out_dir, exist_ok=True)
    bubbles, upstream = postproc.load_bubbles(bubbles_fp)
    grouper, classifier, model_prov = postproc.load_models(
        artifacts_dir=artifacts_dir)
    if thr is None:
        thr = postproc._grouper_threshold(model_prov)
    brightness_mode = postproc._classifier_brightness(model_prov)
    brightness_cell_m = (None if brightness_mode == "abs"
                         else postproc._default_brightness_cell_m(upstream))

    # Screen once, outside the loop: the crack screen is a deterministic
    # geometry filter, not a sampled term, and re-running it every draw would
    # cost time without changing anything.
    bubbles = bubbles.reset_index(drop=True)
    bubbles, _ = postproc.screen_bubbles(bubbles, progress=False)
    bubbles = bubbles.reset_index(drop=True)

    print(f"[mc] {len(bubbles)} screened bubbles, {draws} draws x {inner} inner, "
          f"terms={','.join(terms)}, thr={thr}, cap={cap}")

    build = dict(thr=thr, cap=cap, precision=precision, recall=recall,
                 fn_jitter_m=fn_jitter_m, brightness_mode=brightness_mode,
                 brightness_cell_m=brightness_cell_m,
                 decision_rule=decision_rule, overcall_penalty=overcall_penalty)

    # With neither upstream term sampled, grouping and dissolve are
    # deterministic, so they run once and every draw reuses the result. That is
    # what makes the `--terms rate` unit test cheap instead of a full rerun.
    cached = None
    if not ({"detector", "grouper"} & set(terms)):
        print("[mc] no upstream term sampled -> grouping once and reusing it")
        cached = build_seeps(bubbles, grouper, classifier,
                             np.random.default_rng(base_seed), terms, **build)

    t0 = time.time()
    rows = []
    for i in range(draws):
        seed = base_seed + offset + i
        rng = np.random.default_rng(seed)
        seeps, n_b = cached or build_seeps(bubbles, grouper, classifier, rng,
                                           terms, **build)
        for rec in draw_rows(seeps, n_b, rng, terms, seasons, inner=inner):
            rows.append({"draw": offset + i, "seed": seed, **rec})
        done = i + 1
        print(f"[mc] draw {done}/{draws}  "
              f"{(time.time() - t0) / done:.1f} s/draw", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{part:04d}" if part is not None else f"{offset:06d}"
    fp = os.path.join(out_dir, f"mc_draws_{tag}.csv")
    df.to_csv(fp, index=False)

    meta = {
        "bubbles": os.path.abspath(bubbles_fp),
        "terms": list(terms), "draws": draws, "inner": inner,
        "offset": offset, "base_seed": base_seed,
        "precision": precision, "recall": recall,
        "fn_jitter_m": fn_jitter_m if recall is not None else None,
        "group_threshold": thr, "agglom_cap_m": cap,
        "decision_rule": decision_rule,
        "brightness": brightness_mode,
        "n_bubbles_screened_in": int(len(bubbles)),
        "surveyed_area_m2": upstream.get("surveyed_area_m2"),
        "models": model_prov,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, f"mc_meta_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[mc] wrote {fp}")
    return df


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def summarize(out_dir, seasons=None):
    """Percentiles over every part file in `out_dir`, plus a convergence check.

    Reports the MEDIAN and the 2.5th / 97.5th percentiles, not mean +/- sigma:
    the distribution is skewed, so an asymmetric interval is the honest shape.

    THE ANCHOR METHOD. Sampling does not produce an ensemble centred on the
    deploy point estimate, because thresholding and argmax are not the same
    question as sampling a posterior: the threshold discards every low-P(same)
    edge while sampling fires each at its own rate, and argmax under-calls a
    rare class while sampling reproduces its marginal. So with `mc_point.json`
    present, this reports the interval as RATIOS to the MC median, applied to
    the point estimate:

        anchored_lo = point_total * (p2.5  / mc_median)
        anchored_hi = point_total * (p97.5 / mc_median)

    Ratios rather than differences because the total is positive and
    right-skewed, so the spread scales with the level.

    This keeps the published centre reproducible by anyone who runs the deploy
    path, and takes only the WIDTH from the ensemble. It does not correct the
    shift -- so `median_shift_pct` and the per-class counts are reported beside
    it, loudly. A large shift in n_C is not a cosmetic difference: C carries
    most of the methane, and a shift there means the point estimate's class mix
    is the thing to investigate, not the error bar.
    """
    parts = sorted(glob.glob(os.path.join(out_dir, "mc_draws_*.csv")))
    if not parts:
        raise SystemExit(f"no mc_draws_*.csv in {out_dir}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if seasons:
        df = df[df["season"].isin(seasons)]

    point_fp = os.path.join(out_dir, "mc_point.json")
    point = {}
    if os.path.exists(point_fp):
        with open(point_fp) as fh:
            point = json.load(fh).get("seasons", {})
    else:
        print(f"[mc] WARNING: no {point_fp}; reporting the UNANCHORED ensemble. "
              "Run --point to anchor the interval on the deploy estimate.")

    rows = []
    for season, sub in df.groupby("season"):
        t = sub["total_mg_CH4_per_day"].to_numpy(float)
        lo, med, hi = np.percentile(t, [2.5, 50, 97.5])
        row = {
            "season": season,
            "n_realizations": len(t),
            "median_mg_CH4_per_day": med,
            "p2.5_mg_CH4_per_day": lo,
            "p97.5_mg_CH4_per_day": hi,
            "interval_width_pct": 100 * (hi - lo) / med if med else np.nan,
            "sampled_sd_mg_CH4_per_day": float(t.std(ddof=1)),
            # The closed-form published-rate floor. With --terms rate alone,
            # sampled_sd converges to this; that equality is the loop's unit
            # test. With every term on, sampled_sd must exceed it.
            "rate_only_sd_mg_CH4_per_day":
                float(sub["rate_std_err_mg_CH4_per_day"].median()),
            **{f"median_n_{c}": float(sub[f"n_{c}"].median()) for c in CLASSES},
            "median_n_seeps": float(sub["n_seeps"].median()),
        }

        p = point.get(season)
        if p and med:
            pt = float(p["total_mg_CH4_per_day"])
            row.update({
                "point_total_mg_CH4_per_day": pt,
                "anchored_lo_mg_CH4_per_day": pt * (lo / med),
                "anchored_hi_mg_CH4_per_day": pt * (hi / med),
                "rel_lo_pct": 100 * (lo / med - 1.0),
                "rel_hi_pct": 100 * (hi / med - 1.0),
                "median_shift_pct": 100 * (med / pt - 1.0),
                "point_n_seeps": int(p["n_seeps"]),
                **{f"point_n_{c}": int(p[f"n_{c}"]) for c in CLASSES},
                **{f"shift_n_{c}_pct":
                   (100 * (float(sub[f"n_{c}"].median()) / p[f"n_{c}"] - 1.0)
                    if p[f"n_{c}"] else np.nan) for c in CLASSES},
            })
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Interval bounds against sample size: the check for whether N was enough.
    # The grid adapts to what is actually there, so a short pilot run still
    # reports something instead of an empty table.
    conv = []
    for season, sub in df.groupby("season"):
        t = sub["total_mg_CH4_per_day"].to_numpy(float)
        grid = sorted({n for n in (10, 25, 50, 100, 200, 300, 400, 500)
                       if n <= len(t)} | {len(t)})
        for n in grid:
            lo, hi = np.percentile(t[:n], [2.5, 97.5])
            conv.append({"season": season, "n": n, "p2.5": lo, "p97.5": hi,
                         "width_pct": 100 * (hi - lo) / np.median(t[:n])})
    conv = pd.DataFrame(conv)

    summary.to_csv(os.path.join(out_dir, "mc_summary.csv"), index=False)
    conv.to_csv(os.path.join(out_dir, "mc_convergence.csv"), index=False)

    anchored = "point_total_mg_CH4_per_day" in summary.columns
    print("\n" + "=" * 72)
    print("MONTE CARLO LAKE FLUX" + ("  (ANCHORED)" if anchored else ""))
    print("=" * 72)

    if anchored:
        head = ["season", "n_realizations", "point_total_mg_CH4_per_day",
                "anchored_lo_mg_CH4_per_day", "anchored_hi_mg_CH4_per_day",
                "rel_lo_pct", "rel_hi_pct"]
        print(summary[head].to_string(index=False,
                                      float_format=lambda v: f"{v:,.2f}"))
        for _, r in summary.iterrows():
            print(f"\n  {r['season']}: "
                  f"{r['point_total_mg_CH4_per_day']:,.0f} mg CH4/day "
                  f"({r['anchored_lo_mg_CH4_per_day']:,.0f} - "
                  f"{r['anchored_hi_mg_CH4_per_day']:,.0f}, 95%), "
                  f"{int(r['n_realizations'])} realizations")
        print("\n" + "-" * 72)
        print("SHIFT BETWEEN THE SAMPLED ENSEMBLE AND THE POINT ESTIMATE")
        print("-" * 72)
        shift = ["season", "median_shift_pct", "point_n_seeps",
                 "median_n_seeps"] + \
                [c for p in CLASSES for c in (f"point_n_{p}", f"median_n_{p}",
                                              f"shift_n_{p}_pct")]
        print(summary[shift].to_string(index=False,
                                       float_format=lambda v: f"{v:,.1f}"))
        print("\nThe anchor takes the WIDTH from the ensemble and the CENTRE "
              "from the deploy\nrun. It does not correct the shift above. A "
              "large shift in n_C is a finding,\nnot a rounding difference: C "
              "carries most of the methane, so it says the point\nestimate's "
              "class mix needs arbitrating against the LOIO confusion matrix.")
    else:
        head = ["season", "n_realizations", "median_mg_CH4_per_day",
                "p2.5_mg_CH4_per_day", "p97.5_mg_CH4_per_day",
                "interval_width_pct", "sampled_sd_mg_CH4_per_day",
                "rate_only_sd_mg_CH4_per_day"]
        print(summary[head].to_string(index=False,
                                      float_format=lambda v: f"{v:,.2f}"))

    print("\nconvergence of the interval bounds with sample size:")
    print(conv.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print("\nName every term that was sampled, and state what the interval "
          "excludes:\nsystematic detector bias, the transect width, and "
          "whether the published rates\ntransfer to this lake.")
    return summary, conv


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", metavar="DIR", default=None,
                    help="aggregate an existing run directory and exit")
    ap.add_argument("--point", action="store_true",
                    help="run the chain once with every term OFF and write "
                         "mc_point.json, which --summarize anchors on")
    ap.add_argument("--bubbles", help="bubbles.gpkg from tools.deploy.detect")
    ap.add_argument("--out-dir")
    ap.add_argument("--draws", type=int, default=20,
                    help="outer draws in THIS task (default %(default)s)")
    ap.add_argument("--inner", type=int, default=1,
                    help="classifier+rate draws per grouping (default "
                         "%(default)s); >1 shares a grouping between them")
    ap.add_argument("--offset", type=int, default=0,
                    help="draw index of the first draw, so array tasks never "
                         "share a seed")
    ap.add_argument("--part", type=int, default=None,
                    help="part number for the output filename")
    ap.add_argument("--base-seed", type=int, default=20260917)
    ap.add_argument("--terms", default=",".join(TERMS),
                    help="comma list from detector,grouper,classifier,rate. "
                         "One term at a time gives the variance decomposition")
    ap.add_argument("--precision", type=float, default=None,
                    help="measured bubble precision, LAKE-ICE domain")
    ap.add_argument("--precision-from", default=None, metavar="CSV",
                    help="read precision_lake_ice from a "
                         "bubble_level_summary.csv instead of typing it")
    ap.add_argument("--recall", type=float, default=None,
                    help="measured bubble recall, LAKE-ICE domain. Omit to "
                         "leave false negatives out; injecting them invents "
                         "geometry")
    ap.add_argument("--fn-jitter-m", type=float, default=DEFAULT_FN_JITTER_M)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--cap", type=float, default=AGGLOM_CAP_M)
    ap.add_argument("--season", default="annual",
                    help="comma list, or 'all'")
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--decision-rule", default=DEFAULT_DECISION_RULE,
                    choices=list(postproc.DECISION_RULES))
    ap.add_argument("--overcall-penalty", type=float,
                    default=DEFAULT_OVERCALL_PENALTY)
    args = ap.parse_args(argv)

    if args.summarize:
        summarize(args.summarize)
        return
    if not args.bubbles or not args.out_dir:
        ap.error("--bubbles and --out-dir are required unless --summarize")

    seasons = (tuple(flux_rates.SEASONS) if args.season == "all"
               else tuple(s.strip() for s in args.season.split(",") if s.strip()))

    precision = args.precision
    if precision is None and args.precision_from:
        precision = precision_from_summary(args.precision_from)
        print(f"[mc] precision_lake_ice = {precision:.4f} "
              f"(from {args.precision_from})")

    if args.point:
        point_estimate(args.bubbles, args.out_dir, thr=args.thr, cap=args.cap,
                       seasons=seasons, artifacts_dir=args.artifacts_dir,
                       decision_rule=args.decision_rule,
                       overcall_penalty=args.overcall_penalty)
        return

    run(args.bubbles, args.out_dir, draws=args.draws, offset=args.offset,
        base_seed=args.base_seed,
        terms=tuple(t.strip() for t in args.terms.split(",") if t.strip()),
        precision=precision, recall=args.recall,
        fn_jitter_m=args.fn_jitter_m, thr=args.thr, cap=args.cap,
        seasons=seasons, artifacts_dir=args.artifacts_dir,
        decision_rule=args.decision_rule,
        overcall_penalty=args.overcall_penalty, inner=args.inner,
        part=args.part)


if __name__ == "__main__":
    main()
