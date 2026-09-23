# tools/deploy/uncertainty_proba.py
"""METHOD 2 -- forward uncertainty for the posterior-based lake total.

Every seep still carries the class the forest ranks highest which is recorded in
`seeps.gpkg`. The headline total is computed a different way, from the class probabilities:

    N_c = sum over seeps of p_c          not  count of seeps labelled c

A seep the forest calls 30% likely to be C contributes 0.3 to N_C and 0 to the
label count. Over many seeps those fractions add up to the rare-class seeps a
hard label discards. On the 2026-09-15 Octopus run the two differ by far more
than rounding, and mostly in C, which carries the most methane per seep.

A map made by taking the most likely class at each location
systematically loses rare classes, so the count read off that map is a biased
estimate of how many there are: Olofsson et al. 2014, Remote
Sensing of Environment 148, 42-57).

WHAT IS SAMPLED, AND WHY EACH ONE

    classifier   Each seep's class is drawn from its own posterior.

                 This term also has a CLOSED FORM, so no simulation is strictly
                 needed for it. The count of class c is a sum of independent
                 yes/no events with different probabilities -- a
                 Poisson-binomial -- so

                     N_c  = sum p_c            sd(N_c) = sqrt(sum p_c(1-p_c))

                 The simulation reproduces those two numbers, and `summarize`
                 checks that it does. If it ever fails, the loop is wrong.

    rate         One draw per CLASS per realization, from a lognormal matched to that
                 mean and standard error. The standard error belongs to the
                 published class mean, so it is shared by every seep of that
                 class and does not average away as more seeps are mapped. This
                 is the only term that genuinely needs simulation, because it
                 is skewed and the useful output is percentiles.

WHAT IS NOT SAMPLED

    grouper      Not an uncertainty here. The P(same) threshold is a CHOICE,
                 made on labeled data because it reproduces hand-delineated
                 seep counts, and the number of seeps is not a sum of
                 independent per-pair events -- it comes out of union-find under
                 a diameter cap, where one edge can merge two large clusters.
                 There is no closed form, and sampling the edges produces a
                 grouping validated against nothing.

                 So `--threshold-sweep` re-runs the whole chain at several
                 thresholds and reports the spread as its own line, beside the
                 interval. The sweep ALWAYS includes the
                 artifact's own operating point, flagged in the output, so the
                 row the interval is centred on is never ambiguous.

    detector     Bias, not width. Report precision and recall as stated
                 bounds beside the interval.

**The closed form assumes seeps are independent of one another. They are not entirely: a miscalibrated
forest is wrong on many seeps the same way, and that correlated error does not
average down. The classifier term here is treated as a floor. The correlated part
is measured by the held-out-chip confusion matrix, treated as a bias.

    python -m tools.deploy.uncertainty_proba --bubbles RUN/bubbles.gpkg \
        --out-dir RUN/uncertainty_proba --draws 500 \
        --threshold-sweep 0.5,0.6,0.7
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from tools.classify.fit_classifier import CLASSES
from tools.deploy import postproc
from tools.deploy.chain import (DEFAULT_DECISION_RULE, DEFAULT_OVERCALL_PENALTY,
                                posterior_matrix, resolve_params, run_chain,
                                screen_bubbles)
from tools.flux import rates as flux_rates
from tools.grouping.train_grouper import AGGLOM_CAP_M

DEFAULT_BASE_SEED = 20260921

# See uncertainty_labels for the reasoning. Method 2's terms are centred
# analytically rather than by construction, so this is tighter: a failure here
# means the sampler and the closed form disagree, which is a bug.
MAX_CENTRE_SHIFT_PCT = 2.0

# How far the simulated mean class counts may sit from the closed form before
# the loop is treated as broken. Monte Carlo noise on a mean of N draws scales
# as 1/sqrt(N), so at 500 draws this is many standard errors wide.
MAX_COUNT_DRIFT_PCT = 2.0


# --------------------------------------------------------------------------- #
# fractional counts -> flux
# --------------------------------------------------------------------------- #
def total_from_counts(counts, season="annual"):
    """Count-based total for POSSIBLY FRACTIONAL class counts.

    `flux_rates.lake_total` casts its counts to int, which is right for a set
    of labelled seeps and wrong for a posterior sum -- truncating 2,830.4 C
    seeps to 2,830 is harmless, but truncating every class silently loses
    methane. This keeps the fractions.

    Returns (total, rate_sigma), both mg CH4/day. `rate_sigma` is the published
    floor: the standard error on each class MEAN times that class's count, in
    quadrature. It is fully correlated across seeps, so it does not shrink as
    more seeps are mapped.
    """
    rates, std_errs = flux_rates.season_rates(season)
    total = float(sum(rates[c] * float(counts.get(c, 0.0)) for c in CLASSES))
    sigma = float(np.sqrt(sum((std_errs[c] * float(counts.get(c, 0.0))) ** 2
                              for c in CLASSES)))
    return total, sigma


def classifier_sigma(count_sd, season="annual"):
    """The classifier's own contribution to the total, in closed form.

    Each class count carries an independent standard deviation from the
    Poisson-binomial, and the rates convert those counts to methane, so the
    contributions add in quadrature as (rate_c * sd_c).
    """
    rates, _ = flux_rates.season_rates(season)
    return float(np.sqrt(sum((rates[c] * float(count_sd.get(c, 0.0))) ** 2
                             for c in CLASSES)))


def sample_classes(p, rng):
    """Draw one class per seep from its posterior. Vectorised.

    One uniform per seep against the posterior's cumulative sum: the equivalent
    of a per-row `rng.choice`, and far faster at 90k rows.
    """
    draw = (p.cumsum(axis=1) < rng.random((len(p), 1))).sum(axis=1)
    idx = np.clip(draw, 0, len(CLASSES) - 1)
    vals, cnt = np.unique(idx, return_counts=True)
    counts = {c: 0 for c in CLASSES}
    for v, n in zip(vals, cnt):
        counts[CLASSES[int(v)]] = int(n)
    return counts


# --------------------------------------------------------------------------- #
# the chain, once
# --------------------------------------------------------------------------- #
def _load(bubbles_fp, artifacts_dir, thr, cap, decision_rule, overcall_penalty):
    bubbles, upstream = postproc.load_bubbles(bubbles_fp)
    grouper, classifier, model_prov = postproc.load_models(
        artifacts_dir=artifacts_dir)
    params = resolve_params(model_prov, upstream, thr=thr, cap=cap,
                            decision_rule=decision_rule,
                            overcall_penalty=overcall_penalty)
    print(f"[unc] P(same) threshold {params.thr} ({params.thr_source}), "
          f"cap {params.cap} m, brightness {params.brightness_mode}")
    return bubbles, upstream, grouper, classifier, model_prov, params


def _point_row(res, season):
    """One season's point estimate, both ways of counting."""
    hard = res.class_counts()
    soft = res.expected_class_counts()
    sd = res.expected_count_sd()
    total, rate_sigma = total_from_counts(soft, season=season)
    hard_total, _ = total_from_counts(hard, season=season)
    cls_sigma = classifier_sigma(sd, season=season)
    return {
        "season": season,
        "n_seeps": int(len(res.seeps)),
        **{f"n_{c}_argmax": int(hard.get(c, 0)) for c in CLASSES},
        **{f"n_{c}": float(soft.get(c, 0.0)) for c in CLASSES},
        **{f"sd_n_{c}": float(sd.get(c, 0.0)) for c in CLASSES},
        "total_mg_CH4_per_day": total,
        "total_argmax_mg_CH4_per_day": hard_total,
        "rate_sigma_mg_CH4_per_day": rate_sigma,
        "classifier_sigma_mg_CH4_per_day": cls_sigma,
        "combined_sigma_mg_CH4_per_day":
            float(np.hypot(rate_sigma, cls_sigma)),
    }


# --------------------------------------------------------------------------- #
# the threshold sweep
# --------------------------------------------------------------------------- #
def threshold_sweep(kept, dropped, grouper, classifier, params, thresholds,
                    seasons, surveyed_area_m2=None):
    """Re-run the whole chain at each threshold. Returns a long DataFrame.

    The artifact's own operating point is always included and flagged, so the
    row the reported interval is centred on is never in doubt. If the artifact
    is later rebuilt at a different threshold, the sweep follows it
    automatically instead of reporting a spread around a value no longer in use.

    This is a SENSITIVITY measurement, not a term in the error bar, and not a
    knob to turn. Report the spread; leave the operating point where it was
    chosen on labeled data.
    """
    wanted = sorted({round(float(t), 6) for t in thresholds}
                    | {round(float(params.thr), 6)})
    rows = []
    for thr in wanted:
        is_deploy = abs(thr - params.thr) < 1e-9
        print(f"[sweep] threshold {thr}" + ("  <- deploy" if is_deploy else ""))
        res = run_chain(kept, grouper, classifier, params.replace(thr=thr),
                        progress=False, screened=(kept, dropped))
        soft = res.expected_class_counts()
        hard = res.class_counts()
        for season in seasons:
            total, _ = total_from_counts(soft, season=season)
            rows.append({
                "group_threshold": thr,
                "is_deploy_operating_point": is_deploy,
                "season": season,
                "n_seeps": int(len(res.seeps)),
                **{f"n_{c}_argmax": int(hard.get(c, 0)) for c in CLASSES},
                **{f"n_{c}": float(soft.get(c, 0.0)) for c in CLASSES},
                "total_mg_CH4_per_day": total,
            })
    return flux_rates.add_per_area_columns(pd.DataFrame(rows),
                                           surveyed_area_m2)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(bubbles_fp, out_dir, *, draws=500, base_seed=DEFAULT_BASE_SEED,
        seasons=("annual",), artifacts_dir=None, thr=None, cap=AGGLOM_CAP_M,
        decision_rule=DEFAULT_DECISION_RULE,
        overcall_penalty=DEFAULT_OVERCALL_PENALTY, sweep=()):
    """Point estimate, draws, optional sweep, and the summary -- one call.

    The chain runs ONCE. Every draw reuses its seeps, because neither sampled
    term touches the grouping: only the per-seep class and the three rates
    move. So the whole of this costs about one `postproc` run plus seconds,
    which is why it needs no GPU and barely needs a compute node.
    """
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    bubbles, upstream, grouper, classifier, model_prov, params = _load(
        bubbles_fp, artifacts_dir, thr, cap, decision_rule, overcall_penalty)
    surveyed_area_m2 = upstream.get("surveyed_area_m2")

    kept, dropped = screen_bubbles(
        bubbles.reset_index(drop=True), max_span_m=params.max_span_m,
        min_thinness=params.min_thinness, progress=True)
    kept = kept.reset_index(drop=True)

    res = run_chain(kept, grouper, classifier, params, progress=True,
                    screened=(kept, dropped))

    point = {s: _point_row(res, s) for s in seasons}
    out = {
        "method": "proba",
        "seasons": point,
        "params": params.as_dict(),
        "n_bubbles_in": res.n_bubbles_in,
        "n_bubbles_screened": res.n_screened,
        "surveyed_area_m2": surveyed_area_m2,
        "bubbles": os.path.abspath(bubbles_fp),
        "models": model_prov,
        "note": "n_<c> are posterior sums and drive the total; n_<c>_argmax "
                "are the hard labels written to seeps.gpkg for mapping. They "
                "differ on purpose -- see the module docstring.",
    }
    with open(os.path.join(out_dir, "point.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    print("\n[unc] point estimate, both ways of counting:")
    for s, r in point.items():
        mix_h = ", ".join(f"{c}={r[f'n_{c}_argmax']}" for c in CLASSES)
        mix_s = ", ".join(f"{c}={r[f'n_{c}']:,.0f}" for c in CLASSES)
        print(f"  {s}: {r['n_seeps']} seeps")
        print(f"    argmax (map)   {mix_h}  -> "
              f"{r['total_argmax_mg_CH4_per_day']:,.0f} mg CH4/day")
        print(f"    posterior sum  {mix_s}  -> "
              f"{r['total_mg_CH4_per_day']:,.0f} mg CH4/day  <- headline")
        print(f"    closed-form sd: rates {r['rate_sigma_mg_CH4_per_day']:,.0f}"
              f", classifier {r['classifier_sigma_mg_CH4_per_day']:,.0f}"
              f", combined {r['combined_sigma_mg_CH4_per_day']:,.0f}")

    # ---------------- the draws ---------------- #
    p = posterior_matrix(res.seeps)
    rows = []
    for i in range(draws):
        rng = np.random.default_rng(base_seed + i)
        counts = sample_classes(p, rng)
        for season in seasons:
            total, sigma = flux_rates.lake_total(counts, season=season, rng=rng)
            rows.append({
                "draw": i, "seed": base_seed + i, "season": season,
                "n_seeps": int(len(res.seeps)),
                **{f"n_{c}": int(counts.get(c, 0)) for c in CLASSES},
                "total_mg_CH4_per_day": total,
                "rate_std_err_mg_CH4_per_day": sigma,
            })
    # Each draw carries its own density and annual mass, so a histogram of the
    # draws reads in whichever unit the figure needs.
    draws_df = flux_rates.add_per_area_columns(pd.DataFrame(rows),
                                               surveyed_area_m2)
    draws_df.to_csv(os.path.join(out_dir, "draws.csv"), index=False)

    # ---------------- the sweep ---------------- #
    sweep_df = None
    if sweep:
        sweep_df = threshold_sweep(kept, dropped, grouper, classifier, params,
                                   sweep, seasons,
                                   surveyed_area_m2=surveyed_area_m2)
        sweep_df.to_csv(os.path.join(out_dir, "threshold_sweep.csv"),
                        index=False)

    summary = summarize(out_dir, point=point, draws_df=draws_df,
                        sweep_df=sweep_df, surveyed_area_m2=surveyed_area_m2)
    print(f"\n[unc] wrote point.json, draws.csv, summary.csv"
          f"{', threshold_sweep.csv' if sweep else ''} -> {out_dir}")
    print(f"[unc] {round(time.time() - t0, 1)} s")
    return summary


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def summarize(out_dir, point=None, draws_df=None, sweep_df=None,
              surveyed_area_m2=None):
    """Percentiles, the closed-form cross-check, and the sweep line.

    Reads from disk when called on its own, so a finished directory can be
    re-summarised without re-running the chain.
    """
    if point is None:
        with open(os.path.join(out_dir, "point.json")) as fh:
            blob = json.load(fh)
        point = blob.get("seasons", {})
        surveyed_area_m2 = surveyed_area_m2 or blob.get("surveyed_area_m2")
    if draws_df is None:
        draws_df = pd.read_csv(os.path.join(out_dir, "draws.csv"))
    if sweep_df is None:
        fp = os.path.join(out_dir, "threshold_sweep.csv")
        sweep_df = pd.read_csv(fp) if os.path.exists(fp) else None

    rows = []
    for season, sub in draws_df.groupby("season"):
        t = sub["total_mg_CH4_per_day"].to_numpy(float)
        lo, med, hi = np.percentile(t, [2.5, 50, 97.5])
        p = point.get(season, {})
        pt = float(p.get("total_mg_CH4_per_day", np.nan))
        row = {
            "season": season,
            "n_realizations": len(t),
            "point_total_mg_CH4_per_day": pt,
            "median_mg_CH4_per_day": med,
            "p2.5_mg_CH4_per_day": lo,
            "p97.5_mg_CH4_per_day": hi,
            "interval_width_pct": 100 * (hi - lo) / med if med else np.nan,
            "mean_mg_CH4_per_day": float(t.mean()),
            "centre_shift_pct": 100 * (med / pt - 1.0) if pt else np.nan,
            "sd_mg_CH4_per_day": float(t.std(ddof=1)),
            # These three carry the unit suffix so they pick up the same
            # per-area twins as the percentiles they are compared against.
            "closed_form_rate_sigma_mg_CH4_per_day":
                float(p.get("rate_sigma_mg_CH4_per_day", np.nan)),
            "closed_form_classifier_sigma_mg_CH4_per_day":
                float(p.get("classifier_sigma_mg_CH4_per_day", np.nan)),
            "closed_form_combined_sigma_mg_CH4_per_day":
                float(p.get("combined_sigma_mg_CH4_per_day", np.nan)),
            "total_argmax_mg_CH4_per_day":
                float(p.get("total_argmax_mg_CH4_per_day", np.nan)),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    # The point estimate, both percentiles, the sampled sd and all three
    # closed-form sigmas, each over the surveyed area and as an annual mass.
    summary = flux_rates.add_per_area_columns(summary, surveyed_area_m2)
    if surveyed_area_m2:
        summary["surveyed_area_m2"] = surveyed_area_m2
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\n" + "=" * 72)
    print("METHOD 2 -- LAKE FLUX, POSTERIOR-BASED COUNTS")
    print("=" * 72)
    head = ["season", "n_realizations", "point_total_mg_CH4_per_day",
            "p2.5_mg_CH4_per_day", "p97.5_mg_CH4_per_day",
            "interval_width_pct"]
    print(summary[head].to_string(index=False,
                                  float_format=lambda v: f"{v:,.2f}"))

    area_cols = [c for c in summary.columns
                 if c.startswith(("point_total_", "median_", "p2.5_", "p97.5_"))
                 and c.endswith(("_mg_CH4_per_m2_per_day",
                                 "_g_CH4_per_m2_per_year"))]
    if area_cols:
        print(f"\nover {surveyed_area_m2:,.0f} m2 of surveyed lake:")
        print(summary[["season"] + area_cols].to_string(
            index=False, float_format=lambda v: f"{v:,.4f}"))
        print("the per-year columns are blank for summer and winter: those "
              "rates cover a regime\nof unrecorded length, so neither "
              "integrates to a year.")

    print("\nwhere the spread comes from (closed form, at the point counts):")
    for _, r in summary.iterrows():
        tot = r["closed_form_combined_sigma_mg_CH4_per_day"]
        rate_sigma = r["closed_form_rate_sigma_mg_CH4_per_day"]
        share = 100 * (rate_sigma / tot) ** 2 if tot else np.nan
        print(f"  {r['season']}: rates {rate_sigma:,.0f}, classifier "
              f"{r['closed_form_classifier_sigma_mg_CH4_per_day']:,.0f}, "
              f"combined {tot:,.0f}  ({share:.1f}% of the variance is rates)")

    _check_counts(point, draws_df)
    _centring_check(summary)

    if sweep_df is not None and len(sweep_df):
        print("\n" + "-" * 72)
        print("GROUPING THRESHOLD SENSITIVITY  (reported beside the interval, "
              "never inside it)")
        print("-" * 72)
        cols = [c for c in ["group_threshold", "is_deploy_operating_point",
                            "season", "n_seeps", "total_mg_CH4_per_day",
                            "total_mg_CH4_per_m2_per_day",
                            "total_g_CH4_per_m2_per_year"]
                if c in sweep_df.columns]
        # Densities run to a few hundredths, totals to five figures, and one
        # format cannot show both -- so the width follows the magnitude.
        print(sweep_df[cols].to_string(
            index=False,
            float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"))
        for season, sub in sweep_df.groupby("season"):
            dep = sub[sub["is_deploy_operating_point"]]
            if not len(dep):
                continue
            base = float(dep["total_mg_CH4_per_day"].iloc[0])
            rel = 100 * (sub["total_mg_CH4_per_day"] / base - 1.0)
            thrs = ", ".join(f"{v:g}"
                             for v in sorted(sub["group_threshold"].unique()))
            print(f"  {season}: moving the threshold over [{thrs}] moves the "
                  f"total {rel.min():+.1f}% to {rel.max():+.1f}%, and the seep "
                  f"count {sub['n_seeps'].min():,} to {sub['n_seeps'].max():,}")

    print("\nName what the interval EXCLUDES: detector false positives (the "
          "count is biased\nhigh), the correlated part of classifier error, the "
          "2014 transect width, and\nwhether the published rates transfer to "
          "this lake. Report those as stated bounds.")
    return summary


def _check_counts(point, draws_df):
    """The simulation must reproduce the closed form. Otherwise the loop is wrong."""
    bad = []
    for season, sub in draws_df.groupby("season"):
        p = point.get(season, {})
        for c in CLASSES:
            want = float(p.get(f"n_{c}", np.nan))
            got = float(sub[f"n_{c}"].mean())
            if not np.isfinite(want) or want <= 0:
                continue
            drift = 100 * (got / want - 1.0)
            if abs(drift) > MAX_COUNT_DRIFT_PCT:
                bad.append((season, c, want, got, drift))
    if not bad:
        print(f"\n[unc] closed-form check PASSED: simulated mean class counts "
              f"match sum(p_c) within {MAX_COUNT_DRIFT_PCT}%.")
        return
    for season, c, want, got, drift in bad:
        print(f"[unc] {season} class {c}: sum(p_c) = {want:,.1f} but the draws "
              f"average {got:,.1f} ({drift:+.1f}%)")
    raise SystemExit(
        "The sampler and the closed form disagree, so one of them is wrong. "
        "They are two computations of the same quantity -- the expected class "
        "count -- and the closed form has no Monte Carlo noise, so suspect the "
        "sampling loop first.")


def _centring_check(summary):
    """Report where the ensemble sits relative to the point estimate.

    The hard test for this method is `_check_counts`, which compares the
    simulated class counts against the closed form. That is the quantity the
    classifier term acts on, it is near-normal, and it converges in a few
    draws.

    The TOTAL is deliberately not tested the same way. It additionally carries
    the rate draw, which is mean-preserving but strongly right-skewed -- class
    A's standard error is 63% of its own mean, so its lognormal median sits 15%
    below its mean. Testing the total's median would flag a correct sampler,
    and testing the total's mean converges slowly because a short run can miss
    the upper tail entirely. So this prints the comparison and names the cause
    instead of raising on it.
    """
    for _, r in summary.iterrows():
        pt = float(r["point_total_mg_CH4_per_day"])
        if pt <= 0:
            continue
        se_pct = 100 * float(r["sd_mg_CH4_per_day"]) / np.sqrt(
            r["n_realizations"]) / pt
        mean_shift = 100 * (float(r["mean_mg_CH4_per_day"]) / pt - 1.0)
        print(f"[unc] {r['season']}: ensemble mean is {mean_shift:+.1f}% from "
              f"the point estimate (Monte Carlo error {se_pct:.1f}% at "
              f"{int(r['n_realizations'])} draws); the median is "
              f"{r['centre_shift_pct']:+.1f}% from it, which is the "
              "right-skewed class-A rate, not a bias.")
        if abs(mean_shift) > MAX_CENTRE_SHIFT_PCT and abs(mean_shift) > 3 * se_pct:
            print(f"[unc] WARNING: that mean shift is {abs(mean_shift) / se_pct:.1f} "
                  "Monte Carlo standard errors. Check the closed-form result "
                  "above before reporting this interval.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", metavar="DIR", default=None,
                    help="re-summarise a finished run directory and exit")
    ap.add_argument("--bubbles", help="bubbles.gpkg from tools.deploy.detect")
    ap.add_argument("--out-dir")
    ap.add_argument("--draws", type=int, default=500,
                    help="realizations (default %(default)s). Cheap here -- "
                         "nothing is re-grouped")
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                    help="draw i is seeded base_seed + i (default %(default)s)")
    ap.add_argument("--threshold-sweep", default=None, metavar="LIST",
                    help="comma list of grouper thresholds to re-run the chain "
                         "at, e.g. 0.5,0.6,0.7. The artifact's own operating "
                         "point is always added and flagged in the output")
    ap.add_argument("--season", default="annual",
                    help="comma list, or 'all'. Seasons are alternative views "
                         "of the same seeps and are NOT additive")
    ap.add_argument("--thr", type=float, default=None,
                    help="grouper P(same) operating point for the point "
                         "estimate. Default is the value versioned with the "
                         "artifact")
    ap.add_argument("--cap", type=float, default=AGGLOM_CAP_M)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--decision-rule", default=DEFAULT_DECISION_RULE,
                    choices=list(postproc.DECISION_RULES),
                    help="only affects the argmax labels written for the map; "
                         "the headline total comes from the posterior either way")
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
    sweep = tuple(float(v) for v in args.threshold_sweep.split(",")
                  if v.strip()) if args.threshold_sweep else ()

    run(args.bubbles, args.out_dir, draws=args.draws, base_seed=args.base_seed,
        seasons=seasons, artifacts_dir=args.artifacts_dir, thr=args.thr,
        cap=args.cap, decision_rule=args.decision_rule,
        overcall_penalty=args.overcall_penalty, sweep=sweep)


if __name__ == "__main__":
    main()
