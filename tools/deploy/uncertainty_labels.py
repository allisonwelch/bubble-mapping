# tools/deploy/uncertainty_labels.py
"""METHOD 1 -- forward uncertainty for the label-based lake total.

Every seep carries the class the forest
ranks highest, the classes are counted, and each count is multiplied by its
published per-seep rate. This module puts an error bar around that number
without changing it.

    point estimate   chain.run_chain with nothing sampled
    each draw        the two forests' TREES resampled, + the three rates drawn
    the interval     percentiles of the draws

    grouper      Trees are resampled with replacement and re-averaged. The P(same)
                 THRESHOLD is held at the deploy operating point, because that
                 threshold was chosen on labeled data to reproduce
                 hand-delineated seep counts. Firing each edge at its own
                 probability instead has been validated against nothing, and it
                 merges systematically more -- a different estimator, not a
                 spread around this one.

    classifier   The decision rule (argmax, or
                 conservative) is held, because sampling the posterior
                 reproduces the model's marginal and argmax deliberately does
                 not.

    rate         One draw per CLASS per realization, from a lognormal matched to that
                 mean and standard error. One draw for all class-A seeps rather
                 than one each: the standard error belongs to the published
                 class mean, so it is shared by every seep of that class and
                 does not average away as more seeps are mapped.

WHAT IS NOT SAMPLED

    detector     Its dominant component is BIAS -- at a precision below 1 the
                 count is systematically high -- and a bias moves the center or
                 is stated as a bound. It never belongs in a width. Report
                 precision and recall beside the interval instead.

    crack screen Deterministic geometry with field-anchored thresholds. It runs
                 once, outside the loop, and every draw shares the result.

    grouping     Not an uncertainty. Use `uncertainty_proba --threshold-sweep`
    threshold    to measure it and report the spread as its own line.

KNOWN LIMIT, state it rather than fix it. Resampling a forest's trees is a
bootstrap over ENSEMBLE MEMBERS, not over data. Every tree saw the same labeler
packs, so this captures the forests' internal averaging noise and NOT how much
the answer would move if different chips had been labeled.

    python -m tools.deploy.uncertainty_labels --bubbles RUN/bubbles.gpkg \
        --out-dir RUN/uncertainty_labels --point

    python -m tools.deploy.uncertainty_labels --bubbles RUN/bubbles.gpkg \
        --out-dir RUN/uncertainty_labels --draws 500 --shard 0/8

    python -m tools.deploy.uncertainty_labels --summarize RUN/uncertainty_labels
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import pandas as pd

from tools.classify.fit_classifier import CLASSES
from tools.deploy import postproc
from tools.deploy.chain import (DEFAULT_DECISION_RULE, DEFAULT_OVERCALL_PENALTY,
                                WOBBLE_TERMS, resolve_params, run_chain,
                                screen_bubbles)
from tools.flux import rates as flux_rates
from tools.grouping.train_grouper import AGGLOM_CAP_M

DEFAULT_BASE_SEED = 20260921

# How far the ensemble median may sit from the point estimate before the run is
# treated as broken rather than noisy. An error bar is a spread AROUND the
# published number; a term that moves the centre is a bias wearing a width's
# clothing, and the whole design exists to keep those apart. 5% is loose enough
# for Monte Carlo noise at a few hundred draws and tight enough to catch a
# miscentred term, which in the retired implementation shifted the seep count
# by 37%.
MAX_CENTRE_SHIFT_PCT = 5.0


# --------------------------------------------------------------------------- #
# one realization
# --------------------------------------------------------------------------- #
def draw_row(counts, n_seeps, n_bubbles, rng, seasons, sample_rates=True):
    """Counts -> one summary dict per season.

    `rng=None` or `sample_rates=False` uses the published point rates, which is
    what makes the unsampled call reproduce the deploy total exactly.
    """
    for season in seasons:
        total, sigma = flux_rates.lake_total(
            counts, season=season, rng=rng if sample_rates else None)
        yield {
            "season": season,
            "n_bubbles": int(n_bubbles),
            "n_seeps": int(n_seeps),
            **{f"n_{c}": int(counts.get(c, 0)) for c in CLASSES},
            "total_mg_CH4_per_day": total,
            "rate_std_err_mg_CH4_per_day": sigma,
        }


def _load(bubbles_fp, artifacts_dir, thr, cap, decision_rule, overcall_penalty):
    """Shared setup: bubbles, both forests, the resolved operating point."""
    bubbles, upstream = postproc.load_bubbles(bubbles_fp)
    grouper, classifier, model_prov = postproc.load_models(
        artifacts_dir=artifacts_dir)
    params = resolve_params(model_prov, upstream, thr=thr, cap=cap,
                            decision_rule=decision_rule,
                            overcall_penalty=overcall_penalty)
    print(f"[unc] P(same) threshold {params.thr} ({params.thr_source}), "
          f"cap {params.cap} m, brightness {params.brightness_mode}, "
          f"decision rule {params.decision_rule}")
    return bubbles, upstream, grouper, classifier, model_prov, params


# --------------------------------------------------------------------------- #
# the point estimate
# --------------------------------------------------------------------------- #
def point_estimate(bubbles_fp, out_dir, *, seasons=("annual",),
                   artifacts_dir=None, thr=None, cap=AGGLOM_CAP_M,
                   decision_rule=DEFAULT_DECISION_RULE,
                   overcall_penalty=DEFAULT_OVERCALL_PENALTY):
    """The unsampled chain, written to `point.json`.

    This goes through the same `run_chain` the draws use, so the centre and the
    spread cannot come from different code. It must also equal the deploy run's
    `run_info_postproc.json` for the same `bubbles.gpkg`; if it does not, one of
    the two is stale and neither number should be quoted until that is settled.
    """
    os.makedirs(out_dir, exist_ok=True)
    bubbles, upstream, grouper, classifier, model_prov, params = _load(
        bubbles_fp, artifacts_dir, thr, cap, decision_rule, overcall_penalty)

    res = run_chain(bubbles, grouper, classifier, params, progress=True)
    counts = res.class_counts()
    rows = {r["season"]: r for r in draw_row(
        counts, len(res.seeps), len(res.bubbles), None, seasons,
        sample_rates=False)}

    out = {
        "method": "labels",
        "seasons": rows,
        "params": params.as_dict(),
        "n_bubbles_in": res.n_bubbles_in,
        "n_bubbles_screened": res.n_screened,
        "surveyed_area_m2": upstream.get("surveyed_area_m2"),
        "bubbles": os.path.abspath(bubbles_fp),
        "models": model_prov,
    }
    fp = os.path.join(out_dir, "point.json")
    with open(fp, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"[unc] point estimate -> {fp}")
    for s, r in rows.items():
        mix = ", ".join(f"{c}={r[f'n_{c}']}" for c in CLASSES)
        print(f"[unc]   {s}: {r['n_seeps']} seeps ({mix}), "
              f"{r['total_mg_CH4_per_day']:,.0f} mg CH4/day")
    return out


# --------------------------------------------------------------------------- #
# the draws
# --------------------------------------------------------------------------- #
def run(bubbles_fp, out_dir, *, draws=500, shard=(0, 1),
        base_seed=DEFAULT_BASE_SEED, terms=WOBBLE_TERMS, seasons=("annual",),
        artifacts_dir=None, thr=None, cap=AGGLOM_CAP_M,
        decision_rule=DEFAULT_DECISION_RULE,
        overcall_penalty=DEFAULT_OVERCALL_PENALTY):
    """Run this shard's slice of `draws` realizations, one part file out.

    `draws` is the TOTAL across every shard, and shard (i, n) takes every nth
    draw starting at i. Strided rather than blocked so a shard that dies still
    leaves a uniform sample of the seed space behind, instead of a contiguous
    chunk that might not be representative.

    Draw i is seeded `base_seed + i` and nothing else, so any single draw
    replays on its own and two shards can never share a stream.
    """
    terms = tuple(terms)
    unknown = set(terms) - set(WOBBLE_TERMS)
    if unknown:
        raise ValueError(f"unknown terms {sorted(unknown)}; expected a subset "
                         f"of {list(WOBBLE_TERMS)} (the rate term is always on "
                         "-- it is the dominant one and there is no reason to "
                         "report an interval without it)")
    si, sn = shard
    os.makedirs(out_dir, exist_ok=True)

    bubbles, upstream, grouper, classifier, model_prov, params = _load(
        bubbles_fp, artifacts_dir, thr, cap, decision_rule, overcall_penalty)

    # Screen once, outside the loop. The crack screen is a deterministic
    # geometry filter with field-anchored thresholds, not a sampled term, so
    # re-running it every draw would cost time without changing anything.
    kept, dropped = screen_bubbles(
        bubbles.reset_index(drop=True), max_span_m=params.max_span_m,
        min_aspect=params.min_aspect, min_shape=params.min_shape, progress=True)
    kept = kept.reset_index(drop=True)

    mine = list(range(draws))[si::sn]
    print(f"[unc] shard {si}/{sn}: {len(mine)} of {draws} draws, "
          f"{len(kept)} screened bubbles, wobbling {','.join(terms) or 'nothing'}"
          f" + rates")

    t0, rows = time.time(), []
    for k, i in enumerate(mine, start=1):
        seed = base_seed + i
        rng = np.random.default_rng(seed)
        res = run_chain(kept, grouper, classifier, params, progress=False,
                        rng=rng, wobble=terms, screened=(kept, dropped))
        counts = res.class_counts()
        for rec in draw_row(counts, len(res.seeps), len(res.bubbles), rng,
                            seasons):
            rows.append({"draw": i, "seed": seed, **rec})
        print(f"[unc] draw {k}/{len(mine)} (index {i})  "
              f"{(time.time() - t0) / k:.1f} s/draw", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{si:03d}of{sn:03d}"
    fp = os.path.join(out_dir, f"draws_{tag}.csv")
    df.to_csv(fp, index=False)

    meta = {
        "method": "labels",
        "bubbles": os.path.abspath(bubbles_fp),
        "terms": list(terms),
        "rate_term": True,
        "draws_total": draws, "draws_here": len(mine),
        "shard": f"{si}/{sn}", "base_seed": base_seed,
        "params": params.as_dict(),
        "n_bubbles_screened_in": int(len(kept)),
        "surveyed_area_m2": upstream.get("surveyed_area_m2"),
        "models": model_prov,
        "runtime_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(out_dir, f"meta_{tag}.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"[unc] wrote {fp}")
    return df


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def _percentile_se(values, q, rng, n_boot=400):
    """Bootstrap standard error of a percentile: is the draw count enough?

    Resamples the finished draws with replacement and recomputes the
    percentile. A standard error that is small next to the interval width means
    more draws would not move the answer.
    """
    v = np.asarray(values, float)
    if len(v) < 10:
        return float("nan")
    boot = [np.percentile(rng.choice(v, len(v), replace=True), q)
            for _ in range(n_boot)]
    return float(np.std(boot, ddof=1))


def summarize(out_dir, seasons=None, base_seed=DEFAULT_BASE_SEED):
    """Percentiles over every part file, with the centring check.

    Reports the MEDIAN and the 2.5th / 97.5th percentiles rather than
    mean +/- sigma: the total is positive and right-skewed, so an asymmetric
    interval is the honest shape and a single standard deviation beside it
    would be misleading. `sd_mg_CH4_per_day` is kept as a diagnostic only.

    NO ANCHORING. The ensemble is centred on the point estimate by
    construction -- every term holds the deploy decision rules and perturbs only
    the models -- so the interval is reported where it falls. If it does not
    land on the point estimate, that is a bug to fix rather than a shift to
    rescale away, and this function says so.
    """
    parts = sorted(glob.glob(os.path.join(out_dir, "draws_*.csv")))
    if not parts:
        raise SystemExit(f"no draws_*.csv in {out_dir}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if seasons:
        df = df[df["season"].isin(seasons)]
    if df["draw"].duplicated().any():
        dupes = int(df["draw"].duplicated().sum())
        raise SystemExit(
            f"{dupes} duplicated draw indices across {len(parts)} part files. "
            "Two shards shared a seed range, so the draws are not independent "
            "and the interval would be too narrow. Delete the part files and "
            "re-run with matching --shard i/N values.")

    point_fp = os.path.join(out_dir, "point.json")
    point = {}
    if os.path.exists(point_fp):
        with open(point_fp) as fh:
            point = json.load(fh).get("seasons", {})
    else:
        print(f"[unc] WARNING: no {point_fp}. Run --point so the centring "
              "check can tell a valid interval from a miscentred one.")

    rng = np.random.default_rng(base_seed)
    rows = []
    for season, sub in df.groupby("season"):
        t = sub["total_mg_CH4_per_day"].to_numpy(float)
        lo, med, hi = np.percentile(t, [2.5, 50, 97.5])
        row = {
            "season": season,
            "n_realizations": len(t),
            "mean_mg_CH4_per_day": float(t.mean()),
            "median_mg_CH4_per_day": med,
            "p2.5_mg_CH4_per_day": lo,
            "p97.5_mg_CH4_per_day": hi,
            "interval_width_pct": 100 * (hi - lo) / med if med else np.nan,
            "sd_mg_CH4_per_day": float(t.std(ddof=1)),
            "mc_se_p2.5": _percentile_se(t, 2.5, rng),
            "mc_se_p97.5": _percentile_se(t, 97.5, rng),
            **{f"mean_n_{c}": float(sub[f"n_{c}"].mean()) for c in CLASSES},
            **{f"sd_n_{c}": float(sub[f"n_{c}"].std(ddof=1)) for c in CLASSES},
            "mean_n_seeps": float(sub["n_seeps"].mean()),
            "sd_n_seeps": float(sub["n_seeps"].std(ddof=1)),
        }
        p = point.get(season)
        if p:
            pt = float(p["total_mg_CH4_per_day"])
            row.update({
                "point_total_mg_CH4_per_day": pt,
                # The closed-form published-rate floor AT THE POINT COUNTS.
                # Computed here rather than averaged over the draws: the draws'
                # counts move, so averaging their floors answers a different
                # question than "how much of the published interval is rates".
                "rate_only_sd_mg_CH4_per_day":
                    float(p["rate_std_err_mg_CH4_per_day"]),
                "point_n_seeps": int(p["n_seeps"]),
                **{f"point_n_{c}": int(p[f"n_{c}"]) for c in CLASSES},
                **{f"shift_n_{c}_pct":
                   (100 * (float(sub[f"n_{c}"].mean()) / p[f"n_{c}"] - 1.0)
                    if p[f"n_{c}"] else np.nan) for c in CLASSES},
                "shift_n_seeps_pct":
                    100 * (float(sub["n_seeps"].mean()) / p["n_seeps"] - 1.0),
            })
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\n" + "=" * 72)
    print("METHOD 1 -- LAKE FLUX, LABEL-BASED COUNTS")
    print("=" * 72)
    head = ["season", "n_realizations", "mean_mg_CH4_per_day",
            "median_mg_CH4_per_day", "p2.5_mg_CH4_per_day",
            "p97.5_mg_CH4_per_day", "interval_width_pct"]
    if "point_total_mg_CH4_per_day" in summary.columns:
        head.insert(2, "point_total_mg_CH4_per_day")
    print(summary[head].to_string(index=False,
                                  float_format=lambda v: f"{v:,.2f}"))
    print("\nThe median sits below the mean because the class-A rate is "
          "strongly right-skewed\n(16 +/- 10, so its lognormal median is 15% "
          "under its mean). That is the error model,\nnot a bias: the draw is "
          "mean-preserving by construction.")

    if "rate_only_sd_mg_CH4_per_day" in summary.columns:
        print("\nhow much of the spread is the published rates:")
        for _, r in summary.iterrows():
            floor = r["rate_only_sd_mg_CH4_per_day"]
            sd = r["sd_mg_CH4_per_day"]
            note = ("" if r["n_realizations"] >= 100 else
                    "  (too few draws for the sampled sd to be reliable)")
            print(f"  {r['season']}: sampled sd {sd:,.0f}, published-rate "
                  f"floor at the point counts {floor:,.0f}{note}")

    _centring_check(summary)

    print("\nName the sampled terms and what the interval EXCLUDES: detector "
          "false positives\n(the count is biased high), the grouping "
          "threshold, the 2014 transect width, and\nwhether the published "
          "rates transfer to this lake. Report those as stated bounds.")
    return summary


def _centring_check(summary):
    """Fail loudly if a sampled term moved the centre instead of setting the width.

    THE CHECK IS ON THE CLASS COUNTS, NOT THE TOTAL, and the reason matters.
    The tree terms act only through the counts, so the counts are where a
    miscentred model term would show. The total additionally carries the rate
    draw, which is MEAN-preserving but strongly right-skewed -- class A's
    standard error is 63% of its own mean, so the lognormal's median sits 15%
    below its mean. A total-level median test would therefore flag a correct
    sampler as biased, and a total-level MEAN test converges slowly enough that
    a short pilot reads low simply for having missed the upper tail.

    Counts are near-normal and converge fast, so this catches a real problem in
    a handful of draws. Two conditions, and BOTH must fire: the shift exceeds
    the fixed tolerance AND three Monte Carlo standard errors, so a pilot run
    is never condemned for noise.
    """
    if "point_n_seeps" not in summary.columns:
        return
    n = summary["n_realizations"].to_numpy(float)
    problems, noisy = [], []
    for _, r in summary.iterrows():
        for key, point_key in ([("n_seeps", "point_n_seeps")]
                               + [(f"n_{c}", f"point_n_{c}") for c in CLASSES]):
            pt = float(r[point_key])
            if pt <= 0:
                continue
            shift = float(r[f"shift_{key}_pct"])
            se = 100 * float(r[f"sd_{key}"]) / np.sqrt(r["n_realizations"]) / pt
            item = (r["season"], key, pt, float(r[f"mean_{key}"]), shift, se)
            if abs(shift) <= MAX_CENTRE_SHIFT_PCT:
                continue
            (noisy if abs(shift) <= 3 * se else problems).append(item)

    for season, key, pt, got, shift, se in noisy:
        print(f"[unc] {season} {key}: {shift:+.1f}% off the point estimate, but "
              f"the Monte Carlo error is {se:.1f}% at {int(n[0])} draws -- too "
              "few to judge. Run more before reporting.")

    if not problems:
        print(f"\n[unc] centring check PASSED: every sampled class count is "
              f"within {MAX_CENTRE_SHIFT_PCT}% of the point estimate, so the "
              f"model terms are a spread around the published number.")
        return
    print("\n" + "!" * 72)
    print("CENTRING CHECK FAILED -- DO NOT REPORT THIS INTERVAL")
    print("!" * 72)
    for season, key, pt, got, shift, se in problems:
        print(f"  {season} {key}: draws average {got:,.1f} against a point "
              f"estimate of {pt:,.0f} ({shift:+.1f}%, Monte Carlo error "
              f"{se:.1f}%)")
    raise SystemExit(
        "An error bar is a spread AROUND the published number. A sampled term "
        "that moves the centre is a bias, not an uncertainty, and rescaling "
        "the interval back onto the point estimate hides that rather than "
        "fixing it. Find which term is miscentred by re-running --terms one at "
        "a time.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_shard(text):
    try:
        i, n = (int(v) for v in str(text).split("/", 1))
    except ValueError:
        raise SystemExit(f"--shard wants i/N, got {text!r}") from None
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"--shard {text}: need 0 <= i < N and N >= 1")
    return i, n


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summarize", metavar="DIR", default=None,
                    help="aggregate a finished run directory and exit")
    ap.add_argument("--point", action="store_true",
                    help="run the chain once with nothing sampled and write "
                         "point.json, which the centring check needs")
    ap.add_argument("--bubbles", help="bubbles.gpkg from tools.deploy.detect")
    ap.add_argument("--out-dir")
    ap.add_argument("--draws", type=int, default=500,
                    help="TOTAL realizations across every shard "
                         "(default %(default)s)")
    ap.add_argument("--shard", default="0/1", metavar="i/N",
                    help="run every Nth draw starting at i (default "
                         "%(default)s, meaning all of them)")
    ap.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED,
                    help="draw i is seeded base_seed + i (default %(default)s)")
    ap.add_argument("--terms", default=",".join(WOBBLE_TERMS),
                    help="which forests to resample trees for, from "
                         "grouper,classifier. One at a time gives the variance "
                         "decomposition. The rate term is always on. Pass an "
                         "empty string for rates only")
    ap.add_argument("--season", default="annual",
                    help="comma list, or 'all'. Seasons are alternative views "
                         "of the same seeps and are NOT additive")
    ap.add_argument("--thr", type=float, default=None,
                    help="grouper P(same) operating point. Default is the "
                         "value versioned with the artifact; override only to "
                         "measure sensitivity")
    ap.add_argument("--cap", type=float, default=AGGLOM_CAP_M)
    ap.add_argument("--artifacts-dir", default=None)
    ap.add_argument("--decision-rule", default=DEFAULT_DECISION_RULE,
                    choices=list(postproc.DECISION_RULES))
    ap.add_argument("--overcall-penalty", type=float,
                    default=DEFAULT_OVERCALL_PENALTY)
    args = ap.parse_args(argv)

    if args.summarize:
        summarize(args.summarize, base_seed=args.base_seed)
        return
    if not args.bubbles or not args.out_dir:
        ap.error("--bubbles and --out-dir are required unless --summarize")

    seasons = (tuple(flux_rates.SEASONS) if args.season == "all"
               else tuple(s.strip() for s in args.season.split(",") if s.strip()))
    common = dict(seasons=seasons, artifacts_dir=args.artifacts_dir,
                  thr=args.thr, cap=args.cap,
                  decision_rule=args.decision_rule,
                  overcall_penalty=args.overcall_penalty)

    if args.point:
        point_estimate(args.bubbles, args.out_dir, **common)
        return

    run(args.bubbles, args.out_dir, draws=args.draws,
        shard=_parse_shard(args.shard), base_seed=args.base_seed,
        terms=tuple(t.strip() for t in args.terms.split(",") if t.strip()),
        **common)


if __name__ == "__main__":
    main()
