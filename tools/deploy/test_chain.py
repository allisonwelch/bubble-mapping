# tools/deploy/test_chain.py
"""Acceptance tests for the single-implementation chain.

Run them after ANY change to `chain.py`, `postproc.py` or either uncertainty
script:

    python -m tools.deploy.test_chain



It needs the frozen artifacts and a finished deploy run, so it SKIPS rather
than fails when they are absent. A skip is not a pass; say so if you report it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_RUN = os.path.join(REPO, "data", "deploy_out",
                           "Octopus_10212025_20260915")


class Skip(Exception):
    """Inputs are not on this machine."""


def _load_run(run_dir):
    """(bubbles_path, recorded_run_info) for a finished postproc run.

    Falls back to the `source` recorded in the run metadata when the directory
    holds no `bubbles.gpkg` of its own, which is the case for a postproc-only
    re-analysis of another run's detections.
    """
    info_fp = os.path.join(run_dir, "run_info_postproc.json")
    if not os.path.exists(info_fp):
        raise Skip(f"no {info_fp}")
    with open(info_fp) as fh:
        info = json.load(fh)

    fp = os.path.join(run_dir, "bubbles.gpkg")
    if not os.path.exists(fp):
        src = info.get("source") or ""
        if src.endswith(".gpkg") and os.path.exists(src):
            fp = src
        else:
            raise Skip(f"no bubbles.gpkg in {run_dir}, and its recorded "
                       f"source is not on this machine ({src!r})")
    return fp, info


def test_chain_reproduces_the_deploy_run(run_dir):
    """run_chain with nothing sampled == what this run already recorded.

    Compares against `run_info_postproc.json` in the same directory. A mismatch
    means either the chain changed or that file is stale -- and BOTH are worth
    stopping for, because every number downstream inherits the difference.
    """
    from tools.deploy import postproc
    from tools.deploy.chain import resolve_params, run_chain

    bubbles_fp, want = _load_run(run_dir)
    bubbles, upstream = postproc.load_bubbles(bubbles_fp)
    grouper, classifier, model_prov = postproc.load_models()
    params = resolve_params(model_prov, upstream)

    # The screen's thresholds moved on 2026-09-17 (a third gate, and the span
    # limit to 1.30 m). An older run_info recorded the older gates, so compare
    # like for like rather than declaring the refactor broken.
    recorded_gates = (want.get("screen_max_span_m"),
                      want.get("screen_min_aspect"),
                      want.get("screen_min_shape"))
    current_gates = (params.max_span_m, params.min_aspect, params.min_shape)
    if any(g is None for g in recorded_gates):
        raise Skip(
            f"{os.path.basename(run_dir)}/run_info_postproc.json predates the "
            "three-gate crack screen (no screen_min_aspect), so its counts are "
            "not comparable. Re-run `deploy.py --stage postproc` on it first.")
    if tuple(float(g) for g in recorded_gates) != tuple(
            float(g) for g in current_gates):
        raise Skip(
            f"screen gates differ: recorded {recorded_gates}, current "
            f"{current_gates}. Re-run `deploy.py --stage postproc` to refresh "
            "the run metadata before comparing.")

    res = run_chain(bubbles, grouper, classifier, params, progress=False)
    got = {
        "n_seeps": int(len(res.seeps)),
        "n_bubbles": int(len(res.bubbles)),
        "n_bubbles_screened": res.n_screened,
        "class_counts": res.class_counts(),
    }
    expect = {k: want[k] for k in got}
    expect["class_counts"] = {c: int(n)
                              for c, n in want["class_counts"].items()}
    assert got == expect, (
        f"chain output drifted from {run_dir}\n  got    {got}\n  "
        f"expect {expect}")
    return got


def test_wobble_off_is_deterministic(run_dir):
    """An rng present but no term sampled must change nothing.

    Guards the centring contract at its root: if passing a generator alone
    perturbs the answer, then "nothing sampled" and "the deploy run" are not
    the same thing and every interval is centred on the wrong number.
    """
    from tools.deploy import postproc
    from tools.deploy.chain import resolve_params, run_chain

    bubbles_fp, _ = _load_run(run_dir)
    bubbles, upstream = postproc.load_bubbles(bubbles_fp)
    grouper, classifier, model_prov = postproc.load_models()
    params = resolve_params(model_prov, upstream)

    a = run_chain(bubbles, grouper, classifier, params, progress=False)
    b = run_chain(bubbles, grouper, classifier, params, progress=False,
                  rng=np.random.default_rng(0), wobble=())
    assert a.class_counts() == b.class_counts()
    assert len(a.seeps) == len(b.seeps)
    return a.class_counts()


def test_tree_resampling_preserves_shape():
    """A resampled forest predicts the same shape, with the same classes.

    Cheap, and needs no run directory: it catches the case where `estimators_`
    is swapped onto a model whose per-tree class order does not match the
    forest's, which would silently permute A/B/C.
    """
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from tools.deploy.chain import resample_trees

    X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                               n_classes=3, random_state=0)
    rf = RandomForestClassifier(n_estimators=25, random_state=0).fit(X, y)
    rng = np.random.default_rng(1)
    sub = resample_trees(rf, rng)

    assert list(sub.classes_) == list(rf.classes_)
    p = sub.predict_proba(X)
    assert p.shape == rf.predict_proba(X).shape
    assert np.allclose(p.sum(axis=1), 1.0)
    # Different trees, so a different answer -- otherwise the term contributes
    # nothing and the interval is silently rate-only.
    assert not np.allclose(p, rf.predict_proba(X))
    return p.shape


def test_expected_counts_match_closed_form():
    """Sampling a posterior reproduces sum(p) and sqrt(sum p(1-p)).

    The two computations of the expected class count must agree, because
    `uncertainty_proba` reports one and validates it with the other.
    """
    from tools.deploy.uncertainty_proba import sample_classes

    rng = np.random.default_rng(7)
    n = 20000
    p = rng.dirichlet([6.0, 2.0, 0.4], size=n)

    want = p.sum(axis=0)
    sd = np.sqrt((p * (1 - p)).sum(axis=0))

    draws = 200
    got = np.zeros((draws, 3))
    for i in range(draws):
        c = sample_classes(p, np.random.default_rng(1000 + i))
        got[i] = [c["A"], c["B"], c["C"]]

    # Mean within a few standard errors of the closed-form mean, and the
    # simulated spread within 10% of the closed-form spread.
    se = sd / np.sqrt(draws)
    assert np.all(np.abs(got.mean(axis=0) - want) < 5 * se), (
        f"mean {got.mean(axis=0)} vs closed form {want}")
    assert np.all(np.abs(got.std(axis=0, ddof=1) / sd - 1.0) < 0.10), (
        f"sd {got.std(axis=0, ddof=1)} vs closed form {sd}")
    return dict(zip("ABC", want.round(1)))


def test_fractional_counts_are_not_truncated():
    """total_from_counts keeps the fractions `flux_rates.lake_total` casts away."""
    from tools.deploy.uncertainty_proba import total_from_counts
    from tools.flux import rates as flux_rates

    counts = {"A": 100.7, "B": 10.4, "C": 2.9}
    got, _ = total_from_counts(counts)
    rates, _ = flux_rates.season_rates("annual")
    want = sum(rates[c] * counts[c] for c in "ABC")
    assert abs(got - want) < 1e-6
    # And it must differ from the truncating path, or this test proves nothing.
    trunc, _ = flux_rates.lake_total(counts)
    assert got > trunc
    return round(got, 2)


def main(argv=None):
    run_dir = (argv or sys.argv[1:] or [DEFAULT_RUN])[0]
    tests = [
        (test_chain_reproduces_the_deploy_run, (run_dir,)),
        (test_wobble_off_is_deterministic, (run_dir,)),
        (test_tree_resampling_preserves_shape, ()),
        (test_expected_counts_match_closed_form, ()),
        (test_fractional_counts_are_not_truncated, ()),
    ]
    failed = skipped = 0
    for fn, args in tests:
        name = fn.__name__
        try:
            out = fn(*args)
        except Skip as e:
            print(f"SKIP  {name}: {e}")
            skipped += 1
        except AssertionError as e:
            print(f"FAIL  {name}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
        else:
            print(f"ok    {name}  {out}")
    print(f"\n{len(tests) - failed - skipped} passed, {failed} failed, "
          f"{skipped} skipped")
    if skipped:
        print("A SKIP is not a pass. The chain-reproduction test is the one "
              "that matters; run it where the artifacts and a finished deploy "
              "run both exist.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
