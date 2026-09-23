# tools/deploy/postproc.py
"""Stage B of the whole-lake runner: detected bubbles -> classified seeps -> flux.

    bubbles.gpkg
      -> learned pairwise grouper, diameter-capped     -> seep_group_id
      -> dissolve + convex hull                        -> per-seep features
      -> A/B/C classifier (the deploy-point forest)    -> class
      -> count-based lake total                        -> flux

CPU only, no torch, so it runs anywhere against a bubbles.gpkg the GPU stage
produced. `--from-pred-dir` points it at an existing prediction directory
instead, which is how the chain gets exercised without a lake-scale run.

BOTH MODELS ARE FROZEN ARTIFACTS, loaded from disk beside the checkpoint
(`tools.deploy.build_artifacts`). This stage deploys; it does not train. Passing
`--refit` fits them from the labeler packs instead, which requires the packs to
be present on this machine and makes the run unreproducible -- so it is opt-in,
never a fallback. The run's model provenance, including the sha256 of every
input pack, is stamped into `run_info_postproc.json` and into the output gpkgs.

The dissolve protocol is not negotiable: most of the classifier's features come
from the dissolve, and it was trained on rows built by
`fit_classifier.dissolve_to_seeps`. `chain.dissolve_bubbles_to_seeps` mirrors
that term for term. If the two drift, the classifier is being asked about a
feature distribution it never saw, and the class balance -- which is the flux --
shifts silently. Change one, change both.

THE CHAIN ITSELF LIVES IN `tools.deploy.chain`. This module loads the
inputs, calls `chain.run_chain` once, and writes the outputs. The uncertainty
scripts call the same function N times with sampling switched on, so the point
estimate and the draws can never drift apart -- which they did, silently, when
the chain was written out three times.

The reported interval covers the published per-class rate uncertainty and
nothing else. For the pipeline terms, run `tools.deploy.uncertainty_labels`
(hard labels drive the total) or `tools.deploy.uncertainty_proba` (the class
posterior does).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time

import pandas as pd

import sklearn

from tools.classify.fit_classifier import CLASSES
from tools.deploy import build_artifacts, runinfo
from tools.deploy.chain import (  # noqa: F401  (re-exported for callers)
    DECISION_RULES, DEFAULT_DECISION_RULE, DEFAULT_OVERCALL_PENALTY, GROUP_THR,
    SCREEN_MAX_SPAN_M, SCREEN_MIN_THINNESS, ChainParams,
    _classifier_brightness, _default_brightness_cell_m, _grouper_threshold,
    classify_seeps, dissolve_bubbles_to_seeps, group_bubbles, resolve_params,
    run_chain, screen_bubbles)
from tools.flux import field_reference, rates as flux_rates
from tools.grouping.train_grouper import AGGLOM_CAP_M

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_bubbles(path: str):
    """Detected bubble polygons + features, as written by tools.deploy.detect.

    Returns (bubbles, info). `info` comes from the run metadata embedded in the
    gpkg, so the surveyed area travels with the file rather than beside it; an
    externally produced gpkg simply yields an empty dict.
    """
    if gpd is None:
        raise RuntimeError("geopandas is required")
    g = gpd.read_file(path, layer="bubbles")
    if "image" not in g.columns:
        g["image"] = os.path.basename(path)
    return g, runinfo.read_run_info(path)


def load_from_pred_dir(pred_dir: str):
    """Rebuild the same frame from an existing prediction directory.

    `bubble_features.csv` carries the features; the geometry comes back from
    each chip's `{stem}_cc.tif`, because a predicted bubble only exists as a
    connected component. Same route tools/grouping/group_predictions.py takes.

    Returns (bubbles, info). There is no detect run behind this path, so the
    surveyed area is the summed footprint of the chips actually used -- whole
    chips, with no lake polygon and no alpha mask, which is the right
    denominator for a chip-based density. It assumes the chips do not overlap.
    """
    import rasterio
    from tools.eval.bubble_features import polygonize_labels

    feats = pd.read_csv(os.path.join(pred_dir, "bubble_features.csv"))
    out, area_m2, used = [], 0.0, []
    for im, sub in feats.groupby("image"):
        stem = im[:-4] if im.endswith(".tif") else im
        cc_fp = os.path.join(pred_dir, f"{stem}_cc.tif")
        if not os.path.exists(cc_fp):
            print(f"  [skip] {im}: no {os.path.basename(cc_fp)}")
            continue
        with rasterio.open(cc_fp) as ds:
            polys = polygonize_labels(ds.read(1), ds.transform, ds.crs)
            res = abs(ds.transform.a)
            area_m2 += ds.width * ds.height * res * res
        polys = polys.rename(columns={"id": "bubble_id"})
        out.append(polys.merge(sub, on="bubble_id", how="inner"))
        used.append(im)
    if not out:
        raise SystemExit(f"no chips with a _cc.tif under {pred_dir}")
    g = pd.concat(out, ignore_index=True)
    info = {"stage": "from_pred_dir",
            "pred_dir": os.path.abspath(pred_dir),
            "n_chips": len(used),
            "chips": sorted(used),
            "surveyed_area_m2": float(area_m2),
            "surveyed_area_note": "sum of whole chip footprints; no lake "
                                  "polygon or alpha mask applied"}
    return gpd.GeoDataFrame(g, geometry="geometry", crs=out[0].crs), info


# --------------------------------------------------------------------------- #
# classify + flux
# --------------------------------------------------------------------------- #
def attach_flux(seeps, season="annual"):
    """Per-seep rate in mg CH4/day, from its class. Flux is count-based, so
    this column is a lookup, not a function of the seep's size."""
    rate, _ = flux_rates.season_rates(season)
    seeps = seeps.copy()
    seeps["flux_mg_per_day"] = seeps["class"].map(rate).astype(float)
    return seeps


def flux_report(seeps, surveyed_area_m2=None, season="annual"):
    """The end-of-chain tables: per-season totals and a per-image breakdown.

    Every mg CH4/day column in the season table is reported three ways -- as
    itself, over the surveyed area, and as an annual mass -- including the two
    uncertainty terms, so the interval and the central estimate are never in
    different units. The per-image table is NOT, on purpose: the surveyed area
    is measured over the whole lake and no per-image area exists, so dividing
    one image's flux by it would read as that image's own density.
    """
    counts = seeps["class"].value_counts().to_dict()
    table = flux_rates.flux_table(counts)
    if surveyed_area_m2:
        table["surveyed_area_m2"] = surveyed_area_m2
        # The per-class published rates are per SEEP, so they are excluded:
        # one seep's rate over the whole lake area is not a quantity.
        table = flux_rates.add_per_area_columns(
            table, surveyed_area_m2,
            exclude=[f"rate_{c}_mg_CH4_per_day" for c in CLASSES])
        table["seeps_per_m2"] = table["n_seeps"] / surveyed_area_m2

    per_image = []
    for im, sub in seeps.groupby("image"):
        c = sub["class"].value_counts().to_dict()
        total, sigma = flux_rates.lake_total(c, season=season)
        row = {"image": im, "n_seeps": len(sub)}
        row.update({f"n_{k}": int(c.get(k, 0)) for k in CLASSES})
        row["total_mg_CH4_per_day"] = total
        row["rate_std_err_mg_CH4_per_day"] = sigma
        per_image.append(row)
    per_image = pd.DataFrame(per_image).sort_values(
        "total_mg_CH4_per_day", ascending=False).reset_index(drop=True)
    return table, per_image


def lake_totals_long(seeps, table, surveyed_area_m2=None, label=None,
                     upstream=None, run_id=""):
    """The per-run flux totals, LONG: one row per (season, class).

    `class` takes A / B / C and `all` for the season total, so a row is always
    "this much methane, from this many seeps, of this class, under this
    season's rates". Run identity is repeated on every row, which is what makes
    several runs concatenate into one table and pivot cleanly -- the point being
    to compare one lake against another, or a lake against itself on a
    different flight date.
    """
    upstream = dict(upstream or {})
    counts = seeps["class"].value_counts().to_dict()
    ident = {
        "run_id": run_id,
        "label": label or upstream.get("image") or "",
        "image": upstream.get("image", ""),
        "lake_polygon": upstream.get("lake_polygon", ""),
        "checkpoint": upstream.get("checkpoint", ""),
        "surveyed_area_m2": surveyed_area_m2,
    }

    rows = []
    for _, r in table.iterrows():
        total = float(r["total_mg_CH4_per_day"])
        for c in list(CLASSES) + ["all"]:
            is_all = c == "all"
            n = int(len(seeps)) if is_all else int(counts.get(c, 0))
            flux = total if is_all else float(r[f"flux_{c}_mg_CH4_per_day"])
            rows.append({
                **ident,
                "season": r["season"],
                "class": c,
                "n_seeps": n,
                # The published per-seep rate. Blank on the `all` row: there is
                # no single rate for a mixed population, only the total.
                "rate_mg_CH4_per_day":
                    None if is_all else float(r[f"rate_{c}_mg_CH4_per_day"]),
                "flux_mg_CH4_per_day": flux,
                "pct_of_season_flux":
                    (100 * flux / total) if total else float("nan"),
                # Standard error on the class mean times the count. Only the
                # `all` row carries the quadrature sum, which is the published
                # floor for the total.
                "rate_std_err_mg_CH4_per_day":
                    float(r["rate_std_err_mg_CH4_per_day"]) if is_all
                    else None,
                "seeps_per_m2":
                    (n / surveyed_area_m2) if surveyed_area_m2 else None,
            })
    # `rate_mg_CH4_per_day` is one seep's published rate, not a lake figure, so
    # it keeps its own unit. Everything else gains the density and annual-mass
    # twins, including the standard error on the `all` row.
    return flux_rates.add_per_area_columns(
        pd.DataFrame(rows), surveyed_area_m2, exclude=["rate_mg_CH4_per_day"])


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def load_models(artifacts_dir=None, labeling_dir=None, refit=False, seed=42):
    """(grouper, classifier, provenance) for a deploy run.

    Frozen artifacts by default. `refit` re-fits both from the labeler packs,
    which requires the packs to be on this machine and makes the run
    unreproducible -- so it is opt-in, never a fallback.
    """
    if not refit:
        grouper, classifier, manifest = build_artifacts.load_artifacts(
            artifacts_dir)
        prov = {
            "source": "artifacts",
            "artifacts_dir": os.path.abspath(
                artifacts_dir or build_artifacts.default_out_dir()),
            "built_utc": manifest.get("built_utc"),
            "sklearn_version_built": manifest.get("sklearn_version"),
            "sklearn_version_running": sklearn.__version__,
            "seed": manifest.get("seed"),
            "grouper": manifest.get("grouper", {}),
            "classifier": manifest.get("classifier", {}),
            "inputs": manifest.get("inputs", {}),
        }
        print(f"[models] loaded frozen artifacts from {prov['artifacts_dir']} "
              f"(built {prov['built_utc']}, sklearn "
              f"{prov['sklearn_version_built']})")
        return grouper, classifier, prov

    from tools.classify.fit_classifier import fit_deploy_model
    from tools.grouping.deploy_grouper import train_model

    print("[models] --refit: fitting both forests from the labeler packs. "
          "This run is NOT reproducible from the artifacts on disk.")
    grouper = train_model(seed=seed)
    classifier, clf_info = fit_deploy_model(labeling_dir, seed=seed)
    prov = {
        "source": "refit",
        "sklearn_version_running": sklearn.__version__,
        "seed": seed,
        "grouper": getattr(grouper, "fit_info_", {}),
        "classifier": clf_info,
    }
    return grouper, classifier, prov


def run(bubbles, out_dir, labeling_dir=None, thr=None, cap=AGGLOM_CAP_M,
        season="annual", surveyed_area_m2=None, upstream=None, progress=True,
        source=None, label=None, artifacts_dir=None, refit=False, seed=42,
        decision_rule=DEFAULT_DECISION_RULE,
        overcall_penalty=DEFAULT_OVERCALL_PENALTY,
        max_span_m=SCREEN_MAX_SPAN_M, min_thinness=SCREEN_MIN_THINNESS,
        brightness_cell_m=None):
    """Group -> dissolve -> classify -> flux. Returns (seeps, table, per_image).

    Both models are loaded frozen from disk (`tools.deploy.build_artifacts`);
    `refit=True` re-fits them from the labeler packs instead. The runner
    deploys, it never trains -- see build_artifacts' docstring for what that
    rule is protecting.

    `upstream` is the run metadata from whichever loader produced `bubbles`.
    It supplies the surveyed area unless `surveyed_area_m2` overrides it, and
    it is merged into the metadata stamped onto the outputs, so seeps.gpkg
    records the checkpoint and lake polygon it ultimately came from.

    `thr=None` means "use the operating point recorded in the artifact", the
    same way `surveyed_area_m2=None` means "use the one recorded upstream". An
    explicit value still wins, which is what makes a threshold sweep possible,
    and `group_threshold_source` in the run metadata records which happened.
    """
    upstream = dict(upstream or {})
    if surveyed_area_m2 is None:
        surveyed_area_m2 = upstream.get("surveyed_area_m2")

    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    # Stamps the totals file so re-running never silently overwrites the last
    # answer -- the whole point of this table is comparing runs to each other.
    run_id = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[postproc] {len(bubbles)} bubbles over "
          f"{bubbles['image'].nunique()} image(s)")
    if surveyed_area_m2:
        print(f"[postproc] surveyed area {surveyed_area_m2:,.0f} m2")
    else:
        print("[postproc] WARNING: no surveyed area recorded, so the total "
              "cannot be turned into a density and is not comparable to "
              "another lake, another flight date, or a field figure.")

    grouper, classifier, model_prov = load_models(
        artifacts_dir=artifacts_dir, labeling_dir=labeling_dir, refit=refit,
        seed=seed)
    params = resolve_params(
        model_prov, upstream, thr=thr, cap=cap, decision_rule=decision_rule,
        overcall_penalty=overcall_penalty, max_span_m=max_span_m,
        min_thinness=min_thinness,
        brightness_cell_m=brightness_cell_m)
    print(f"[group] P(same) threshold {params.thr} ({params.thr_source})")

    # One call, and it is the SAME call the uncertainty scripts make with
    # sampling switched on. Nothing about the chain is re-implemented here.
    res = run_chain(bubbles, grouper, classifier, params, progress=progress)
    bubbles, screened, seeps = res.bubbles, res.screened, res.seeps
    n_in = res.n_bubbles_in

    seeps = attach_flux(seeps, season=season)
    rule_note = (f"{decision_rule}" if decision_rule == "argmax"
                 else f"{decision_rule} (overcall penalty {overcall_penalty})")
    print(f"[classify] decision rule: {rule_note}")
    print(f"[classify] {len(seeps)} seeps: "
          f"{seeps['class'].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()}")

    table, per_image = flux_report(seeps, surveyed_area_m2, season=season)

    bubbles_out = os.path.join(out_dir, "bubbles_grouped.gpkg")
    seeps_out = os.path.join(out_dir, "seeps.gpkg")
    for fp in (bubbles_out, seeps_out):
        if os.path.exists(fp):
            os.remove(fp)
    bubbles.to_file(bubbles_out, layer="bubbles", driver="GPKG")
    seeps.to_file(seeps_out, layer="seeps", driver="GPKG")
    seeps.drop(columns="geometry").to_csv(
        os.path.join(out_dir, "seeps.csv"), index=False)
    table.to_csv(os.path.join(out_dir, "flux_summary.csv"), index=False)
    per_image.to_csv(os.path.join(out_dir, "flux_per_image.csv"), index=False)

    if len(screened):
        screened_fp = os.path.join(out_dir, "screened_bubbles.gpkg")
        if os.path.exists(screened_fp):
            os.remove(screened_fp)
        screened.to_file(screened_fp, layer="screened", driver="GPKG")

    bench = field_reference.compare(
        seeps["class"].value_counts().to_dict(), surveyed_area_m2,
        median_area_m2=seeps.groupby("class")["hull_area_m2"].median().to_dict())
    bench.to_csv(os.path.join(out_dir, "field_benchmark.csv"), index=False)
    totals = lake_totals_long(seeps, table, surveyed_area_m2, label=label,
                              upstream=upstream, run_id=run_id)
    totals_fp = os.path.join(out_dir, f"lake_flux_totals_{run_id}.csv")
    totals.to_csv(totals_fp, index=False)

    # Upstream first so the stage-B keys win on any collision.
    info = {**upstream, **{
        "stage": "postproc",
        "source": source,
        "group_threshold": params.thr,
        "group_threshold_source": params.thr_source,
        "agglom_cap_m": params.cap,
        "season": season,
        "screen_max_span_m": params.max_span_m,
        "screen_min_thinness": params.min_thinness,
        "n_bubbles_in": int(n_in),
        "n_bubbles_screened": res.n_screened,
        "brightness": params.brightness_mode,
        "brightness_cell_m": params.brightness_cell_m,
        "n_bubbles": int(len(bubbles)),
        "n_seeps": int(len(seeps)),
        "class_counts": {c: int((seeps["class"] == c).sum()) for c in CLASSES},
        "surveyed_area_m2": surveyed_area_m2,
        "decision_rule": decision_rule,
        "overcall_penalty": (overcall_penalty if decision_rule == "conservative"
                             else None),
        "models": model_prov,
        "run_id": run_id,
        "runtime_s": round(time.time() - t0, 1),
    }}
    for fp in (bubbles_out, seeps_out):
        runinfo.write_run_info(fp, info)
    with open(os.path.join(out_dir, "run_info_postproc.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)

    _print_report(table, per_image, surveyed_area_m2, bench)
    print(f"[postproc] wrote {seeps_out} (+ seeps.csv, flux_summary.csv, "
          f"flux_per_image.csv, field_benchmark.csv)")
    print(f"[postproc] wrote {totals_fp}")
    return seeps, table, per_image


def _print_report(table, per_image, surveyed_area_m2, bench=None):
    print("\n" + "=" * 72)
    print("COUNT-BASED FLUX  (sum over classes of seep count x per-class rate)")
    print("=" * 72)
    cols = ["season", "n_A", "n_B", "n_C", "n_seeps", "total_mg_CH4_per_day",
            "rate_std_err_pct", "pct_flux_from_C"]
    if surveyed_area_m2:
        cols += ["total_mg_CH4_per_m2_per_day", "total_g_CH4_per_m2_per_year"]
    print(table[cols].to_string(
        index=False,
        float_format=lambda v: f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"))
    if surveyed_area_m2:
        print("\ntotal_g_CH4_per_m2_per_year is blank for summer and winter: "
              "those are per-day rates\nwithin a regime whose length is not "
              "recorded, so neither integrates to a year.")
    print("\nper image:")
    print(per_image.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print("\nrate_std_err_pct is the PUBLISHED RATE uncertainty only. Detector, "
          "grouper and\nclassifier error are not in it. For the pipeline terms "
          "run tools.deploy.uncertainty_labels\nor tools.deploy.uncertainty_proba "
          "against this run's bubbles.gpkg. Quote this as a\npoint estimate, "
          "with the surveyed area named.")

    if bench is not None and len(bench):
        print("\n" + "=" * 72)
        print("AGAINST THE 2014 FIELD TRANSECTS  (external check, never fitted)")
        print("=" * 72)
        print(bench.to_string(index=False))
        print(f"\nsource: {field_reference.SOURCE}")
        print("The density row assumes a 1 m transect width (2 m halves it) -- "
              "the workbooks\nrecord seep counts but not the width. Treat a "
              "disagreement here as a lead to\nchase, NOT a parameter to tune: "
              "these numbers are the only validation on this\nlake that no "
              "stage of the pipeline was fitted to.")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bubbles", help="bubbles.gpkg from tools.deploy.detect")
    src.add_argument("--from-pred-dir",
                     help="a canonical prediction directory "
                          "(bubble_features.csv + {stem}_cc.tif per chip)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifacts-dir", default=None,
                    help="where grouper_rf.joblib / classifier_rf.joblib live "
                         f"(default: {build_artifacts.default_out_dir()})")
    ap.add_argument("--refit", action="store_true",
                    help="re-fit both forests from the labeler packs instead "
                         "of loading the frozen artifacts. Requires the packs "
                         "to be present and makes the run unreproducible.")
    ap.add_argument("--seed", type=int, default=42,
                    help="only used with --refit")
    ap.add_argument("--labeling-dir", default=None,
                    help="the three final labeler packs (only used with --refit)")
    ap.add_argument("--decision-rule", default=DEFAULT_DECISION_RULE,
                    choices=DECISION_RULES,
                    help="how the class posterior becomes a label "
                         "(default %(default)s)")
    ap.add_argument("--overcall-penalty", type=float,
                    default=DEFAULT_OVERCALL_PENALTY,
                    help="only used by --decision-rule conservative "
                         "(default %(default)s)")
    ap.add_argument("--thr", type=float, default=None,
                    help="grouper P(same) operating point. Default is whatever "
                         "the artifact was built with; pass a value only to "
                         "measure sensitivity, and report the spread rather "
                         "than adopting the one that flatters the total.")
    ap.add_argument("--cap", type=float, default=AGGLOM_CAP_M)
    ap.add_argument("--max-span-m", type=float, default=SCREEN_MAX_SPAN_M,
                    help="drop single connected components longer than this "
                         "AND thin. Default %(default)s m is the largest major "
                         "axis in the field workbooks, not a tuned value. Pass "
                         "0 to disable the screen.")
    ap.add_argument("--min-thinness", type=float, default=SCREEN_MIN_THINNESS,
                    help="span / (2 * largest inscribed circle radius) above "
                         "which a long component counts as a crack (default "
                         "%(default)s). 1 is a circle, and the measure is "
                         "local, so a component that is fat anywhere survives "
                         "however ragged or elongated its outline is.")
    ap.add_argument("--brightness-cell-m", type=float, default=None,
                    help="neighbourhood size for relative brightness. Only "
                         "used when the loaded classifier was fit with "
                         "--brightness rel; defaults to the detector's tile "
                         "size, which is the scale its training chips were cut "
                         "at.")
    ap.add_argument("--season", default="annual",
                    choices=sorted(flux_rates.SEASONS))
    ap.add_argument("--surveyed-area-m2", type=float, default=None,
                    help="override the valid area recorded by the loader; "
                         "normally it travels inside the source gpkg")
    args = ap.parse_args(argv)

    if args.bubbles:
        bubbles, upstream = load_bubbles(args.bubbles)
        source = os.path.abspath(args.bubbles)
    else:
        bubbles, upstream = load_from_pred_dir(args.from_pred_dir)
        source = os.path.abspath(args.from_pred_dir)

    run(bubbles, args.out_dir, labeling_dir=args.labeling_dir, thr=args.thr,
        cap=args.cap, season=args.season,
        surveyed_area_m2=args.surveyed_area_m2, upstream=upstream,
        source=source, artifacts_dir=args.artifacts_dir, refit=args.refit,
        seed=args.seed, decision_rule=args.decision_rule,
        overcall_penalty=args.overcall_penalty,
        max_span_m=args.max_span_m or None, min_thinness=args.min_thinness,
        brightness_cell_m=args.brightness_cell_m)


if __name__ == "__main__":
    main()