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

The dissolve protocol is not negotiable: most of the classifier's features come
from the dissolve, and it was trained on rows built by
`fit_classifier.dissolve_to_seeps`. `dissolve_bubbles_to_seeps` below mirrors
that term for term. If the two drift, the classifier is being asked about a
feature distribution it never saw, and the class balance -- which is the flux --
shifts silently. Change one, change both.

The reported interval covers the published per-class rate uncertainty and
nothing else. Detector, grouper and classifier error have no closed form here
and need the Monte Carlo chain: this runner called N times with sampling
switched on. The interface is built for it; the sampling is not written yet.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time

import numpy as np
import pandas as pd
from shapely.ops import unary_union
from tqdm import tqdm

from tools.classify.fit_classifier import (CLASSES, FEATURES as CLASS_FEATURES,
                                           fit_deploy_model)
from tools.deploy import runinfo
from tools.flux import rates as flux_rates
from tools.grouping.deploy_grouper import _pair_features, constrained_cluster
from tools.grouping.train_grouper import AGGLOM_CAP_M, FEATURES as PAIR_FEATURES

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

GROUP_THR = 0.6   # RF P(same) operating point, per deploy_grouper


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
# grouping
# --------------------------------------------------------------------------- #
def group_bubbles(clf, bubbles, thr=GROUP_THR, cap=AGGLOM_CAP_M, progress=True):
    """Assign `seep_group_id` within each `image`, anchor-id convention.

    Candidate pairs are generated per image only, so cross-image id reuse can
    never merge bubbles across chips. A whole lake is one image, so the
    partition is the lake and a seep spanning a tile seam groups correctly --
    the tile grid is deliberately not the grouping partition.

    The diameter cap is enforced (`constrained_cluster`): plain connected
    components let chains of short edges bridge into runaway seeps, and flux is
    count-based, so one runaway seep is a real error in the total.
    """
    sgid = bubbles["bubble_id"].astype(np.int64).copy()
    n_multi = 0
    groups = list(bubbles.groupby("image"))
    for im, sub in tqdm(groups, desc="group", disable=not progress):
        if len(sub) < 2:
            continue
        fx = sub["centroid_x_m"].to_numpy(float)
        fy = sub["centroid_y_m"].to_numpy(float)
        fa = sub["area_m2"].to_numpy(float)
        pp, feat = _pair_features(sub, fx, fy, fa)
        if len(pp) == 0:
            continue
        proba = clf.predict_proba(feat[PAIR_FEATURES].to_numpy(float))[:, 1]
        keep = proba >= thr
        comp = constrained_cluster(len(sub), pp[keep], proba[keep],
                                   np.column_stack([fx, fy]), cap)
        ids = sub["bubble_id"].to_numpy(np.int64)
        for c in np.unique(comp):
            m = np.where(comp == c)[0]
            sgid.iloc[sub.index[m]] = int(ids[m].max())   # collision-safe anchor
            if len(m) > 1:
                n_multi += 1
    print(f"[group] {len(bubbles)} bubbles -> "
          f"{bubbles.assign(_g=sgid).groupby(['image', '_g']).ngroups} seeps "
          f"({n_multi} multi-bubble)")
    return sgid


# --------------------------------------------------------------------------- #
# dissolve
# --------------------------------------------------------------------------- #
def dissolve_bubbles_to_seeps(bubbles, progress=True):
    """One row per (image, seep_group_id): hull geometry + classifier features.

    Mirrors `fit_classifier.dissolve_to_seeps` on every shared term -- hull of
    the unary union, brightness weighted by member area_m2 -- minus the
    labeler-only columns. Read that function before touching this one.
    """
    rows, geoms = [], []
    keys = list(bubbles.groupby(["image", "seep_group_id"], sort=True))
    for (img, gid), sub in tqdm(keys, desc="dissolve", disable=not progress):
        geom = unary_union(sub.geometry.values)
        hull = geom.convex_hull
        w = sub["area_m2"].to_numpy(dtype=float)
        w = w / w.sum() if w.sum() > 0 else np.full(len(w), 1.0 / len(w))
        c = hull.centroid
        rows.append({
            "image": img,
            "seep_group_id": int(gid),
            "n_bubbles": len(sub),
            "area_m2": float(geom.area),
            "hull_area_m2": float(hull.area),
            "perim_m": float(geom.length),
            "centroid_x_m": float(c.x), "centroid_y_m": float(c.y),
            "mean_R": float(np.dot(w, sub["mean_R"].to_numpy(dtype=float))),
            "mean_G": float(np.dot(w, sub["mean_G"].to_numpy(dtype=float))),
            "mean_B": float(np.dot(w, sub["mean_B"].to_numpy(dtype=float))),
        })
        geoms.append(hull)
    seeps = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms,
                             crs=bubbles.crs)
    return seeps


# --------------------------------------------------------------------------- #
# classify + flux
# --------------------------------------------------------------------------- #
def classify_seeps(seeps, clf):
    """Attach `class` and the per-class posterior columns."""
    X = seeps[CLASS_FEATURES].to_numpy(dtype=float)
    seeps = seeps.copy()
    seeps["class"] = clf.predict(X)
    proba = clf.predict_proba(X)
    for i, c in enumerate(clf.classes_):
        seeps[f"p_{c}"] = proba[:, i]
    return seeps


def attach_flux(seeps, season="annual"):
    """Per-seep rate in mg CH4/day, from its class. Flux is count-based, so
    this column is a lookup, not a function of the seep's size."""
    rate, _ = flux_rates.season_rates(season)
    seeps = seeps.copy()
    seeps["flux_mg_per_day"] = seeps["class"].map(rate).astype(float)
    return seeps


def flux_report(seeps, surveyed_area_m2=None, season="annual"):
    """The end-of-chain tables: per-season totals and a per-image breakdown."""
    counts = seeps["class"].value_counts().to_dict()
    table = flux_rates.flux_table(counts)
    if surveyed_area_m2:
        table["surveyed_area_m2"] = surveyed_area_m2
        table["mg_CH4_per_m2_per_day"] = (
            table["total_mg_CH4_per_day"] / surveyed_area_m2)
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
                "flux_mg_CH4_per_m2_per_day":
                    (flux / surveyed_area_m2) if surveyed_area_m2 else None,
                "seeps_per_m2":
                    (n / surveyed_area_m2) if surveyed_area_m2 else None,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(bubbles, out_dir, labeling_dir=None, thr=GROUP_THR, cap=AGGLOM_CAP_M,
        season="annual", surveyed_area_m2=None, upstream=None, progress=True,
        source=None, label=None):
    """Group -> dissolve -> classify -> flux. Returns (seeps, table, per_image).

    `upstream` is the run metadata from whichever loader produced `bubbles`.
    It supplies the surveyed area unless `surveyed_area_m2` overrides it, and
    it is merged into the metadata stamped onto the outputs, so seeps.gpkg
    records the checkpoint and lake polygon it ultimately came from.
    """
    from tools.grouping.deploy_grouper import train_model

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

    grouper = train_model()
    bubbles = bubbles.reset_index(drop=True)
    bubbles["seep_group_id"] = group_bubbles(grouper, bubbles, thr=thr, cap=cap,
                                             progress=progress)

    seeps = dissolve_bubbles_to_seeps(bubbles, progress=progress)

    classifier, clf_info = fit_deploy_model(labeling_dir)
    seeps = attach_flux(classify_seeps(seeps, classifier), season=season)
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
    totals = lake_totals_long(seeps, table, surveyed_area_m2, label=label,
                              upstream=upstream, run_id=run_id)
    totals_fp = os.path.join(out_dir, f"lake_flux_totals_{run_id}.csv")
    totals.to_csv(totals_fp, index=False)

    # Upstream first so the stage-B keys win on any collision.
    info = {**upstream, **{
        "stage": "postproc",
        "source": source,
        "group_threshold": thr,
        "agglom_cap_m": cap,
        "season": season,
        "n_bubbles": int(len(bubbles)),
        "n_seeps": int(len(seeps)),
        "class_counts": {c: int((seeps["class"] == c).sum()) for c in CLASSES},
        "surveyed_area_m2": surveyed_area_m2,
        "classifier": clf_info,
        "run_id": run_id,
        "runtime_s": round(time.time() - t0, 1),
    }}
    for fp in (bubbles_out, seeps_out):
        runinfo.write_run_info(fp, info)
    with open(os.path.join(out_dir, "run_info_postproc.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)

    _print_report(table, per_image, surveyed_area_m2)
    print(f"[postproc] wrote {seeps_out} (+ seeps.csv, flux_summary.csv, "
          f"flux_per_image.csv)")
    print(f"[postproc] wrote {totals_fp}")
    return seeps, table, per_image


def _print_report(table, per_image, surveyed_area_m2):
    print("\n" + "=" * 72)
    print("COUNT-BASED FLUX  (sum over classes of seep count x per-class rate)")
    print("=" * 72)
    cols = ["season", "n_A", "n_B", "n_C", "n_seeps", "total_mg_CH4_per_day",
            "rate_std_err_pct", "pct_flux_from_C"]
    if surveyed_area_m2:
        cols.append("mg_CH4_per_m2_per_day")
    print(table[cols].to_string(index=False,
                                float_format=lambda v: f"{v:,.3f}"))
    print("\nper image:")
    print(per_image.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    print("\nrate_std_err_pct is the PUBLISHED RATE uncertainty only. Detector, "
          "grouper and\nclassifier error are not in it -- those need the Monte "
          "Carlo chain, which is not\nwritten yet. Quote this as a point "
          "estimate, with the surveyed area named.")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--bubbles", help="bubbles.gpkg from tools.deploy.detect")
    src.add_argument("--from-pred-dir",
                     help="a canonical prediction directory "
                          "(bubble_features.csv + {stem}_cc.tif per chip)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--labeling-dir", default=None,
                    help="the three final labeler packs (classifier training)")
    ap.add_argument("--thr", type=float, default=GROUP_THR)
    ap.add_argument("--cap", type=float, default=AGGLOM_CAP_M)
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
        source=source)


if __name__ == "__main__":
    main()