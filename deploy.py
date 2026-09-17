#!/usr/bin/env python
"""Whole-lake deployment: orthomosaic -> classified seeps -> methane flux.

The single entry point for the deploy chain.

    python deploy.py --image ORTHO.tif --lake LAKE.gpkg:layer --out-dir OUT

Stages, in order (tools/deploy/):
    tiles     grid the lake into detector-sized tiles inside the lake polygon
    detect    sliding-window inference -> smooth -> connected components   [GPU]
    postproc  learned grouper -> seeps -> A/B/C classifier -> count-based flux

`--stage` runs only part of it. `detect` needs a GPU; `postproc` needs neither
a GPU nor torch, so a finished detect run can be re-analysed on any machine
against its own bubbles.gpkg.

    python deploy.py --out-dir OUT --stage postproc          # re-analyse
    python deploy.py --from-pred-dir DIR --out-dir OUT       # chips, not a lake

OUTPUTS in --out-dir
    bubbles.gpkg           detected bubbles + run metadata     (detect)
    tiles.gpkg             the tile grid, for QGIS             (detect)
    seeps.gpkg             hulls, class, per-seep rate         (postproc)
    seeps.csv              the same table without geometry
    lake_flux_totals_<YYYYmmdd-HHMMSS>.csv
                           per-run flux totals, long (one row per season x
                           class). Timestamped, so re-running adds a file
                           rather than overwriting the last answer.
    flux_summary.csv       per-season totals and uncertainty terms
    flux_per_image.csv     per-source breakdown

WHAT THE NUMBER IS
A point estimate with the published per-class rate uncertainty, over a measured
surveyed area. Every stage of the chain was validated in isolation; detector,
grouper and classifier error are NOT in the interval. Treat it accordingly.
"""
from __future__ import annotations

import argparse
import os
import sys

from tools.deploy import build_artifacts
from tools.deploy import postproc as postproc_mod
from tools.deploy import tiles as tiles_mod
from tools.flux import rates as flux_rates
from tools.grouping.train_grouper import AGGLOM_CAP_M
from tools.paths import CANONICAL_CHECKPOINT_RELPATH


def build_parser():
    ap = argparse.ArgumentParser(
        prog="deploy.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    src = ap.add_argument_group("input")
    src.add_argument("--image", help="whole-lake orthomosaic (detect stage)")
    src.add_argument("--lake", "--lake-polygon", dest="lake", default=None,
                     help="lake outline as PATH or PATH:LAYER. Inference is "
                          "cropped to it. Without one, shore and snow-covered "
                          "bank are detected too, and they do not all classify "
                          "into the lowest-rate class -- the total reads high.")
    src.add_argument("--from-pred-dir", default=None,
                     help="skip detection; run the rest over an existing "
                          "prediction directory of chips")
    src.add_argument("--out-dir", required=True)
    src.add_argument("--label", default=None,
                     help="name for this run in the flux-totals CSV "
                          "(defaults to the image filename)")

    mdl = ap.add_argument_group("model")
    mdl.add_argument("--checkpoint", default=CANONICAL_CHECKPOINT_RELPATH)
    mdl.add_argument("--config", default="config.configSwinUnet",
                     help="module supplying architecture + eval settings. Only "
                          "model construction and eval knobs are read; all "
                          "paths come from the arguments above.")
    mdl.add_argument("--threshold", type=float, default=None,
                     help="pixel probability cut (default: config.eval_threshold)")
    mdl.add_argument("--mc-samples", type=int, default=None,
                     help="override config.eval_mc_samples; lower is "
                          "proportionally faster but no longer matches the "
                          "config the reference predictions were made under")

    grd = ap.add_argument_group("tiling")
    grd.add_argument("--tile-m", type=float, default=tiles_mod.DEFAULT_TILE_M,
                     help="normalization window in metres. NOT a chunk size -- "
                          "it changes the detections. See tools/deploy/tiles.py")
    grd.add_argument("--halo-m", type=float, default=tiles_mod.DEFAULT_HALO_M)
    grd.add_argument("--limit-tiles", type=int, default=None,
                     help="stop after N tiles, for a smoke test")

    grp = ap.add_argument_group("grouping + flux")
    # Both defaults are physical priors from the historical field workbooks
    # (candidate radius = p95 of seep major axis, cap = p99), not fitted knobs,
    # and they are versioned with the checkpoint. Because flux is count-based,
    # moving them moves the headline directly -- so they are exposed for
    # SENSITIVITY ANALYSIS, and for re-selecting an operating point after a
    # grouper retrain. Do not adjust them until a lake total looks right: that
    # is fitting a free parameter to the answer you are using it to test.
    grp.add_argument("--thr", type=float, default=postproc_mod.GROUP_THR,
                     help="grouper P(same seep) edge threshold (default %(default)s)")
    grp.add_argument("--cap", type=float, default=AGGLOM_CAP_M,
                     help="max seep centroid span in metres (default %(default)s)")
    grp.add_argument("--season", default="annual",
                     choices=sorted(flux_rates.SEASONS))
    # The grouper and classifier are FROZEN ARTIFACTS, loaded from disk beside
    # the checkpoint. A deploy run must not train: refitting from the packs on
    # every run makes deployment depend on the training data being present, lets
    # a pack edit silently change the deployed model, and makes an old number
    # irreproducible. Build them with `python -m tools.deploy.build_artifacts`.
    grp.add_argument("--artifacts-dir", default=None,
                     help="where grouper_rf.joblib / classifier_rf.joblib live "
                          f"(default: {build_artifacts.default_out_dir()})")
    grp.add_argument("--refit", action="store_true",
                     help="re-fit both forests from the labeler packs instead "
                          "of loading the frozen artifacts. Needs the packs "
                          "present; makes the run unreproducible.")
    grp.add_argument("--seed", type=int, default=42,
                     help="only used with --refit")
    grp.add_argument("--labeling-dir", default=None,
                     help="the three labeler packs the classifier is fit on "
                          "(only used with --refit)")

    # ---------------------------------------------------------------- #
    # CRACK SCREEN -- ice cracks detected as one long connected component.
    # ---------------------------------------------------------------- #
    # All three gates must fire for a component to be dropped, and all three
    # defaults come from the 2429 hand-measured field seeps, not from a lake
    # total. See tools/deploy/postproc.py for why elongation is measured
    # separately from roughness.
    scr = ap.add_argument_group("crack screen")
    scr.add_argument("--max-span-m", type=float,
                     default=postproc_mod.SCREEN_MAX_SPAN_M,
                     help="minimum rotated rectangle major axis, metres "
                          "(default %(default)s, the largest ever measured). "
                          "Pass 0 to disable the screen entirely.")
    scr.add_argument("--min-aspect", type=float,
                     default=postproc_mod.SCREEN_MIN_ASPECT,
                     help="major/minor of that rectangle (default %(default)s; "
                          "0.5%% of field seeps reach it)")
    scr.add_argument("--min-shape", type=float,
                     default=postproc_mod.SCREEN_MIN_SHAPE,
                     help="perimeter^2/(4 pi area), a roughness measure "
                          "(default %(default)s; a circle is 1, real bubbles "
                          "~2). Elongation is --min-aspect, not this.")

    # ---------------------------------------------------------------- #
    # DECISION RULE -- how a class posterior becomes an A/B/C label.
    # ---------------------------------------------------------------- #
    # Separated from the model on purpose: the forest emits a posterior, and
    # turning that into one label is an independent choice that moves the
    # reported flux without any refitting or new labels.
    #
    #   argmax        highest-posterior class. Bayes-optimal under 0/1 loss,
    #                 which treats an A-called-C exactly like a C-called-A.
    #                 Flux does not -- those are 16 vs 971 mg CH4/day. This is
    #                 the default and the regime every recorded number is in.
    #   conservative  minimum expected cost over the ORDERED classes, with
    #                 over-calling penalised, so the model errs toward the
    #                 SMALLER class. Measured at penalty 1.5 on the LOIO eval:
    #                 accuracy 0.837 -> 0.856, over-call bias 1.93:1 -> 1.17:1,
    #                 classifier flux error +11.9% -> +0.9%, at the cost of C
    #                 recall 0.563 -> 0.514. Numbers in SECRET_CLAUDE.md §3.
    #
    # Both rules write p_A / p_B / p_C to seeps.gpkg, so a finished run can be
    # re-decided without re-running anything.
    dec = ap.add_argument_group("decision rule")
    dec.add_argument("--decision-rule", default=postproc_mod.DEFAULT_DECISION_RULE,
                     choices=postproc_mod.DECISION_RULES,
                     help="posterior -> A/B/C label. 'argmax' (default) is the "
                          "regime every recorded metric is in; 'conservative' "
                          "errs toward the smaller class.")
    dec.add_argument("--overcall-penalty", type=float,
                     default=postproc_mod.DEFAULT_OVERCALL_PENALTY,
                     help="how much worse a promotion is than a demotion. Only "
                          "read by --decision-rule conservative. 1.0 is "
                          "argmax-like; above ~2 the model stops predicting C "
                          "(default %(default)s)")

    ap.add_argument("--stage", default="all",
                    choices=["all", "detect", "postproc"])
    return ap


def run_detect(args):
    """Import torch only here, so a postproc-only run never needs it."""
    import importlib
    from tools.deploy.detect import detect_lake

    config = importlib.import_module(args.config).Configuration()
    # configSwinUnet sets CUDA_VISIBLE_DEVICES from selected_GPU on construction.
    # Under SLURM the allocated GPU is already the only visible one and is always
    # index 0, so anything else asks for a device the job does not have. Report
    # what we ended up with rather than letting it be silently wrong.
    print(f"[deploy] CUDA_VISIBLE_DEVICES="
          f"{os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
          f"(config.selected_GPU={getattr(config, 'selected_GPU', None)})")
    return detect_lake(
        args.image, args.checkpoint, config, args.out_dir,
        lake_polygon=args.lake, tile_m=args.tile_m, halo_m=args.halo_m,
        threshold=args.threshold, mc_samples=args.mc_samples,
        limit_tiles=args.limit_tiles)


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)

    if args.from_pred_dir:
        args.stage = "postproc"
    elif args.stage in ("all", "detect"):
        if not args.image:
            ap.error("--image is required for the detect stage "
                     "(or use --from-pred-dir / --stage postproc)")
        if not os.path.exists(args.checkpoint):
            ap.error(f"checkpoint not found: {args.checkpoint}")
        if not args.lake:
            print("[deploy] WARNING: no --lake polygon; inference will cover "
                  "the whole raster footprint, shore included.", file=sys.stderr)

    os.makedirs(args.out_dir, exist_ok=True)
    info = None

    if args.stage in ("all", "detect"):
        info = run_detect(args)

    if args.stage in ("all", "postproc"):
        if args.from_pred_dir:
            bubbles, upstream = postproc_mod.load_from_pred_dir(args.from_pred_dir)
            source = os.path.abspath(args.from_pred_dir)
        else:
            fp = os.path.join(args.out_dir, "bubbles.gpkg")
            if not os.path.exists(fp):
                raise SystemExit(
                    f"{fp} not found -- run --stage detect first (needs a GPU), "
                    "or point --out-dir at a finished detect run")
            bubbles, upstream = postproc_mod.load_bubbles(fp)
            source = os.path.abspath(fp)
        postproc_mod.run(
            bubbles, args.out_dir, labeling_dir=args.labeling_dir,
            thr=args.thr, cap=args.cap, season=args.season,
            upstream=info or upstream, source=source, label=args.label,
            artifacts_dir=args.artifacts_dir, refit=args.refit, seed=args.seed,
            decision_rule=args.decision_rule,
            overcall_penalty=args.overcall_penalty,
            max_span_m=args.max_span_m or None, min_aspect=args.min_aspect,
            min_shape=args.min_shape)

    print(f"\n[deploy] done -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()