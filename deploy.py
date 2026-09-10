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
    grp.add_argument("--labeling-dir", default=None,
                     help="the three labeler packs the classifier is fit on")

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
            upstream=info or upstream, source=source, label=args.label)

    print(f"\n[deploy] done -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()