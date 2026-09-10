# tools/deploy/run_lake.py
"""The composed end-to-end runner: orthomosaic -> lake methane flux.

    python -m tools.deploy.run_lake \
        --image data/training_images/AE/Octopus_10212025_..._cog_rectified.tif \
        --lake-polygon data/training/AE/lakes.gpkg:octopus \
        --checkpoint data/models/SWIN/AE/20260428-1537_SWINxAE_continued/20260428-1537_SWINxAE.weights.pt \
        --out-dir data/results/SWIN/AE/deploy/octopus_10212025

`--stage detect` needs torch and a GPU; `--stage postproc` needs neither, so
the usual split is detect on HPC, postproc wherever you are reading the number.
`--stage all` (the default) does both in one process.

The output is a point estimate with the published per-class rate uncertainty
attached, over a named surveyed area. It is not an end-to-end validated number:
each stage was validated in isolation, and detector, grouper and classifier
error all sit outside the reported interval.

The three things most likely to make the total wrong, in order:
  1. no lake polygon -- shore and snow are detected and classified like ice,
     and they do not all land in the lowest-rate class;
  2. a checkpoint / config mismatch, which loads partially and reports
     plausible nonsense (detect.load_detector refuses on key and shape
     mismatches, but `preprocessed_dir` discipline is still on you);
  3. quoting a total without its surveyed area.
"""
from __future__ import annotations

import argparse
import os

from tools.deploy import postproc as postproc_mod
from tools.deploy import tiles as tiles_mod
from tools.flux import rates as flux_rates
from tools.grouping.train_grouper import AGGLOM_CAP_M


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="whole-lake orthomosaic (detect stage)")
    ap.add_argument("--checkpoint", help="detector weights (detect stage)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage", default="all",
                    choices=["all", "detect", "postproc"])
    ap.add_argument("--lake-polygon", default=None,
                    help="path[:layer] of the lake outline. Inference is "
                         "cropped to it; without one every shore detection is "
                         "a false positive by construction.")
    ap.add_argument("--lake-layer", default=None)
    ap.add_argument("--tile-m", type=float, default=tiles_mod.DEFAULT_TILE_M,
                    help="normalization window, NOT a chunking size -- it "
                         "changes the detections (see tools/deploy/tiles.py)")
    ap.add_argument("--halo-m", type=float, default=tiles_mod.DEFAULT_HALO_M)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--mc-samples", type=int, default=None)
    ap.add_argument("--limit-tiles", type=int, default=None)
    ap.add_argument("--thr", type=float, default=postproc_mod.GROUP_THR,
                    help="grouper P(same) edge threshold")
    ap.add_argument("--cap", type=float, default=AGGLOM_CAP_M,
                    help="max seep centroid span (m)")
    ap.add_argument("--labeling-dir", default=None)
    ap.add_argument("--season", default="annual",
                    choices=sorted(flux_rates.SEASONS))
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    info = None

    if args.stage in ("all", "detect"):
        if not args.image or not args.checkpoint:
            ap.error("--image and --checkpoint are required for the detect stage")
        # Imported here, not at module scope: postproc-only runs must not need
        # torch, and this box does not have it.
        from tools.deploy.detect import detect_lake
        from config import configSwinUnet

        config = configSwinUnet.Configuration()
        info = detect_lake(
            args.image, args.checkpoint, config, args.out_dir,
            lake_polygon=args.lake_polygon, lake_layer=args.lake_layer,
            tile_m=args.tile_m, halo_m=args.halo_m,
            threshold=args.threshold, mc_samples=args.mc_samples,
            limit_tiles=args.limit_tiles)

    if args.stage in ("all", "postproc"):
        bubbles_fp = os.path.join(args.out_dir, "bubbles.gpkg")
        if not os.path.exists(bubbles_fp):
            raise SystemExit(
                f"{bubbles_fp} not found -- run --stage detect first (on a "
                "machine with a GPU), or point --out-dir at a finished detect run")
        # Detect returns its metadata in memory; a standalone postproc run gets
        # it back out of the gpkg, which is where it was written to live.
        bubbles, upstream = postproc_mod.load_bubbles(bubbles_fp)
        postproc_mod.run(
            bubbles, args.out_dir, labeling_dir=args.labeling_dir,
            thr=args.thr, cap=args.cap, season=args.season,
            upstream=info or upstream, source=os.path.abspath(bubbles_fp))


if __name__ == "__main__":
    main()