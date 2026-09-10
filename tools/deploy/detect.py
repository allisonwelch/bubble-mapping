# tools/deploy/detect.py
"""Stage A of the whole-lake runner: orthomosaic -> detected bubble polygons.

Needs torch and a GPU, so in practice it runs on HPC (`deploy_lake.slurm`). It
writes one thing the rest of the chain consumes, `bubbles.gpkg`, which the
CPU-only stage B (`tools.deploy.postproc`) then runs against anywhere.

The forward pass, sliding window, blending and MC-dropout averaging are
imported from `evaluation.py::_infer_full_image` rather than reimplemented, and
smoothing / connected components / per-bubble features come from
`tools.eval.*`. A second copy of any of those would be a second pipeline, and
`mean_R/G/B` in particular has to mean the same thing here as in the
classifier's training rows. All this module decides is what counts as a frame
(tools/deploy/tiles.py) and where the pixels come from.

The ortho is NOT expected to be pre-cropped. Two masks are applied to the
PREDICTION -- never to the imagery going in, so normalization statistics stay
those of the real scene:
  * pixels outside `--lake-polygon`, rasterized per tile;
  * pixels the ortho's alpha band marks invalid (outside the flight footprint).

`surveyed_area_m2` in run_info.json is the valid, in-lake area actually seen.
Carry it with any total: a count-based flux figure without its denominator is
not comparable to a field campaign, to another lake, or to another flight date.
"""
from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize as rio_rasterize
from tqdm import tqdm

from tools.deploy import runinfo
from tools.deploy import tiles as tiles_mod
from tools.eval.bubble_features import compute_bubble_features, polygonize_labels
from tools.eval.write_bubble_rasters import cc_label, smooth_pred

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


def load_detector(config, checkpoint: str):
    """Build the architecture from `config` and load `checkpoint` into it.

    Routed through evaluation.py's own builders so a checkpoint loads exactly
    as it does under evaluation -- except for its silent-failure mode.
    `_load_model_from_checkpoint` falls back from strict to non-strict on a key
    or shape mismatch, which loads a mismatched checkpoint partially and
    returns plausible nonsense. We check strictness first and refuse, because a
    runner that reports a lake total from a half-loaded model is worse than one
    that stops.
    """
    import torch  # local: stage B must import this module-free of torch
    from evaluation import (_build_swin, _build_unet,
                            _load_model_from_checkpoint)

    arch = str(getattr(config, "run_name", "")).upper()
    if "SWIN" in arch:
        model = _build_swin(config)
    elif "UNET" in arch:
        model = _build_unet(config)
    else:
        raise SystemExit(
            f"cannot infer architecture from config.run_name={config.run_name!r}; "
            "the runner supports the Swin and UNet families")

    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state["model_state"] if isinstance(state, dict) and \
        "model_state" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise SystemExit(
            f"checkpoint does not match the config's architecture:\n"
            f"  {len(missing)} missing keys, {len(unexpected)} unexpected\n"
            f"  first missing: {list(missing)[:3]}\n"
            f"  first unexpected: {list(unexpected)[:3]}\n"
            "evaluation.py would load this partially and report plausible but "
            "meaningless numbers. Check patch_size / channels_used / the swin_* "
            "family against the checkpoint's .metadata.json.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model_from_checkpoint(model, checkpoint, device)
    print(f"[detect] {os.path.basename(checkpoint)} -> {device}")
    return model, device


def _tile_valid_mask(alpha, lake_geom, transform, shape):
    """Boolean mask of pixels the detector is allowed to claim a bubble in."""
    valid = np.ones(shape, dtype=bool) if alpha is None else (alpha > 0)
    if lake_geom is not None:
        inside = rio_rasterize(
            [(lake_geom, 1)], out_shape=shape, transform=transform,
            fill=0, all_touched=True, dtype=np.uint8).astype(bool)
        valid &= inside
    return valid


def detect_lake(image_path, checkpoint, config, out_dir,
                lake_polygon=None, lake_layer=None,
                tile_m=tiles_mod.DEFAULT_TILE_M,
                halo_m=tiles_mod.DEFAULT_HALO_M,
                threshold=None, mc_samples=None,
                limit_tiles=None, progress=True):
    """Run the detector over the lake and write `bubbles.gpkg` + `run_info.json`.

    Returns the run_info dict.
    """
    if gpd is None:
        raise RuntimeError("geopandas is required")
    os.makedirs(out_dir, exist_ok=True)

    threshold = float(getattr(config, "eval_threshold", 0.5)
                      if threshold is None else threshold)
    if mc_samples is not None:
        # MC dropout costs one forward pass per sample per patch. Lowering it
        # is a legitimate speed trade, but it no longer matches the config the
        # reference predictions were made under, so it goes in run_info.json.
        config.eval_mc_samples = int(mc_samples)
        config.eval_mc_dropout = int(mc_samples) > 1

    with rasterio.open(image_path) as src:
        crs, res = src.crs, float(abs(src.transform.a))
        has_alpha = src.count >= 4

    lake_geom = None
    if lake_polygon:
        lake_geom = tiles_mod.read_lake_geom(lake_polygon, lake_layer, crs)
    else:
        print("[detect] WARNING: no lake polygon. Inference will cover the whole "
              "raster footprint, so shore and snow-covered bank are included, "
              "and every detection there is a false positive by construction. "
              "The flux total from this run will read high.")

    patch_px = int(config.patch_size[0])
    tiles = tiles_mod.build_tiles(image_path, lake_geom, tile_m=tile_m,
                                  halo_m=halo_m, snap_px=patch_px)
    if limit_tiles:
        tiles = tiles[:int(limit_tiles)]
    grid = tiles_mod.summarize(tiles, tile_m, halo_m, res)
    print(f"[detect] {grid['n_tiles']} tiles of {grid['tile_px']} px "
          f"({grid['tile_px'] * res:.2f} m), normalization window "
          f"{grid['norm_window_px']} px ({grid['norm_window_m']:.2f} m), "
          f"{grid['n_edge_tiles']} touching the shore")
    tiles_mod.write_tiles_gpkg(tiles, crs, os.path.join(out_dir, "tiles.gpkg"))

    model, device = load_detector(config, checkpoint)
    from evaluation import _infer_full_image

    feat_rows, geom_rows = [], []
    surveyed_px = 0
    t0 = time.time()
    it = tqdm(tiles, desc="detect", disable=not progress)
    with rasterio.open(image_path) as src:
        for t in it:
            pw = tiles_mod.pad_window(t)
            img = src.read((1, 2, 3), window=pw).transpose(1, 2, 0)
            alpha = src.read(4, window=pw) if has_alpha else None
            pad_transform = src.window_transform(pw)

            valid = _tile_valid_mask(alpha, lake_geom, pad_transform,
                                     img.shape[:2])
            r0, c0 = tiles_mod.core_offset_in_pad(t)
            core = (slice(r0, r0 + t["core_height"]),
                    slice(c0, c0 + t["core_width"]))
            n_core_valid = int(valid[core].sum())
            if n_core_valid == 0:
                continue
            surveyed_px += n_core_valid

            frame = SimpleNamespace(img=img.astype(np.float32),
                                    annotations=np.zeros(img.shape[:2], np.uint8))
            out = _infer_full_image(model, frame, device, config)
            prob = out[0] if isinstance(out, tuple) else out

            pred = (prob >= threshold) & valid
            # Smooth and label on the PADDED array, then keep only components
            # whose centroid is in the core: a bubble on a seam is emitted
            # whole, by exactly one tile.
            cc = cc_label(smooth_pred(pred.astype(np.uint8)))
            if int(cc.max()) == 0:
                continue

            f = compute_bubble_features(cc, img, pad_transform)
            if f.empty:
                continue
            left, bottom, right, top = t["bounds"]
            in_core = ((f["centroid_x_m"] >= left) & (f["centroid_x_m"] < right)
                       & (f["centroid_y_m"] >= bottom) & (f["centroid_y_m"] < top))
            f = f[in_core]
            if f.empty:
                continue

            polys = polygonize_labels(cc, pad_transform, crs)
            polys = polys.rename(columns={"id": "bubble_id"})
            polys = polys[polys["bubble_id"].isin(f["bubble_id"])]

            f = f.assign(tile_id=t["tile_id"])
            feat_rows.append(f)
            geom_rows.append(polys.merge(f, on="bubble_id", how="inner"))

    if not feat_rows:
        raise SystemExit("no bubbles detected anywhere in the lake -- check the "
                         "lake polygon, the checkpoint and the threshold")

    bubbles = pd.concat(geom_rows, ignore_index=True)
    # bubble_id is per-tile out of the CC labeller; renumber to a single lake-wide
    # namespace so (image, bubble_id) is a key, per CLAUDE.md's ID convention.
    bubbles["bubble_id"] = np.arange(1, len(bubbles) + 1, dtype=np.int64)
    bubbles["image"] = os.path.basename(image_path)
    bubbles = gpd.GeoDataFrame(bubbles, geometry="geometry", crs=crs)

    out_gpkg = os.path.join(out_dir, "bubbles.gpkg")
    if os.path.exists(out_gpkg):
        os.remove(out_gpkg)
    bubbles.to_file(out_gpkg, layer="bubbles", driver="GPKG")
    bubbles.drop(columns="geometry").to_csv(
        os.path.join(out_dir, "bubble_features.csv"), index=False)

    surveyed_m2 = surveyed_px * res * res
    info = {
        "stage": "detect",
        "image": os.path.abspath(image_path),
        "checkpoint": os.path.abspath(checkpoint),
        "lake_polygon": os.path.abspath(lake_polygon) if lake_polygon else None,
        "crs": str(crs),
        "eval_threshold": threshold,
        "eval_mc_dropout": bool(getattr(config, "eval_mc_dropout", False)),
        "eval_mc_samples": int(getattr(config, "eval_mc_samples", 1)),
        "patch_size": list(config.patch_size),
        "eval_patch_stride": getattr(config, "eval_patch_stride", None),
        "grid": grid,
        "n_bubbles": int(len(bubbles)),
        "surveyed_area_m2": float(surveyed_m2),
        "bubbles_per_m2": float(len(bubbles) / surveyed_m2) if surveyed_m2 else None,
        "runtime_s": round(time.time() - t0, 1),
    }
    # Authoritative copy rides inside the gpkg, so the surveyed area cannot be
    # separated from the bubbles it belongs to. The json beside it is a
    # human-readable dump of the same dict; nothing reads it back.
    runinfo.write_run_info(out_gpkg, info)
    with open(os.path.join(out_dir, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)

    print(f"[detect] {len(bubbles)} bubbles over "
          f"{surveyed_m2:,.0f} m2 of valid in-lake ice "
          f"({info['bubbles_per_m2']:.2f} /m2) in {info['runtime_s']:.0f}s")
    print(f"[detect] wrote {out_gpkg}")
    return info


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="whole-lake orthomosaic")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lake-polygon", default=None)
    ap.add_argument("--lake-layer", default=None)
    ap.add_argument("--tile-m", type=float, default=tiles_mod.DEFAULT_TILE_M)
    ap.add_argument("--halo-m", type=float, default=tiles_mod.DEFAULT_HALO_M)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--mc-samples", type=int, default=None,
                    help="override config.eval_mc_samples; 1 disables MC dropout "
                         "(20x faster, but no longer the canonical detector)")
    ap.add_argument("--limit-tiles", type=int, default=None,
                    help="stop after N tiles -- for a smoke test")
    args = ap.parse_args(argv)

    from config import configSwinUnet
    config = configSwinUnet.Configuration()
    detect_lake(args.image, args.checkpoint, config, args.out_dir,
                lake_polygon=args.lake_polygon, lake_layer=args.lake_layer,
                tile_m=args.tile_m, halo_m=args.halo_m,
                threshold=args.threshold, mc_samples=args.mc_samples,
                limit_tiles=args.limit_tiles)


if __name__ == "__main__":
    main()