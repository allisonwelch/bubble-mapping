# tools/write_seep_rasters.py
"""
Write seep post-processing rasters (_smoothed.tif and _cc.tif) for each prediction
in a directory. Standalone-runnable, also importable from seep_level_eval.py.

Smoothing: morphological closing + opening with disk(1) per CLAUDE.md 2026-04-28.
"""
import os, glob
import numpy as np
import rasterio
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, disk
from tqdm import tqdm

# Auxiliary outputs that should be skipped when walking a prediction directory.
_AUX_SUFFIXES = ("_prob.tif", "_epistemic.tif", "_aleatoric.tif",
                 "_smoothed.tif", "_cc.tif")


def smooth_pred(mask, radius=1):
    """Closing+opening with disk(radius). Removes per-pixel boundary noise."""
    m = mask.astype(bool)
    m = binary_closing(m, disk(radius))
    m = binary_opening(m, disk(radius))
    return m.astype(np.uint8)


def cc_label(mask):
    """8-connectivity connected components."""
    return label(mask, connectivity=2)


def _write_raster(path, arr, base_profile, dtype, nodata=None):
    prof = base_profile.copy()
    prof.update(driver="GTiff", count=1, dtype=dtype, compress="lzw")
    if nodata is not None:
        prof["nodata"] = nodata
    else:
        prof.pop("nodata", None)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


def write_rasters_for_pred(pred_fp, smoothed=None, cc=None, profile=None):
    """
    Write {stem}_smoothed.tif and {stem}_cc.tif next to pred_fp.

    If smoothed / cc / profile are supplied, they're reused — lets a caller that
    already computed them (e.g. seep_level_eval.main) avoid re-reading pred_fp.
    """
    if smoothed is None or profile is None:
        with rasterio.open(pred_fp) as src:
            if smoothed is None:
                smoothed = smooth_pred(src.read(1))
            if profile is None:
                profile = src.profile.copy()
    if cc is None:
        cc = cc_label(smoothed)

    out_dir = os.path.dirname(pred_fp)
    stem = os.path.splitext(os.path.basename(pred_fp))[0]
    _write_raster(os.path.join(out_dir, f"{stem}_smoothed.tif"),
                  smoothed, profile, dtype="uint8")
    cc_dtype = "uint16" if int(cc.max()) <= 65535 else "uint32"
    _write_raster(os.path.join(out_dir, f"{stem}_cc.tif"),
                  cc, profile, dtype=cc_dtype, nodata=0)


def write_rasters_for_dir(pred_dir):
    pred_fps = [
        fp for fp in sorted(glob.glob(os.path.join(pred_dir, "*.tif")))
        if not fp.endswith(_AUX_SUFFIXES)
    ]
    for pred_fp in tqdm(pred_fps, desc="Writing seep rasters"):
        write_rasters_for_pred(pred_fp)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import configSwinUnet
    config = configSwinUnet.Configuration().validate()
    ckpt_pred_dir = os.path.join(config.results_dir, "20260428-1537_SWINxAE.weights")
    write_rasters_for_dir(ckpt_pred_dir)