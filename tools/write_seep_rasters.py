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
from skimage.morphology import binary_closing, binary_dilation, binary_opening, disk
from tqdm import tqdm

# Auxiliary outputs that should be skipped when walking a prediction directory.
_AUX_SUFFIXES = ("_prob.tif", "_epistemic.tif", "_aleatoric.tif",
                 "_smoothed.tif", "_cc.tif", "_snow.tif")


def smooth_pred(mask, radius=1):
    """Closing+opening with disk(radius). Removes per-pixel boundary noise."""
    m = mask.astype(bool)
    m = binary_closing(m, disk(radius))
    m = binary_opening(m, disk(radius))
    return m.astype(np.uint8)


def snow_mask_hsv(image, v_thresh=0.85, s_thresh=0.15, dilate_px=0):
    """Boolean snow mask from RGB image: high V (brightness) AND low S
    (saturation) in HSV. Optionally dilate by `dilate_px` to grow the mask
    into snow-fringe pixels.

    image: (H, W, C) with the first 3 channels treated as R, G, B. Integer
    dtypes are normalized by their dtype max; float inputs assumed already in
    [0, 1] unless data exceeds 1.5 (then normalized by observed max).

    Returns (H, W) bool, or None if `image` lacks 3+ channels.
    """
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        return None
    rgb = image[..., :3]
    if np.issubdtype(rgb.dtype, np.integer):
        scale = float(np.iinfo(rgb.dtype).max)
    else:
        m = float(rgb.max()) if rgb.size else 1.0
        scale = 1.0 if m <= 1.5 else m
    rgb = rgb.astype(np.float32) / max(scale, 1e-9)
    cmax = rgb.max(axis=2)
    cmin = rgb.min(axis=2)
    v = cmax
    s = np.where(cmax > 1e-9, (cmax - cmin) / np.maximum(cmax, 1e-9), 0.0)
    mask = (v >= float(v_thresh)) & (s <= float(s_thresh))
    if dilate_px and int(dilate_px) > 0:
        mask = binary_dilation(mask, disk(int(dilate_px)))
    return mask


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


def write_rasters_for_pred(pred_fp, smoothed=None, cc=None, profile=None,
                           out_dir=None):
    """
    Write {stem}_smoothed.tif and {stem}_cc.tif. By default writes next to
    pred_fp; pass `out_dir` to redirect to a separate directory (caller is
    responsible for ensuring it exists).

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

    target_dir = out_dir if out_dir is not None else os.path.dirname(pred_fp)
    stem = os.path.splitext(os.path.basename(pred_fp))[0]
    _write_raster(os.path.join(target_dir, f"{stem}_smoothed.tif"),
                  smoothed, profile, dtype="uint8")
    cc_dtype = "uint16" if int(cc.max()) <= 65535 else "uint32"
    _write_raster(os.path.join(target_dir, f"{stem}_cc.tif"),
                  cc, profile, dtype=cc_dtype, nodata=0)


def write_snow_raster(pred_fp, snow, profile, out_dir=None):
    """Write {stem}_snow.tif (uint8 0/1) for QGIS overlay against the chip.
    Pass `out_dir` to redirect away from the prediction's directory."""
    if snow is None or profile is None:
        return
    target_dir = out_dir if out_dir is not None else os.path.dirname(pred_fp)
    stem = os.path.splitext(os.path.basename(pred_fp))[0]
    _write_raster(os.path.join(target_dir, f"{stem}_snow.tif"),
                  snow.astype(np.uint8), profile, dtype="uint8")


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