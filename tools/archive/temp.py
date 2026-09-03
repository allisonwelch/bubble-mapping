"""
Make a contact-sheet visualization from a folder of GeoTIFF/TIFF files.

For each TIFF:
- read the first 3 bands
- reorder them to (2,1,0)  [i.e., band3, band2, band1]
- contrast-stretch each channel (percentile stretch)
- read the last band and binarize it (mask > threshold)

Outputs:
1) mosaic_rgb.png   : RGB thumbnails (optionally with mask overlay)
2) mosaic_mask.png  : binary mask thumbnails
3) mosaic_combined.png (optional): RGB mosaic stacked above mask mosaic

Dependencies:
  pip install rasterio numpy pillow
"""

from __future__ import annotations

import math
from pathlib import Path
from tqdm import tqdm

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont


def percentile_stretch_to_uint8(img: np.ndarray, p_low=2.0, p_high=98.0) -> np.ndarray:
    """
    img: (H, W, C) float/uint array
    returns: uint8 (H, W, C) in [0,255]
    """
    img = img.astype(np.float32, copy=False)

    out = np.empty_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        ch = img[:, :, c]
        lo = np.nanpercentile(ch, p_low)
        hi = np.nanpercentile(ch, p_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            # fallback: min/max
            lo = np.nanmin(ch)
            hi = np.nanmax(ch)
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                out[:, :, c] = 0.0
                continue

        out[:, :, c] = (ch - lo) / (hi - lo)

    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def choose_grid(n: int) -> tuple[int, int]:
    """Pick cols/rows to get a near-square mosaic."""
    if n <= 0:
        return 0, 0
    root = int(round(math.sqrt(n)))
    candidates = range(max(1, root - 8), root + 9)
    best = None
    for cols in candidates:
        rows = math.ceil(n / cols)
        # score: how close to square, then prefer fewer rows
        score = abs(cols - rows) * 10 + rows
        if best is None or score < best[0]:
            best = (score, cols, rows)
    _, cols, rows = best
    return cols, rows


def to_pil_rgb(arr_uint8_hwc: np.ndarray) -> Image.Image:
    return Image.fromarray(arr_uint8_hwc, mode="RGB")


def make_contact_sheet(
    tiles: list[Image.Image],
    labels: list[str],
    out_path: Path,
    tile_size: int = 256,
    padding: int = 8,
    label_height: int = 18,
    bg=(20, 20, 20),
    label_color=(230, 230, 230),
):
    n = len(tiles)
    cols, rows = choose_grid(n)

    # use a small default font (portable)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    sheet_w = padding + cols * (tile_size + padding)
    sheet_h = padding + rows * (tile_size + label_height + padding)

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=bg)
    draw = ImageDraw.Draw(sheet)

    for i, (tile, lab) in enumerate(zip(tiles, labels)):
        r = i // cols
        c = i % cols
        x = padding + c * (tile_size + padding)
        y = padding + r * (tile_size + label_height + padding)

        sheet.paste(tile, (x, y))

        # label below tile
        ly = y + tile_size + 2
        draw.text((x, ly), lab, fill=label_color, font=font)

    sheet.save(out_path)
    return out_path


def overlay_mask_on_rgb(rgb_img: Image.Image, mask_img: Image.Image, alpha: int = 90) -> Image.Image:
    """
    Overlay binary mask as a semi-transparent red layer on RGB.
    mask_img: mode 'L' where 0/255
    """
    rgb = rgb_img.convert("RGBA")
    mask = mask_img.convert("L")

    # red overlay where mask is on
    red = Image.new("RGBA", rgb.size, (255, 0, 0, alpha))
    # use mask as alpha selector
    rgb = Image.composite(red, rgb, mask).convert("RGB")
    return rgb


def build_mosaics(
    folder: str | Path,
    out_rgb: str = "mosaic_rgb.png",
    out_mask: str = "mosaic_mask.png",
    out_combined: str = "mosaic_combined.png",
    tile_size: int = 256,
    p_low: float = 2.0,
    p_high: float = 98.0,
    mask_threshold: float = 0.0,
    overlay_mask: bool = True,
):
    folder = Path(folder)
    files = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    if not files:
        raise FileNotFoundError(f"No .tif/.tiff files found in: {folder}")

    rgb_tiles: list[Image.Image] = []
    mask_tiles: list[Image.Image] = []
    labels: list[str] = []

    for fp in tqdm(files):
        with rasterio.open(fp) as ds:
            if ds.count < 3:
                raise ValueError(f"{fp.name} has only {ds.count} band(s); need at least 3.")

            # First 3 bands (rasterio is 1-indexed): [1,2,3] then reorder to (3,2,1)
            b = ds.read([1, 2, 3]).astype(np.float32)  # (3,H,W)
            b = b[[2, 1, 0], :, :]  # reorder to (2,1,0)
            rgb = np.transpose(b, (1, 2, 0))  # (H,W,3)

            rgb_u8 = percentile_stretch_to_uint8(rgb, p_low=p_low, p_high=p_high)
            rgb_pil = to_pil_rgb(rgb_u8).resize((tile_size, tile_size), Image.Resampling.BILINEAR)

            # last band -> binary
            last = ds.read(ds.count)
            mask = (last > mask_threshold).astype(np.uint8) * 255
            mask_pil = Image.fromarray(mask, mode="L").resize((tile_size, tile_size), Image.Resampling.NEAREST)

            # optional overlay (still "RGB", but shows binary mask)
            if overlay_mask:
                rgb_pil = overlay_mask_on_rgb(rgb_pil, mask_pil, alpha=90)

            # for the mask contact sheet, keep it RGB so everything displays consistently
            mask_rgb = Image.merge("RGB", (mask_pil, mask_pil, mask_pil))

        rgb_tiles.append(rgb_pil)
        mask_tiles.append(mask_rgb)
        labels.append(fp.stem)

    out_rgb_path = folder / out_rgb
    out_mask_path = folder / out_mask

    make_contact_sheet(rgb_tiles, labels, out_rgb_path, tile_size=tile_size)
    make_contact_sheet(mask_tiles, labels, out_mask_path, tile_size=tile_size)

    # combined (RGB sheet stacked above mask sheet)
    rgb_sheet = Image.open(out_rgb_path).convert("RGB")
    mask_sheet = Image.open(out_mask_path).convert("RGB")
    gap = 16
    combined = Image.new("RGB", (max(rgb_sheet.width, mask_sheet.width), rgb_sheet.height + gap + mask_sheet.height), (20, 20, 20))
    combined.paste(rgb_sheet, (0, 0))
    combined.paste(mask_sheet, (0, rgb_sheet.height + gap))
    combined_path = folder / out_combined
    combined.save(combined_path)

    print("Saved:")
    print(" ", out_rgb_path)
    print(" ", out_mask_path)
    print(" ", combined_path)


if __name__ == "__main__":
    # EDIT THIS:
    build_mosaics(
        folder="/isipd/projects/p_planetdw/data/methods_test/preprocessed/20260109-1434_UNETxPS/",
        tile_size=256,         # thumbnails size in the mosaic
        p_low=2, p_high=98,    # stretch percentiles
        mask_threshold=0.0,    # binarize: last_band > threshold
        overlay_mask=True,     # show binary mask over RGB in the RGB mosaic
    )
