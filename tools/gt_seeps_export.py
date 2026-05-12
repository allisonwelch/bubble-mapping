# tools/gt_seeps_export.py
"""
Export per-chip ground-truth seeps to a single labelable vector layer.

Each test chip's last band is the rasterized GT label mask (drawn at the
seep level — one polygon per seep, per CLAUDE.md 2026-05-11). This script:
  1. Walks chip_dir (or the chips referenced by pred_dir).
  2. For each chip: label() the GT band → one CC per seep; compute
     polygon-level features (area, perim, circularity, solidity,
     eccentricity, mean_R/G/B) via seep_feature_table.compute_bubble_features.
  3. Polygonizes the same labels and merges features by id.
  4. Adds an empty `class` column for QGIS attribute-table editing.
  5. Writes ONE gt_seeps.gpkg containing all chips' seeps.

QGIS workflow:
  - Load gt_seeps.gpkg in QGIS, chip imagery as basemap underneath.
  - Toggle editing on the layer → open attribute table → fill `class`.
  - Save edits → ship the .gpkg back.

Output path defaults to pred_dir (so all eval artifacts live together);
override via --out_dir or out_dir=. CLI passes through to main(...).
"""
import os
import sys
import glob
import argparse
import numpy as np
import rasterio
from skimage.measure import label as cc_label
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seep_feature_table import (
    labels_to_seep_gdf,
    write_seeps_gpkg,
    _AUX_SUFFIXES,
)

try:
    import geopandas as gpd  # noqa: F401
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False


def _load_gt_and_image(chip_fp):
    """Read last band as GT mask, all but-last bands as the chip image."""
    with rasterio.open(chip_fp) as src:
        n = src.count
        gt = src.read(n)
        if n > 1:
            image = np.transpose(src.read(list(range(1, n))), (1, 2, 0))
        else:
            image = gt
        transform = src.transform
        crs = src.crs
    if gt.max() > 1.5:
        gt = (gt / 255.0)
    gt_bin = (gt >= 0.5).astype(np.uint8)
    return gt_bin, image, transform, crs


def build_gt_seeps_for_chip(chip_fp):
    """Return a GeoDataFrame of GT seep polygons for one chip, or None if
    the chip has no labels or no CRS."""
    if not _HAS_GPD:
        raise RuntimeError("geopandas is required")
    gt_bin, image, transform, crs = _load_gt_and_image(chip_fp)
    if crs is None:
        return None
    lab = cc_label(gt_bin, connectivity=2)
    if int(lab.max()) == 0:
        return None
    gdf = labels_to_seep_gdf(lab, image, transform, crs, id_name="seep_id")
    if gdf.empty:
        return None
    gdf.insert(0, "image", os.path.basename(chip_fp))
    return gdf


def main(chip_source_dir, out_dir, glob_pattern="*.tif"):
    """Walk chip_source_dir, build one GeoDataFrame per chip, concat, write.

    chip_source_dir can be either:
      - the chip directory (every *.tif in there is a chip), or
      - the pred directory (every non-aux *.tif has a matching chip filename
        in some other directory — caller should pass the actual chip dir).
    """
    if not _HAS_GPD:
        raise RuntimeError("geopandas is required")
    fps = [
        fp for fp in sorted(glob.glob(os.path.join(chip_source_dir, glob_pattern)))
        if not fp.endswith(_AUX_SUFFIXES)
        and not fp.endswith(("_seep_cluster.tif",))
        and "_r125_r45_r10_lonely_seep_cluster" not in fp
    ]
    gdfs = []
    for fp in tqdm(fps, desc="GT seep polygons"):
        g = build_gt_seeps_for_chip(fp)
        if g is not None:
            gdfs.append(g)
    if not gdfs:
        raise RuntimeError(
            f"No GT polygons extracted from {chip_source_dir}.\n"
            f"Check that the chips' last band is the rasterized GT mask."
        )
    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, "gt_seeps.gpkg")
    write_seeps_gpkg(out_fp, gdfs, class_column=True)
    total = sum(len(g) for g in gdfs)
    print(f"\nWrote {total} GT seeps from {len(gdfs)} chips to {out_fp}")
    print("Open in QGIS, toggle editing, fill the `class` column per row, save.")
    return out_fp


def _parse_args():
    p = argparse.ArgumentParser(description="Export GT seep polygons to GPKG")
    p.add_argument("--chip_dir", default=None,
                   help="Directory of chip .tif files (default: config.preprocessed_dir)")
    p.add_argument("--out_dir", default=None,
                   help="Where to write gt_seeps.gpkg (default: pred_dir)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import configSwinUnet
    config = configSwinUnet.Configuration().validate()
    chip_dir = args.chip_dir or config.preprocessed_dir
    out_dir = args.out_dir or os.path.join(
        config.results_dir, "20260428-1537_SWINxAE.weights"
    )
    main(chip_source_dir=chip_dir, out_dir=out_dir)