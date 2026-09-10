# tools/deploy/tiles.py
"""Grid a whole-lake orthomosaic into tiles inside the lake polygon.

The tile is the normalization window, not a chunking convenience. Training
normalizes each patch independently (`core/frame_info.py::getPatch`), while
`evaluation.py::_infer_full_image` normalizes a whole frame before sliding the
window over it -- and the frames it was built for are the chip-sized training
areas. Tiling the lake at the chip size keeps the detector in the regime it was
evaluated in, so `tile_m` changes the detections and is recorded in
run_info.json alongside the checkpoint.

Tiles carry a `halo_m` margin. Detection runs on the padded window and keeps
only components whose centroid lands in the core, so a bubble on a seam is
emitted whole by exactly one tile, and the sliding window sees real imagery at
the core edge instead of reflect padding.

Tiles are kept when their core INTERSECTS the lake polygon, so a tile that is
mostly shore is still processed and its in-lake part still contributes; the
shore part is masked out of the prediction downstream (see detect.py).
"""
from __future__ import annotations

import math
import os

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box as shp_box
from shapely.ops import unary_union

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None


DEFAULT_TILE_M = 15.0     # the training-area / chip edge length
DEFAULT_HALO_M = 0.5      # == CAND_RADIUS, so a seam bubble keeps its neighbours


def read_lake_geom(path: str, layer: str | None = None, target_crs=None):
    """Union of the lake polygon layer, reprojected to `target_crs`.

    `path` may carry a layer suffix: "lakes.gpkg:octopus".
    """
    if gpd is None:
        raise RuntimeError("geopandas is required to read a lake polygon")
    if layer is None and path.count(":") > 1:
        # "C:/x/lakes.gpkg:octopus" -- only the LAST colon is a layer separator
        head, _, tail = path.rpartition(":")
        if head and not tail.endswith((".gpkg", ".shp", ".geojson")):
            path, layer = head, tail
    g = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if g.empty:
        raise SystemExit(f"lake polygon layer is empty: {path}")
    if target_crs is not None and g.crs is not None and g.crs != target_crs:
        g = g.to_crs(target_crs)
    return unary_union(list(g.geometry.values))


def build_tiles(image_path: str, lake_geom=None,
                tile_m: float = DEFAULT_TILE_M,
                halo_m: float = DEFAULT_HALO_M,
                snap_px: int | None = None):
    """Grid the raster into tiles, keeping those that touch the lake polygon.

    `snap_px` (pass the detector's patch size) rounds the padded window to a
    whole number of patches and shrinks the core to match, so the sliding
    window tiles it exactly, `_infer_full_image` never reflect-pads, and no
    forward passes are spent on invented pixels.

    Returns a list of dicts, one per tile:
        tile_id                   "r0007c0012", stable and grid-derived
        core_*  / pad_*           pixel windows (col_off, row_off, width, height)
        bounds                    core extent in CRS units (left, bottom, right, top)
        touches_lake_edge         True when the core is not fully inside the lake
                                  polygon, i.e. shore is in frame for this tile

    The grid is anchored to the raster origin, so a rerun with the same
    tile_m/halo_m reproduces the same tile_ids and the same detections.
    """
    with rasterio.open(image_path) as src:
        res = float(abs(src.transform.a))
        W, H = src.width, src.height
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    tile_px = max(1, int(round(tile_m / res)))
    halo_px = int(round(halo_m / res))
    if snap_px:
        # NEAREST multiple, not the next one up: rounding up can widen the
        # normalization window well past `tile_m` at some pixel sizes. Nearest
        # keeps it within half a patch either way.
        pad_px = int(max(1, round(tile_px / snap_px)) * snap_px)
        tile_px = max(1, pad_px - 2 * halo_px)

    # Restrict the grid to the lake's bounding box; the per-tile intersection
    # test below does the exact filtering.
    if lake_geom is not None:
        lb = lake_geom.bounds
        col0 = max(0, int(math.floor((lb[0] - bounds.left) / res)))
        col1 = min(W, int(math.ceil((lb[2] - bounds.left) / res)))
        row0 = max(0, int(math.floor((bounds.top - lb[3]) / res)))
        row1 = min(H, int(math.ceil((bounds.top - lb[1]) / res)))
    else:
        col0, col1, row0, row1 = 0, W, 0, H

    tiles = []
    for row in range(row0 // tile_px, math.ceil(row1 / tile_px)):
        for col in range(col0 // tile_px, math.ceil(col1 / tile_px)):
            c_off, r_off = col * tile_px, row * tile_px
            c_w = min(tile_px, W - c_off)
            c_h = min(tile_px, H - r_off)
            if c_w <= 0 or c_h <= 0:
                continue
            left, top = transform * (c_off, r_off)
            right, bottom = transform * (c_off + c_w, r_off + c_h)
            core_box = shp_box(min(left, right), min(top, bottom),
                               max(left, right), max(top, bottom))
            if lake_geom is not None and not lake_geom.intersects(core_box):
                continue

            p_off_c = max(0, c_off - halo_px)
            p_off_r = max(0, r_off - halo_px)
            p_w = min(W, c_off + c_w + halo_px) - p_off_c
            p_h = min(H, r_off + c_h + halo_px) - p_off_r

            tiles.append({
                "tile_id": f"r{row:04d}c{col:04d}",
                "row": row, "col": col,
                "core_col_off": c_off, "core_row_off": r_off,
                "core_width": c_w, "core_height": c_h,
                "pad_col_off": p_off_c, "pad_row_off": p_off_r,
                "pad_width": p_w, "pad_height": p_h,
                "bounds": core_box.bounds,
                "touches_lake_edge": bool(
                    lake_geom is not None and not lake_geom.contains(core_box)),
            })
    if not tiles:
        raise SystemExit(
            "no tiles intersect the lake polygon -- check that the polygon and "
            f"the raster share a CRS (raster is {crs})")
    return tiles


def core_window(t) -> Window:
    return Window(t["core_col_off"], t["core_row_off"],
                  t["core_width"], t["core_height"])


def pad_window(t) -> Window:
    return Window(t["pad_col_off"], t["pad_row_off"],
                  t["pad_width"], t["pad_height"])


def core_offset_in_pad(t):
    """(row, col) of the core's top-left corner within the padded window."""
    return (t["core_row_off"] - t["pad_row_off"],
            t["core_col_off"] - t["pad_col_off"])


def tiles_to_gdf(tiles, crs):
    """Tile cores as polygons, for QGIS and for the run's provenance."""
    if gpd is None:
        raise RuntimeError("geopandas is required")
    geoms = [shp_box(*t["bounds"]) for t in tiles]
    cols = {k: [t[k] for t in tiles]
            for k in ("tile_id", "row", "col", "touches_lake_edge")}
    return gpd.GeoDataFrame(cols, geometry=geoms, crs=crs)


def write_tiles_gpkg(tiles, crs, out_fp):
    gdf = tiles_to_gdf(tiles, crs)
    if os.path.exists(out_fp):
        os.remove(out_fp)
    gdf.to_file(out_fp, layer="tiles", driver="GPKG")
    return out_fp


def summarize(tiles, tile_m, halo_m, res):
    """One-line description of the grid, for the console and run_info.json.

    `tile_px` / `norm_window_px` are read back off the built tiles rather than
    recomputed from `tile_m`, so they stay honest when `snap_px` moved them.
    """
    tile_px = int(max(t["core_width"] for t in tiles))
    norm_px = int(max(t["pad_width"] for t in tiles))
    return {
        "n_tiles": len(tiles),
        "tile_m": tile_m,
        "halo_m": halo_m,
        "pixel_size_m": res,
        "tile_px": tile_px,
        "norm_window_px": norm_px,
        "norm_window_m": norm_px * res,
        "n_edge_tiles": int(sum(t["touches_lake_edge"] for t in tiles)),
        "grid_area_m2": float(sum(
            (t["core_width"] * res) * (t["core_height"] * res) for t in tiles)),
    }