# tools/seep_feature_table.py
"""
Per-BUBBLE feature tables for detected bubbles, and ground-truth bubble
polygons rebuilt from the original drawn shapes.

A "bubble" here is one connected component of the smoothed prediction, or one
hand-drawn ground-truth polygon. Nothing in this module is a *seep*: bubbles
become seeps only after the learned pairwise grouper runs
(tools/seep_grouper_*.py). See the naming convention in CLAUDE.md.

Inputs (already on disk after evaluation + write_seep_rasters):
  - {stem}_cc.tif      per-bubble connected components (uint16/uint32)
  - {basename}.tif     matching chip in chip_dir (RGB bands for brightness)

Outputs (written to pred_dir):
  - seep_features_per_bubble.csv   one row per detected bubble

REMOVED 2026-09-03: the anchor-conditional + "lonely" clustering rule
(anchor_cluster / lonely_cluster / aggregate_clusters / build_cluster_raster /
build_pred_seeps_gdf / process_dir) and its outputs
(seep_features_per_cluster.csv, pred_seeps.gpkg, *_seep_cluster.tif). It was a
hand-tuned placeholder superseded by the learned random-forest grouper, and it
is the reason the retired cluster_f1=0.672 was circular -- the same rule was
applied to BOTH the ground-truth and predicted sides, so grouping error
cancelled out and the metric measured detection only. Recoverable at tag
`pre-cleanup`.
"""
import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes as rio_shapes
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import xy as rio_xy
from shapely.geometry import shape as shp_shape
from shapely.geometry import box as shp_box
from skimage.measure import regionprops

try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    gpd = None
    _HAS_GPD = False


_AUX_SUFFIXES = ("_prob.tif", "_epistemic.tif", "_aleatoric.tif",
                 "_smoothed.tif", "_cc.tif", "_seep_cluster.tif",
                 "_snow.tif")


def _aux_path(pred_fp, suffix):
    stem = os.path.splitext(pred_fp)[0]
    return f"{stem}{suffix}"

def compute_bubble_features(cc_lab, image, transform):
    """One row per CC. Morphological + RGB-brightness features.

    Adds `solidity` (area / convex-hull area) and `eccentricity` (from the
    fitted ellipse; both come straight from skimage.regionprops). Both are
    more robust to boundary-roughness noise than circularity — see
    CLAUDE.md 2026-04-28 for why we recommend them for downstream
    classification work.
    """
    pix_m = abs(transform.a)
    has_rgb = image.ndim == 3 and image.shape[2] >= 3
    rows = []
    for p in regionprops(cc_lab):
        ar_m2 = p.area * (pix_m ** 2)
        per_m = max(p.perimeter * pix_m, 1e-9)
        circ = 4 * np.pi * ar_m2 / (per_m ** 2)
        cy_px, cx_px = p.centroid
        cx_m, cy_m = rio_xy(transform, cy_px, cx_px)
        ys, xs = np.where(cc_lab == p.label)
        if has_rgb:
            mR = float(image[ys, xs, 0].mean())
            mG = float(image[ys, xs, 1].mean())
            mB = float(image[ys, xs, 2].mean())
        else:
            v = float(image[ys, xs].mean()) if image.ndim == 2 \
                else float(image[ys, xs, 0].mean())
            mR = mG = mB = v
        rows.append({
            "bubble_id": int(p.label),
            "centroid_x_m": float(cx_m),
            "centroid_y_m": float(cy_m),
            "area_m2": float(ar_m2),
            "perim_m": float(per_m),
            "circularity": float(circ),
            "solidity": float(p.solidity),
            "eccentricity": float(p.eccentricity),
            "mean_R": mR, "mean_G": mG, "mean_B": mB,
        })
    return pd.DataFrame(rows, columns=[
        "bubble_id", "centroid_x_m", "centroid_y_m",
        "area_m2", "perim_m", "circularity",
        "solidity", "eccentricity",
        "mean_R", "mean_G", "mean_B",
    ])


def polygonize_labels(label_array, transform, crs):
    """Polygonize a uint label array (0=background). Same-label pieces are
    dissolved into a single (Multi)Polygon per id."""
    if not _HAS_GPD:
        raise RuntimeError("geopandas is required for polygonization")
    arr = label_array.astype(np.int32)
    mask = (arr > 0).astype(np.uint8)
    geoms, ids = [], []
    for geom, val in rio_shapes(arr, mask=mask, transform=transform):
        if int(val) == 0:
            continue
        geoms.append(shp_shape(geom))
        ids.append(int(val))
    if not geoms:
        return gpd.GeoDataFrame({"id": [], "geometry": []}, crs=crs, geometry="geometry")
    gdf = gpd.GeoDataFrame({"id": ids, "geometry": geoms}, crs=crs)
    if gdf["id"].duplicated().any():
        gdf = gdf.dissolve(by="id", as_index=False)
    return gdf


def labels_to_seep_gdf(label_array, image, transform, crs, id_name="seep_id"):
    """End-to-end: labels → polygons + features → GeoDataFrame (one row per
    label, with shapely geometry, area/perim/circ/solidity/ecc, mean_R/G/B).
    Use for both GT polygons (label = polygon CC) and pred cluster rasters."""
    if not _HAS_GPD:
        raise RuntimeError("geopandas is required for labels_to_seep_gdf")
    features = compute_bubble_features(label_array, image, transform)
    features = features.rename(columns={"bubble_id": id_name})
    gdf = polygonize_labels(label_array, transform, crs)
    gdf = gdf.rename(columns={"id": id_name})
    return gdf.merge(features, on=id_name, how="left")


def build_gt_seeps_from_source(chip_fp, source_polygons_gdf,
                               id_name="seep_id"):
    """Build per-chip GT-seep GeoDataFrame from ORIGINAL drawn polygons.

    Use this instead of running labels_to_seep_gdf on the rasterized GT mask:
    the rasterize -> CC -> repolygonize round-trip merges any two original
    polygons that touch (or sit within 1 px diagonally with 8-connectivity)
    into a single output polygon, which destroys per-seep class labels.

    For each source polygon intersecting the chip footprint:
      - geometry: original polygon, clipped to chip extent
      - area_m2 / perim_m / circularity / solidity: from shapely on the
        clipped geometry (CRS is meters)
      - eccentricity + mean_R/G/B: rasterize each polygon with a UNIQUE
        per-polygon label (so adjacent polygons never merge into one CC)
        and read regionprops + image pixels off that label raster

    Returns a GeoDataFrame with the same column schema as labels_to_seep_gdf
    (id_name, centroid_x_m, centroid_y_m, area_m2, perim_m, circularity,
    solidity, eccentricity, mean_R, mean_G, mean_B, geometry). Returns None
    when the chip has no CRS or no polygons intersect it.
    """
    if not _HAS_GPD:
        raise RuntimeError("geopandas is required")
    with rasterio.open(chip_fp) as src:
        n = src.count
        chip_crs = src.crs
        transform = src.transform
        H, W = src.height, src.width
        bounds = src.bounds
        # Image bands = all but the last (last is the rasterized GT mask).
        if n > 2:
            image = np.transpose(src.read(list(range(1, n))), (1, 2, 0))
        elif n == 2:
            image = src.read(1)
        else:
            image = src.read(1)

    if chip_crs is None:
        return None

    src_gdf = source_polygons_gdf
    if src_gdf.crs != chip_crs:
        src_gdf = src_gdf.to_crs(chip_crs)

    chip_box = shp_box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    cand_idx = list(src_gdf.sindex.query(chip_box))
    if not cand_idx:
        return None
    cand = src_gdf.iloc[cand_idx][["geometry"]].copy()
    cand["geometry"] = cand.geometry.intersection(chip_box)
    cand = cand[~cand.geometry.is_empty & cand.geometry.notna()]
    cand = cand[cand.geom_type.isin(("Polygon", "MultiPolygon"))]
    if cand.empty:
        return None
    cand = cand.reset_index(drop=True)
    cand[id_name] = np.arange(1, len(cand) + 1, dtype=np.int64)

    has_rgb = image.ndim == 3 and image.shape[2] >= 3

    # Unique per-polygon labels keep adjacent polygons in their own CCs.
    shapes = list(zip(cand.geometry, cand[id_name].astype(int).values))
    label_arr = rio_rasterize(
        shapes, out_shape=(H, W), transform=transform,
        fill=0, all_touched=False, dtype=np.int32,
    )
    rp_by_label = {p.label: p for p in regionprops(label_arr)}

    rows = []
    for _, r in cand.iterrows():
        sid = int(r[id_name])
        geom = r.geometry
        ar_m2 = float(geom.area)
        per_m = max(float(geom.length), 1e-9)
        circ = 4 * np.pi * ar_m2 / (per_m ** 2)
        hull_area = float(geom.convex_hull.area)
        sol = float(ar_m2 / hull_area) if hull_area > 0 else 0.0
        c = geom.centroid
        cx_m, cy_m = float(c.x), float(c.y)

        p = rp_by_label.get(sid)
        if p is not None:
            ecc = float(p.eccentricity)
            ys, xs = np.where(label_arr == sid)
            if has_rgb and len(ys):
                mR = float(image[ys, xs, 0].mean())
                mG = float(image[ys, xs, 1].mean())
                mB = float(image[ys, xs, 2].mean())
            elif len(ys):
                v = float(image[ys, xs].mean()) if image.ndim == 2 \
                    else float(image[ys, xs, 0].mean())
                mR = mG = mB = v
            else:
                mR = mG = mB = float("nan")
        else:
            # Polygon was sub-pixel and didn't paint any cells; geometry
            # features still come from shapely so we keep the row.
            ecc = float("nan")
            mR = mG = mB = float("nan")

        rows.append({
            id_name: sid,
            "centroid_x_m": cx_m, "centroid_y_m": cy_m,
            "area_m2": ar_m2, "perim_m": per_m,
            "circularity": circ, "solidity": sol, "eccentricity": ecc,
            "mean_R": mR, "mean_G": mG, "mean_B": mB,
        })

    feat_df = pd.DataFrame(rows)
    out = gpd.GeoDataFrame(
        feat_df.merge(cand[[id_name, "geometry"]], on=id_name),
        geometry="geometry", crs=chip_crs,
    )
    return out


def write_seeps_gpkg(out_fp, gdfs, class_column=False):
    """Concatenate per-chip GeoDataFrames and write to a single GPKG.
    Adds an empty `class` text column when class_column=True (used for GT)."""
    if not _HAS_GPD:
        raise RuntimeError("geopandas is required for write_seeps_gpkg")
    valid = [g for g in gdfs if g is not None and not g.empty]
    if not valid:
        return None
    out = pd.concat(valid, ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=valid[0].crs)
    if class_column and "class" not in out.columns:
        out["class"] = ""
    out.to_file(out_fp, driver="GPKG")
    return out_fp


def _load_inputs(pred_fp, chip_fp):
    """Standalone-mode loader: read _cc.tif (must exist) and the chip image."""
    cc_fp = _aux_path(pred_fp, "_cc.tif")
    if not os.path.exists(cc_fp):
        raise FileNotFoundError(
            f"_cc.tif missing for {os.path.basename(pred_fp)}: {cc_fp}\n"
            f"Run tools/write_seep_rasters.py first to generate per-bubble CCs."
        )
    with rasterio.open(cc_fp) as src:
        cc = src.read(1).astype(np.int64)
        transform = src.transform
        profile = src.profile.copy()
    with rasterio.open(chip_fp) as src:
        n = src.count
        if n > 1:
            image = np.transpose(src.read(list(range(1, n))), (1, 2, 0))
        else:
            image = src.read(1)
    return cc, image, transform, profile


def process_pred(pred_fp, chip_fp,
                 cc=None, image=None, transform=None, profile=None):
    """Per-image per-bubble feature extraction.

    Pass pre-computed cc/image/transform/profile to skip disk reads (used by
    the bubble-level eval during its main loop). Otherwise reads _cc.tif and
    the chip from disk.

    Returns a bubbles DataFrame prefixed with an 'image' column.
    """
    if cc is None or image is None or transform is None or profile is None:
        _cc, _image, _transform, _profile = _load_inputs(pred_fp, chip_fp)
        cc = _cc if cc is None else cc
        image = _image if image is None else image
        transform = _transform if transform is None else transform
        profile = _profile if profile is None else profile

    bubbles = compute_bubble_features(cc, image, transform)
    bubbles.insert(0, "image", os.path.basename(pred_fp))
    return bubbles


def write_feature_csvs(pred_dir, bubbles):
    bubbles.to_csv(os.path.join(pred_dir, "seep_features_per_bubble.csv"),
                   index=False)
    _print_summary(bubbles)


def _print_summary(bubbles):
    n_bub = len(bubbles)
    area = bubbles["area_m2"]
    print(f"\nBUBBLE FEATURES: n_bubbles={n_bub}")
    if n_bub:
        print(f"  area_m2  median={area.median():.5f}  "
              f"p90={area.quantile(0.90):.5f}  max={area.max():.5f}")
