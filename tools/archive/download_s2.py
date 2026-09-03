from pathlib import Path
import re
import json
import math
import pandas as pd
import geopandas as gpd
import os
from shapely.geometry import mapping, Polygon, MultiPolygon, GeometryCollection
from shapely.validation import make_valid
from shapely import wkb as _wkb
import ee

# ==========================
# CONFIG
# ==========================
AOI_FOLDER = r"/isipd/projects/p_planetdw/data/methods_test/aois/study_area"
DRIVE_FOLDER = "S2_GEE_Exports"                     # base Google Drive folder name (AOI name will be appended)

DATE_WINDOW_DAYS = 90
CLOUD_THRESH_PROP = 80.0                            # scene-level CLOUDY_PIXEL_PERCENTAGE filter
PER_TILE_MAX_AOI_CLOUD = 0.30                       # 0..1; skip a tile if best image still cloudier than this. set None to disable

# All optical bands (S2 L2A)
BANDS = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
SCL_CLOUD_CLASSES = [3, 8, 9, 10]                   # shadow, cloud medium, cloud high, cirrus

MAX_PIXELS = 1e13
EXPORT_REGION_SIMPLIFY_M = 10                       # meters; light simplification of export region
SUMMARY_CSV = "S2_GEE_export_summary.csv"           # written next to AOI folder

single_file_path = "/isipd/projects/p_planetdw/data/methods_test/auxilliary_data/aoi_ext.gpkg"  # optional path to a single AOI file (overrides AOI_FOLDER)

print(AOI_FOLDER)

# ==========================
# Helpers (AOIs & UTM)
# ==========================
def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"

def _safe_str(s, fallback="unknown"):
    s = str(s) if s is not None else ""
    s = s.strip()
    return s if s else fallback

def _sanitize_for_fs(name: str) -> str:
    # Safe for Drive folder and file names
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "unknown"

def create_aois(input_folder, target_crs="EPSG:4326"):
    """
    Read all AOIs (no dissolve). Adds a per-target running number "aoi_idx"
    and per-row UTM, centroid, etc.
    """
    input_folder = Path(input_folder)
    files = sorted([f for pat in ("*.geojson", "*.gpkg") for f in input_folder.glob(pat)])
    if not files:
        raise FileNotFoundError("No .geojson or .gpkg files found in AOI_FOLDER")

    gdfs = []
    for file in files:
        gdf = gpd.read_file(file)
        if gdf.crs is None:
            if file.suffix.lower() == ".geojson":
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            else:
                print(f"WARNING: {file.name} has no CRS; skipping.")
                continue
        if gdf.crs.to_string() != target_crs:
            gdf = gdf.to_crs(target_crs)

        parts = file.stem.split("_")
        gdf["filename"]   = file.name
        gdf["region"]     = parts[0] if len(parts) > 0 else None
        gdf["target"]     = parts[1] if len(parts) > 1 else None
        gdf["date_raw"]   = parts[2] if len(parts) > 2 else None
        gdf["resolution"] = parts[3] if len(parts) > 3 else None
        gdf["tile"]       = parts[4] if len(parts) > 4 else None
        gdfs.append(gdf)

    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=target_crs)
    merged["date"] = pd.to_datetime(
        merged["date_raw"].where(merged["date_raw"].astype(str).str.fullmatch(r"\d{8}")),
        format="%Y%m%d", errors="coerce"
    )
    cols = ["filename", "region", "target", "date", "resolution", "tile", "geometry"]
    merged = merged[[c for c in cols if c in merged.columns]].copy()

    # Ensure target has a value for grouping
    merged["target"] = merged["target"].apply(lambda x: _safe_str(x, "unknown"))

    # ---- centroid in projected CRS to avoid geographic centroid warnings (use Web Mercator)
    tmp_proj = merged.to_crs("EPSG:3857")
    cent = tmp_proj.geometry.centroid.to_crs("EPSG:4326")
    merged["centroid_lon"] = cent.x
    merged["centroid_lat"] = cent.y
    merged["utm_epsg"] = merged.apply(
        lambda r: utm_epsg_from_lonlat(r["centroid_lon"], r["centroid_lat"]), axis=1
    )

    # Per-target running number: 1..N (aoi_idx)
    merged = merged.sort_values(["target"]).reset_index(drop=True)
    merged["aoi_idx"] = merged.groupby("target").cumcount() + 1

    return merged

def date_window(aoi_date, days=DATE_WINDOW_DAYS):
    if pd.isna(aoi_date):
        end = pd.Timestamp.utcnow().normalize()
        start = end - pd.Timedelta(days=365)
    else:
        start = (aoi_date - pd.Timedelta(days=days))
        end   = (aoi_date + pd.Timedelta(days=days))
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

# ==========================
# Robust Shapely → EE geometry
# ==========================
_MAX_COORDS = 60000
_SIMPLIFY_DEG = 1e-4  # ~11 m at equator

def _drop_z(geom):
    return _wkb.loads(_wkb.dumps(geom, output_dimension=2))

def _poly_parts(geom):
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        parts = []
        for g in geom.geoms:
            parts.extend(_poly_parts(g))
        return parts
    return []

def _coord_count(geom):
    total = 0
    for poly in _poly_parts(geom):
        total += len(poly.exterior.coords)
        for ring in poly.interiors:
            total += len(ring.coords)
    return total

def _to_valid_simple_2d_polygon_or_multipolygon(geom):
    if geom.is_empty:
        raise ValueError("AOI is empty.")
    g = make_valid(geom).buffer(0)
    g = _drop_z(g)
    parts = _poly_parts(g)
    if not parts:
        raise ValueError("AOI has no polygonal area after cleaning.")
    g2 = MultiPolygon(parts) if len(parts) > 1 else parts[0]
    if _coord_count(g2) > _MAX_COORDS:
        g2 = make_valid(g2.simplify(_SIMPLIFY_DEG, preserve_topology=True)).buffer(0)
        parts2 = _poly_parts(g2)
        if not parts2:
            raise ValueError("AOI lost area after simplification.")
        g2 = MultiPolygon(parts2) if len(parts2) > 1 else parts2[0]
    return g2

def to_ee_geometry(geom):
    g2 = _to_valid_simple_2d_polygon_or_multipolygon(geom)
    gj = json.loads(json.dumps(mapping(g2)))
    return ee.Geometry(gj, proj="EPSG:4326", geodesic=False, evenOdd=True)

def to_ee_geometry_safe(geom):
    try:
        return to_ee_geometry(geom)
    except Exception as e:
        print(f"[to_ee_geometry_safe] primary failed: {e}  → using bounding box.")
        minx, miny, maxx, maxy = geom.bounds
        return ee.Geometry.Rectangle([minx, miny, maxx, maxy],
                                     proj="EPSG:4326", geodesic=False)

def simplify_for_export(region, meters=EXPORT_REGION_SIMPLIFY_M):
    if not meters or meters <= 0:
        return region
    return (region.transform("EPSG:3857", 1)
                  .simplify(meters)
                  .transform("EPSG:4326", 1))

# ==========================
# GEE init
# ==========================
def gee_init():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

# ==========================
# AOI Cloud metric + rank key (with null-safe defaults)
# ==========================
def compute_aoi_cloud_frac(img, region):
    """AOI-specific cloud fraction from SCL (fraction 0..1), may return null if no pixels."""
    scl = img.select("SCL")
    is_cloud = ee.Image(0)
    for c in SCL_CLOUD_CLASSES:
        is_cloud = is_cloud.Or(scl.eq(c))
    stats = is_cloud.rename("c").reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=20, bestEffort=True, maxPixels=MAX_PIXELS
    )
    return ee.Number(stats.get("c"))  # may be null

def _coalesce_number(val, fallback):
    # Equivalent of COALESCE for numbers: if null → fallback
    return ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(val, None), fallback, val))

def add_rank_properties(col, region, aoi_date):
    """
    Adds (null-safe):
      - aoi_cloud_filled (fraction, default 1.0 if null)
      - date_delta_days_filled (default 9999 if null/no AOI date)
      - scene_cloud_filled (default 100 if missing)
      - rank_key = aoi_cloud*1e6 + dateDelta*1e3 + sceneCloud
      - date_str
      - tile_id (coalesced MGRS)
    """
    target_date = ee.Date(aoi_date.strftime("%Y-%m-%d")) if aoi_date is not None else None

    def _with_props(im):
        ac = compute_aoi_cloud_frac(im, region)
        ac_filled = _coalesce_number(ac, 1.0)

        t  = ee.Date(im.get("system:time_start"))
        if target_date:
            dd = t.difference(target_date, "day").abs()
        else:
            dd = ee.Number(9999)
        dd_filled = _coalesce_number(dd, 9999)

        scene_cloud = _coalesce_number(ee.Number(im.get("CLOUDY_PIXEL_PERCENTAGE")), 100)

        rank_key = ac_filled.multiply(1e6).add(dd_filled.multiply(1e3)).add(scene_cloud)

        # tile id (some collections/images can miss the property)
        tile = ee.String(ee.Algorithms.If(im.get("MGRS_TILE"), im.get("MGRS_TILE"), "TNA"))

        return (im
                .set("aoi_cloud", ac)
                .set("aoi_cloud_filled", ac_filled)
                .set("date_delta_days", dd)
                .set("date_delta_days_filled", dd_filled)
                .set("scene_cloud_filled", scene_cloud)
                .set("rank_key", rank_key)
                .set("tile_id", tile)
                .set("date_str", t.format("YYYY-MM-dd")))

    return col.map(_with_props)

# ==========================
# Build collection (NO per-pixel mixing)
# ==========================
def build_s2_collection(region, start_date, end_date, aoi_date):
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region)
           .filterDate(start_date, end_date)
           .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", CLOUD_THRESH_PROP))
           .select(BANDS + ["SCL"]))
    return add_rank_properties(col, region, aoi_date)

# ==========================
# Select whole best scene per tile (crisp borders)
# ==========================
def select_best_per_tile(col, per_tile_max_cloud=None):
    """
    Returns an ImageCollection consisting of the single best scene per tile_id.
    Best = min rank_key (aoi_cloud → date distance → scene cloud).
    Optionally drops tiles whose best aoi_cloud_filled > per_tile_max_cloud.
    """
    tiles = ee.List(col.aggregate_array("tile_id")).distinct()

    def _best_for_tile(tile):
        tc = col.filter(ee.Filter.eq("tile_id", tile)).sort("rank_key")
        best = ee.Image(tc.first())
        return best.set("tile_id", tile)

    best_list = tiles.map(_best_for_tile)
    best_col = ee.ImageCollection.fromImages(best_list)

    if per_tile_max_cloud is not None:
        best_col = best_col.filter(ee.Filter.lte("aoi_cloud_filled", per_tile_max_cloud))

    return best_col

# ==========================
# Cloud masking (SCL) + gap fill
# ==========================
def mask_clouds_scl(img):
    """Mask clouds using SCL; keep valid, non-cloud pixels only."""
    scl = img.select("SCL")
    clear = ee.Image(1)
    for c in SCL_CLOUD_CLASSES:
        clear = clear.And(scl.neq(c))
    # also drop nodata (SCL==0)
    clear = clear.And(scl.gt(0))
    return img.updateMask(clear)

def build_gap_fill_image(col, base_comp, region):
    """
    Build a cloud-free filler image to cover only the pixels missing from base_comp.
    'Closest in time' wins inside the gap.
    """
    # Where the base composite has no data (per-band masks are identical; use first band)
    base_mask = base_comp.select(0).mask()
    gap_mask  = base_mask.Not()

    # Sort so the temporally-closest image is LAST (mosaic keeps last)
    fill_col = (col
                .sort("date_delta_days_filled", False)   # False => descending; closest ends up last
                .map(lambda im: (mask_clouds_scl(im)
                                  .resample("bilinear")
                                  .select(BANDS)
                                  .multiply(0.0001)
                                  .toFloat())))

    # Mosaic cloud-free pixels and apply only to the gap
    fill_img = (fill_col
                .mosaic()
                .updateMask(gap_mask)
                .clip(region))
    return fill_img

# ==========================
# Export
# ==========================
def export_rgb(img, region, file_prefix, utm_epsg, drive_folder):
    export_region = simplify_for_export(region, EXPORT_REGION_SIMPLIFY_M)
    task = ee.batch.Export.image.toDrive(
        image = img,
        description = file_prefix,
        folder = drive_folder,           # AOI-specific folder
        fileNamePrefix = file_prefix,
        region = export_region,
        crs = utm_epsg,
        scale = 10,
        maxPixels = MAX_PIXELS,
        fileFormat = "GeoTIFF",
        formatOptions = {"cloudOptimized": True}    # COG export
    )
    task.start()
    print(f"[Export] Started: {file_prefix} → Drive/{drive_folder}  (CRS={utm_epsg}, 10 m)")
    return task

# ==========================
# Summary building (null-safe)
# ==========================
def summarize_selected_images(selected_col, region, target, aoi_idx, utm, start_date, end_date, task_name):
    """
    Summarize images actually used (best per tile). Null-safe to avoid compute errors.
    """
    def per_image_feature(im):
        # clear area via SCL (QC only)
        scl = im.select("SCL")
        is_cloud = ee.Image(0)
        for c in SCL_CLOUD_CLASSES:
            is_cloud = is_cloud.Or(scl.eq(c))
        keep = is_cloud.Not().And(scl.gt(0))
        clear_area = ee.Image.pixelArea().updateMask(keep).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=region, scale=20, bestEffort=True, maxPixels=MAX_PIXELS
        ).get("area")
        clear_area = _coalesce_number(clear_area, 0)

        props = ee.Dictionary({
            "AOI": target,
            "AOI_Index": aoi_idx,
            "UTM": utm,
            "ExportTask": task_name,
            "WindowStart": start_date,
            "WindowEnd": end_date,
            "SystemIndex": im.get("system:index"),
            "Date": im.get("date_str"),
            "SceneCloudPct": im.get("scene_cloud_filled"),
            "AOICloudFrac": _coalesce_number(im.get("aoi_cloud"), 1.0),
            "DateDeltaDays": im.get("date_delta_days_filled"),
            "MGRS": im.get("tile_id"),
            "RankKey": im.get("rank_key"),
            "ClearArea_m2": clear_area
        })
        return ee.Feature(None, props)

    fc = ee.FeatureCollection(selected_col.map(per_image_feature))
    feats = fc.getInfo()["features"]

    rows = []
    for f in feats:
        p = f["properties"]
        rows.append({
            "AOI": p.get("AOI"),
            "AOI_Index": p.get("AOI_Index"),
            "UTM": p.get("UTM"),
            "ExportTask": p.get("ExportTask"),
            "WindowStart": p.get("WindowStart"),
            "WindowEnd": p.get("WindowEnd"),
            "SystemIndex": p.get("SystemIndex"),
            "Date": p.get("Date"),
            "SceneCloudPct": p.get("SceneCloudPct"),
            "AOICloudFrac": p.get("AOICloudFrac"),
            "DateDeltaDays": p.get("DateDeltaDays"),
            "MGRS": p.get("MGRS"),
            "RankKey": p.get("RankKey"),
            "ClearArea_m2": p.get("ClearArea_m2")
        })
    return rows

# ==========================
# Main
# ==========================
def main():
    gee_init()

    if single_file_path:
        combined_aois = gpd.read_file(single_file_path)
        crs = combined_aois.crs
        
        # single file has multiple aois, so explode and name file according to 'layer' field
        names = combined_aois['layer'].astype(str).tolist()
        for n in names:
            out_path = os.path.join(AOI_FOLDER, f"{n}.gpkg")

            subset = combined_aois[combined_aois['layer'] == n]
            # save to file
            subset.to_file(out_path, driver="GPKG")
            print(f"Wrote subset AOI to {out_path}")

    aois = create_aois(AOI_FOLDER)
    print(f"[AOIs] {len(aois)} individual AOI(s) (no dissolve)")

    all_rows = []
    tasks = []

    for _, row in aois.iterrows():
        target = _safe_str(row.get("target"), "unknown")
        target_safe = _sanitize_for_fs(target)
        aoi_idx = int(row.get("aoi_idx") or 1)

        geom = row.geometry
        utm = row.get("utm_epsg") or "EPSG:32633"
        aoi_date = row.get("date") if pd.notna(row.get("date")) else None
        start, end = date_window(row.get("date"), DATE_WINDOW_DAYS)
        region = to_ee_geometry_safe(geom)

        print(f"\n=== {target} [#{aoi_idx}] | window={start}..{end} | {utm} ===")

        # Build collection & select whole best scene per tile
        col = build_s2_collection(region, start, end, aoi_date)
        best_col = select_best_per_tile(col, PER_TILE_MAX_AOI_CLOUD)

        # Whole-scene mosaic (base), reflectance scaling, clip AOI
        # NOTE: resample to 10 m with bilinear so 20 m / 60 m optical bands upscale cleanly
        comp_base = (best_col
                     .map(lambda im: im.resample("bilinear").select(BANDS))
                     .mosaic()
                     .multiply(0.0001)
                     .toFloat()
                     .clip(region))

        # Fill only the missing pixels with next-closest-in-time cloud-free imagery
        fill_img  = build_gap_fill_image(col, comp_base, region)
        comp_out  = comp_base.unmask(fill_img)  # keep what's already there; fill where missing

        # AOI-specific Drive folder
        drive_folder = f"{DRIVE_FOLDER}_{target_safe}"

        # Name & export
        date_tag = (row.get("date").strftime("%Y%m%d") if pd.notna(row.get("date")) else "nodate")
        prefix = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            f"{target_safe}_{aoi_idx:03d}_S2L2A_{date_tag}_SR_10m_utm"
        )
        task = export_rgb(comp_out, region, prefix, utm, drive_folder)
        tasks.append(task)

        # Summarize selected (used) images: base (best per tile)
        rows = summarize_selected_images(best_col, region, target, aoi_idx, utm, start, end, prefix)
        rows = sorted(rows, key=lambda r: (r["AOI"], r["AOI_Index"], r["MGRS"], r["DateDeltaDays"], r["AOICloudFrac"]))
        all_rows.extend(rows)

    # Write summary CSV
    df = pd.DataFrame(all_rows)

    print("=====SUMMARY=====")
    print(df["DateDeltaDays"].describe())
    out_csv = Path(AOI_FOLDER).parent / SUMMARY_CSV
    df.to_csv(out_csv, index=False)
    print(f"\n[Summary] Wrote selected-image summary: {out_csv}")
    if not df.empty:
        print(df.head(12).to_string(index=False))

    print(f"\nSubmitted {len(tasks)} export task(s). Open https://code.earthengine.google.com/tasks to monitor.")

if __name__ == "__main__":
    main()
