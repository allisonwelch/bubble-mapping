from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import geopandas as gpd
import requests
from requests.auth import HTTPBasicAuth

from shapely.geometry import mapping, Polygon, MultiPolygon, GeometryCollection, shape
from shapely.validation import make_valid
from shapely import wkb as _wkb
from shapely.ops import transform as shp_transform
from pyproj import Transformer

print("Imports complete.")


# =============================================================================
# CONFIG
# =============================================================================

# AOIs
AOI_FOLDER = Path(r"N:\isipd\projects\p_planetdw\data\methods_test\aois\study_area")
SINGLE_FILE_PATH = Path(
    r"N:\isipd\projects\p_planetdw\data\methods_test\auxilliary_data\aoi_ext.gpkg"
)  # optional

# Output (ONLY used rows)
OUT_CSV = AOI_FOLDER.parent / "PSScene_used_selection_summary.csv"

# Planet auth (Data API)
PLANET_API_KEY = ""
PLANET_CREDENTIAL = PLANET_API_KEY

ITEM_TYPE = "PSScene"
PRODUCT_BUNDLE = "analytic_sr_udm2"

DATE_WINDOW_DAYS = 90
MAX_SCENE_CLOUD_COVER = 0.80

# AOI-specific gating using clear_percent from /coverage
# AOICloudFrac = 1 - clear_percent/100
PER_STRIP_MAX_AOI_CLOUD = 0.30  # set None to disable

# "Prefer one per strip", but allow more if it adds new AOI footprint area
SELECT_ONE_PER_STRIP = True

MAX_CANDIDATES_PER_AOI = 250

# Coverage endpoint settings
COVERAGE_MODE = "estimate"  # "estimate" or "UDM2"
COVERAGE_POLL_SECONDS = 10
COVERAGE_TIMEOUT_SECONDS = 600

PAGE_SIZE = 250
SORT = "acquired asc"

# -------- NEW: REAL footprint coverage controls (this is the core fix) --------
TARGET_FOOTPRINT_FRAC = 0.999   # stop when AOI is ~fully covered by footprints
MIN_FOOTPRINT_GAIN = 0.002      # skip candidates that add <0.2% new AOI area
# If SELECT_ONE_PER_STRIP=True, allow same-strip scenes only if they add >= this much new area
MIN_GAIN_TO_ALLOW_SAME_STRIP = 0.002

# Clear-coverage heuristic (optional stop condition)
TARGET_CLEAR_FRAC = 0.98        # estimated clear-covered AOI fraction (based on new footprint area * clear_frac)
MAX_SELECTED_PER_AOI = 12
PRE_RANK_TOP_K = 120            # footprint filtering is free; bump this to avoid missing “edge” scenes


DATA_API = "https://api.planet.com/data/v1"


def _safe_str(s: Any, fallback: str = "unknown") -> str:
    s = str(s) if s is not None else ""
    s = s.strip()
    return s if s else fallback


def utm_epsg_from_lonlat(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


def date_window(aoi_date: Optional[pd.Timestamp], days: int = DATE_WINDOW_DAYS) -> Tuple[str, str]:
    if aoi_date is None or pd.isna(aoi_date):
        end = pd.Timestamp.utcnow().tz_localize("UTC")
        start = end - pd.Timedelta(days=365)
    else:
        if aoi_date.tzinfo is None:
            aoi_date = aoi_date.tz_localize("UTC")
        start = aoi_date - pd.Timedelta(days=days)
        end = aoi_date + pd.Timedelta(days=days)
    return start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# Geometry cleanup for Planet (clip/coverage/search)
# =============================================================================

_MAX_VERTICES = 1500
_SIMPLIFY_DEG = 1e-4  # ~11m at equator


def _drop_z(geom):
    return _wkb.loads(_wkb.dumps(geom, output_dimension=2))


def _poly_parts(geom) -> List[Polygon]:
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if isinstance(geom, GeometryCollection):
        out: List[Polygon] = []
        for g in geom.geoms:
            out.extend(_poly_parts(g))
        return out
    return []


def _strip_holes(poly: Polygon) -> Polygon:
    return Polygon(poly.exterior)


def _vertex_count(poly: Polygon) -> int:
    n = len(poly.exterior.coords)
    for r in poly.interiors:
        n += len(r.coords)
    return n


def _clean_aoi_for_planet(geom) -> MultiPolygon | Polygon:
    if geom.is_empty:
        raise ValueError("AOI is empty")

    g = make_valid(geom)
    g = _drop_z(g)
    g = g.buffer(0)

    parts = _poly_parts(g)
    if not parts:
        raise ValueError("AOI has no polygonal area after cleaning")

    parts = [_strip_holes(p) for p in parts]
    out: MultiPolygon | Polygon = MultiPolygon(parts) if len(parts) > 1 else parts[0]

    def total_vertices(x: MultiPolygon | Polygon) -> int:
        if isinstance(x, Polygon):
            return _vertex_count(x)
        return sum(_vertex_count(p) for p in x.geoms)

    if total_vertices(out) > _MAX_VERTICES:
        simp_parts: List[Polygon] = []
        for p in (out.geoms if isinstance(out, MultiPolygon) else [out]):
            sp = make_valid(p.simplify(_SIMPLIFY_DEG, preserve_topology=True)).buffer(0)
            sp = _strip_holes(sp) if isinstance(sp, Polygon) else Polygon(sp.exterior)
            simp_parts.append(sp)
        out = MultiPolygon(simp_parts) if len(simp_parts) > 1 else simp_parts[0]

    return out


def shapely_to_geojson_geom(geom) -> Dict[str, Any]:
    cleaned = _clean_aoi_for_planet(geom)
    return json.loads(json.dumps(mapping(cleaned)))


def _projector(src_crs: str, dst_crs: str):
    tfm = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return lambda g: shp_transform(tfm.transform, g)


# =============================================================================
# AOI ingestion
# =============================================================================

def create_aois(input_folder: Path, target_crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    files = sorted([f for pat in ("*.geojson", "*.gpkg") for f in input_folder.glob(pat)])
    if not files:
        raise FileNotFoundError(f"No .geojson/.gpkg found in {input_folder}")

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
        gdf["filename"] = file.name
        gdf["region"] = parts[0] if len(parts) > 0 else None
        gdf["target"] = parts[1] if len(parts) > 1 else None
        gdf["date_raw"] = parts[2] if len(parts) > 2 else None
        gdf["resolution"] = parts[3] if len(parts) > 3 else None
        gdf["tile"] = parts[4] if len(parts) > 4 else None
        gdfs.append(gdf)

    merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=target_crs)

    merged["date"] = pd.to_datetime(
        merged["date_raw"].where(merged["date_raw"].astype(str).str.fullmatch(r"\d{8}")),
        format="%Y%m%d",
        errors="coerce",
        utc=True,
    )

    cols = ["filename", "region", "target", "date", "resolution", "tile", "geometry"]
    merged = merged[[c for c in cols if c in merged.columns]].copy()
    merged["target"] = merged["target"].apply(lambda x: _safe_str(x, "unknown"))

    tmp_proj = merged.to_crs("EPSG:3857")
    cent = tmp_proj.geometry.centroid.to_crs("EPSG:4326")
    merged["centroid_lon"] = cent.x
    merged["centroid_lat"] = cent.y
    merged["utm_epsg"] = merged.apply(
        lambda r: utm_epsg_from_lonlat(r["centroid_lon"], r["centroid_lat"]),
        axis=1
    )

    merged = merged.sort_values(["target"]).reset_index(drop=True)
    merged["aoi_idx"] = merged.groupby("target").cumcount() + 1
    return merged


# =============================================================================
# Planet Data API client
# =============================================================================

@dataclass
class Candidate:
    item_id: str
    acquired: pd.Timestamp
    scene_cloud_cover: float  # 0..1
    strip_id: str
    footprint: Optional[Dict[str, Any]] = None  # <- NEW: footprint geometry from quick-search

    coverage_status: str = "unknown"
    clear_percent: Optional[float] = None      # 0..100
    aoi_cloud_frac: Optional[float] = None     # 0..1
    date_delta_days: Optional[float] = None
    rank_key: Optional[float] = None

    # NEW: footprint metrics
    footprint_cover_frac: Optional[float] = None     # overlap(AOI, footprint)/AOI
    footprint_gain_frac: Optional[float] = None      # incremental gain when selected


class PlanetClient:
    def __init__(self, credential: str):
        cred = (credential or "").strip()
        if not cred:
            raise ValueError("Planet credential missing. Set PL_API_KEY / PLANET_API_KEY or paste token in PLANET_CREDENTIAL.")
        self.session = requests.Session()

        if cred.startswith("eyJ"):
            self.auth = None
            self.headers = {"Authorization": f"Bearer {cred}"}
            self.auth_mode = "bearer"
        else:
            self.auth = HTTPBasicAuth(cred, "")
            self.headers = None
            self.auth_mode = "basic"

    def _post(self, url: str, *, params: Optional[dict] = None, json_body: Any = None, timeout: int = 120) -> requests.Response:
        return self.session.post(
            url,
            params=params,
            json=json_body,
            auth=self.auth,
            headers=self.headers,
            timeout=timeout,
        )

    def quick_search(
        self,
        aoi_geojson: Dict[str, Any],
        start_iso: str,
        end_iso: str,
        max_cloud_cover: float,
        page_size: int = PAGE_SIZE,
        sort: str = SORT,
        max_items: int = MAX_CANDIDATES_PER_AOI,
    ) -> List[Candidate]:
        url = f"{DATA_API}/quick-search"
        params = {"_page_size": page_size, "_sort": sort}

        body = {
            "item_types": [ITEM_TYPE],
            "geometry": aoi_geojson,
            "filter": {
                "type": "AndFilter",
                "config": [
                    {
                        "type": "DateRangeFilter",
                        "field_name": "acquired",
                        "config": {"gte": start_iso, "lte": end_iso},
                    },
                    {
                        "type": "RangeFilter",
                        "field_name": "cloud_cover",
                        "config": {"lte": float(max_cloud_cover)},
                    },
                ],
            },
        }

        out: List[Candidate] = []
        next_url = url
        next_params = params

        while True:
            r = self._post(next_url, params=next_params, json_body=body, timeout=120)
            r.raise_for_status()
            data = r.json()

            for feat in data.get("features", []):
                if len(out) >= max_items:
                    return out

                item_id = feat.get("id")
                props = feat.get("properties", {}) or {}
                acquired_raw = props.get("acquired")
                cloud_cover = float(props.get("cloud_cover") or 1.0)
                strip_id = _safe_str(props.get("strip_id"), fallback=item_id)

                # NEW: footprint geometry comes directly from the feature
                footprint = feat.get("geometry")

                if not item_id or not acquired_raw:
                    continue

                acquired = pd.to_datetime(acquired_raw, utc=True, errors="coerce")
                if pd.isna(acquired):
                    continue

                out.append(
                    Candidate(
                        item_id=item_id,
                        acquired=acquired,
                        scene_cloud_cover=cloud_cover,
                        strip_id=strip_id,
                        footprint=footprint,
                    )
                )

            next_link = (data.get("_links") or {}).get("_next")
            if not next_link or len(out) >= max_items:
                break

            next_url = next_link
            next_params = None

        return out

    def coverage_clear_percent(
        self,
        item_id: str,
        aoi_geojson: Dict[str, Any],
        mode: str = COVERAGE_MODE,
        poll_seconds: int = COVERAGE_POLL_SECONDS,
        timeout_seconds: int = COVERAGE_TIMEOUT_SECONDS,
    ) -> Tuple[str, Optional[float]]:
        url = f"{DATA_API}/item-types/{ITEM_TYPE}/items/{item_id}/coverage"
        params = {"mode": mode}
        payload = {"geometry": aoi_geojson}

        def _call() -> Dict[str, Any]:
            rr = self._post(url, params=params, json_body=payload, timeout=120)
            rr.raise_for_status()
            return rr.json()

        start = time.time()
        resp = _call()
        status = _safe_str(resp.get("status"), "unknown")
        clear_pct = resp.get("clear_percent")
        clear_pct = float(clear_pct) if clear_pct is not None else None

        if mode != "UDM2":
            return status, clear_pct

        while status == "activating":
            if time.time() - start > timeout_seconds:
                return "timeout", None
            time.sleep(poll_seconds)
            resp = _call()
            status = _safe_str(resp.get("status"), "unknown")
            clear_pct = resp.get("clear_percent")
            clear_pct = float(clear_pct) if clear_pct is not None else None

        return status, clear_pct


# =============================================================================
# Ranking + selection
# =============================================================================

def compute_rank(
    aoi_date: Optional[pd.Timestamp],
    acquired: pd.Timestamp,
    scene_cloud_cover: float,
    clear_percent: Optional[float],
) -> Tuple[float, float, float]:
    if clear_percent is None:
        aoi_cloud = 1.0
    else:
        aoi_cloud = max(0.0, min(1.0, 1.0 - float(clear_percent) / 100.0))

    if aoi_date is None or pd.isna(aoi_date):
        dd = 9999.0
    else:
        if aoi_date.tzinfo is None:
            aoi_date = aoi_date.tz_localize("UTC")
        dd = abs((acquired - aoi_date).total_seconds()) / 86400.0

    scene_cloud_pct = float(scene_cloud_cover) * 100.0 if scene_cloud_cover is not None else 100.0
    rank_key = aoi_cloud * 1e6 + dd * 1e3 + scene_cloud_pct
    return aoi_cloud, dd, rank_key


def cheap_rank_candidates(candidates: List[Candidate], aoi_date: Optional[pd.Timestamp]) -> List[Candidate]:
    if aoi_date is not None and not pd.isna(aoi_date) and aoi_date.tzinfo is None:
        aoi_date = aoi_date.tz_localize("UTC")

    def key(c: Candidate) -> Tuple[float, float]:
        if aoi_date is None or pd.isna(aoi_date):
            dd = 9999.0
        else:
            dd = abs((c.acquired - aoi_date).total_seconds()) / 86400.0
        return (dd, c.scene_cloud_cover)

    return sorted(candidates, key=key)


def _safe_make_shapely(geojson_geom: Optional[Dict[str, Any]]):
    if not geojson_geom:
        return None
    try:
        g = make_valid(shape(geojson_geom)).buffer(0)
        return g
    except Exception:
        return None


def select_scenes_low_overlap_until_covered(
    client: PlanetClient,
    candidates: List[Candidate],
    aoi_geojson: Dict[str, Any],
    aoi_geom4326,              # <- NEW: cleaned AOI shapely geometry (EPSG:4326)
    utm_epsg: str,             # <- NEW: for accurate area
    aoi_date: Optional[pd.Timestamp],
    target_footprint_frac: float = TARGET_FOOTPRINT_FRAC,
    min_footprint_gain: float = MIN_FOOTPRINT_GAIN,
    target_clear_frac: float = TARGET_CLEAR_FRAC,
    max_selected: int = MAX_SELECTED_PER_AOI,
    one_per_strip: bool = SELECT_ONE_PER_STRIP,
) -> Tuple[List[Candidate], float, float, int]:
    """
    REAL coverage selection:
      - Tracks true AOI footprint union coverage using footprints from quick-search.
      - Only calls /coverage (paid) after confirming the scene adds new footprint area.
      - Estimates "clear-covered AOI" as: sum(new_footprint_area_frac * clear_frac)

    Returns:
      selected, final_footprint_union_frac, final_est_clear_union_frac, n_coverage_calls
    """
    if aoi_date is not None and not pd.isna(aoi_date) and aoi_date.tzinfo is None:
        aoi_date = aoi_date.tz_localize("UTC")

    proj = _projector("EPSG:4326", utm_epsg)
    aoi_utm = proj(aoi_geom4326)
    aoi_area = float(aoi_utm.area) if not aoi_utm.is_empty else 0.0
    if aoi_area <= 0:
        return [], 0.0, 0.0, 0

    selected: List[Candidate] = []
    seen_strips: set[str] = set()

    covered_utm = None  # union of (footprint ∩ AOI) in UTM
    union_footprint = 0.0
    union_clear_est = 0.0
    coverage_calls = 0

    for c in candidates:
        if len(selected) >= max_selected:
            break

        fp4326 = _safe_make_shapely(c.footprint)
        if fp4326 is None or fp4326.is_empty:
            continue

        fp_utm = proj(fp4326)
        inter = fp_utm.intersection(aoi_utm)
        if inter.is_empty:
            continue

        # Full overlap fraction of this scene footprint with AOI
        c.footprint_cover_frac = float(inter.area) / aoi_area

        # Incremental NEW area this scene would add beyond already-covered footprint union
        if covered_utm is None:
            new_area = float(inter.area)
        else:
            new_area = float(inter.difference(covered_utm).area)

        new_gain_frac = 0.0 if aoi_area <= 0 else new_area / aoi_area
        c.footprint_gain_frac = new_gain_frac

        # Skip scenes that add essentially no new AOI footprint
        if new_gain_frac < min_footprint_gain:
            continue

        # Prefer one per strip, but allow same-strip scenes if they add real new area
        if one_per_strip and c.strip_id in seen_strips and new_gain_frac < MIN_GAIN_TO_ALLOW_SAME_STRIP:
            continue

        # Only NOW pay for /coverage (cloud/clear over AOI overlap)
        try:
            status, clear_pct = client.coverage_clear_percent(c.item_id, aoi_geojson, mode=COVERAGE_MODE)
            coverage_calls += 1
        except Exception as e:
            print(f"[Coverage] {c.item_id} failed: {e}")
            continue

        aoi_cloud, dd, rk = compute_rank(aoi_date, c.acquired, c.scene_cloud_cover, clear_pct)

        c.coverage_status = status
        c.clear_percent = clear_pct
        c.aoi_cloud_frac = aoi_cloud
        c.date_delta_days = dd
        c.rank_key = rk

        # AOI-specific cloudiness gate (based on clear_percent over overlap)
        if PER_STRIP_MAX_AOI_CLOUD is not None and c.aoi_cloud_frac is not None:
            if c.aoi_cloud_frac > PER_STRIP_MAX_AOI_CLOUD:
                continue

        # Accept: update unions
        covered_utm = inter if covered_utm is None else covered_utm.union(inter)
        union_footprint = float(covered_utm.area) / aoi_area

        clear_frac = 0.0
        if clear_pct is not None:
            clear_frac = max(0.0, min(1.0, float(clear_pct) / 100.0))

        # Estimated clear-covered AOI union:
        # only credit the newly-added footprint area by this scene * its clear fraction
        union_clear_est = min(1.0, union_clear_est + new_gain_frac * clear_frac)

        selected.append(c)
        seen_strips.add(c.strip_id)

        clear_pct_str = f"{clear_pct:.1f}%" if clear_pct is not None else "None"
        print(
            f"[Pick] {len(selected):02d} | add_footprint={new_gain_frac*100:.2f}% | "
            f"footprint_union={union_footprint*100:.2f}% | clear_percent={clear_pct_str} | "
            f"clear_union_est={union_clear_est*100:.2f}% | strip={c.strip_id} | item={c.item_id}"
        )

        # Stop once AOI is actually covered by footprints (and optionally clear-covered)
        if union_footprint >= target_footprint_frac:
            if target_clear_frac is None or union_clear_est >= target_clear_frac:
                break

    selected.sort(key=lambda x: x.rank_key if x.rank_key is not None else math.inf)
    return selected, union_footprint, union_clear_est, coverage_calls


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    if not PLANET_CREDENTIAL:
        raise RuntimeError(
            "Missing Planet credential. Set PL_API_KEY / PLANET_API_KEY or paste token in PLANET_CREDENTIAL."
        )

    # Optional: split multi-layer AOI file into per-layer GPKGs
    if SINGLE_FILE_PATH and SINGLE_FILE_PATH.exists():
        combined = gpd.read_file(SINGLE_FILE_PATH)
        if "layer" in combined.columns:
            AOI_FOLDER.mkdir(parents=True, exist_ok=True)
            for layer_name in combined["layer"].astype(str).unique().tolist():
                subset = combined[combined["layer"].astype(str) == str(layer_name)]
                out_path = AOI_FOLDER / f"{layer_name}.gpkg"
                subset.to_file(out_path, driver="GPKG")
                print(f"Wrote subset AOI to {out_path}")

    aois = create_aois(AOI_FOLDER)
    print(f"[AOIs] {len(aois)} AOI(s)")

    client = PlanetClient(PLANET_CREDENTIAL)
    print(f"[Auth] mode={client.auth_mode}")

    used_rows: List[Dict[str, Any]] = []

    for _, aoi in aois.iterrows():
        target = _safe_str(aoi.get("target"), "unknown")
        aoi_idx = int(aoi.get("aoi_idx") or 1)

        geom = aoi.geometry
        utm = _safe_str(aoi.get("utm_epsg"), "EPSG:32633")

        aoi_date = aoi.get("date")
        if pd.isna(aoi_date):
            aoi_date = None

        start_iso, end_iso = date_window(aoi_date, DATE_WINDOW_DAYS)

        # IMPORTANT: use the cleaned AOI geometry consistently for API + footprint computations
        aoi_clean = _clean_aoi_for_planet(geom)
        aoi_geojson = json.loads(json.dumps(mapping(aoi_clean)))

        print(f"\n=== {target} [#{aoi_idx}] | window={start_iso}..{end_iso} | {utm} ===")

        candidates = client.quick_search(
            aoi_geojson=aoi_geojson,
            start_iso=start_iso,
            end_iso=end_iso,
            max_cloud_cover=MAX_SCENE_CLOUD_COVER,
            max_items=MAX_CANDIDATES_PER_AOI,
        )
        print(f"[Search] {len(candidates)} candidate(s) after scene-level cloud filter")

        if not candidates:
            continue

        cheap = cheap_rank_candidates(candidates, aoi_date)[:PRE_RANK_TOP_K]
        print(f"[PreRank] considering top {len(cheap)} candidate(s) (PRE_RANK_TOP_K={PRE_RANK_TOP_K})")

        selected, final_footprint, final_clear_est, n_cov = select_scenes_low_overlap_until_covered(
            client=client,
            candidates=cheap,
            aoi_geojson=aoi_geojson,
            aoi_geom4326=aoi_clean,
            utm_epsg=utm,
            aoi_date=aoi_date,
            target_footprint_frac=TARGET_FOOTPRINT_FRAC,
            min_footprint_gain=MIN_FOOTPRINT_GAIN,
            target_clear_frac=TARGET_CLEAR_FRAC,
            max_selected=MAX_SELECTED_PER_AOI,
            one_per_strip=SELECT_ONE_PER_STRIP,
        )

        print(
            f"[Select] used={len(selected)} | footprint_union={final_footprint*100:.2f}% | "
            f"clear_union_est={final_clear_est*100:.2f}% | coverage_calls={n_cov} | "
            f"one_per_strip(prefer)={SELECT_ONE_PER_STRIP}"
        )

        for c in selected:
            used_rows.append(
                {
                    "AOI": target,
                    "AOI_Index": aoi_idx,
                    "UTM": utm,
                    "WindowStart": start_iso,
                    "WindowEnd": end_iso,
                    "ProductBundle": PRODUCT_BUNDLE,

                    "ItemID": c.item_id,
                    "StripID": c.strip_id,
                    "Acquired": c.acquired.isoformat(),
                    "DateDeltaDays": c.date_delta_days,

                    "SceneCloudCover": c.scene_cloud_cover,      # 0..1

                    # This is clear over (AOI ∩ scene), NOT footprint completeness:
                    "AOIClearPct_overlap": c.clear_percent,       # 0..100
                    "AOICloudFrac_overlap": c.aoi_cloud_frac,     # 0..1
                    "CoverageStatus": c.coverage_status,

                    "RankKey": c.rank_key,

                    # NEW: footprint truth
                    "FootprintCoverPct": None if c.footprint_cover_frac is None else c.footprint_cover_frac * 100.0,
                    "FootprintGainPct": None if c.footprint_gain_frac is None else c.footprint_gain_frac * 100.0,

                    # NEW: final unions for the AOI
                    "FinalFootprintCoveragePct": final_footprint * 100.0,
                    "FinalClearCoverage_estPct": final_clear_est * 100.0,
                }
            )

    df = pd.DataFrame(used_rows)
    if not df.empty:
        sort_cols = [c for c in ["AOI", "AOI_Index", "StripID", "DateDeltaDays", "AOICloudFrac_overlap", "RankKey"] if c in df.columns]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n[Summary] Wrote USED-scene CSV: {OUT_CSV}")
    if not df.empty:
        print(df.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
