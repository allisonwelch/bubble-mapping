"""Build per-labeler quarter-chip labeling packs for the grouping + A/B/C exercise.

Allocation (2026-06-11): chips quartered at the median centroid x/y; three
labelers; every pack <= ~1000 label polygons.

  shared calibration (all three labelers): 4-SE, 38-SW, 21-NE
  L1 unique: 39-NW, 4-NW
  L2 unique: 39-SW, 4-NE
  L3 unique: 4-SW, 38-NW+NE, 41-NW+SW, 21-NW+SW+SE, 27, 52, 25 (whole chips)

Each output GPKG (one per labeler) contains two layers:
  * ``labels``      -- every polygon to label, schema identical to
                       gt_seeps_label_chip39.gpkg plus two extra columns:
                       ``unit`` (e.g. "39-NW") and ``is_context``.
                       Context rows (is_context=1) are polygons from outside
                       the unit but within 1.0 m (the historical p99 group
                       extent) of its boundary -- visible for grouping context,
                       NOT to be classified.
  * ``label_areas`` -- one rectangle per unit, the bounding box of ALL the
                       unit's polygons (label + context rows), so every
                       polygon in the labels layer sits inside a box. Black
                       outline style embedded via layer_styles. Adjacent
                       units' boxes may overlap slightly where context rings
                       meet.

Labeler rules (same as chip-39, plus the boundary rule):
  * group a cluster by setting every member's seep_group_id to the anchor
    (largest) bubble's own seep_id;
  * a group is YOURS iff its anchor is a non-context row (is_context = 0).
    If your group extends to context polygons, include them in the group;
    if the anchor itself is a context row, leave your non-context satellites
    alone (the group belongs to the neighboring unit);
  * classify (A/B/C) only non-context rows.
"""

import os
import sqlite3
import sys

import geopandas as gpd
import pandas as pd
from shapely.geometry import box
from shapely.ops import unary_union

REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
sys.path.insert(0, REPO_PATH)
from seep_classification_allocation import assign_strata  # noqa: E402

PRED_DIR = os.path.join(
    REPO_PATH, "data", "results", "SWIN", "AE", "20260428-1537_SWINxAE.weights"
)
GT_PATH = os.path.join(PRED_DIR, "gt_seeps.gpkg")
OUT_DIR = os.path.join(PRED_DIR, "labeling")

CONTEXT_RADIUS_M = 1.0   # historical p99 group extent: any group anchored
                         # inside a unit is fully visible within this buffer
BOX_PAD_M = 0.2          # visual breathing room around the label-area boxes

# (chip, quads) -- quads is a tuple of quadrant names, or "ALL" for the whole chip
CALIBRATION_UNITS = [
    ("4.tif", ("SE",)),
    ("38.tif", ("SW",)),
    ("21.tif", ("NE",)),
]
UNIQUE_UNITS = {
    "L1": [("39.tif", ("NW",)), ("4.tif", ("NW",))],
    "L2": [("39.tif", ("SW",)), ("4.tif", ("NE",))],
    "L3": [
        ("4.tif", ("SW",)),
        ("38.tif", ("NW", "NE")),
        ("41.tif", ("NW", "SW")),
        ("21.tif", ("NW", "SW", "SE")),
        ("27.tif", "ALL"),
        ("52.tif", "ALL"),
        ("25.tif", "ALL"),
    ],
}

# gt_seeps_label_chip39.gpkg column order (the schema to reproduce), with the
# two pack-specific additions slotted before geometry.
PACK_COLS = [
    "image", "seep_id", "class", "labeler", "is_calibration",
    "area_bin", "area_label", "sol_bin", "sol_label",
    "centroid_x_m", "centroid_y_m", "area_m2", "perim_m", "circularity",
    "solidity", "eccentricity", "mean_R", "mean_G", "mean_B",
    "seep_group_id", "is_pregrouped", "is_overgrouped",
    "unit", "is_context", "geometry",
]

def outline_qml(width):
    """Outline-only fill symbol QML (transparent fill, black border).

    width is in mm; 0 renders as a hairline in QGIS.
    """
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="Symbology" version="3.28">
 <renderer-v2 type="singleSymbol" enableorderby="0" forceraster="0" symbollevels="0">
  <symbols>
   <symbol type="fill" name="0" alpha="1" clip_to_extent="1" force_rhr="0">
    <layer class="SimpleFill" enabled="1" locked="0" pass="0">
     <prop k="color" v="0,0,0,0"/>
     <prop k="joinstyle" v="miter"/>
     <prop k="offset" v="0,0"/>
     <prop k="offset_unit" v="MM"/>
     <prop k="outline_color" v="0,0,0,255"/>
     <prop k="outline_style" v="solid"/>
     <prop k="outline_width" v="{width}"/>
     <prop k="outline_width_unit" v="MM"/>
     <prop k="style" v="no"/>
    </layer>
   </symbol>
  </symbols>
 </renderer-v2>
</qgis>
"""


def quadrant_masks(chip_gdf):
    """Boolean centroid-quadrant masks, split at the median centroid x/y."""
    mx = chip_gdf["centroid_x_m"].median()
    my = chip_gdf["centroid_y_m"].median()
    x, y = chip_gdf["centroid_x_m"], chip_gdf["centroid_y_m"]
    return {
        "NW": (x < mx) & (y >= my),
        "NE": (x >= mx) & (y >= my),
        "SW": (x < mx) & (y < my),
        "SE": (x >= mx) & (y < my),
    }, mx, my


def quadrant_rect(chip_gdf, quad, mx, my):
    """Quadrant rectangle (context-selection only; not the drawn outline)."""
    xmin, ymin, xmax, ymax = chip_gdf.total_bounds
    return {
        "NW": box(xmin, my, mx, ymax),
        "NE": box(mx, my, xmax, ymax),
        "SW": box(xmin, ymin, mx, my),
        "SE": box(mx, ymin, xmax, my),
    }[quad]


def bounds_box(*gdfs, pad=BOX_PAD_M):
    """Padded bounding box around every geometry in the given frames."""
    frames = [g for g in gdfs if len(g)]
    xmin, ymin, xmax, ymax = pd.concat(frames).total_bounds
    return box(xmin - pad, ymin - pad, xmax + pad, ymax + pad)


def build_unit(gt, chip, quads):
    """Return (members_gdf, context_gdf, area_geom, unit_name) for one unit.

    The drawn label-area box is the bounding box of all the unit's polygons
    (members + context) so nothing in the labels layer falls outside it.
    """
    g = gt[gt["image"] == chip]
    if g.empty:
        raise SystemExit(f"no polygons for chip {chip!r} in gt_seeps.gpkg")
    chip_stem = os.path.splitext(chip)[0]

    if quads == "ALL":
        return g.copy(), g.iloc[0:0].copy(), bounds_box(g), f"{chip_stem}-ALL"

    masks, mx, my = quadrant_masks(g)
    member_mask = pd.concat([masks[q] for q in quads], axis=1).any(axis=1)
    members = g[member_mask].copy()
    quad_geom = unary_union([quadrant_rect(g, q, mx, my) for q in quads])

    near = g.geometry.intersects(quad_geom.buffer(CONTEXT_RADIUS_M))
    context = g[near & ~member_mask].copy()
    area_geom = bounds_box(members, context)
    return members, context, area_geom, f"{chip_stem}-{'+'.join(quads)}"


def to_pack_rows(gdf, labeler, unit_name, is_calibration, is_context):
    rows = gdf.copy()
    rows["class"] = ""
    rows["labeler"] = labeler
    rows["is_calibration"] = is_calibration
    rows["seep_group_id"] = rows["seep_id"]
    rows["is_pregrouped"] = 0
    rows["is_overgrouped"] = 0
    rows["unit"] = unit_name
    rows["is_context"] = int(is_context)
    rows["area_label"] = rows["area_label"].astype(str)
    rows["sol_label"] = rows["sol_label"].astype(str)
    return rows


def embed_outline_styles(gpkg_path, layer_widths):
    """Register default black-outline QMLs in layer_styles.

    layer_widths: {layer_name: outline_width_mm} (0 = hairline).
    """
    con = sqlite3.connect(gpkg_path)
    try:
        con.execute(
            """CREATE TABLE IF NOT EXISTS layer_styles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                f_table_catalog TEXT(256), f_table_schema TEXT(256),
                f_table_name TEXT(256), f_geometry_column TEXT(256),
                styleName TEXT(30), styleQML TEXT, styleSLD TEXT,
                useAsDefault BOOLEAN, description TEXT, owner TEXT(30),
                ui TEXT(30), update_time DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )
        for layer_name, width in layer_widths.items():
            con.execute(
                """INSERT INTO layer_styles
                   (f_table_catalog, f_table_schema, f_table_name, f_geometry_column,
                    styleName, styleQML, styleSLD, useAsDefault, description, owner)
                   VALUES ('', '', ?, 'geom', 'black_outline', ?, '', 1,
                           'black outline, no fill', '')""",
                (layer_name, outline_qml(width)),
            )
        registered = con.execute(
            "SELECT 1 FROM gpkg_contents WHERE table_name = 'layer_styles'"
        ).fetchone()
        if not registered:
            con.execute(
                """INSERT INTO gpkg_contents (table_name, data_type, identifier)
                   VALUES ('layer_styles', 'attributes', 'layer_styles')"""
            )
        con.commit()
    finally:
        con.close()


def main():
    print(f"loading {os.path.basename(GT_PATH)} ...")
    gt = gpd.read_file(GT_PATH)
    print(f"  -> {len(gt)} GT polygons, CRS {gt.crs}")
    gt = assign_strata(gt)

    os.makedirs(OUT_DIR, exist_ok=True)
    grand_total = 0
    for labeler, unique_units in UNIQUE_UNITS.items():
        units = [(c, q, True) for c, q in CALIBRATION_UNITS] + [
            (c, q, False) for c, q in unique_units
        ]
        member_frames, context_frames = [], []
        area_rows = []
        print(f"\n=== {labeler} ===")
        for chip, quads, is_cal in units:
            members, context, area_geom, unit_name = build_unit(gt, chip, quads)
            member_frames.append(to_pack_rows(members, labeler, unit_name, is_cal, False))
            if len(context):
                context_frames.append(to_pack_rows(context, labeler, unit_name, is_cal, True))
            area_rows.append(
                {"unit": unit_name, "image": chip, "is_calibration": is_cal,
                 "n_label_polys": len(members), "geometry": area_geom}
            )
            tag = "cal " if is_cal else "uniq"
            print(f"  [{tag}] {unit_name:>12}: {len(members):>4} to label, {len(context):>3} context")

        members_all = pd.concat(member_frames, ignore_index=True)
        # context rows that are members of another unit in the same pack, or
        # duplicated across units, are dropped (each polygon appears once)
        if context_frames:
            context_all = pd.concat(context_frames, ignore_index=True)
            member_keys = set(zip(members_all["image"], members_all["seep_id"]))
            context_all = context_all[
                ~context_all.apply(lambda r: (r["image"], r["seep_id"]) in member_keys, axis=1)
            ]
            context_all = context_all.drop_duplicates(subset=["image", "seep_id"])
            labels = pd.concat([members_all, context_all], ignore_index=True)
        else:
            labels = members_all

        labels = gpd.GeoDataFrame(labels[PACK_COLS], geometry="geometry", crs=gt.crs)
        areas = gpd.GeoDataFrame(area_rows, geometry="geometry", crs=gt.crs)

        out_path = os.path.join(OUT_DIR, f"gt_seeps_label_quarters_{labeler}.gpkg")
        if os.path.exists(out_path):
            os.remove(out_path)
        labels.to_file(out_path, driver="GPKG", layer="labels")
        areas.to_file(out_path, driver="GPKG", layer="label_areas")
        embed_outline_styles(out_path, {"labels": 0, "label_areas": 0.8})

        n_label = int((labels["is_context"] == 0).sum())
        n_ctx = int((labels["is_context"] == 1).sum())
        grand_total += n_label
        print(f"  -> {os.path.basename(out_path)}: {n_label} to label "
              f"(+{n_ctx} context rows), {len(areas)} label-area outlines")

    print(f"\ntotal label-slots across labelers: {grand_total}")
    print("reminder: context rows (is_context=1) are grouping context only -- "
          "no class; a group is yours iff its anchor row has is_context=0")


if __name__ == "__main__":
    main()
