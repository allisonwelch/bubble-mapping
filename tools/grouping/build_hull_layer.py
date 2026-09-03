"""Precompute the per-seep convex hulls of a label pack into a STATIC layer, so
QGIS stops recomputing them on every pan/zoom.

WHY THIS EXISTS
  The `group_hull_by_class` style injected by tools/grouping/deploy_grouper.py draws
  the seep footprint with a geometry generator:

      convex_hull(collect($geometry, group_by:=concat("image",'-',"seep_group_id")))

  `collect(..., group_by:=)` is an AGGREGATE evaluated per rendered feature. QGIS
  caches it per group within one render pass, but each distinct group still costs
  a FULL-LAYER SCAN -- the group filter is a concat() expression, so no index can
  serve it. Cost per repaint is therefore
      (groups on screen) x (features in the WHOLE layer),
  independent of the map extent (zooming in does not shrink the scan). That is
  fine on a 1-4 chip quarter pack (~1-3k features) and unusable on the 48-chip
  all_chips pack (18,762 features / 11,888 groups -> millions of geometry fetches
  per pan).

  This tool dissolves the groups ONCE and writes the hulls as real polygons.
  Rendering then costs a spatial-index lookup like any normal vector layer.

TRADE-OFF (read this before using it for grouping work)
  The hull layer is a SNAPSHOT. It does NOT update when a labeler edits
  seep_group_id -- that live feedback is the entire point of the generator style.
  So:
    * REVIEW / QA / figures / whole-lake overview  -> use this static layer.
    * ACTIVELY CORRECTING GROUPING                 -> keep the generator style and
      instead put a subset filter on `labels` (QGIS Layer Properties -> Source ->
      Query Builder: "image" = '40.tif'). Provider subset strings ARE applied to
      the aggregate's feature requests, so the scan drops to that one chip.
  Re-run this tool after a grouping pass to refresh the snapshot (--layer is
  dropped and rebuilt, so re-running is idempotent).

WHAT IT WRITES (into the pack itself unless -o is given)
  * layer `seep_hulls`: one polygon per (image, seep_group_id) with the size /
    shape / class attributes, styled (default) as a dark footprint outline plus a
    thicker class-coloured outline -- the same visual language as the generator
    style it replaces.
  * a NON-default style `bubbles_plain` on `labels`: transparent fill, thin light
    outline. Switch `labels` to this (Layer Styling -> style dropdown) to actually
    get the speedup -- adding the hull layer does nothing on its own while
    `labels` is still rendering the generator.
  * SQLite indexes on labels(image, seep_group_id) and labels(image, seep_id).
    These do NOT speed the generator (its concat() filter is not index-usable)
    but they do speed the subset filter above, attribute-table sorting, and the
    (image, seep_group_id) joins every tool here does.

CLASS ON A GROUP: promoted by MAX (C > B > A), matching the promotion rule
established for the Prajna pack -- a seep is one physical feature, so members
that disagree are an artifact of judging bubbles in isolation, and max is the
flux-conservative read. Groups whose members disagreed are counted and flagged
in `class_mixed` so the label-QA slips stay visible instead of being smoothed
away.

Usage:
  python -m tools.grouping.build_hull_layer PACK.gpkg                 # in place (+ backup)
  python -m tools.grouping.build_hull_layer PACK.gpkg -o HULLS.gpkg   # sidecar, pack untouched
  python -m tools.grouping.build_hull_layer PACK.gpkg --no-backup --no-index
"""
import os
import sys
import time
import shutil
import sqlite3
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely

HULL_LAYER = "seep_hulls"
HULL_STYLE = "seep_hulls_by_class"
BUBBLE_STYLE = "bubbles_plain"
# class outline colors (r,g,b) -- ColorBrewer Dark2, same as deploy_grouper.
CLASS_RGB = {"A": "27,158,119", "B": "217,95,2", "C": "117,112,179"}
CLASS_RANK = {"": 0, "A": 1, "B": 2, "C": 3}
RANK_CLASS = {v: k for k, v in CLASS_RANK.items()}


# ---------------------------------------------------------------------------
# hull construction
# ---------------------------------------------------------------------------
def _oriented_axes(geoms):
    """(major_m, minor_m) of each geometry's minimum rotated rectangle.

    This is the L x W the historical field workbooks actually measured, so it is
    the axis pair the classifier features use -- computed here on the SAME hull
    the map shows, so the layer's attributes agree with what the eye sees.
    """
    major = np.full(len(geoms), np.nan)
    minor = np.full(len(geoms), np.nan)
    for i, g in enumerate(geoms):
        if g is None or g.is_empty:
            continue
        xy = np.asarray(shapely.get_coordinates(g.minimum_rotated_rectangle))
        if len(xy) < 4:
            # degenerate hull (single bubble collapsed to a point/line)
            major[i] = minor[i] = 0.0
            continue
        seg = np.linalg.norm(np.diff(xy[:5], axis=0), axis=1)
        major[i], minor[i] = float(seg[:2].max()), float(seg[:2].min())
    return major, minor


def build_hulls(pack):
    """labels -> one convex-hull polygon per (image, seep_group_id)."""
    t0 = time.time()
    gdf = gpd.read_file(pack, layer="labels")
    n_in = len(gdf)

    # seep_group_id defaults to the row's own seep_id (the pack convention); a
    # NULL here means "never grouped", not "group 0".
    sg = pd.to_numeric(gdf.get("seep_group_id"), errors="coerce")
    sg = sg.fillna(pd.to_numeric(gdf["seep_id"], errors="coerce")).astype("int64")
    gdf = gdf.assign(_sg=sg)
    # ids restart at 1 in every chip, so the group key is the (image, id) TUPLE.
    # Never key on seep_group_id alone (it silently merges seeps across chips).
    gdf["seep_uid"] = gdf["image"].astype(str) + "::" + gdf["_sg"].astype(str)

    hull = (gdf.dissolve(by="seep_uid")
               .geometry.convex_hull
               .rename("geometry"))

    cls = gdf["class"].fillna("").astype(str).str.strip().str.upper()
    gdf["_rank"] = cls.map(CLASS_RANK).fillna(0).astype(int)

    def _any_flag(col):
        if col not in gdf.columns:
            return None
        return (pd.to_numeric(gdf[col], errors="coerce").fillna(0).astype(int)
                .groupby(gdf["seep_uid"]).max())

    agg = pd.DataFrame({
        "image": gdf.groupby("seep_uid")["image"].first(),
        "seep_group_id": gdf.groupby("seep_uid")["_sg"].first(),
        "n_bubbles": gdf.groupby("seep_uid").size(),
        "bubble_area_m2": (pd.to_numeric(gdf["area_m2"], errors="coerce")
                           .groupby(gdf["seep_uid"]).sum()),
        "_rank_max": gdf.groupby("seep_uid")["_rank"].max(),
        # a group is "mixed" only if two members carry DIFFERENT non-blank
        # classes; blank members are unlabeled, not disagreement.
        "class_mixed": (gdf[gdf["_rank"] > 0].groupby("seep_uid")["_rank"]
                        .nunique().reindex(hull.index).fillna(0).gt(1)
                        .astype(int)),
    })
    for col in ("is_pregrouped", "is_overgrouped", "is_context"):
        flag = _any_flag(col)
        if flag is not None:
            agg[col] = flag
    if "group_source" in gdf.columns:
        # 'fixed' wins: a group holding any human-fixed bubble is human work.
        agg["group_source"] = np.where(
            gdf.assign(_f=gdf["group_source"].astype(str).eq("fixed"))
               .groupby("seep_uid")["_f"].max().reindex(hull.index).fillna(False),
            "fixed", "model")

    agg["class"] = agg.pop("_rank_max").map(RANK_CLASS).fillna("")

    out = gpd.GeoDataFrame(agg, geometry=hull.values, crs=gdf.crs)
    out["hull_area_m2"] = out.geometry.area
    maj, mnr = _oriented_axes(out.geometry.values)
    out["major_axis_m"], out["minor_axis_m"] = maj, mnr
    # gas vs interstitial ice; computed from the GROUPING, not drawn vertices.
    # bubble_area_m2 SUMS the member polygons, so where two drawn bubbles overlap
    # the shared area is double-counted and fill_ratio can land just above 1.0
    # (4 of 11,888 groups on the all-chips pack, all n_bubbles==2, max 1.18).
    # That is a property of the drawn labels, not of the hull -- left as-is so it
    # matches the classifier's fill_ratio definition rather than silently clipping.
    with np.errstate(invalid="ignore", divide="ignore"):
        out["fill_ratio"] = np.where(out["hull_area_m2"] > 0,
                                     out["bubble_area_m2"] / out["hull_area_m2"],
                                     np.nan)
    out = out.reset_index()  # seep_uid becomes a column

    cols = ["seep_uid", "image", "seep_group_id", "class", "class_mixed",
            "n_bubbles", "hull_area_m2", "bubble_area_m2", "fill_ratio",
            "major_axis_m", "minor_axis_m"]
    cols += [c for c in ("is_pregrouped", "is_overgrouped", "is_context",
                         "group_source") if c in out.columns]
    out = out[cols + ["geometry"]]
    print(f"[hull] {n_in} bubbles -> {len(out)} seeps "
          f"({int((out['n_bubbles'] > 1).sum())} multi-bubble) "
          f"in {time.time() - t0:.1f}s")
    mixed = int(out["class_mixed"].sum())
    if mixed:
        print(f"[hull]   WARNING {mixed} groups mix non-blank classes; "
              f"promoted by MAX (C>B>A) and flagged in class_mixed")
    return out


# ---------------------------------------------------------------------------
# QGIS styles (static -- no geometry generator, no aggregates)
# ---------------------------------------------------------------------------
def _x(s):
    return (s.replace("&", "&amp;").replace('"', "&quot;")
             .replace("<", "&lt;").replace(">", "&gt;"))


def _opt(name, value, typ="QString"):
    return f'<Option type="{typ}" name="{name}" value="{value}"/>'


def _fill_layer(opts, ddp="", pass_=0):
    return (f'<layer class="SimpleFill" enabled="1" pass="{pass_}" locked="0">'
            f'<Option type="Map">{opts}</Option>{ddp}</layer>')


def _qml(layers):
    return f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
 <renderer-v2 type="singleSymbol" forceraster="0" symbollevels="0" enableorderby="0" referencescale="-1">
  <symbols>
   <symbol type="fill" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10">
    {layers}
   </symbol>
  </symbols>
 </renderer-v2>
 <blendMode>0</blendMode>
 <featureBlendMode>0</featureBlendMode>
</qgis>'''


def hull_qml():
    """Dark footprint outline + a thicker class-coloured outline on top.

    Deliberately the same look as deploy_grouper's generator style, so
    swapping to the precomputed layer is invisible on the map. Outline widths in
    MM (screen-constant) -- MapUnit renders metre-thick borders on a UTM layer.
    """
    case = ("CASE"
            + "".join(f" WHEN \"class\"='{k}' THEN '{rgb},255'"
                      for k, rgb in CLASS_RGB.items())
            + " ELSE '0,0,0,0' END")
    base = "".join([
        _opt("color", "255,255,255,0"), _opt("outline_color", "50,50,50,255"),
        _opt("outline_width", "0.3"), _opt("outline_width_unit", "MM"),
        _opt("style", "no"), _opt("outline_style", "solid"),
        _opt("offset", "0,0"), _opt("offset_unit", "MM"),
    ])
    klass = "".join([
        _opt("color", "255,255,255,0"), _opt("outline_color", "0,0,0,0"),
        _opt("outline_width", "0.6"), _opt("outline_width_unit", "MM"),
        _opt("style", "no"), _opt("outline_style", "solid"),
        _opt("offset", "0,0"), _opt("offset_unit", "MM"),
    ])
    ddp = ('<data_defined_properties><Option type="Map">'
           '<Option type="QString" name="name" value=""/>'
           '<Option type="Map" name="properties">'
           '<Option type="Map" name="outlineColor">'
           '<Option type="bool" name="active" value="true"/>'
           f'<Option type="QString" name="expression" value="{_x(case)}"/>'
           '<Option type="int" name="type" value="3"/>'
           '</Option></Option>'
           '<Option type="QString" name="type" value="collection"/>'
           '</Option></data_defined_properties>')
    return _qml(_fill_layer(base) + _fill_layer(klass, ddp, pass_=1))


def bubble_qml():
    """Plain bubbles: transparent fill, thin light outline, no aggregates."""
    opts = "".join([
        _opt("color", "255,255,255,0"), _opt("outline_color", "120,120,120,200"),
        _opt("outline_width", "0.2"), _opt("outline_width_unit", "MM"),
        _opt("style", "no"), _opt("outline_style", "solid"),
        _opt("offset", "0,0"), _opt("offset_unit", "MM"),
    ])
    return _qml(_fill_layer(opts))


# ---------------------------------------------------------------------------
# sqlite plumbing
# ---------------------------------------------------------------------------
def _table_exists(cur, name):
    return cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (name,)).fetchone() is not None


def drop_layer(gpkg, layer):
    """Remove a features layer + all its GPKG bookkeeping, so a rebuild appends
    into a clean slot instead of duplicating features."""
    con = sqlite3.connect(gpkg)
    cur = con.cursor()
    known = cur.execute("SELECT 1 FROM gpkg_contents WHERE table_name=?",
                        (layer,)).fetchone()
    if not (known or _table_exists(cur, layer)):
        con.close()
        return False
    try:
        # triggers first (rtree triggers fire ON the feature table and call ST_*
        # functions a plain sqlite3 connection does not have).
        trigs = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name=?",
            (layer,)).fetchall()]
        aux = [(r[0], r[1]) for r in cur.execute(
            "SELECT name, type FROM sqlite_master WHERE name LIKE ?",
            (f"rtree_{layer}_%",)).fetchall()]
        for name in trigs + [n for n, t in aux if t == "trigger"]:
            cur.execute(f'DROP TRIGGER IF EXISTS "{name}"')
        # dropping the rtree VIRTUAL table takes its _rowid/_node/_parent
        # shadow tables with it; IF EXISTS covers whatever order they land in.
        for name, typ in aux:
            if typ == "table":
                cur.execute(f'DROP TABLE IF EXISTS "{name}"')
        cur.execute(f'DROP TABLE IF EXISTS "{layer}"')
        cur.execute("DELETE FROM gpkg_geometry_columns WHERE table_name=?", (layer,))
        cur.execute("DELETE FROM gpkg_contents WHERE table_name=?", (layer,))
        for t in ("gpkg_extensions", "gpkg_ogr_contents", "layer_styles"):
            if _table_exists(cur, t):
                col = "f_table_name" if t == "layer_styles" else "table_name"
                cur.execute(f'DELETE FROM "{t}" WHERE {col}=?', (layer,))
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()
    return True


def inject_style(gpkg, table, name, qml, as_default, description=""):
    con = sqlite3.connect(gpkg)
    cur = con.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS layer_styles ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT, "
        "f_table_schema TEXT, f_table_name TEXT, f_geometry_column TEXT, "
        "styleName TEXT, styleQML TEXT, styleSLD TEXT, useAsDefault BOOLEAN, "
        "description TEXT, owner TEXT, ui TEXT, update_time DATETIME)")
    cur.execute("DELETE FROM layer_styles WHERE styleName=? AND f_table_name=?",
                (name, table))
    if as_default:
        cur.execute("UPDATE layer_styles SET useAsDefault=0 WHERE f_table_name=?",
                    (table,))
    cur.execute(
        "INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, "
        "f_geometry_column, styleName, styleQML, styleSLD, useAsDefault, "
        "description, owner, ui, update_time) "
        "VALUES ('','',?,'geom',?,?,'',?,?,'','',datetime('now'))",
        (table, name, qml, 1 if as_default else 0, description))
    con.commit()
    con.close()


def add_indexes(gpkg, table="labels"):
    """Attribute indexes for the (image, seep_group_id) / (image, seep_id) keys
    every tool here joins on. GDAL ignores extra indexes, so this is safe."""
    con = sqlite3.connect(gpkg)
    cur = con.cursor()
    have = {r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')}
    made = []
    for cols in (("image", "seep_group_id"), ("image", "seep_id")):
        if not set(cols) <= have:
            continue
        idx = f"{table}_{'_'.join(cols)}_idx"
        cur.execute(f'CREATE INDEX IF NOT EXISTS "{idx}" '
                    f'ON "{table}" ({", ".join(cols)})')
        made.append(idx)
    cur.execute("ANALYZE")
    con.commit()
    con.close()
    return made


def verify(gpkg, hull_layer):
    """Confirm the write added a layer and took nothing away.

    In sidecar mode the target holds only the hulls, so `labels` is absent -- its
    counts come back as None rather than failing the check.
    """
    con = sqlite3.connect(gpkg)
    cur = con.cursor()
    has_labels = _table_exists(cur, "labels")
    out = {
        "labels": (cur.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
                   if has_labels else None),
        hull_layer: cur.execute(
            f'SELECT COUNT(*) FROM "{hull_layer}"').fetchone()[0],
        "styles": (cur.execute(
            "SELECT f_table_name, styleName, useAsDefault FROM layer_styles "
            "ORDER BY f_table_name, styleName").fetchall()
            if _table_exists(cur, "layer_styles") else []),
        "labels_triggers": cur.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
            "AND tbl_name='labels'").fetchone()[0],
        "labels_rtree": cur.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE name='rtree_labels_geom'"
        ).fetchone()[0],
    }
    con.close()
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Precompute per-seep convex hulls into a static GPKG layer "
                    "so QGIS stops recomputing them on every pan/zoom.")
    ap.add_argument("pack", help="label pack .gpkg (must have a `labels` layer)")
    ap.add_argument("-o", "--out", default=None,
                    help="write the hull layer to this NEW gpkg instead of into "
                         "the pack (leaves the pack completely untouched)")
    ap.add_argument("--layer", default=HULL_LAYER)
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the pre-write copy of the pack (in-place mode only)")
    ap.add_argument("--no-index", action="store_true",
                    help="skip the labels(image,...) attribute indexes")
    args = ap.parse_args()

    if not os.path.exists(args.pack):
        raise SystemExit(f"no such pack: {args.pack}")
    hulls = build_hulls(args.pack)

    sidecar = args.out is not None
    target = args.out or args.pack
    if sidecar:
        if os.path.exists(target):
            os.remove(target)
    elif not args.no_backup:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bak = f"{os.path.splitext(args.pack)[0]}.pre_hulls_{stamp}.gpkg"
        shutil.copyfile(args.pack, bak)
        print(f"[hull] backup -> {bak}")

    # A rebuild must not append into the old layer; drop it first.
    if os.path.exists(target) and drop_layer(target, args.layer):
        print(f"[hull] dropped existing layer '{args.layer}'")

    # mode='a' opens the existing gpkg for UPDATE and creates just this layer,
    # so `labels`, its RTree index and layer_styles are never rewritten.
    hulls.to_file(target, layer=args.layer, driver="GPKG",
                  mode="a" if os.path.exists(target) else "w")

    inject_style(target, args.layer, HULL_STYLE, hull_qml(), as_default=True,
                 description="precomputed seep hull; class = thicker outline")
    if not sidecar:
        # NOT default: leaves the pack opening the way the labeler left it.
        inject_style(target, "labels", BUBBLE_STYLE, bubble_qml(),
                     as_default=False,
                     description="plain bubbles (no hull generator) -- pair with "
                                 f"the {args.layer} layer")
        if not args.no_index:
            made = add_indexes(target)
            print(f"[hull] indexes: {', '.join(made) if made else 'none added'}")

    v = verify(target, args.layer)
    print(f"[hull] wrote {target}")
    print(f"[hull]   {args.layer}={v[args.layer]} rows")
    if v["labels"] is not None:
        print(f"[hull]   labels={v['labels']} rows (untouched), triggers="
              f"{v['labels_triggers']}, rtree_labels_geom="
              f"{'present' if v['labels_rtree'] else 'MISSING'}")
    for t, s, d in v["styles"]:
        print(f"[hull]   style {t}:{s}{' (default)' if d else ''}")
    if sidecar:
        print(f"[hull] NOTE sidecar mode: load {target} alongside the pack, and "
              f"turn OFF the pack's hull-generator style to get the speedup.")
    else:
        print(f"[hull] NEXT in QGIS: add the '{args.layer}' layer, then switch "
              f"`labels` to the '{BUBBLE_STYLE}' style (Layer Styling panel -> "
              f"style dropdown). Leaving `labels` on the generator style keeps "
              f"the slow render.")
    print("[hull] REMINDER this layer is a SNAPSHOT -- re-run after any grouping "
          "edits. For live hulls while editing, keep the generator style and put "
          "a subset filter (\"image\" = '39.tif') on `labels` instead.")


if __name__ == "__main__":
    sys.exit(main())