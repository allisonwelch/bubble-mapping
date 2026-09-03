"""Deploy the learned pairwise grouper onto a label pack: assign seep_group_id
to its bubbles and bake a 'group-hull-by-class' QGIS style, so labelers only
review/correct grouping and fill `class` (the first-order flux lever).

Pipeline
  1. train the pairwise 'same-seep?' model on the COMBINED Katey labels
     (chip-39 + quarter packs) -- see seep_pairwise_grouper_quarters.py.
  2. for the target pack, per image: candidate pairs within 0.5 m among
     eligible bubbles -> P(same) -> threshold -> connected components -> seeps.
     seep_group_id = the MAX seep_id among a component's members (anchor-id
     convention; collision-safe within a chip; matches the labeler rubric).
  3. write seep_group_id back IN PLACE on a copy (sqlite UPDATE by fid, so the
     geometry, spatial index, and existing layer_styles are untouched). Also
     freeze the model's proposal in an immutable seep_group_id_pred column +
     a group_source (model|fixed) flag, so a returned (labeler-edited) pack can
     be diffed back into grouper training data (tools/seep_grouper_corrections.py).
  4. inject a named style 'group_hull_by_class':
        * live convex-hull geometry generator per (image, seep_group_id) ->
          consistent dark outline = the SEEP footprint;
        * hull fill data-defined by `class`: blank=transparent, A/B/C = fixed
          colors at 50% opacity;
        * the bubble polygons drawn on top with a thin light outline (context +
          the edit target).

ELIGIBILITY: is_pregrouped==1 or is_overgrouped==1 bubbles stay as their own
seep (fixed singleton; never merged). NOTE pregrouped is a human judgment not
set in fresh packs -- pass --flag-envelope-area-m2 to auto-flag big envelope
polygons as fixed singletons before grouping, or flag them in QGIS first.

Usage:
  python tools/seep_grouper_deploy.py IN.gpkg [-o OUT.gpkg] [--thr 0.6]
         [--flag-envelope-area-m2 0.15]
"""
import os
import sys
import shutil
import sqlite3
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seep_pairwise_grouper_quarters import (  # noqa: E402
    load_labeled, build_pairs, FEATURES, CAND_RADIUS, AGGLOM_CAP_M)

STYLE_NAME = "group_hull_by_class"
# class fill colors (r,g,b) -- ColorBrewer Dark2; alpha 128 == 50% opacity.
CLASS_RGB = {"A": "27,158,119", "B": "217,95,2", "C": "117,112,179"}


def train_model(thr_note=True):
    L, fields = load_labeled()
    L, feat, ii, jj, y, img = build_pairs(L, fields)
    X = feat[FEATURES].to_numpy(float)
    clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(X, y)
    print(f"[train] {len(y)} pairs ({int(y.sum())} same) -> RandomForest fit "
          f"on {L['image'].nunique()} images")
    return clf


def _pair_features(sub_elig, fx, fy, fa):
    """Within-image candidate pairs + the FEATURES for an eligible-bubble frame.
    Density/anchor context is measured against the full per-image field
    (fx, fy, fa)."""
    xy = sub_elig[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
    if len(xy) < 2:
        return np.empty((0, 2), int), pd.DataFrame(columns=FEATURES)
    tree = cKDTree(xy)
    pp = tree.query_pairs(r=CAND_RADIUS, output_type="ndarray")
    if len(pp) == 0:
        return np.empty((0, 2), int), pd.DataFrame(columns=FEATURES)
    ftree = cKDTree(np.column_stack([fx, fy]))
    a = sub_elig["area_m2"].to_numpy(float)
    b = ((sub_elig["mean_R"] + sub_elig["mean_G"] + sub_elig["mean_B"]) / 3.0).to_numpy(float)
    c = sub_elig["circularity"].fillna(sub_elig["circularity"].median()).to_numpy(float)
    e = sub_elig["eccentricity"].fillna(sub_elig["eccentricity"].median()).to_numpy(float)
    s = sub_elig["solidity"].fillna(sub_elig["solidity"].median()).to_numpy(float)
    rows = []
    for p, q in pp:
        mid = (xy[p] + xy[q]) / 2.0
        d025 = max(0, len(ftree.query_ball_point(mid, 0.25)) - 2)
        d050 = max(0, len(ftree.query_ball_point(mid, 0.50)) - 2)
        nbr = ftree.query_ball_point(mid, 0.50)
        loc_max = float(fa[nbr].max()) if len(nbr) else 0.0
        pair_max = max(a[p], a[q])
        rows.append({
            "dist": float(np.hypot(xy[p, 0] - xy[q, 0], xy[p, 1] - xy[q, 1])),
            "size_ratio": min(a[p], a[q]) / max(a[p], a[q]),
            "max_area": pair_max, "min_area": min(a[p], a[q]),
            "bright_diff": abs(b[p] - b[q]),
            "dens_025": d025, "dens_050": d050,
            "circ_mean": (c[p] + c[q]) / 2.0, "ecc_mean": (e[p] + e[q]) / 2.0,
            "sol_mean": (s[p] + s[q]) / 2.0, "circ_diff": abs(c[p] - c[q]),
            "loc_max_area": loc_max,
            "anchor_ratio": loc_max / max(pair_max, 1e-9),
        })
    return pp, pd.DataFrame(rows)


def constrained_cluster(m, pairs, proba, xy, cap):
    """Diameter-capped agglomeration (complete-linkage style).

    Merge candidate pairs in DESCENDING P(same), but reject any merge whose
    resulting cluster would have a max pairwise member-centroid span exceeding
    `cap` (the historical agglomeration major-axis cap). Prevents chains of
    short (<=CAND_RADIUS) edges from bridging into a seep wider than physically
    plausible. Returns an integer label per bubble (0..m-1)."""
    parent = list(range(m))
    members = {i: [i] for i in range(m)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k in np.argsort(-proba):
        i, j = int(pairs[k, 0]), int(pairs[k, 1])
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        union = members[ri] + members[rj]
        pts = xy[union]
        # max pairwise distance within the proposed merged cluster
        dmax = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)).max()
        if dmax <= cap:
            parent[rj] = ri
            members[ri] = union
            del members[rj]
    roots = np.array([find(i) for i in range(m)])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def group_pack(df, clf, thr, cap=AGGLOM_CAP_M, flag_envelope_area_m2=None):
    """Return a Series fid->seep_group_id for the pack frame `df`."""
    df = df.copy()
    if flag_envelope_area_m2 is not None:
        big = df["area_m2"] >= float(flag_envelope_area_m2)
        df.loc[big, "is_pregrouped"] = 1
        print(f"[deploy] auto-flagged {int(big.sum())} envelope polygons "
              f"(area >= {flag_envelope_area_m2} m2) as fixed singletons")
    # FIXED bubbles are excluded from model pairing and keep their EXISTING
    # seep_group_id (the model never touches human work). A bubble is fixed if:
    #   * it is already human-processed (class non-empty) -> preserves existing
    #     human groupings on partially-done packs like katey_kwa; or
    #   * is_overgrouped==1 (already wrongly spans seeps; must not absorb more).
    # class / is_pregrouped / is_overgrouped are never written by deploy, so
    # they are carried through unchanged. is_pregrouped envelopes that are NOT
    # yet processed stay eligible anchors (allison/prajna); processed envelopes
    # are fixed via the class rule.
    cls = df["class"].fillna("").astype(str).str.strip()
    fixed = (cls != "") | (df["is_overgrouped"].fillna(0).astype(int) == 1)
    out = df["seep_group_id"].astype(np.int64).copy()   # preserve existing ids
    n_merged_groups = 0
    for im, sub in df.groupby("image"):
        fx = sub["centroid_x_m"].to_numpy(float)
        fy = sub["centroid_y_m"].to_numpy(float)
        fa = sub["area_m2"].to_numpy(float)
        elig = sub[~fixed.loc[sub.index]]
        pp, feat = _pair_features(elig, fx, fy, fa)
        if len(pp) == 0:
            continue
        proba = clf.predict_proba(feat[FEATURES].to_numpy(float))[:, 1]
        keep = proba >= thr
        m = len(elig)
        elig_xy = elig[["centroid_x_m", "centroid_y_m"]].to_numpy(float)
        comp = constrained_cluster(m, pp[keep], proba[keep], elig_xy, cap)
        elig_ids = elig["seep_id"].to_numpy(np.int64)
        for cid in np.unique(comp):
            members = np.where(comp == cid)[0]
            anchor = int(elig_ids[members].max())          # collision-safe
            if len(members) > 1:
                n_merged_groups += 1
            out.loc[elig.index[members]] = anchor
    n_fixed_groups = df.loc[fixed, "seep_group_id"].nunique()
    print(f"[deploy] preserved {int(fixed.sum())} human-processed/overgrouped "
          f"bubbles in {n_fixed_groups} existing groups (untouched)")
    print(f"[deploy] model-grouped {int((~fixed).sum())} remaining bubbles -> "
          f"{n_merged_groups} new multi-bubble seeps; "
          f"{int(((~fixed) & (df['is_pregrouped'].fillna(0).astype(int)==1)).sum())} "
          f"unprocessed envelopes eligible as anchors")
    return out


# ---------------------------------------------------------------------------
# QGIS style (QML): group-hull-by-class
# ---------------------------------------------------------------------------
def build_qml():
    # class -> OUTLINE color CASE (solid for A/B/C, transparent when blank).
    # Class is now shown as a thicker outline on top of the hull outline,
    # NOT as a fill (so the imagery under the hull stays visible).
    case = ("CASE"
            + "".join(f" WHEN \"class\"='{k}' THEN '{rgb},255'"
                      for k, rgb in CLASS_RGB.items())
            + " ELSE '0,0,0,0' END")
    hull_expr = ("convex_hull(collect($geometry, "
                 "group_by:=concat(\"image\",'-',\"seep_group_id\")))")

    def opt(name, value, typ="QString"):
        return f'<Option type="{typ}" name="{name}" value="{value}"/>'

    # bubble fill sub-options (transparent fill, thin light outline).
    # Outline width in MM (screen-constant) -- MapUnit on a UTM layer renders
    # metre-thick borders that swamp the map.
    bubble_opts = "".join([
        opt("color", "255,255,255,0"), opt("outline_color", "120,120,120,200"),
        opt("outline_width", "0.2"), opt("outline_width_unit", "MM"),
        opt("style", "no"), opt("outline_style", "solid"),
        opt("offset", "0,0"), opt("offset_unit", "MM"),
    ])
    # hull base outline: transparent fill, thin constant dark outline -- the
    # consistent seep footprint, drawn for every group regardless of class.
    hull_base_opts = "".join([
        opt("color", "255,255,255,0"), opt("outline_color", "50,50,50,255"),
        opt("outline_width", "0.3"), opt("outline_width_unit", "MM"),
        opt("style", "no"), opt("outline_style", "solid"),
        opt("offset", "0,0"), opt("offset_unit", "MM"),
    ])
    # hull class outline: no fill, a slightly thicker outline whose color is
    # data-defined by `class` (transparent when blank). Drawn AFTER the base
    # layer so it renders on top of the hull area outline.
    hull_class_opts = "".join([
        opt("color", "255,255,255,0"), opt("outline_color", "0,0,0,0"),
        opt("outline_width", "0.6"), opt("outline_width_unit", "MM"),
        opt("style", "no"), opt("outline_style", "solid"),
        opt("offset", "0,0"), opt("offset_unit", "MM"),
    ])
    ddp = (
        '<data_defined_properties><Option type="Map">'
        '<Option type="QString" name="name" value=""/>'
        '<Option type="Map" name="properties">'
        '<Option type="Map" name="outlineColor">'
        '<Option type="bool" name="active" value="true"/>'
        f'<Option type="QString" name="expression" value="{_x(case)}"/>'
        '<Option type="int" name="type" value="3"/>'
        '</Option></Option>'
        '<Option type="QString" name="type" value="collection"/>'
        '</Option></data_defined_properties>')

    qml = f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="Symbology">
 <renderer-v2 type="singleSymbol" forceraster="0" symbollevels="0" enableorderby="0" referencescale="-1">
  <symbols>
   <symbol type="fill" name="0" alpha="1" clip_to_extent="1" force_rhr="0" frame_rate="10">
    <layer class="GeometryGenerator" enabled="1" pass="0" locked="0">
     <Option type="Map">
      <Option type="QString" name="SymbolType" value="Fill"/>
      <Option type="QString" name="geometryModifier" value="{_x(hull_expr)}"/>
      <Option type="QString" name="units" value="MapUnit"/>
     </Option>
     <symbol type="fill" name="@0@0" alpha="1" clip_to_extent="1" force_rhr="0">
      <layer class="SimpleFill" enabled="1" pass="0" locked="0">
       <Option type="Map">{hull_base_opts}</Option>
      </layer>
      <layer class="SimpleFill" enabled="1" pass="1" locked="0">
       <Option type="Map">{hull_class_opts}</Option>
       {ddp}
      </layer>
     </symbol>
    </layer>
    <layer class="SimpleFill" enabled="1" pass="0" locked="0">
     <Option type="Map">{bubble_opts}</Option>
    </layer>
   </symbol>
  </symbols>
 </renderer-v2>
 <blendMode>0</blendMode>
 <featureBlendMode>0</featureBlendMode>
</qgis>'''
    return qml


def _x(s):
    return (s.replace("&", "&amp;").replace('"', "&quot;")
             .replace("<", "&lt;").replace(">", "&gt;"))


def _ensure_columns(cur, table, cols):
    """ADD COLUMN for any of `cols` (name->sqltype) missing from `table`.
    Idempotent so re-deploying a pack doesn't error."""
    have = {r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')}
    for name, typ in cols.items():
        if name not in have:
            cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {typ}')


def inject_style(gpkg, qml, name=STYLE_NAME):
    con = sqlite3.connect(gpkg)
    cur = con.cursor()
    # geopandas-written gpkgs have no layer_styles table; create it if absent.
    cur.execute(
        "CREATE TABLE IF NOT EXISTS layer_styles ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT, "
        "f_table_schema TEXT, f_table_name TEXT, f_geometry_column TEXT, "
        "styleName TEXT, styleQML TEXT, styleSLD TEXT, useAsDefault BOOLEAN, "
        "description TEXT, owner TEXT, ui TEXT, update_time DATETIME)")
    cur.execute("DELETE FROM layer_styles WHERE styleName=? AND f_table_name='labels'",
                (name,))
    # make ours the default so the pack opens straight into the hull view
    cur.execute("UPDATE layer_styles SET useAsDefault=0 WHERE f_table_name='labels'")
    cur.execute(
        "INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name, "
        "f_geometry_column, styleName, styleQML, styleSLD, useAsDefault, description, "
        "owner, ui, update_time) VALUES ('','','labels','geom',?,?,?,1,?,'','', "
        "datetime('now'))",
        (name, qml, "", "live hull per (image,seep_group_id); class=thicker outline"))
    con.commit()
    con.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--thr", type=float, default=0.6,
                    help="P(same) edge threshold (RF best-bias op point ~0.6)")
    ap.add_argument("--cap", type=float, default=AGGLOM_CAP_M,
                    help="max seep span (m); merges exceeding it are rejected "
                         "(historical agglomeration major-axis cap, default 1.0)")
    ap.add_argument("--flag-envelope-area-m2", type=float, default=None)
    args = ap.parse_args()
    out = args.out or args.pack.replace(".gpkg", "_grouped.gpkg")
    if os.path.abspath(out) == os.path.abspath(args.pack):
        raise SystemExit("refusing to overwrite the input in place; pass -o")

    clf = train_model()

    con = sqlite3.connect(args.pack)
    df = pd.read_sql_query(
        "SELECT fid, image, seep_id, COALESCE(seep_group_id, seep_id) AS seep_group_id, "
        "class, centroid_x_m, centroid_y_m, area_m2, "
        "circularity, eccentricity, solidity, mean_R, mean_G, mean_B, "
        "is_pregrouped, COALESCE(is_overgrouped,0) AS is_overgrouped "
        "FROM labels", con)
    con.close()
    print(f"[deploy] {args.pack}: {len(df)} bubbles, "
          f"{df['image'].nunique()} images {df['image'].value_counts().to_dict()}")

    sgid = group_pack(df, clf, args.thr, cap=args.cap,
                      flag_envelope_area_m2=args.flag_envelope_area_m2)

    # Provenance for FUTURE grouper tuning: freeze the model's proposed grouping
    # in an immutable column (seep_group_id_pred) the labeler never edits, plus a
    # group_source flag (model|fixed). Labeler corrections land in seep_group_id;
    # the pair-level diff (seep_group_id vs seep_group_id_pred) on group_source==
    # 'model' rows is the correction record consumed by
    # tools/seep_grouper_corrections.py. fixed rows are human work (class set) or
    # is_overgrouped singletons the model was told to preserve, so they are NOT
    # scored model-vs-human (would be circular).
    cls = df["class"].fillna("").astype(str).str.strip()
    fixed = (cls != "") | (df["is_overgrouped"].fillna(0).astype(int) == 1)
    src = np.where(fixed.to_numpy(), "fixed", "model")

    shutil.copyfile(args.pack, out)
    con = sqlite3.connect(out)
    cur = con.cursor()
    _ensure_columns(cur, "labels",
                    {"seep_group_id_pred": "INTEGER", "group_source": "TEXT"})
    # GDAL RTree triggers call ST_* functions a plain sqlite3 conn lacks and
    # fire on UPDATE; drop them around the non-spatial write, recreate after.
    # Geometry is never touched, so the spatial index stays valid.
    trigs = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name='labels' AND sql IS NOT NULL").fetchall()
    try:
        for name, _ in trigs:
            con.execute(f'DROP TRIGGER IF EXISTS "{name}"')
        sg = sgid.astype(int).tolist()
        cur.executemany(
            "UPDATE labels SET seep_group_id=?, seep_group_id_pred=?, "
            "group_source=? WHERE fid=?",
            list(zip(sg, sg, src.tolist(), df["fid"].astype(int).tolist())))
        for _, sql in trigs:
            con.execute(sql)
        con.commit()
    except Exception:
        con.rollback()
        for _, sql in trigs:
            try:
                con.execute(sql)
            except sqlite3.OperationalError:
                pass
        con.commit()
        raise
    finally:
        con.close()
    inject_style(out, build_qml())

    # seep_group_id is unique only WITHIN an image -- every chip restarts ids at
    # 1,2,... so counting it globally silently merges distinct seeps that happen
    # to share an id across chips. Always key on the (image, seep_group_id)
    # tuple. (Grouping itself was never affected: candidate pairs are built
    # within-image only. This was a reporting bug -- on the 10-chip validation
    # pack it under-reported 2711 seeps as 1305.)
    sizes = df.assign(_g=sgid).groupby(["image", "_g"]).size()
    n_groups = len(sizes)
    n_multi = int((sizes > 1).sum())
    print(f"[deploy] wrote {out}")
    print(f"[deploy]   {n_groups} seeps ({n_multi} multi-bubble) from {len(df)} bubbles")
    print(f"[deploy]   style '{STYLE_NAME}' injected (pick it in QGIS Layer Styles)")


if __name__ == "__main__":
    main()