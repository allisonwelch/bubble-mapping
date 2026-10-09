"""One-shot reconciliation of `pack_rf_grouped` after the 2026-10-08 chip-4 regrouping.

Run it from the repo root:

    python -m tools.labeling.reconcile_pack_rf [--dry-run]

It does four things, in this order, because each one feeds the next:

1. **Stamps `group_source='fixed'` on the regrouped chip-4 bubbles.** The rule is the
   one the 2026-10-07 pass already established over the other 19 chips: a bubble is
   `fixed` when the set of bubbles in its FINAL group differs from the set in its
   `seep_group_id_pred` group. The script reproduces the existing `group_source` column
   on those 19 chips before it writes, and aborts on any disagreement, so the chip-4
   rows land on the same convention rather than a reinvented one.

   `group_source` therefore records whether the GROUPING is human work, not whether the
   class is. A bubble whose model grouping the labeler accepted and then classified
   stays `model`, which is what keeps it scorable in
   `tools/grouping/grouper_corrections.py`.

2. **Unions `is_pregrouped` and `is_overgrouped` in from the final labeler packs.**
   Those flags describe the polygon, not the grouping, so they survive a regrouping and
   should never have been dropped. `is_pregrouped` comes from allison's pack (the pack
   this one is labeled with) unioned with whatever `pack_rf_grouped` already carries, so
   the five chip-39 flags she added under the tightened envelope-plus-ice convention
   stay. `is_overgrouped` is a union across all three labelers: a polygon spanning more
   than one seep is an annotation defect whoever spots it.

3. **Propagates the chip-4 changes into `pack_rf_disputed_classes`.** That pack is a
   `fid`-aligned subset of `pack_rf_grouped`, so the join is on `fid`. Class propagation
   is one-directional and non-destructive: `pack_rf_grouped` is the class authority for
   chip 4, so its class wins wherever it is set, and a row it leaves blank keeps the
   dispute resolution already recorded in the disputed pack.

4. **Rebuilds `seep_hulls` in both packs.** `pack_rf_grouped` goes through
   `tools.grouping.build_hull_layer`. The disputed pack's hull layer is then the subset
   of those hulls whose `seep_uid` its own rows belong to, which is how that layer was
   built in the first place -- hulling the 138-row subset on its own would truncate every
   group that reaches outside it.

JOIN KEY: `(image, centroid rounded to 1 cm)`, never `bubble_id`. `pack_rf_grouped`
renumbered `bubble_id`, so the ids do not line up with the final packs. Four polygons
miss the rounded-centroid join (they were redrawn, and their areas moved by 1-3%); each
resolves to one `pack_rf_grouped` row within 4 mm while the next candidate sits at least
29 cm away, so a 5 cm nearest-neighbour fallback picks them up unambiguously. The script
asserts that margin instead of trusting it.

WRITES: non-spatial `UPDATE` by `fid` on a plain `sqlite3` connection, with the GDAL
RTree triggers dropped and recreated around the write. Geometry is never touched, so the
spatial index and the `layer_styles` symbology both stay valid. Both packs are copied to
a timestamped backup first.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from tools.paths import CANONICAL_PRED_RELDIR

RF_DIR = os.path.join("data", "training", "AE", "2026-09-24_RandomForestTraining")
GROUPED = os.path.join(RF_DIR, "pack_rf_grouped.gpkg")
DISPUTED = os.path.join(RF_DIR, "pack_rf_disputed_classes.gpkg")
FINAL = os.path.join(CANONICAL_PRED_RELDIR, "labeling", "final_labeler_packs")
LABELERS = ("allison", "katey", "prajna")

CHIP = "4.tif"
# `is_pregrouped` follows the pack's own labeler; `is_overgrouped` takes every labeler.
PREGROUP_FROM = ("allison",)
OVERGROUP_FROM = LABELERS

READ = ("fid, image, bubble_id, class, seep_group_id, seep_group_id_pred, "
        "group_source, is_pregrouped, is_overgrouped, centroid_x_m, centroid_y_m, "
        "area_m2")
# round-to-cm join key, and the fallback radius for polygons it misses. A fallback
# counts only when the runner-up sits NEAR_MARGIN times further out, so a redrawn
# polygon is distinguished from a genuinely crowded pair.
KEY_DP = 2
NEAR_TOL_M = 0.05
NEAR_MARGIN = 10.0


def read_labels(path: str, cols: str = READ) -> pd.DataFrame:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(f"SELECT {cols} FROM labels", con)
    finally:
        con.close()
    if "centroid_x_m" in df.columns:
        df["kx"] = df["centroid_x_m"].round(KEY_DP)
        df["ky"] = df["centroid_y_m"].round(KEY_DP)
    return df


def update_by_fid(path: str, sets: dict[str, dict[int, object]]) -> int:
    """Apply {column: {fid: value}} to `labels`, RTree triggers parked.

    Returns the number of rows touched. Columns are written one statement per fid so a
    row appearing under two columns still gets both.
    """
    con = sqlite3.connect(path)
    trigs = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name='labels' AND sql IS NOT NULL").fetchall()
    n = 0
    try:
        for name, _ in trigs:
            con.execute(f'DROP TRIGGER IF EXISTS "{name}"')
        for col, by_fid in sets.items():
            if not by_fid:
                continue
            con.executemany(f'UPDATE labels SET "{col}"=? WHERE fid=?',
                            [(v, int(f)) for f, v in by_fid.items()])
            n += len(by_fid)
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
    return n


# ---------------------------------------------------------------------------
# step 1 -- group_source
# ---------------------------------------------------------------------------
def membership_changed(df: pd.DataFrame) -> pd.Series:
    """True where the row's final group holds a different bubble set than its
    `seep_group_id_pred` group. Both sides are keyed per image, because
    `seep_group_id` restarts at 1 in every chip."""
    sg = (pd.to_numeric(df["seep_group_id"], errors="coerce")
          .fillna(pd.to_numeric(df["bubble_id"], errors="coerce")).astype("int64"))
    out = pd.Series(False, index=df.index)
    for im, sub in df.groupby("image"):
        s = sg.loc[sub.index]
        fin = sub.assign(_g=s).groupby("_g")["bubble_id"].apply(frozenset)
        pre = sub.groupby("seep_group_id_pred")["bubble_id"].apply(frozenset)
        out.loc[sub.index] = [fin[a] != pre[b]
                              for a, b in zip(s, sub["seep_group_id_pred"])]
    return out


def check_group_source_rule(df: pd.DataFrame) -> None:
    """Confirm the membership rule reproduces `group_source` on every chip but CHIP."""
    other = df[df["image"] != CHIP]
    want = membership_changed(other).map({True: "fixed", False: "model"})
    have = other["group_source"].astype(str)
    bad = want != have
    if bad.any():
        print(df[df["image"] != CHIP][bad][
            ["image", "bubble_id", "seep_group_id", "seep_group_id_pred",
             "group_source"]].to_string())
        raise SystemExit(
            f"group_source rule disagrees on {int(bad.sum())} of {len(other)} rows "
            f"outside {CHIP}; the 2026-10-07 convention is not what this script "
            f"thinks it is -- stopping before any write")
    print(f"[gs] membership rule reproduces group_source on all {len(other)} rows "
          f"outside {CHIP}")


# ---------------------------------------------------------------------------
# step 2 -- flag carry-over
# ---------------------------------------------------------------------------
def match_to_grouped(pack: pd.DataFrame, g: pd.DataFrame) -> pd.Series:
    """pack row index -> `pack_rf_grouped` fid, on (image, cm centroid) with a
    nearest-neighbour fallback. Raises when a fallback match is ambiguous."""
    key = g.set_index(["image", "kx", "ky"])["fid"]
    if key.index.duplicated().any():
        raise SystemExit("pack_rf_grouped has duplicate (image, cm centroid) keys; "
                         "the join is not 1:1")
    idx = pd.MultiIndex.from_frame(pack[["image", "kx", "ky"]])
    out = pd.Series(key.reindex(idx).to_numpy(), index=pack.index, dtype="float64")
    for i in out.index[out.isna()]:
        r = pack.loc[i]
        sub = g[g["image"] == r["image"]]
        d = np.hypot(sub["centroid_x_m"] - r["centroid_x_m"],
                     sub["centroid_y_m"] - r["centroid_y_m"]).nsmallest(2)
        if d.iat[0] > NEAR_TOL_M or d.iat[1] < NEAR_MARGIN * max(d.iat[0], 1e-4):
            raise SystemExit(
                f"ambiguous fallback for {r['image']} bubble_id={r['bubble_id']}: "
                f"nearest {d.iat[0]:.3f} m, runner-up {d.iat[1]:.3f} m")
        out.loc[i] = sub.loc[d.index[0], "fid"]
        print(f"[flag]   fallback {r['image']} bubble_id={r['bubble_id']} -> fid="
              f"{int(out.loc[i])} at {d.iat[0] * 100:.1f} cm "
              f"(runner-up {d.iat[1]:.2f} m)")
    return out.astype("int64")


def flag_union(g: pd.DataFrame, packs: dict[str, pd.DataFrame]
               ) -> dict[str, dict[int, int]]:
    """fids to set to 1 for each flag, unioning the named packs with what `g` has."""
    sets: dict[str, dict[int, int]] = {}
    for col, sources in (("is_pregrouped", PREGROUP_FROM),
                         ("is_overgrouped", OVERGROUP_FROM)):
        have = set(g.loc[pd.to_numeric(g[col], errors="coerce").fillna(0) == 1, "fid"])
        add: set[int] = set()
        for name in sources:
            p = packs[name]
            flagged = p[pd.to_numeric(p[col], errors="coerce").fillna(0) == 1]
            if flagged.empty:
                continue
            fids = set(match_to_grouped(flagged, g))
            new = fids - have - add
            add |= new
            per_image = (g[g["fid"].isin(new)].groupby("image").size().to_dict())
            print(f"[flag] {col}: {name} contributes {len(new)} new "
                  f"({len(fids)} flagged) {per_image}")
        sets[col] = {f: 1 for f in sorted(add)}
        print(f"[flag] {col}: {len(have)} already set, {len(add)} added -> "
              f"{len(have) + len(add)} total")
    return sets


# ---------------------------------------------------------------------------
# step 4 -- whole seeps in the disputed pack
# ---------------------------------------------------------------------------
def seep_uid(df: pd.DataFrame) -> pd.Series:
    sg = (pd.to_numeric(df["seep_group_id"], errors="coerce")
          .fillna(pd.to_numeric(df["bubble_id"], errors="coerce")).astype("int64"))
    return df["image"].astype(str) + "::" + sg.astype(str)


def labeler_columns(rows: pd.DataFrame, packs: dict[str, pd.DataFrame]
                    ) -> pd.DataFrame:
    """Rebuild the disputed pack's per-labeler columns for `rows`.

    Verified 2026-10-08 to reproduce all three `class_<labeler>` columns, `dispute`
    and `is_disputed` on all 138 original rows, so new rows land on the same basis:
    `dispute` is the sorted distinct non-blank classes joined by '/', null when the
    labelers who cover the polygon agree or only one of them does.
    """
    out = pd.DataFrame(index=rows.index)
    idx = pd.MultiIndex.from_frame(rows[["image", "kx", "ky"]])
    for name, p in packs.items():
        key = p.set_index(["image", "kx", "ky"])["class"]
        out[f"class_{name}"] = key.reindex(idx).to_numpy()
    seen = out.apply(lambda r: sorted({str(v).strip() for v in r
                                       if pd.notna(v) and str(v).strip()}), axis=1)
    out["dispute"] = ["/".join(s) if len(s) > 1 else None for s in seen]
    out["is_disputed"] = [int(len(s) > 1) for s in seen]
    out["class_gt"] = None
    return out


def expand_to_whole_seeps(src: str, dst: str, packs: dict[str, pd.DataFrame],
                          dry_run: bool = False) -> int:
    """Pull into `dst` every `src` bubble sharing a seep with a bubble already there.

    Regrouping can merge a bubble in the disputed pack with bubbles outside it, which
    leaves the pack holding a slice of the seep -- the class is a judgment about the
    whole seep, so a truncated group shows the wrong footprint to judge it by. This
    adds the missing members and touches no existing row.

    Rows copy over with their `src` fid and geometry bytes intact, so the pack stays a
    fid-aligned subset of `pack_rf_grouped`. The per-labeler columns are rebuilt from
    the final packs; a bubble no final quarter covers gets nulls and `is_disputed = 0`,
    which is honest -- it is carried for seep completeness, not as a dispute.
    """
    g = read_labels(src)
    d = read_labels(dst)
    g = g.assign(uid=seep_uid(g))
    uids = set(g.loc[g["fid"].isin(d["fid"]), "uid"])
    tgt = g[g["uid"].isin(uids)]
    missing = sorted(set(tgt["fid"]) - set(d["fid"]))
    trunc = tgt[tgt["fid"].isin(missing)]["uid"].nunique()
    print(f"[seep] {len(d)} rows in {len(uids)} seeps; whole membership is "
          f"{len(tgt)} rows -> {len(missing)} to add across {trunc} truncated seep(s)")
    if not missing:
        return 0
    add = tgt[tgt["fid"].isin(missing)].copy()
    for u, sub in tgt[tgt["uid"].isin(set(add["uid"]))].groupby("uid"):
        have = int(sub["fid"].isin(d["fid"]).sum())
        print(f"[seep]   {u}: {have} of {len(sub)} present, +{len(sub) - have}")

    extra = labeler_columns(add, packs)
    per_labeler = [f"class_{n}" for n in packs]
    cover = extra[per_labeler].notna().sum(axis=1).value_counts().sort_index()
    print(f"[seep] added rows: {int(extra['is_disputed'].sum())} carry a labeler "
          f"disagreement; final-pack coverage {cover.to_dict()} labeler(s) per row")
    if dry_run:
        print("[seep] dry-run: nothing inserted")
        return 0

    # Read the rows out over a READ-ONLY connection and insert them as parameters.
    # ATTACH would be tidier, but it takes a write lock on the source, which fails
    # outright while the pack is open in QGIS.
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    src_cols = [r[1] for r in con.execute('PRAGMA table_info("labels")')]
    marks0 = ", ".join("?" * len(missing))
    src_rows = con.execute(
        f'SELECT {", ".join(chr(34) + c + chr(34) for c in src_cols)} FROM labels '
        f"WHERE fid IN ({marks0})", missing).fetchall()
    src_rtree = con.execute(f"SELECT id, minx, maxx, miny, maxy FROM "
                            f"rtree_labels_geom WHERE id IN ({marks0})",
                            missing).fetchall()
    con.close()
    if len(src_rows) != len(missing) or len(src_rtree) != len(missing):
        raise SystemExit(f"read {len(src_rows)} rows and {len(src_rtree)} rtree "
                         f"entries for {len(missing)} fids")
    con = sqlite3.connect(dst)
    have = {r[1] for r in con.execute('PRAGMA table_info("labels")')}
    shared = [c for c in src_cols if c in have]
    trigs = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name='labels' AND sql IS NOT NULL").fetchall()
    keep = [i for i, c in enumerate(src_cols) if c in shared]
    cols = ", ".join(f'"{src_cols[i]}"' for i in keep)
    try:
        for name, _ in trigs:
            con.execute(f'DROP TRIGGER IF EXISTS "{name}"')
        con.executemany(
            f"INSERT INTO labels ({cols}) VALUES "
            f"({', '.join('?' * len(keep))})",
            [tuple(r[i] for i in keep) for r in src_rows])
        # the rtree is maintained by the triggers that are parked, so copy the
        # source entries rather than recomputing envelopes from the geometry blobs.
        con.executemany("INSERT INTO rtree_labels_geom (id, minx, maxx, miny, maxy) "
                        "VALUES (?, ?, ?, ?, ?)", src_rtree)
        for col in extra.columns:
            con.executemany(f'UPDATE labels SET "{col}"=? WHERE fid=?',
                            [(None if pd.isna(v) else v, int(f))
                             for f, v in zip(add["fid"], extra[col])])
        for _, sql in trigs:
            con.execute(sql)
        # gpkg_ogr_contents is trigger-maintained too, and gpkg_contents caches the
        # layer bbox; both go stale behind a trigger-free insert.
        con.execute("UPDATE gpkg_ogr_contents SET feature_count="
                    "(SELECT COUNT(*) FROM labels) WHERE table_name='labels'")
        con.execute("UPDATE gpkg_contents SET min_x=(SELECT MIN(minx) FROM "
                    "rtree_labels_geom), max_x=(SELECT MAX(maxx) FROM "
                    "rtree_labels_geom), min_y=(SELECT MIN(miny) FROM "
                    "rtree_labels_geom), max_y=(SELECT MAX(maxy) FROM "
                    "rtree_labels_geom) WHERE table_name='labels'")
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
    print(f"[seep] inserted {len(missing)} row(s) into {dst}")
    return len(missing)


# ---------------------------------------------------------------------------
# step 5 -- hulls
# ---------------------------------------------------------------------------
def subset_hulls(src: str, dst: str) -> None:
    """Replace `dst`'s seep_hulls with the `src` hulls its own rows belong to.

    The disputed pack holds a slice of the bubbles, so its groups reach outside it.
    Taking the hulls from the full pack keeps `n_bubbles`, `hull_area_m2` and the
    promoted class describing the whole seep rather than the slice.
    """
    import geopandas as gpd

    from tools.grouping import build_hull_layer as bhl

    lab = read_labels(dst, "fid, image, bubble_id, seep_group_id")
    sg = (pd.to_numeric(lab["seep_group_id"], errors="coerce")
          .fillna(pd.to_numeric(lab["bubble_id"], errors="coerce")).astype("int64"))
    uids = set(lab["image"].astype(str) + "::" + sg.astype(str))

    hulls = gpd.read_file(src, layer=bhl.HULL_LAYER)
    keep = hulls[hulls["seep_uid"].isin(uids)].reset_index(drop=True)
    missing = uids - set(keep["seep_uid"])
    if missing:
        raise SystemExit(f"{len(missing)} seep_uid(s) absent from {src} hulls: "
                         f"{sorted(missing)[:5]}")

    bhl.drop_layer(dst, bhl.HULL_LAYER)
    keep.to_file(dst, layer=bhl.HULL_LAYER, driver="GPKG", mode="a")
    bhl.inject_style(dst, bhl.HULL_LAYER, bhl.HULL_STYLE, bhl.hull_qml(),
                     as_default=True,
                     description=f"seep hulls subset from {os.path.basename(src)}")
    bhl.inject_style(dst, "labels", bhl.BUBBLE_STYLE, bhl.bubble_qml(),
                     as_default=False,
                     description="plain bubbles (no hull generator)")
    print(f"[hull] {dst}: {len(uids)} groups -> {len(keep)} hulls "
          f"(subset of {len(hulls)} in the full pack)")


def backup(path: str, tag: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = f"{os.path.splitext(path)[0]}.pre_{tag}_{stamp}.gpkg"
    shutil.copyfile(path, dst)
    print(f"[bak] {dst}")
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="report every change and write nothing")
    args = ap.parse_args()

    for p in (GROUPED, DISPUTED, *(os.path.join(
            FINAL, f"gt_seeps_label_quarters_{n}_grouped.gpkg") for n in LABELERS)):
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")

    g = read_labels(GROUPED)
    packs = {n: read_labels(os.path.join(
        FINAL, f"gt_seeps_label_quarters_{n}_grouped.gpkg")) for n in LABELERS}

    # --- step 1 -------------------------------------------------------------
    check_group_source_rule(g)
    chip = g[g["image"] == CHIP]
    chg = membership_changed(chip)
    gs_fids = sorted(chip.loc[chg & (chip["group_source"].astype(str) != "fixed"),
                              "fid"])
    print(f"[gs] {CHIP}: {int(chg.sum())} of {len(chip)} bubbles regrouped; "
          f"{len(gs_fids)} need group_source model -> fixed")
    kept = chip.loc[~chg & chip["class"].fillna("").str.strip().ne(""), "fid"]
    if len(kept):
        print(f"[gs] {CHIP}: {len(kept)} classified bubble(s) keep group_source="
              f"'model' (grouping accepted, only the class is human): "
              f"fid {sorted(kept)}")

    # --- step 2 -------------------------------------------------------------
    flags = flag_union(g, packs)

    # --- step 3 -------------------------------------------------------------
    d = read_labels(DISPUTED)
    if not set(d["fid"]) <= set(g["fid"]):
        raise SystemExit("pack_rf_disputed_classes holds fids absent from "
                         "pack_rf_grouped; the fid join does not hold")
    gi = g.set_index("fid")
    di = d.set_index("fid")
    src = gi.loc[di.index]
    prop: dict[str, dict[int, object]] = {}

    sg_new = src["seep_group_id"]
    sg_hit = sg_new.fillna(-1) != di["seep_group_id"].fillna(-1)
    prop["seep_group_id"] = {f: (None if pd.isna(v) else int(v))
                             for f, v in sg_new[sg_hit].items()}

    # group_source as it stands AFTER step 1
    gs_new = src["group_source"].astype(str).mask(src.index.isin(gs_fids), "fixed")
    gs_hit = gs_new != di["group_source"].astype(str)
    prop["group_source"] = dict(gs_new[gs_hit].items())

    # class: pack_rf_grouped wins where it is set; a blank there never erases a
    # dispute resolution already recorded in the disputed pack.
    g_cls = src["class"].fillna("").str.strip()
    d_cls = di["class"].fillna("").str.strip()
    cls_hit = g_cls.ne("") & g_cls.ne(d_cls)
    prop["class"] = dict(g_cls[cls_hit].items())
    clash = cls_hit & d_cls.ne("")
    if clash.any():
        print(f"[dsp] {int(clash.sum())} class(es) OVERWRITTEN in the disputed pack "
              f"({CHIP} authority):")
        print(pd.DataFrame({"image": di.loc[clash, "image"],
                            "bubble_id": di.loc[clash, "bubble_id"],
                            "was": d_cls[clash], "now": g_cls[clash]}).to_string())
    held = d_cls.ne("") & g_cls.eq("")
    print(f"[dsp] seep_group_id {len(prop['seep_group_id'])}, group_source "
          f"{len(prop['group_source'])}, class {len(prop['class'])} row(s) to update; "
          f"{int(held.sum())} dispute resolution(s) kept where pack_rf_grouped is blank")

    # Mirror the flags from pack_rf_grouped as they stand after step 1, not just the
    # rows step 2 added -- otherwise a re-run against an already-stamped pack leaves
    # the disputed pack behind.
    for col in ("is_pregrouped", "is_overgrouped"):
        now = (pd.to_numeric(src[col], errors="coerce").fillna(0).astype(int)
               .mask(src.index.isin(flags[col]), 1))
        hit = now != pd.to_numeric(di[col], errors="coerce").fillna(0).astype(int)
        prop[col] = {f: int(v) for f, v in now[hit].items()}
        if prop[col]:
            print(f"[dsp] {col}: {len(prop[col])} row(s) to set "
                  f"({sum(prop[col].values())} to 1)")

    if args.dry_run:
        expand_to_whole_seeps(GROUPED, DISPUTED, packs, dry_run=True)
        print("[dry-run] nothing written")
        return 0

    # --- write --------------------------------------------------------------
    backup(GROUPED, "reconcile")
    n = update_by_fid(GROUPED, {
        "group_source": {f: "fixed" for f in gs_fids}, **flags})
    print(f"[write] {GROUPED}: {n} column-updates")

    backup(DISPUTED, "reconcile")
    n = update_by_fid(DISPUTED, prop)
    print(f"[write] {DISPUTED}: {n} column-updates")

    # --- step 4 -------------------------------------------------------------
    # After the propagation, so the inserted rows are measured against a disputed
    # pack whose grouping already matches pack_rf_grouped.
    expand_to_whole_seeps(GROUPED, DISPUTED, packs)

    # --- step 5 -------------------------------------------------------------
    print("[hull] rebuilding pack_rf_grouped hulls")
    subprocess.run([sys.executable, "-m", "tools.grouping.build_hull_layer",
                    GROUPED, "--no-backup"], check=True)
    subset_hulls(GROUPED, DISPUTED)
    return 0


if __name__ == "__main__":
    sys.exit(main())
