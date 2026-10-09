"""Build `final_complete_class_pack` -- the frozen, fully classified label set.

Run it from the repo root:

    python -m tools.labeling.build_final_class_pack [--dry-run]

It composes three class sources into one pack with no blank classes, then drops the
rows no source can classify. `pack_rf_grouped` is the base and is never modified:

1. **Carry the hand-resolved disputes in** from `pack_rf_disputed_classes`, which is a
   `fid`-aligned subset of the base. Only rows the base leaves blank are written, so a
   class the base already carries always wins -- the base is the class authority for
   chip 4 after the 2026-10-08 regrouping.
2. **Backfill the unanimous classes** from the three final labeler packs. A row
   qualifies when every labeler who classified that polygon chose the same class.
   Rows where labelers disagree are not backfilled: those are the disputes, and step 1
   already resolved them.
3. **Delete the rows no pack can classify** -- either absent from the final packs or
   blank there too. These were left blank on purpose (unsure, or on the training-area
   edge), so a guessed class is worse than a missing row.
4. **Rebuild `seep_hulls`** through `tools.grouping.build_hull_layer`, because deleting
   a member changes its group's hull footprint and can empty a group outright.

A `class_source` column records which step set each class, so the provenance survives in
the file rather than only in the manifest.

JOIN KEYS: `fid` against the disputed pack, which was pruned from the base and keeps its
fids. `(image, centroid rounded to 1 cm)` against the final labeler packs -- never
`bubble_id`, because the base renumbered it. The script asserts both: the disputed join
must agree on image and centroid, and the centroid key must be unique in all four packs.

WRITES: non-spatial `UPDATE` by `fid` with the RTree triggers parked, then `DELETE` with
the triggers live. The delete triggers are plain SQL and keep the RTree and
`gpkg_ogr_contents` consistent; the insert and update triggers call `ST_*` functions that
plain `sqlite3` lacks, which is why only the update path parks them. Geometry is never
touched, so the spatial index and the `layer_styles` symbology both stay valid.

VERIFIES, after writing: every surviving row's geometry blob and every non-class
attribute are byte-identical to the base, no class is blank, the RTree holds exactly the
surviving fids, and the trigger and style inventories came back.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import pandas as pd

from tools.paths import CANONICAL_PRED_RELDIR

RF_DIR = os.path.join("data", "training", "AE", "2026-09-24_RandomForestTraining")
FINAL_DIR = os.path.join(CANONICAL_PRED_RELDIR, "labeling", "final_labeler_packs")

BASE = os.path.join(RF_DIR, "pack_rf_grouped.gpkg")
DISPUTED = os.path.join(RF_DIR, "pack_rf_disputed_classes.gpkg")
LABELERS = ("allison", "katey", "prajna")

OUT_NAME = "final_complete_class_pack.gpkg"
OUT = os.path.join(FINAL_DIR, OUT_NAME)
ALIAS = os.path.join(RF_DIR, OUT_NAME)
MANIFEST = os.path.join(FINAL_DIR, "final_complete_class_pack.manifest.json")

KEY_DP = 2
SOURCE_COL = "class_source"
VALID = ("A", "B", "C")

# The three things the pack cannot carry, plus the labeling dates the drift sentence
# needs. `--manifest-only` re-stamps these, so edit them here and refresh.
NOTES = [
    "class_source='pack' means pack_rf_grouped already carried the class; 'dispute' "
    "means the 2026-10-08 hand resolution of the 126 disputed bubbles; "
    "'final_unanimous' means every labeler who classified that polygon agreed.",

    "Grouping review status: group_source='fixed' marks groups a human regrouped. It "
    "does NOT mark the rest as unreviewed -- the grouping was reviewed on every chip "
    "and edited only where the grouper was wrong, so the 'model' rows are accepted "
    "labels. Load-bearing for the chain bias.",

    "48.tif bubbles 953 and 954 were inspected and is_pregrouped=0 is correct. The QA "
    "geometry check flags them on every run.",

    "Labeling dates. EARLIER session, taken from the other two labelers' packs: "
    "katey's pack last written 2026-09-03, prajna's 2026-09-09, and prajna's notes "
    "carry an in-pack stamp of 2026-07-22 on 27 rows, so the session spans roughly "
    "2026-07-22 to 2026-09-09. Allison's pack also records a salvage on 2026-09-11 "
    "from a 2026-06 chip-39 session. LATER session: 2026-09-24 (pack_rf.gpkg created) "
    "to 2026-10-08 (this freeze). No pack carries a per-row date column, so both are "
    "bounded ranges rather than exact dates.",

    "The 11 class-mixed groups were resolved by hand on 2026-10-08 after the first "
    "build, so no group relies on MAX promotion. Three went against MAX: 4.tif 445 "
    "and 810 and 48.tif 801 resolved to A.",

    "The 5x5 cm floor is undecided and gates step 2, not this freeze.",
]


def final_pack(name: str) -> str:
    return os.path.join(FINAL_DIR, f"gt_seeps_label_quarters_{name}_grouped.gpkg")


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------
def read_labels(path: str, cols: str = "*") -> pd.DataFrame:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(f"SELECT {cols} FROM labels", con)
    finally:
        con.close()
    df["cls"] = df["class"].fillna("").astype(str).str.strip().str.upper()
    if "centroid_x_m" in df.columns:
        df["k"] = list(zip(df["image"].astype(str),
                           df["centroid_x_m"].round(KEY_DP),
                           df["centroid_y_m"].round(KEY_DP)))
    return df


def require_unique_key(df: pd.DataFrame, what: str) -> None:
    dup = int(df["k"].duplicated().sum())
    if dup:
        raise SystemExit(f"{what}: {dup} duplicate (image, cm centroid) keys; the "
                         f"centroid join is not 1:1 -- stopping before any write")


def check_classes(df: pd.DataFrame, what: str) -> None:
    bad = sorted(set(df.loc[df["cls"] != "", "cls"]) - set(VALID))
    if bad:
        raise SystemExit(f"{what}: unexpected class values {bad}")


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------
def build_plan(base: pd.DataFrame, disp: pd.DataFrame,
               packs: dict[str, pd.DataFrame]) -> tuple[dict[int, str], list[int]]:
    """Return ({fid: (class, source)}, [fids to delete]) for the blank base rows."""
    # the disputed pack is a fid-aligned prune of the base; prove it before trusting it
    j = disp.merge(base, on="fid", how="left", suffixes=("_d", "_b"))
    misaligned = j[j["image_b"].isna()
                   | (j["image_d"] != j["image_b"])
                   | ((j["centroid_x_m_d"] - j["centroid_x_m_b"]).abs() > 1e-6)
                   | ((j["centroid_y_m_d"] - j["centroid_y_m_b"]).abs() > 1e-6)]
    if len(misaligned):
        raise SystemExit(f"{len(misaligned)} disputed rows do not line up with the "
                         f"base on fid; the pack is no longer a fid-aligned subset")
    blank_disp = disp[disp["cls"] == ""]
    if len(blank_disp):
        raise SystemExit(
            f"{len(blank_disp)} disputed rows still carry no class "
            f"(fids {sorted(blank_disp['fid'])[:10]}...); resolve them in QGIS first")
    conflict = j[(j["cls_d"] != "") & (j["cls_b"] != "") & (j["cls_d"] != j["cls_b"])]
    if len(conflict):
        print(conflict[["fid", "image_d", "bubble_id_d", "cls_b", "cls_d"]].to_string())
        raise SystemExit(f"{len(conflict)} rows disagree between the base and the "
                         f"disputed pack; decide the authority before writing")

    plan: dict[int, tuple[str, str]] = {}
    blank = base[base["cls"] == ""]
    print(f"[plan] base {len(base)} rows, {len(blank)} blank")

    # step 1 -- the hand-resolved disputes
    d_cls = dict(zip(disp["fid"], disp["cls"]))
    for f in blank["fid"]:
        if f in d_cls:
            plan[int(f)] = (d_cls[f], "dispute")
    n1 = len(plan)
    print(f"[plan] step 1  dispute      -> {n1} rows "
          f"({len(disp)} in the disputed pack, "
          f"{len(disp) - n1} already classified in the base)")

    # step 2 -- unanimous across whichever labelers classified the polygon
    votes: dict[object, set[str]] = {}
    seen: set[object] = set()
    for name in LABELERS:
        p = packs[name]
        seen |= set(p["k"])
        for k, c in zip(p.loc[p["cls"] != "", "k"], p.loc[p["cls"] != "", "cls"]):
            votes.setdefault(k, set()).add(c)

    drop: list[int] = []
    split = {"unanimous": 0, "disagree": 0, "unclassified": 0, "absent": 0}
    for f, k in zip(blank["fid"], blank["k"]):
        f = int(f)
        if f in plan:
            continue
        v = votes.get(k, set())
        if len(v) == 1:
            plan[f] = (next(iter(v)), "final_unanimous")
            split["unanimous"] += 1
        elif len(v) > 1:
            split["disagree"] += 1
        else:
            drop.append(f)
            split["unclassified" if k in seen else "absent"] += 1
    print(f"[plan] step 2  unanimous    -> {split['unanimous']} rows")
    print(f"[plan] step 3  delete       -> {len(drop)} rows "
          f"({split['unclassified']} blank in the final packs, "
          f"{split['absent']} with no final-pack row)")

    if split["disagree"]:
        left = blank[~blank["fid"].astype(int).isin(plan)
                     & ~blank["fid"].astype(int).isin(drop)]
        print(left[["fid", "image", "bubble_id", "seep_group_id"]].to_string())
        raise SystemExit(
            f"{split['disagree']} blank rows have labelers disagreeing but are absent "
            f"from the disputed pack; they belong in a dispute pass, not a backfill")

    if len(plan) + len(drop) != len(blank):
        raise SystemExit(f"plan covers {len(plan) + len(drop)} of {len(blank)} blanks")

    by_image = (blank[blank["fid"].astype(int).isin(plan)]
                .assign(src=[plan[int(f)][1] for f in
                             blank.loc[blank["fid"].astype(int).isin(plan), "fid"]])
                .groupby(["src", "image"]).size())
    print(f"[plan] fills by source and chip:\n{by_image.to_string()}")
    return plan, drop


# ---------------------------------------------------------------------------
# writes
# ---------------------------------------------------------------------------
def park_triggers(con: sqlite3.Connection, table: str):
    """Drop `table`'s triggers and return their DDL, for recreation after the write."""
    trigs = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name=? AND sql IS NOT NULL", (table,)).fetchall()
    for name, _ in trigs:
        con.execute(f'DROP TRIGGER IF EXISTS "{name}"')
    return trigs


def write_classes(path: str, plan: dict[int, tuple[str, str]]) -> None:
    """Set `class` and `class_source` by fid, RTree triggers parked."""
    con = sqlite3.connect(path)
    trigs = park_triggers(con, "labels")
    try:
        cols = [r[1] for r in con.execute("PRAGMA table_info(labels)")]
        if SOURCE_COL not in cols:
            con.execute(f'ALTER TABLE labels ADD COLUMN "{SOURCE_COL}" TEXT')
        con.execute(f'UPDATE labels SET "{SOURCE_COL}"=\'pack\' '
                    f'WHERE class IS NOT NULL AND trim(class)<>\'\'')
        con.executemany(
            f'UPDATE labels SET class=?, "{SOURCE_COL}"=? WHERE fid=?',
            [(c, src, f) for f, (c, src) in plan.items()])
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


def delete_rows(path: str, fids: list[int]) -> None:
    """Delete by fid with the triggers LIVE, so the RTree and the feature count follow.

    Both delete triggers on `labels` are plain SQL, so they run under `sqlite3` without
    the `ST_*` functions the insert and update triggers need.
    """
    con = sqlite3.connect(path)
    try:
        con.executemany("DELETE FROM labels WHERE fid=?", [(f,) for f in fids])
        con.commit()
    finally:
        con.close()


def rebuild_hulls(path: str) -> None:
    cmd = [sys.executable, "-m", "tools.grouping.build_hull_layer", path,
           "--no-backup"]
    print(f"[hull] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def geom_hashes(path: str) -> pd.Series:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT fid, geom FROM labels").fetchall()
    finally:
        con.close()
    return pd.Series({f: hashlib.sha256(g).hexdigest() for f, g in rows})


def verify(base_path: str, out_path: str, plan: dict, drop: list[int]) -> dict:
    base = read_labels(base_path)
    out = read_labels(out_path)
    keep = sorted(set(base["fid"]) - set(drop))

    if len(out) != len(keep):
        raise SystemExit(f"[verify] FAIL {len(out)} rows, expected {len(keep)}")
    if sorted(out["fid"]) != keep:
        raise SystemExit("[verify] FAIL the surviving fids are not the kept fids")
    blank = out[out["cls"] == ""]
    if len(blank):
        raise SystemExit(f"[verify] FAIL {len(blank)} rows still carry no class")
    check_classes(out, "[verify] output")

    bh, oh = geom_hashes(base_path), geom_hashes(out_path)
    if not bh.reindex(keep).equals(oh.reindex(keep)):
        raise SystemExit("[verify] FAIL geometry changed on a surviving row")

    # every attribute but `class` and `class_source` must be untouched
    frozen = [c for c in base.columns
              if c not in ("fid", "class", "cls", "k", SOURCE_COL)]
    b = base[base["fid"].isin(keep)].set_index("fid")[frozen].sort_index()
    o = out.set_index("fid")[frozen].sort_index()
    for c in frozen:
        if not b[c].equals(o[c]):
            raise SystemExit(f"[verify] FAIL column '{c}' changed on a surviving row")

    # the classes the base already carried must be unchanged
    had = base[(base["cls"] != "") & base["fid"].isin(keep)].set_index("fid")["cls"]
    now = out.set_index("fid")["cls"].reindex(had.index)
    if not had.equals(now):
        raise SystemExit("[verify] FAIL a class the base already carried moved")

    con = sqlite3.connect(f"file:{out_path}?mode=ro", uri=True)
    rtree = [r[0] for r in con.execute("SELECT id FROM rtree_labels_geom")]
    counts = dict(con.execute("SELECT table_name, feature_count FROM gpkg_ogr_contents"))
    trig = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='trigger' "
                       "AND tbl_name='labels'").fetchone()[0]
    styles = con.execute("SELECT f_table_name, styleName FROM layer_styles "
                         "ORDER BY 1, 2").fetchall()
    hulls = con.execute("SELECT COUNT(*) FROM seep_hulls").fetchone()[0]
    con.close()
    base_con = sqlite3.connect(f"file:{base_path}?mode=ro", uri=True)
    base_trig = base_con.execute("SELECT COUNT(*) FROM sqlite_master WHERE "
                                 "type='trigger' AND tbl_name='labels'").fetchone()[0]
    base_styles = base_con.execute("SELECT f_table_name, styleName FROM layer_styles "
                                   "ORDER BY 1, 2").fetchall()
    base_con.close()

    if sorted(rtree) != keep:
        raise SystemExit(f"[verify] FAIL RTree holds {len(rtree)} ids, "
                         f"expected the {len(keep)} surviving fids")
    if counts.get("labels") != len(keep):
        raise SystemExit(f"[verify] FAIL gpkg_ogr_contents says "
                         f"{counts.get('labels')} labels, expected {len(keep)}")
    if trig != base_trig:
        raise SystemExit(f"[verify] FAIL {trig} labels triggers, expected {base_trig}")
    missing = sorted(set(base_styles) - set(styles))
    if missing:
        raise SystemExit(f"[verify] FAIL styles lost: {missing}")

    # group integrity -- reported, not fatal: a deleted row can empty a group or take
    # the anchor whose bubble_id the group is named for.
    def groups(df):
        sg = (pd.to_numeric(df["seep_group_id"], errors="coerce")
              .fillna(pd.to_numeric(df["bubble_id"], errors="coerce")).astype("int64"))
        return df.assign(_sg=sg, uid=df["image"].astype(str) + "::" + sg.astype(str))
    gb, go = groups(base), groups(out)
    emptied = sorted(set(gb["uid"]) - set(go["uid"]))
    anchorless = sorted(
        {uid for uid, sub in go.groupby("uid")
         if sub["_sg"].iat[0] not in set(pd.to_numeric(sub["bubble_id"]))}
        - {uid for uid, sub in gb.groupby("uid")
           if sub["_sg"].iat[0] not in set(pd.to_numeric(sub["bubble_id"]))})

    print(f"[verify] PASS {len(out)} rows, 0 blank classes, geometry and every "
          f"non-class attribute byte-identical on all {len(keep)} survivors")
    print(f"[verify]   RTree {len(rtree)} ids, feature_count {counts}, "
          f"{trig} labels triggers, {len(styles)} styles, {hulls} hulls")
    print(f"[verify]   groups {len(set(gb['uid']))} -> {len(set(go['uid']))}; "
          f"{len(emptied)} emptied by the delete, {len(anchorless)} newly anchorless")
    if emptied:
        print(f"[verify]   emptied: {emptied}")
    if anchorless:
        print(f"[verify]   anchorless (keyed on (image, seep_group_id), so still "
              f"valid): {anchorless}")

    return {
        "rows": len(out),
        "classes": out["cls"].value_counts().sort_index().to_dict(),
        "class_source": (out[SOURCE_COL].fillna("").value_counts()
                         .sort_index().to_dict()),
        "seeps": len(set(go["uid"])),
        "hulls": hulls,
        "groups_emptied": emptied,
        "groups_newly_anchorless": anchorless,
    }


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_alias(target: str, link: str) -> str:
    """Point `link` at `target`. Prefers a symlink; falls back to a hard link."""
    if os.path.lexists(link):
        os.remove(link)
    rel = os.path.relpath(os.path.abspath(target), os.path.dirname(os.path.abspath(link)))
    try:
        os.symlink(rel, link)
        return "symlink"
    except OSError as exc:
        print(f"[alias] symlink refused ({exc.__class__.__name__}: {exc}); "
              f"falling back to a hard link")
    os.link(os.path.abspath(target), link)
    return "hardlink"


def refresh_manifest() -> int:
    """Re-stamp the sha256 and the class counts from the pack as it stands.

    Hand edits in QGIS and hull rebuilds both change the file, which retires the
    recorded sha256. Nothing here reads the source packs or writes to `labels`.
    """
    if not os.path.exists(MANIFEST):
        raise SystemExit(f"no manifest to refresh: {MANIFEST}")
    with open(MANIFEST, encoding="utf-8") as fh:
        m = json.load(fh)

    out = read_labels(OUT)
    blank = int((out["cls"] == "").sum())
    if blank:
        raise SystemExit(f"{blank} rows carry no class; the pack is not frozen")
    check_classes(out, "[refresh] output")

    con = sqlite3.connect(f"file:{OUT}?mode=ro", uri=True)
    hulls = pd.read_sql_query(
        'SELECT seep_uid, class, class_mixed FROM seep_hulls', con)
    rtree = con.execute("SELECT COUNT(*) FROM rtree_labels_geom").fetchone()[0]
    con.close()
    if rtree != len(out):
        raise SystemExit(f"[refresh] RTree holds {rtree} ids against {len(out)} rows")

    m["refreshed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    m["sha256"] = sha256(OUT)
    m["rows"] = len(out)
    m["classes"] = out["cls"].value_counts().sort_index().to_dict()
    m["class_source"] = (out[SOURCE_COL].fillna("").value_counts()
                         .sort_index().to_dict())
    m["seeps"] = len(hulls)
    m["hulls"] = len(hulls)
    m["hull_classes"] = hulls["class"].value_counts().sort_index().to_dict()
    m["groups_class_mixed"] = int(hulls["class_mixed"].sum())
    m["notes"] = NOTES
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    print(f"[refresh] {len(out)} rows, 0 blank, {m['classes']}")
    print(f"[refresh] {len(hulls)} hulls {m['hull_classes']}, "
          f"{m['groups_class_mixed']} class-mixed")
    print(f"[refresh] sha256 {m['sha256']}")
    return 0


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compose pack_rf_grouped, the resolved disputes and the unanimous "
                    "final-pack classes into one fully classified pack.")
    ap.add_argument("--dry-run", action="store_true",
                    help="plan and report, write nothing")
    ap.add_argument("--manifest-only", action="store_true",
                    help="re-stamp the manifest from the pack as it stands, after a "
                         "QGIS edit or a hull rebuild; touches no label data")
    args = ap.parse_args()

    if args.manifest_only:
        return refresh_manifest()

    for p in [BASE, DISPUTED] + [final_pack(n) for n in LABELERS]:
        if not os.path.exists(p):
            raise SystemExit(f"no such pack: {p}")

    base = read_labels(BASE)
    disp = read_labels(DISPUTED)
    packs = {n: read_labels(final_pack(n)) for n in LABELERS}
    require_unique_key(base, "pack_rf_grouped")
    for n, p in packs.items():
        require_unique_key(p, f"{n} final pack")
        check_classes(p, f"{n} final pack")
    check_classes(base, "pack_rf_grouped")
    check_classes(disp, "pack_rf_disputed_classes")

    plan, drop = build_plan(base, disp, packs)
    print(f"[plan] result: {len(base)} - {len(drop)} = {len(base) - len(drop)} rows, "
          f"{len(plan)} classes written")
    if args.dry_run:
        print("[dry-run] nothing written")
        return 0

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    if os.path.exists(OUT):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bak = f"{os.path.splitext(OUT)[0]}.pre_build_{stamp}.gpkg"
        shutil.copyfile(OUT, bak)
        print(f"[out] existing pack backed up -> {bak}")
    shutil.copyfile(BASE, OUT)
    print(f"[out] {BASE} -> {OUT}")

    write_classes(OUT, plan)
    print(f"[out] wrote {len(plan)} classes and the {SOURCE_COL} column")
    delete_rows(OUT, drop)
    print(f"[out] deleted {len(drop)} unclassifiable rows")
    rebuild_hulls(OUT)

    summary = verify(BASE, OUT, plan, drop)
    kind = make_alias(OUT, ALIAS)
    print(f"[alias] {ALIAS} -> {OUT} ({kind})")

    manifest = {
        "pack": os.path.relpath(OUT).replace("\\", "/"),
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "built_by": "tools.labeling.build_final_class_pack",
        "sha256": sha256(OUT),
        "alias": {"path": os.path.relpath(ALIAS).replace("\\", "/"), "kind": kind},
        "sources": {
            os.path.relpath(p).replace("\\", "/"): sha256(p)
            for p in [BASE, DISPUTED] + [final_pack(n) for n in LABELERS]},
        "composition": {
            "base_rows": len(base),
            "classes_written": len(plan),
            "rows_deleted": len(drop),
            "deleted_fids": drop,
        },
        **summary,
        "notes": NOTES,
    }
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[manifest] {MANIFEST}")
    print(f"[manifest] sha256 {manifest['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
