"""Merge a returned labeler pack's per-polygon `class` into a master label pack,
then promote each class to its seep group.

Motivating case (2026-07-29, Prajna): the labeler classified every polygon on an
OLD, UNGROUPED copy of the pack, so their `seep_group_id` is ~all singletons and
must be discarded -- only their `class` (and `notes`) are salvageable. The master
pack carries the authoritative model grouping, so classes are joined back on
(image, bubble_id) and then dissolved within each (image, seep_group_id).

"Dissolve" here means: every polygon sharing a seep group gets the SAME class,
taken as the LARGEST class present in that group (C > B > A). A seep is one
physical feature, so a group whose members disagree is a labeling artifact of
having judged bubbles in isolation; the largest class is the flux-conservative
read (see the count-based flux model in CLAUDE.md).

Writes are done as non-spatial sqlite UPDATEs by fid on a file COPY, with the
GDAL RTree triggers dropped and recreated around them (they call ST_* functions
a plain sqlite3 connection lacks). Geometry is never touched, so the spatial
index, layer_styles, and every untouched column survive intact.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3

import pandas as pd

try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    raise SystemExit("geopandas is required; run inside venv-bubble")

# Class ordering for the group promotion. Higher wins.
CLASS_RANK = {"A": 1, "B": 2, "C": 3}


def _ensure_columns(cur, table: str, cols: dict[str, str]) -> None:
    """ALTER TABLE ... ADD COLUMN for any column not already present."""
    have = {r[1] for r in cur.execute(f'PRAGMA table_info("{table}")')}
    for name, sqltype in cols.items():
        if name not in have:
            cur.execute(f'ALTER TABLE "{table}" ADD COLUMN "{name}" {sqltype}')


def _clean(s: pd.Series) -> pd.Series:
    """NaN/None -> '' and strip, so blank and missing are one thing."""
    return s.fillna("").astype(str).str.strip()


def promote_to_group(df: pd.DataFrame) -> pd.Series:
    """Largest class (C>B>A) per (image, seep_group_id), broadcast to members.

    Groups where every member is blank stay blank. Blank members of a
    partially-classified group inherit the group's class like anyone else.
    """
    rank = df["class_labeler"].map(CLASS_RANK)
    grp_rank = rank.groupby([df["image"], df["seep_group_id"]]).transform("max")
    inv = {v: k for k, v in CLASS_RANK.items()}
    return grp_rank.map(inv).fillna("")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--master", required=True,
                    help="master pack .gpkg (authoritative seep_group_id)")
    ap.add_argument("--returned", required=True,
                    help="labeler's returned pack (.shp or .gpkg) holding class")
    ap.add_argument("--out", required=True, help="output .gpkg to create")
    ap.add_argument("--layer", default="labels", help="feature table name")
    ap.add_argument("--blank-note", default="",
                    help="if set, stamp this into notes for rows the labeler "
                         "left unclassified (e.g. 'unclassified_20260722')")
    args = ap.parse_args()

    # --- master attributes, straight from sqlite so we keep the real fid ------
    con = sqlite3.connect(args.master)
    master = pd.read_sql(
        f'SELECT fid, image, bubble_id, seep_group_id, is_context FROM "{args.layer}"',
        con)
    con.close()

    ret = gpd.read_file(args.returned)
    if "geometry" in ret.columns:
        ret = ret.drop(columns="geometry")
    # Shapefiles truncate field names to 10 chars and store ints as floats.
    ret = ret.rename(columns={"seep_group": "seep_group_id_labeler"})
    ret["bubble_id"] = ret["bubble_id"].astype("int64")

    keys = ["image", "bubble_id"]
    for name, frame in (("master", master), ("returned", ret)):
        dup = frame.duplicated(keys).sum()
        if dup:
            raise SystemExit(f"[fail] {name} has {dup} duplicate {keys} rows")

    df = master.merge(ret[keys + ["class", "notes"]], on=keys, how="left",
                      indicator=True)
    missing = int((df["_merge"] == "left_only").sum())
    extra = len(ret) - int((df["_merge"] == "both").sum())
    if missing or extra:
        raise SystemExit(
            f"[fail] join is not 1:1 -- {missing} master rows unmatched, "
            f"{extra} returned rows unused")
    df = df.drop(columns="_merge")

    df["class_labeler"] = _clean(df["class"])
    bad = sorted(set(df["class_labeler"]) - set(CLASS_RANK) - {""})
    if bad:
        raise SystemExit(f"[fail] unrecognized class values: {bad}")

    df["class"] = promote_to_group(df)
    df["notes"] = _clean(df["notes"])
    blank = df["class_labeler"] == ""
    if args.blank_note:
        df.loc[blank & (df["notes"] == ""), "notes"] = args.blank_note
        df.loc[blank & (df["notes"] != ""), "notes"] += f"; {args.blank_note}"

    promoted = df["class"] != df["class_labeler"]
    n_grp_mixed = (df[df["class_labeler"] != ""]
                   .groupby(["image", "seep_group_id"])["class_labeler"]
                   .nunique().gt(1).sum())

    # --- write ---------------------------------------------------------------
    shutil.copyfile(args.master, args.out)
    con = sqlite3.connect(args.out)
    cur = con.cursor()
    _ensure_columns(cur, args.layer, {"class_labeler": "TEXT"})
    trigs = con.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
        "AND tbl_name=? AND sql IS NOT NULL", (args.layer,)).fetchall()
    try:
        for name, _ in trigs:
            con.execute(f'DROP TRIGGER IF EXISTS "{name}"')
        cur.executemany(
            f'UPDATE "{args.layer}" SET class=?, class_labeler=?, notes=? '
            f"WHERE fid=?",
            list(zip(df["class"], df["class_labeler"], df["notes"],
                     df["fid"].astype(int))))
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

    print(f"[apply] wrote {args.out}")
    print(f"[apply]   {len(df)} rows; class from labeler: "
          f"{(~blank).sum()} classified, {blank.sum()} blank")
    print(f"[apply]   {n_grp_mixed} mixed-class groups -> "
          f"{promoted.sum()} polygons promoted")
    print(f"[apply]   final class: "
          f"{df['class'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()