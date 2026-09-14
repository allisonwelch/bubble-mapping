"""One-shot migration: fold the still-unique part of the 2026-06 chip-39 pack
into allison's final labeler pack, so `final_labeler_packs/` is the only place
the grouper and classifier ever read from.

APPLIED 2026-09-11 -- allison's pack went 1240 -> 1440 rows. Kept for the record
and because it documents the reconciliation; it refuses to run twice.

WHY
`gt_seeps_label_chip39_classified.gpkg` (406 rows) was the chip-39 grouping /
classification session. Its bbox is the north half of 39.tif -- allison's
`39-NW` quarter -- and 206 of its rows carry the SAME (image, bubble_id) keys as
that quarter, which she later relabeled in full. On those 206 the two disagree
on 21 classes and on 31 `is_pregrouped` flags, and the final pack is the newer,
complete labeling. So those 206 are a superseded duplicate and are DROPPED.

The other 200 rows sit outside every final quarter and are the only labeling
those polygons have (134 A / 64 B / 2 C in 69 groups). No group straddles the
covered/uncovered boundary, so the 200 come out whole -- nothing is truncated.
They are appended to allison's pack under `unit = '39-N-extra'`, which is what
makes the older vintage traceable and droppable later.

Verified before writing (see the checks in `main`): the salvaged bubble_ids and
seep_group_ids are disjoint from both 39.tif quarters, and no feature column is
null. `seep_group_id` is renormalised to the anchor rubric (max member
bubble_id) -- 40 of the 69 groups predate it. Still collision-free, because the
anchor is itself a salvaged bubble_id.

Appending goes through pyogrio (`mode="a"`), not a geopandas rewrite: a rewrite
would drop the pack's `layer_styles` layer and with it every labeler's QGIS
symbology.

    python -m tools.labeling.salvage_chip39_north [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import datetime as _dt

import geopandas as gpd
import pandas as pd

from tools.paths import CANONICAL_PRED_RELDIR

LAB = os.path.join(CANONICAL_PRED_RELDIR, "labeling")
FIN = os.path.join(LAB, "final_labeler_packs")
SRC = os.path.join(LAB, "archive", "backup_pre_classified_20260624",
                   "gt_seeps_label_chip39_classified.gpkg")
ALLISON = os.path.join(FIN, "gt_seeps_label_quarters_allison_grouped.gpkg")
KATEY = os.path.join(FIN, "gt_seeps_label_quarters_katey_grouped.gpkg")

UNIT = "39-N-extra"
NOTE = ("salvaged 2026-09-11 from gt_seeps_label_chip39_classified.gpkg "
        "(2026-06 chip-39 session); the 206 rows superseded by 39-NW were dropped")


def build_salvage() -> gpd.GeoDataFrame:
    """The 200 chip-39 rows no final quarter covers, in allison's schema."""
    src = gpd.read_file(SRC)
    ali = gpd.read_file(ALLISON, layer="labels")
    kat = gpd.read_file(KATEY, layer="labels")

    covered = set(ali.loc[ali["image"] == "39.tif", "bubble_id"].astype(int))
    covered |= set(kat.loc[kat["image"] == "39.tif", "bubble_id"].astype(int))
    keep = src[~src["bubble_id"].astype(int).isin(covered)].copy()

    # Groups must come out whole; a straddling group would arrive truncated,
    # which is the one thing that would make the salvaged hulls wrong.
    dropped_gids = set(src.loc[src["bubble_id"].astype(int).isin(covered),
                               "seep_group_id"].astype(int))
    straddle = set(keep["seep_group_id"].astype(int)) & dropped_gids
    if straddle:
        raise SystemExit(f"[fail] {len(straddle)} group(s) straddle the "
                         f"covered/uncovered boundary: {sorted(straddle)[:10]}")

    # Anchor rubric: seep_group_id = max member bubble_id. 40 of the 69 groups
    # predate it. The anchor is a salvaged bubble_id, so this stays disjoint
    # from both quarters' id spaces.
    anchor = (keep.groupby("seep_group_id")["bubble_id"]
              .transform(lambda s: s.astype(int).max()))
    keep["seep_group_id"] = anchor.astype(int)

    keep["labeler"] = "allison"
    keep["unit"] = UNIT
    keep["is_calibration"] = False
    keep["is_context"] = 0
    keep["notes"] = NOTE
    # Never model-grouped, so there is no frozen proposal to diff against.
    keep["seep_group_id_pred"] = None
    keep["group_source"] = None

    cols = [c for c in ali.columns if c != "geometry"] + ["geometry"]
    missing = [c for c in cols if c not in keep.columns]
    if missing:
        raise SystemExit(f"[fail] salvage frame is missing {missing}")
    keep = keep[cols]
    return gpd.GeoDataFrame(keep, geometry="geometry", crs=ali.crs), ali, kat


def check(keep, ali, kat) -> None:
    a39 = ali[ali["image"] == "39.tif"]
    k39 = kat[kat["image"] == "39.tif"]
    bid = set(keep["bubble_id"].astype(int))
    gid = set(keep["seep_group_id"].astype(int))
    problems = []
    for name, other in (("allison[39]", a39), ("katey[39]", k39)):
        if bid & set(other["bubble_id"].astype(int)):
            problems.append(f"bubble_id collision with {name}")
        if gid & set(other["seep_group_id"].astype(int)):
            problems.append(f"seep_group_id collision with {name}")
    feats = ["area_m2", "mean_R", "mean_G", "mean_B", "circularity",
             "solidity", "eccentricity", "centroid_x_m", "centroid_y_m"]
    n_null = int(keep[feats].isna().sum().sum())
    if n_null:
        problems.append(f"{n_null} null feature value(s)")
    if (keep["class"].fillna("").str.strip() == "").any():
        problems.append("unclassified row(s) in the salvage set")
    if problems:
        raise SystemExit("[fail] " + "; ".join(problems))
    print(f"[check] {len(keep)} rows, {keep.groupby('seep_group_id').ngroups} "
          f"groups, classes {keep['class'].value_counts().to_dict()}")
    print("[check] no bubble_id / seep_group_id collision with either quarter; "
          "no null features")


def layer_names(path):
    con = sqlite3.connect(path)
    try:
        return sorted(r[0] for r in
                      con.execute("select table_name from gpkg_contents"))
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    keep, ali, kat = build_salvage()
    if (ali["unit"] == UNIT).any():
        raise SystemExit(
            f"[skip] allison's pack already carries "
            f"{int((ali['unit'] == UNIT).sum())} '{UNIT}' rows -- this "
            f"migration has run. Re-running would duplicate them.")
    check(keep, ali, kat)
    before_layers = layer_names(ALLISON)
    print(f"[before] {os.path.basename(ALLISON)}: {len(ali)} rows, "
          f"layers {before_layers}")

    if args.dry_run:
        print("[dry-run] nothing written")
        return

    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = os.path.join(LAB, "archive", f"pre_chip39_salvage_{stamp}")
    os.makedirs(bak, exist_ok=True)
    shutil.copy2(ALLISON, os.path.join(bak, os.path.basename(ALLISON)))
    print(f"[backup] {os.path.join(bak, os.path.basename(ALLISON))}")

    keep.to_file(ALLISON, layer="labels", driver="GPKG", mode="a")

    after = gpd.read_file(ALLISON, layer="labels")
    after_layers = layer_names(ALLISON)
    print(f"[after ] {len(after)} rows (+{len(after) - len(ali)}), "
          f"layers {after_layers}")
    if after_layers != before_layers:
        raise SystemExit("[fail] layer set changed -- layer_styles may be gone")
    n_extra = int((after["unit"] == UNIT).sum())
    dup = after.duplicated(["image", "bubble_id"]).sum()
    print(f"[after ] unit={UNIT}: {n_extra} rows; "
          f"duplicate (image, bubble_id): {dup}")
    if n_extra != len(keep) or dup:
        raise SystemExit("[fail] append did not land cleanly")
    print(f"[after ] 39.tif now: "
          f"{after[after['image'] == '39.tif']['unit'].value_counts().to_dict()}")
    print("[done]")


if __name__ == "__main__":
    main()
