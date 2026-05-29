"""Phase 1 (2026-05-28 plan): add seep_group_id / is_pregrouped to label packs.

Adds the two grouping columns to EXISTING labeler GPKGs *in place*, without
re-running the allocation sampler -- so the exact same polygons stay in each
pack (we only add fields, never reshuffle membership).

A GeoPackage is a SQLite database, so we use `ALTER TABLE ... ADD COLUMN` and
plain `UPDATE`s on the feature table directly. This is deliberately NOT done via
geopandas `to_file`, because rewriting the file would drop the `layer_styles`
layer (QGIS symbology) and any other sidecar layers. ALTER TABLE touches only
the feature table and leaves everything else byte-intact.

Columns added (per the 2026-05-28 CLAUDE.md plan, Phase 1):
  * seep_group_id INTEGER  -- default = that row's seep_id (every polygon starts
    as its own singleton group). seep_id is PER-CHIP, so the dissolve key
    downstream is (image, seep_group_id), never seep_group_id alone.
  * is_pregrouped INTEGER  -- 0/1 boolean; default 0. Labelers flip to 1 for
    polygons that already enclose multiple bubbles as one envelope shape.

Existing `class` values are DISCARDED here (reset to '') per the 2026-05-29
decision to redo classification *after* grouping rather than reconcile labels
that were entered at the individual-polygon level before grouping existed.

Each target file is copied into a timestamped backup folder before any change.
The script is idempotent: re-running skips columns that already exist (but will
re-clear `class` each time -- intended, since this is a reset step).

LABELER NOTE encoded in the docstring of the produced packs / rubric:
when assigning a NEW shared seep_group_id in QGIS, pick an integer LARGER than
the largest seep_id anywhere in the layer. That guarantees it cannot collide
with another polygon's default (own-seep_id) value and accidentally absorb it.
"""

import os
import shutil
import sqlite3
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Config: which files get the fields. Per the 2026-05-29 decision this is the
# numbered packs + master (a clean grouping-first restart); the person-named
# allison/katey copies are left as the archive of the old polygon-level labels.
# ---------------------------------------------------------------------------
REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
LABELING_DIR = os.path.join(
    REPO_PATH, "data", "results", "SWIN", "AE",
    "20260428-1537_SWINxAE.weights", "labeling",
)
TARGET_FILES = [
    "gt_seeps_label_labeler_1.gpkg",
    "gt_seeps_label_labeler_2.gpkg",
    "gt_seeps_label_labeler_3.gpkg",
    "gt_seeps_label_master.gpkg",
]

DISCARD_EXISTING_CLASS = True   # reset `class` to '' (redo after grouping)

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


def _feature_table(con):
    """Return the GPKG feature-table name (the data layer, not layer_styles)."""
    rows = con.execute(
        "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
    ).fetchall()
    if not rows:
        raise RuntimeError("no feature table found in gpkg_contents")
    if len(rows) > 1:
        raise RuntimeError(f"expected one feature table, found {[r[0] for r in rows]}")
    return rows[0][0]


def _columns(con, table):
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _table_triggers(con, table):
    """Return [(name, sql), ...] for triggers defined on `table`.

    GDAL-written GPKGs carry RTree spatial-index triggers that call ST_*
    functions (ST_IsEmpty, ST_MinX, ...) which a plain sqlite3 connection does
    not provide. They fire on UPDATE and would abort our non-spatial column
    edits, so we drop them for the duration and recreate them afterwards. The
    spatial index stays valid because we never touch geometry.
    """
    return con.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type='trigger' AND tbl_name=? AND sql IS NOT NULL",
        (table,),
    ).fetchall()


def add_group_fields(path, discard_class=DISCARD_EXISTING_CLASS):
    con = sqlite3.connect(path)
    triggers = []
    try:
        tbl = _feature_table(con)
        cols = _columns(con, tbl)
        if "seep_id" not in cols:
            raise RuntimeError(f"{os.path.basename(path)}: no seep_id column to seed from")

        triggers = _table_triggers(con, tbl)
        for name, _sql in triggers:
            con.execute(f'DROP TRIGGER IF EXISTS "{name}"')

        added = []
        if "seep_group_id" not in cols:
            con.execute(f'ALTER TABLE "{tbl}" ADD COLUMN seep_group_id INTEGER')
            added.append("seep_group_id")
        if "is_pregrouped" not in cols:
            con.execute(f'ALTER TABLE "{tbl}" ADD COLUMN is_pregrouped INTEGER')
            added.append("is_pregrouped")

        # Seed defaults. seep_group_id only seeded where NULL so a re-run does
        # not clobber grouping work already entered by a labeler.
        con.execute(
            f'UPDATE "{tbl}" SET seep_group_id = seep_id WHERE seep_group_id IS NULL'
        )
        con.execute(
            f'UPDATE "{tbl}" SET is_pregrouped = 0 WHERE is_pregrouped IS NULL'
        )
        n_class = 0
        if discard_class and "class" in cols:
            n_class = con.execute(
                f"UPDATE \"{tbl}\" SET class = '' "
                f"WHERE class IS NOT NULL AND TRIM(class) <> ''"
            ).rowcount

        for _name, sql in triggers:
            con.execute(sql)   # recreate exactly as captured
        triggers = []          # recreated; nothing to restore in finally
        con.commit()

        n_rows = con.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()[0]
        n_groups = con.execute(
            f'SELECT COUNT(*) FROM (SELECT DISTINCT image, seep_group_id FROM "{tbl}")'
        ).fetchone()[0]
        print(f"  {os.path.basename(path):38s} table={tbl}")
        print(f"     added cols: {added or '(none, already present)'}")
        print(f"     rows={n_rows}  distinct (image,seep_group_id)={n_groups}  "
              f"class cleared={n_class}")
    except Exception:
        # Restore any triggers we dropped but did not get to recreate.
        con.rollback()
        for _name, sql in triggers:
            try:
                con.execute(sql)
            except sqlite3.OperationalError:
                pass
        con.commit()
        raise
    finally:
        con.close()


def main():
    if not os.path.isdir(LABELING_DIR):
        raise SystemExit(f"labeling dir not found: {LABELING_DIR}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = os.path.join(LABELING_DIR, f"backup_pre_group_fields_{stamp}")
    os.makedirs(backup_dir, exist_ok=True)
    print(f"backing up targets -> {backup_dir}")
    for fn in TARGET_FILES:
        src = os.path.join(LABELING_DIR, fn)
        if not os.path.exists(src):
            print(f"  WARNING: {fn} not found, skipping")
            continue
        shutil.copy2(src, os.path.join(backup_dir, fn))

    print("\nadding group fields:")
    for fn in TARGET_FILES:
        src = os.path.join(LABELING_DIR, fn)
        if os.path.exists(src):
            add_group_fields(src)

    print("\ndone. New columns: seep_group_id (default=seep_id), is_pregrouped (0).")
    print("Existing class values were cleared (redo classification after grouping).")


if __name__ == "__main__":
    main()