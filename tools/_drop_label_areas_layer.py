"""One-off: delete the `label_areas` layer from the quarter label packs,
leaving the (labeler-edited) `labels` layer untouched.

A GPKG layer drop = drop the feature table + its rtree index (virtual table
and triggers) and deregister it from the gpkg_* registry tables and
layer_styles. Backups are written next to the originals first.
"""
import glob
import os
import shutil
import sqlite3

PRED_DIR = os.path.join(
    "data", "results", "SWIN", "AE", "20260428-1537_SWINxAE.weights", "labeling"
)
LAYER = "label_areas"

for path in sorted(glob.glob(os.path.join(PRED_DIR, "gt_seeps_label_quarters_*.gpkg"))):
    bak = path + ".pre_drop_label_areas.bak"
    shutil.copy2(path, bak)

    con = sqlite3.connect(path)
    cur = con.cursor()
    try:
        # triggers referencing the layer's rtree (drop before the tables)
        trig = [r[0] for r in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE ?",
            (f"rtree_{LAYER}_%",),
        )]
        for t in trig:
            cur.execute(f'DROP TRIGGER IF EXISTS "{t}"')
        # rtree virtual table (dropping it removes its shadow tables)
        cur.execute(f'DROP TABLE IF EXISTS "rtree_{LAYER}_geom"')
        # the feature table itself
        cur.execute(f'DROP TABLE IF EXISTS "{LAYER}"')
        # registry + style rows
        for table, col in [
            ("gpkg_contents", "table_name"),
            ("gpkg_geometry_columns", "table_name"),
            ("gpkg_ogr_contents", "table_name"),
            ("gpkg_extensions", "table_name"),
            ("layer_styles", "f_table_name"),
        ]:
            exists = cur.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone()
            if exists:
                cur.execute(f'DELETE FROM "{table}" WHERE "{col}" = ?', (LAYER,))
        con.commit()
        cur.execute("VACUUM")
        left = [r[0] for r in cur.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'"
        )]
        n = cur.execute('SELECT COUNT(*) FROM "labels"').fetchone()[0]
        print(f"{os.path.basename(path)}: dropped {LAYER}; feature layers now {left}, "
              f"labels rows = {n}  (backup: {os.path.basename(bak)})")
    finally:
        con.close()
