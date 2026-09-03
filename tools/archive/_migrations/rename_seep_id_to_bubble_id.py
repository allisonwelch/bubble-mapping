"""ONE-SHOT MIGRATION (2026-09-03): rename the `seep_id` column to `bubble_id`
in every GeoPackage under data/, and fix the QGIS styles that reference it.

WHY
`seep_id` is 1:1 with drawn polygons -- it has always been a BUBBLE id. The
actual seep id is `seep_group_id` (1163 polygons -> 834 groups in a typical
pack). Under the pixel/bubble/seep convention the column was simply misnamed.

WHY ONLY THIS COLUMN
`seep_group_id` is deliberately NOT renamed. Renaming both would ROTATE the
names -- `seep_id` would be reused to mean something new -- and a missed
reference would then silently read bubble ids where seep ids are expected, with
no error. Renaming only the misnamed one means any missed reference raises
KeyError instead. It also lands the label packs on the exact schema
`pred_bubbles_grouped.gpkg` already uses (bubble_id + seep_group_id), so both
sides of the pipeline converge.

WHAT IT TOUCHES
  * the feature table column, via ALTER TABLE ... RENAME COLUMN
  * the embedded QGIS QML styles in layer_styles (30 records reference the
    column; a rename without this silently breaks every labeler's symbology)
  * the attribute index labels_image_seep_id_idx, renamed to match

GDAL's RTree triggers call ST_* functions that a plain sqlite3 connection does
not have. SQLite re-parses every trigger during RENAME COLUMN, so the triggers
are dropped and recreated around the rename -- the established pattern in this
repo. Geometry is never read or written, so the spatial index stays valid.

Idempotent: a file already carrying bubble_id is skipped.

    python -m tools.archive._migrations.rename_seep_id_to_bubble_id --dry-run
    python -m tools.archive._migrations.rename_seep_id_to_bubble_id --apply
"""
import argparse
import glob
import hashlib
import os
import shutil
import sqlite3
import sys

OLD, NEW = "seep_id", "bubble_id"
OLD_IDX, NEW_IDX = "labels_image_seep_id_idx", "labels_image_bubble_id_idx"


def geom_sha(con, layer):
    """SHA over the geometry blobs in rowid order -- proves the rename touched
    only column names."""
    h = hashlib.sha256()
    try:
        for (g,) in con.execute('select "geom" from "%s" order by rowid' % layer):
            h.update(g if g else b"")
    except sqlite3.Error:
        return None
    return h.hexdigest()[:16]


def layers_with(con, col):
    out = []
    for (L,) in con.execute("select table_name from gpkg_contents"):
        cols = [r[1] for r in con.execute('PRAGMA table_info("%s")' % L)]
        if col in cols:
            out.append(L)
    return out


def migrate(path, apply=False, backup_dir=None):
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys=OFF")
    try:
        targets = layers_with(con, OLD)
        if not targets:
            return "skip: no %s column" % OLD
        for L in targets:
            cols = [r[1] for r in con.execute('PRAGMA table_info("%s")' % L)]
            if NEW in cols:
                return "skip: %s already present in %s" % (NEW, L)

        n_qml = con.execute(
            "select count(*) from sqlite_master where name='layer_styles'"
        ).fetchone()[0]
        if n_qml:
            n_qml = con.execute(
                "select count(*) from layer_styles where styleQML like ?",
                ("%" + OLD + "%",)).fetchone()[0]

        before = {L: (con.execute('select count(*) from "%s"' % L).fetchone()[0],
                      geom_sha(con, L)) for L in targets}

        if not apply:
            return "WOULD rename in %s | %d style record(s)" % (
                ", ".join(targets), n_qml)

        if backup_dir:
            os.makedirs(backup_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))

        # 1. drop triggers (they call ST_* which plain sqlite3 lacks, and
        #    RENAME COLUMN re-parses every trigger in the schema)
        trigs = con.execute(
            "select name, sql from sqlite_master where type='trigger' "
            "and sql is not null").fetchall()
        for name, _ in trigs:
            con.execute('DROP TRIGGER IF EXISTS "%s"' % name)

        # 2. the rename itself (SQLite rewrites dependent index definitions)
        for L in targets:
            con.execute('ALTER TABLE "%s" RENAME COLUMN "%s" TO "%s"'
                        % (L, OLD, NEW))

        # 3. give the attribute index a name that matches its column
        has_idx = con.execute(
            "select count(*) from sqlite_master where type='index' and name=?",
            (OLD_IDX,)).fetchone()[0]
        if has_idx:
            tbl = con.execute(
                "select tbl_name from sqlite_master where name=?",
                (OLD_IDX,)).fetchone()[0]
            con.execute('DROP INDEX "%s"' % OLD_IDX)
            con.execute('CREATE INDEX "%s" ON "%s"("image","%s")'
                        % (NEW_IDX, tbl, NEW))

        # 4. rewrite the embedded QGIS styles.  "seep_group_id" does NOT
        #    contain "seep_id" as a substring, so a plain replace is safe.
        if n_qml:
            con.execute(
                "update layer_styles set styleQML = replace(styleQML, ?, ?) "
                "where styleQML like ?", (OLD, NEW, "%" + OLD + "%"))
            for col in ("styleSLD",):
                try:
                    con.execute(
                        "update layer_styles set %s = replace(%s, ?, ?) "
                        "where %s like ?" % (col, col, col),
                        (OLD, NEW, "%" + OLD + "%"))
                except sqlite3.Error:
                    pass

        # 5. put the triggers back exactly as they were
        for name, sql in trigs:
            con.execute(sql)

        con.commit()

        # 6. verify: same rows, same geometry, column renamed
        for L in targets:
            cols = [r[1] for r in con.execute('PRAGMA table_info("%s")' % L)]
            assert NEW in cols and OLD not in cols, "rename failed on %s" % L
            n, sha = before[L]
            n2 = con.execute('select count(*) from "%s"' % L).fetchone()[0]
            sha2 = geom_sha(con, L)
            assert n == n2, "row count changed in %s: %d -> %d" % (L, n, n2)
            assert sha == sha2, "GEOMETRY CHANGED in %s" % L
        n_trig = con.execute(
            "select count(*) from sqlite_master where type='trigger'").fetchone()[0]
        assert n_trig == len(trigs), \
            "trigger count %d != %d" % (n_trig, len(trigs))

        return "OK  %s | %d styles | %d triggers restored" % (
            ", ".join(targets), n_qml, n_trig)
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually write (default is a dry run)")
    ap.add_argument("--root", default="data",
                    help="tree to walk for .gpkg files")
    ap.add_argument("--backup-dir", default=None,
                    help="copy each file here before writing")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.root, "**", "*.gpkg"),
                             recursive=True))
    print(f"{'APPLY' if a.apply else 'DRY RUN'}: {len(files)} gpkg under {a.root}/\n")
    counts = {}
    for f in files:
        try:
            r = migrate(f, apply=a.apply, backup_dir=a.backup_dir)
        except Exception as e:
            r = "ERROR: %s: %s" % (type(e).__name__, e)
        key = r.split(":")[0].split("|")[0].strip()
        counts[key] = counts.get(key, 0) + 1
        if not r.startswith("skip"):
            print(f"  {r:<62} {f}")
    print("\nsummary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if any(k.startswith("ERROR") for k in counts):
        sys.exit(1)


if __name__ == "__main__":
    main()
