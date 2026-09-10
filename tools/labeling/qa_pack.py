"""Integrity check for a labeler GeoPackage before it feeds the grouper or classifier.

Four categories of failure, in rough order of how silently they bite:

  C. FILE INTEGRITY   -- damage from the sqlite UPDATE-by-fid write pattern.
                         Invisible in QGIS; breaks other people's symbology or
                         the spatial index.
  B. PROVENANCE       -- the pack is not the file you think it is: an ungrouped
                         source copy, a stale duplicate under another directory,
                         classes silently reset, mixed grouping provenance.
  A. LABELER SLIPS    -- group ids belonging to no member, class disagreement
                         inside a group, unflagged envelopes, notes claiming a
                         flag that is not set.
  D. READINESS        -- coverage, class balance, context rows.

What this CANNOT do: tell you an A/B/C call is wrong (that is the subjective
judgment where the model already sits at the human ceiling), or find seeps
nobody labeled. Geometry integrity beyond internal consistency needs a
pre-migration manifest to compare against.

Usage:
    python -m tools.labeling.qa_pack PACK.gpkg
    python -m tools.labeling.qa_pack PACK.gpkg --siblings labeling/final_labeler_packs
    python -m tools.labeling.qa_pack --all labeling/final_labeler_packs
"""
import argparse
import glob
import os
import sqlite3
import sys
from collections import Counter, defaultdict

# A grouped pack whose seep_group_id equals bubble_id on ~every row has not been
# grouped -- this is the signature of the pack handed out on an ungrouped copy.
UNGROUPED_PCT = 97.0
# Hull major axis implied by the 1.0 m centroid-span agglomeration cap, plus the
# end-bubble radii. A SINGLETON wider than this is almost certainly an envelope
# polygon that nobody flagged.
SINGLETON_MAX_M = 1.2
OVERGROUP_HINTS = ("multiple seep", "overgroup", "two seeps", "more than one seep")
REQUIRED = ("image", "bubble_id", "seep_group_id", "class")

OK, WARN, FAIL = "ok", "WARN", "FAIL"


class Report:
    def __init__(self):
        self.rows = []

    def add(self, cat, level, msg, detail=None):
        self.rows.append((cat, level, msg, detail or []))

    def worst(self):
        levels = [r[1] for r in self.rows]
        return FAIL if FAIL in levels else (WARN if WARN in levels else OK)

    def render(self, name):
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        for cat in ("C", "B", "A", "D"):
            rows = [r for r in self.rows if r[0] == cat]
            if not rows:
                continue
            title = {"C": "C. FILE INTEGRITY", "B": "B. PROVENANCE",
                     "A": "A. LABELER SLIPS", "D": "D. READINESS"}[cat]
            print(f"\n{title}")
            for _, level, msg, detail in rows:
                mark = {OK: "  ok  ", WARN: " WARN ", FAIL: " FAIL "}[level]
                print(f"  [{mark}] {msg}")
                for d in detail[:12]:
                    print(f"            {d}")
                if len(detail) > 12:
                    print(f"            ... and {len(detail) - 12} more")
        print(f"\nVERDICT: {self.worst()}")
        return self.worst()


def cols(con, table="labels"):
    return [r[1] for r in con.execute(f'pragma table_info("{table}")')]


def check_integrity(con, fp, rep, sibling_triggers=None):
    layers = [r[0] for r in con.execute(
        "select table_name from gpkg_contents where data_type='features'")]
    if "labels" not in layers:
        rep.add("C", FAIL, f"no `labels` layer (found {layers or 'none'})")
        return False
    have = cols(con)
    missing = [c for c in REQUIRED if c not in have]
    if missing:
        rep.add("C", FAIL, f"missing required columns: {missing}")
    if "seep_id" in have:
        rep.add("C", FAIL, "column `seep_id` still present -- should be `bubble_id` "
                           "(renamed 2026-09-03)")

    n = con.execute("select count(*) from labels").fetchone()[0]

    trig = con.execute("select count(*) from sqlite_master "
                       "where type='trigger' and tbl_name='labels'").fetchone()[0]
    if sibling_triggers and trig != sibling_triggers:
        rep.add("C", FAIL, f"{trig} triggers on `labels`, siblings have "
                           f"{sibling_triggers} -- a write dropped them and did not "
                           f"recreate them")
    elif trig == 0:
        rep.add("C", FAIL, "no triggers on `labels` -- RTree maintenance is gone")
    else:
        rep.add("C", OK, f"{trig} triggers present")

    try:
        rt = con.execute("select count(*) from rtree_labels_geom").fetchone()[0]
        if rt != n:
            rep.add("C", FAIL, f"RTree has {rt} entries for {n} rows -- spatial index stale")
        else:
            rep.add("C", OK, f"RTree consistent ({rt} entries)")
    except sqlite3.Error:
        rep.add("C", WARN, "no RTree index on `labels`")

    try:
        ns = con.execute("select count(*) from layer_styles").fetchone()[0]
        rep.add("C", OK if ns else WARN,
                f"layer_styles: {ns} record(s)" if ns else "layer_styles table is empty")
        stale = con.execute(
            "select count(*) from layer_styles where styleQML like '%seep_id%'"
        ).fetchone()[0]
        if stale:
            rep.add("C", FAIL, f"{stale} QML style(s) still reference `seep_id` -- "
                               "labeler symbology is broken")
    except sqlite3.Error:
        rep.add("C", WARN, "no layer_styles table -- QGIS symbology will not travel "
                           "with this pack (typical of an ungrouped source copy)")

    if "image" in have and "bubble_id" in have:
        dupes = con.execute(
            "select image, bubble_id, count(*) c from labels "
            "group by image, bubble_id having c > 1").fetchall()
        if dupes:
            rep.add("C", FAIL, f"{len(dupes)} duplicate (image, bubble_id) keys",
                    [f"{i} / {b} x{c}" for i, b, c in dupes])
        else:
            rep.add("C", OK, "(image, bubble_id) unique")
    return True


def check_provenance(con, fp, rep, all_packs):
    have = cols(con)
    n = con.execute("select count(*) from labels").fetchone()[0]
    k = con.execute('select count(*) from labels '
                    'where class is not null and class != ""').fetchone()[0]
    rep.add("B", OK if k else WARN,
            f"{k} of {n} rows classified ({100 * k / max(n, 1):.1f}%)")
    if n and not k:
        rep.add("B", WARN, "pack carries NO classes -- if this should hold labels, a "
                           "regeneration or hull rebuild may have dropped them")

    eq = con.execute("select count(*) from labels "
                     "where seep_group_id = bubble_id").fetchone()[0]
    pct = 100 * eq / max(n, 1)
    if pct >= UNGROUPED_PCT:
        rep.add("B", FAIL, f"seep_group_id == bubble_id on {pct:.1f}% of rows -- this is "
                           f"an UNGROUPED source copy, not a grouped pack. If a labeler "
                           f"classified this, only `class` is salvageable.")
    else:
        rep.add("B", OK, f"grouping present (seep_group_id == bubble_id on {pct:.1f}%)")

    if "group_source" in have:
        gs = dict(con.execute(
            "select coalesce(group_source,'(null)'), count(*) from labels "
            "group by 1").fetchall())
        if len(gs) > 1:
            rep.add("B", WARN, f"mixed grouping provenance: {gs} -- model-proposed and "
                               f"human-corrected grouping in one file")
        else:
            rep.add("B", OK, f"group_source: {gs}")

    # A twin under archive/ or backup*/ is deliberate history, not a hazard. Only a
    # LIVE duplicate can be picked up by a wrong --labeling-dir, and only a
    # differing one is dangerous -- that is how the stale July katey pack (289
    # classified vs 1138) still sits beside the completed August one.
    base = os.path.basename(fp)
    twins = [p for p in all_packs
             if os.path.basename(p) == base and os.path.abspath(p) != os.path.abspath(fp)]
    live, archived = [], []
    for t in twins:
        # This repo marks history both ways: `archive/`, `backup_pre_*/` and also
        # `<packname>_archive/`. Match either end of a path segment.
        parts = [q.lower() for q in os.path.normpath(t).split(os.sep)]
        marked = any(q.startswith(("archive", "backup")) or
                     q.endswith(("archive", "_backup", "backup"))
                     for q in parts)
        (archived if marked else live).append(t)
    stale = []
    for t in live:
        try:
            c2 = sqlite3.connect(t)
            k2 = c2.execute('select count(*) from labels '
                            'where class is not null and class != ""').fetchone()[0]
            c2.close()
        except sqlite3.Error:
            k2 = -1
        stale.append((t, k2))
    differing = [(t, k2) for t, k2 in stale if k2 != k]
    if differing:
        rep.add("B", FAIL,
                f"{len(differing)} LIVE duplicate basename(s) with a different label "
                f"count ({k} classified here) -- a wrong --labeling-dir silently "
                f"picks the other",
                [f"{t}  ({k2} classified)" for t, k2 in differing])
    elif stale:
        rep.add("B", WARN, f"{len(stale)} live duplicate basename(s), same label count",
                [t for t, _ in stale])
    else:
        rep.add("B", OK, f"no live duplicate basename"
                         + (f" ({len(archived)} archived copy/copies)" if archived else ""))


def _major_axis(geom):
    try:
        mrr = geom.minimum_rotated_rectangle
        xs, ys = mrr.exterior.coords.xy
        pts = list(zip(xs, ys))
        d = [((pts[i][0] - pts[i + 1][0]) ** 2 + (pts[i][1] - pts[i + 1][1]) ** 2) ** 0.5
             for i in range(len(pts) - 1)]
        return max(d) if d else 0.0
    except Exception:
        return 0.0


def check_labeler(con, fp, rep):
    have = cols(con)
    rows = con.execute(
        "select image, bubble_id, seep_group_id, "
        "       coalesce(class,''), coalesce(notes,''), "
        f"      {'is_pregrouped' if 'is_pregrouped' in have else '0'}, "
        f"      {'is_overgrouped' if 'is_overgrouped' in have else '0'} "
        "from labels").fetchall()

    members = defaultdict(list)
    for img, bid, gid, cls, notes, preg, over in rows:
        members[(img, gid)].append((bid, cls, notes, preg, over))

    orphan = [f"{img} group {gid} (members {[m[0] for m in ms][:6]})"
              for (img, gid), ms in members.items()
              if gid not in {m[0] for m in ms}]
    rep.add("A", FAIL if orphan else OK,
            f"{len(orphan)} group id(s) belong to no member -- the mid-range-id "
            f"collision; set seep_group_id to the anchor's own bubble_id"
            if orphan else "every seep_group_id belongs to one of its members", orphan)

    mixed = []
    for (img, gid), ms in members.items():
        cs = {m[1] for m in ms if m[1]}
        if len(cs) > 1:
            mixed.append(f"{img} group {gid}: {sorted(cs)} "
                         f"({[m[1] or '-' for m in ms]})")
    rep.add("A", FAIL if mixed else OK,
            f"{len(mixed)} group(s) mix classes -- promotion is MAX (C > B > A)"
            if mixed else "no class disagreement within a group", mixed)

    hinted = [f"{img} / {bid}: {notes[:60]}"
              for img, bid, gid, cls, notes, preg, over in rows
              if any(h in notes.lower() for h in OVERGROUP_HINTS) and not over]
    rep.add("A", WARN if hinted else OK,
            f"{len(hinted)} row(s) have notes claiming overgrouping with "
            f"is_overgrouped unset" if hinted else "no unflagged overgroup claims", hinted)

    try:
        import geopandas as gpd
        g = gpd.read_file(fp, layer="labels")
        sizes = Counter(zip(g["image"], g["seep_group_id"]))
        singles = g[[sizes[(i, s)] == 1 for i, s in zip(g["image"], g["seep_group_id"])]]
        if "is_pregrouped" in g.columns:
            singles = singles[singles["is_pregrouped"].fillna(0).astype(int) == 0]
        big = [(r["image"], r["bubble_id"], _major_axis(r.geometry))
               for _, r in singles.iterrows()]
        big = [b for b in big if b[2] > SINGLETON_MAX_M]
        rep.add("A", WARN if big else OK,
                f"{len(big)} unflagged singleton(s) wider than {SINGLETON_MAX_M} m -- "
                f"likely envelope polygons; set is_pregrouped" if big
                else f"no unflagged singletons above {SINGLETON_MAX_M} m",
                [f"{i} / {b}: {m:.2f} m" for i, b, m in sorted(big, key=lambda x: -x[2])])
    except ImportError:
        rep.add("A", WARN, "geopandas unavailable -- skipped the oversized-singleton check")
    except Exception as e:
        rep.add("A", WARN, f"oversized-singleton check failed: {e}")


def check_readiness(con, rep):
    have = cols(con)
    bal = dict(con.execute('select class, count(*) from labels '
                           'where class is not null and class != "" group by class'))
    rep.add("D", OK, f"class balance: {bal or 'none'}")
    per = con.execute(
        'select image, count(*), sum(case when class is not null and class != "" '
        'then 1 else 0 end) from labels group by image order by image').fetchall()
    rep.add("D", OK, f"{len(per)} chip(s)",
            [f"{i}: {n} rows, {k} classified" for i, n, k in per])
    if "is_context" in have:
        nc = con.execute("select count(*) from labels "
                         "where is_context = 1").fetchone()[0]
        rep.add("D", OK, f"{nc} context row(s) -- excluded from fitting, kept as neighbours")


def qa(fp, all_packs, sibling_triggers=None):
    rep = Report()
    con = sqlite3.connect(fp)
    try:
        if check_integrity(con, fp, rep, sibling_triggers):
            check_provenance(con, fp, rep, all_packs)
            check_labeler(con, fp, rep)
            check_readiness(con, rep)
    finally:
        con.close()
    return rep.render(os.path.relpath(fp))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pack", nargs="?", help="pack to check")
    ap.add_argument("--all", metavar="DIR", help="check every .gpkg in DIR")
    ap.add_argument("--siblings", metavar="DIR",
                    help="directory of known-good packs, for the trigger-count baseline")
    ap.add_argument("--tree", default="data",
                    help="root to scan for duplicate basenames (default: data)")
    a = ap.parse_args()

    all_packs = glob.glob(os.path.join(a.tree, "**", "*.gpkg"), recursive=True)

    baseline = None
    if a.siblings:
        counts = []
        for p in glob.glob(os.path.join(a.siblings, "*.gpkg")):
            try:
                c = sqlite3.connect(p)
                counts.append(c.execute(
                    "select count(*) from sqlite_master where type='trigger' "
                    "and tbl_name='labels'").fetchone()[0])
                c.close()
            except sqlite3.Error:
                pass
        if counts:
            baseline = Counter(counts).most_common(1)[0][0]

    targets = ([a.pack] if a.pack else
               sorted(glob.glob(os.path.join(a.all or ".", "*.gpkg"))))
    if not targets:
        ap.error("nothing to check")

    verdicts = [qa(t, all_packs, baseline) for t in targets]
    print(f"\n{len(verdicts)} pack(s): "
          f"{verdicts.count(FAIL)} FAIL, {verdicts.count(WARN)} WARN, "
          f"{verdicts.count(OK)} ok")
    return 1 if FAIL in verdicts else 0


if __name__ == "__main__":
    sys.exit(main())
