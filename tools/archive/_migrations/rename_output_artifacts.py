"""ONE-SHOT MIGRATION (2026-09-03): rename bubble-tier output artifacts.

These files are all per-BUBBLE but were named `seep_*` from before the grouper
existed. They are renamed on disk here so nothing is left orphaned when the
code starts writing the new names.

    gt_seeps.gpkg                  -> gt_bubbles.gpkg
    seep_features_per_bubble.csv   -> bubble_features.csv
    seep_level_summary.csv         -> bubble_level_summary.csv
    seep_level_per_image.csv       -> bubble_level_per_image.csv
    seep_level_pairs.csv           -> bubble_level_pairs.csv
    historical_seep_size_priors.json -> historical_size_priors.json

NOT renamed, deliberately:
  * `gt_seeps_label_*.gpkg` -- the labeler packs. Labelers have these open in
    QGIS and the name is cosmetic; renaming them buys nothing and risks
    breaking someone's session mid-labeling.
  * anything under a `20260903_defunct_clustering` archive dir -- those are
    the retired cluster-level artifacts and keep their historical names.

Idempotent: a destination that already exists is skipped, and a source that is
absent is skipped.

    python -m tools.archive._migrations.rename_output_artifacts
    python -m tools.archive._migrations.rename_output_artifacts --apply
"""
import argparse
import os

RENAMES = {
    "gt_seeps.gpkg": "gt_bubbles.gpkg",
    "seep_features_per_bubble.csv": "bubble_features.csv",
    "seep_level_summary.csv": "bubble_level_summary.csv",
    "seep_level_per_image.csv": "bubble_level_per_image.csv",
    "seep_level_pairs.csv": "bubble_level_pairs.csv",
    "historical_seep_size_priors.json": "historical_size_priors.json",
}

# directories whose contents keep their historical names
FROZEN = ("20260903_defunct_clustering", "_pre_cleanup_manifest")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="actually rename (default is a dry run)")
    ap.add_argument("--root", default="data")
    a = ap.parse_args()

    hits = []
    for root, dirs, files in os.walk(a.root):
        if any(f in root for f in FROZEN):
            continue
        for f in files:
            if f in RENAMES:
                hits.append((os.path.join(root, f),
                             os.path.join(root, RENAMES[f])))

    print(f"{'APPLY' if a.apply else 'DRY RUN'}: {len(hits)} artifact(s)\n")
    done = skipped = 0
    for src, dst in hits:
        if os.path.exists(dst):
            print(f"  SKIP (dest exists)  {dst}")
            skipped += 1
            continue
        print(f"  {'RENAMED' if a.apply else 'WOULD RENAME'}  "
              f"{os.path.basename(src)} -> {os.path.basename(dst)}"
              f"   in {os.path.dirname(src)}")
        if a.apply:
            os.rename(src, dst)
        done += 1
    print(f"\n{done} renamed, {skipped} skipped")


if __name__ == "__main__":
    main()
