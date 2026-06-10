"""Build a labeling pack for ALL GT label polygons intersecting a single chip.

Produces a GeoPackage with the *exact* same field schema as the
`gt_seeps_label_labeler_*.gpkg` packs (post `seep_add_group_fields.py`), but
scoped to every label polygon in one chip rather than a stratified sample. Use
this when you want to run the grouping + A/B/C classification exercise over a
whole dense chip (e.g. chip 39, the over-merge canary) instead of the sampled
750-seep allocation.

Source polygons come straight from `gt_seeps.gpkg`, whose rows are the original
drawn polygons clipped per-chip (built by `build_gt_seeps_from_source`), tagged
with an `image` column. "Intersects chip N" is therefore exactly the rows whose
`image == "<N>.tif"` -- those are already clipped to the chip footprint and carry
the per-polygon morphology/brightness features.

Columns added on top of gt_seeps to match the labeler-pack schema:
  * allocation tags:  labeler, is_calibration, area_bin, area_label,
                      sol_bin, sol_label  (strata via the allocation script's
                      assign_strata, computed over the FULL gt population so the
                      cuts match the existing packs)
  * grouping fields:  seep_group_id (default = seep_id), is_pregrouped (0)
  * class reset to '' for QGIS attribute-table labeling

The labeler grouping rule is unchanged: to group a cluster, set every member's
seep_group_id to the anchor (largest) bubble's own seep_id. seep_id is unique
within a chip, so this is collision-free.
"""

import os
import sys

import geopandas as gpd

# Reuse the allocation script's strata logic so bin cuts match the real packs.
REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
sys.path.insert(0, REPO_PATH)
from seep_classification_allocation import assign_strata  # noqa: E402

PRED_DIR = os.path.join(
    REPO_PATH, "data", "results", "SWIN", "AE", "20260428-1537_SWINxAE.weights"
)
GT_PATH = os.path.join(PRED_DIR, "gt_seeps.gpkg")
OUT_DIR = os.path.join(PRED_DIR, "labeling")

CHIP = "39.tif"          # which chip's polygons to pack
LABELER_TAG = "chip39"   # value for the `labeler` column

# Exact column order of gt_seeps_label_labeler_*.gpkg (post group-fields add).
PACK_COLS = [
    "image", "seep_id", "class", "labeler", "is_calibration",
    "area_bin", "area_label", "sol_bin", "sol_label",
    "centroid_x_m", "centroid_y_m", "area_m2", "perim_m", "circularity",
    "solidity", "eccentricity", "mean_R", "mean_G", "mean_B",
    "seep_group_id", "is_pregrouped", "geometry",
]


def main():
    print(f"loading {os.path.basename(GT_PATH)} ...")
    gt = gpd.read_file(GT_PATH)
    print(f"  -> {len(gt)} GT polygons")

    # Strata over the full population so area/solidity cuts match the packs.
    gt = assign_strata(gt)

    pack = gt[gt["image"] == CHIP].copy()
    print(f"\n{CHIP}: {len(pack)} label polygons intersecting the chip")
    if pack.empty:
        raise SystemExit(f"no polygons with image == {CHIP!r} in gt_seeps.gpkg")

    # Reset class for labeling; add allocation + grouping fields.
    pack["class"] = ""
    pack["labeler"] = LABELER_TAG
    pack["is_calibration"] = False
    pack["seep_group_id"] = pack["seep_id"]   # singleton groups to start
    pack["is_pregrouped"] = 0
    # Categorical strata labels -> plain str, matching the existing packs.
    pack["area_label"] = pack["area_label"].astype(str)
    pack["sol_label"] = pack["sol_label"].astype(str)

    missing = [c for c in PACK_COLS if c not in pack.columns]
    if missing:
        raise SystemExit(f"gt_seeps is missing expected columns: {missing}")
    pack = pack[PACK_COLS]

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"gt_seeps_label_{LABELER_TAG}.gpkg")
    pack.to_file(out_path, driver="GPKG")
    print(f"\nwrote {os.path.basename(out_path)}  ({len(pack)} polygons)")
    print(f"  -> {out_path}")
    print(f"  seep_group_id seeded = seep_id (all singletons), is_pregrouped = 0, class blank")


if __name__ == "__main__":
    main()