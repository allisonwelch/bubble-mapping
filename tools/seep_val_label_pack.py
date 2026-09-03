"""Build a labeling pack for every GT label polygon on the EVALUATION chips.

Motivation (2026-07-29): with the real per-seep flux rates in hand
(A=16, B=131, C=971 mg CH4/day), the C class turns out to be ~4% of seeps but
~52% of total flux, and the classifier's weak cross-chip C recall is the
dominant error term. The fix is more C examples -- so this pack exists to be
C-HUNTED, not labeled uniformly. Open it, find the big bright seeps, group and
class those first.

WHICH CHIPS: the 9 chips in `labeling/chips` (21 25 27 33 38 39 4 41 52) -- the
set the SWIN model was EVALUATED on. Take the chip list from that directory, not
from `aa_frames_list.json`: the split json's `validation_frames` are frames the
detector trained alongside for model selection and are NOT the evaluation chips.

These chips are already partly labeled -- the quarter packs cover only sub-quarters
of some of them (4-SE, 21-NE, 38-SW, ...), so most polygons here are still blank.
The pack is built fresh from source, so its per-chip `seep_id` numbering is its
own; it does NOT carry over the quarter packs' existing class/grouping. Treat this
as a separate labeling surface, and reconcile on geometry (not on seep_id) if you
later need to merge it with the quarter packs.

Chips in `labeling/chips` are RGB+alpha (band 4 is a constant 255 alpha, not a GT
mask); `build_gt_seeps_from_source` reads all-but-the-last band, so it correctly
picks up RGB.

Output schema is byte-compatible with `gt_seeps_label_quarters_*_grouped.gpkg` so
the same QGIS workflow, style, and downstream tools apply unchanged. `class` is
blank and `seep_group_id` defaults to `seep_id`; run
`tools/seep_grouper_deploy.py` on the result to pre-group it (that step also adds
`seep_group_id_pred` / `group_source` and injects the hull style).

Polygons come from the ORIGINAL drawn shapes via `build_gt_seeps_from_source`,
NOT from re-polygonized label rasters -- the raster round-trip merges touching
polygons (2026-05-14).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import geopandas as gpd  # noqa: E402
from seep_feature_table import build_gt_seeps_from_source  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Exact column order of gt_seeps_label_quarters_*_grouped.gpkg.
PACK_COLS = [
    "image", "seep_id", "class", "seep_group_id", "is_pregrouped",
    "is_overgrouped", "labeler", "is_calibration",
    "area_bin", "area_label", "sol_bin", "sol_label",
    "centroid_x_m", "centroid_y_m", "area_m2", "perim_m", "circularity",
    "solidity", "eccentricity", "mean_R", "mean_G", "mean_B",
    "unit", "is_context", "notes", "geometry",
]

# Area strata cuts -- PROVISIONAL sampling constructs, NOT class thresholds
# (2026-05-13). Radii 10 / 12.5 / 25 cm. Kept only so this pack's bin columns
# line up with the existing packs.
PROV_BINS = [np.pi * 0.10 ** 2, np.pi * 0.125 ** 2, np.pi * 0.25 ** 2]


def assign_strata(gt: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """area_bin/area_label + solidity-quartile sol_bin/sol_label, matching packs."""
    a = gt["area_m2"].to_numpy()
    gt["area_bin"] = np.digitize(a, PROV_BINS)
    gt["area_label"] = pd.Categorical.from_codes(
        gt["area_bin"],
        categories=[f"1 (<{PROV_BINS[0]:.3f} m^2)", "2", "3",
                    f"4 (>{PROV_BINS[2]:.3f} m^2)"]).astype(str)
    s = gt["solidity"].to_numpy()
    q25, q75 = np.percentile(s, [25, 75])
    gt["sol_bin"] = np.digitize(s, [q25, q75])
    gt["sol_label"] = pd.Categorical.from_codes(
        gt["sol_bin"],
        categories=[f"low (<{q25:.3f})", "mid", f"high (>={q75:.3f})"]).astype(str)
    print(f"  solidity quartile cuts: [{q25:.3f}, {q75:.3f}]")
    return gt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chip-dir", default=os.path.join(
        REPO, "data", "results", "SWIN", "AE", "20260428-1537_SWINxAE.weights",
        "labeling", "chips"),
        help="dir of chip .tif files (default = the 9 evaluation chips)")
    ap.add_argument("--labels", default=os.path.join(
        REPO, "data", "training", "AE", "2026-04-16_UNETxAE",
        "labels_full_dataset_no_toolik_AE.gpkg"))
    # "dir" = take every .tif in --chip-dir. That is the right mode for
    # labeling/chips, which holds exactly the 9 EVALUATION chips (21 25 27 33
    # 38 39 4 41 52) already exported for QGIS. Do NOT use the split-json
    # `validation_frames` for this: those frames are model-selection data the
    # detector trained alongside, not the held-out chips the model was
    # evaluated on.
    ap.add_argument("--split", default="dir",
                    choices=["dir", "validation_frames", "testing_frames",
                             "training_frames"])
    ap.add_argument("--labeler", default="eval_chunt",
                    help="value for the `labeler` column")
    ap.add_argument("-o", "--out", default=os.path.join(
        REPO, "data", "results", "SWIN", "AE", "20260428-1537_SWINxAE.weights",
        "labeling", "gt_seeps_label_eval_chips.gpkg"))
    args = ap.parse_args()

    if args.split == "dir":
        ids = sorted(int(os.path.splitext(os.path.basename(p))[0])
                     for p in glob.glob(os.path.join(args.chip_dir, "*.tif")))
        print(f"chips in {args.chip_dir}: {len(ids)} -> {ids}")
    else:
        frames = json.load(open(os.path.join(args.chip_dir,
                                             "aa_frames_list.json")))
        ids = sorted(int(x) for x in frames[args.split])
        print(f"{args.split}: {len(ids)} chips -> {ids}")

    print(f"\nloading {os.path.basename(args.labels)} ...")
    src = gpd.read_file(args.labels)
    print(f"  {len(src)} source polygons, CRS {src.crs.to_epsg()}")

    parts = []
    for i in ids:
        chip_fp = os.path.join(args.chip_dir, f"{i}.tif")
        if not os.path.exists(chip_fp):
            print(f"  [skip] {i}.tif not found")
            continue
        gdf = build_gt_seeps_from_source(chip_fp, src)
        if gdf is None or gdf.empty:
            print(f"  {i}.tif: 0 polygons")
            continue
        gdf["image"] = f"{i}.tif"
        print(f"  {i}.tif: {len(gdf)} polygons")
        parts.append(gdf)

    if not parts:
        raise SystemExit("[fail] no polygons found on any validation chip")

    gt = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=src.crs)

    # seep_id is PER-CHIP (2026-05-29): renumber 1..N within each image so the
    # anchor-id grouping rule stays collision-free inside a chip. Groups must
    # always be keyed on the (image, seep_group_id) TUPLE downstream.
    gt["seep_id"] = gt.groupby("image").cumcount() + 1

    gt = assign_strata(gt)
    gt["class"] = ""
    gt["seep_group_id"] = gt["seep_id"]          # every polygon its own singleton
    gt["is_pregrouped"] = 0                      # human judgment; set in QGIS
    gt["is_overgrouped"] = 0
    gt["labeler"] = args.labeler
    gt["is_calibration"] = False
    gt["unit"] = gt["image"].str.replace(".tif", "-ALL", regex=False)
    gt["is_context"] = 0                         # whole chips, nothing truncated
    gt["notes"] = None

    gt = gpd.GeoDataFrame(gt[PACK_COLS], geometry="geometry", crs=src.crs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    gt.to_file(args.out, layer="labels", driver="GPKG")

    print(f"\n[out] {args.out}")
    print(f"  {len(gt)} polygons over {gt['image'].nunique()} chips")
    print("  per chip:", gt["image"].value_counts().sort_index().to_dict())
    big = gt.nlargest(10, "area_m2")[["image", "seep_id", "area_m2"]]
    print("\n  10 largest polygons (start C-hunting here):")
    print("   " + big.to_string(index=False).replace("\n", "\n   "))
    print("\nnext: python tools/seep_grouper_deploy.py "
          f'"{args.out}" -o "{args.out.replace(".gpkg", "_grouped.gpkg")}"')


if __name__ == "__main__":
    main()