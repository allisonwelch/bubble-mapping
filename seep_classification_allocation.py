"""Sample GT seeps for the classification-threshold labeling exercise.

Produces N_LABELERS labeling packs of TARGET_PER_LABELER seeps each, structured as:

  * a shared CALIBRATION set (everyone labels these) -- used to measure
    inter-rater agreement (Cohen's / Fleiss' kappa) and surface taxonomy
    disagreements before they pollute the training set.
  * UNIQUE per-labeler sets (disjoint across labelers) -- maximize feature-space
    coverage so the downstream decision tree has a usable training distribution.

Stratification is on (area x solidity) bins computed from gt_seeps.gpkg.
IMPORTANT: the area-bin boundaries here are EXPLORATORY sampling strata.
Their only job is to give the decision tree
a representative spread of seep sizes to fit on.

Outputs:
  * gt_seeps_label_master.gpkg -- one row per (seep, labeler) assignment;
    includes `labeler` (str) and `is_calibration` (bool) columns
  * gt_seeps_label_labeler{i}.gpkg -- one per labeler, ready for QGIS
    attribute-table editing of the `class` column
"""

import os
import sys
from math import pi

import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

# Windows + PyCharm/IPython aggressively buffer stdout. Force line-buffering so
# progress prints surface in real time instead of after the whole script ends.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass  # python < 3.7 fallback; not expected on this project

REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
PRED_DIR  = f"{REPO_PATH}/data/results/SWIN/AE/20260428-1537_SWINxAE.weights"
GT_PATH   = f"{PRED_DIR}/gt_seeps.gpkg"
OUT_DIR   = PRED_DIR

# Labeler-pack budget.
N_LABELERS = 3
CALIBRATION_N = 75          # seeps every labeler labels (shared)
UNIQUE_PER_LABELER = 225    # seeps unique to each labeler (disjoint)
TARGET_PER_LABELER = CALIBRATION_N + UNIQUE_PER_LABELER   # = 300

# Exploratory area-strata bounds (NOT classification thresholds -- the labeling
# exercise will derive those). Chosen at round radii spanning the dataset.
# Bins are numbered 1..4 from smallest to largest; each PROV_R_BIN{n}_MAX is the
# upper-edge radius of bin n (bin 4 has no upper bound). Numeric labels chosen
# deliberately to avoid collision with Walter Anthony's A/B/C class taxonomy.
PROV_R_BIN1_MAX = 0.10   # m   -- bin 1 (smallest): r < 10 cm
PROV_R_BIN2_MAX = 0.125  # m   -- bin 2:            10 <= r < 12.5 cm
PROV_R_BIN3_MAX = 0.25   # m   -- bin 3:            12.5 <= r < 25 cm
                         #       bin 4 (largest):   r >= 25 cm
PROV_A_BIN1_MAX = pi * PROV_R_BIN1_MAX ** 2   # 0.0314 m^2
PROV_A_BIN2_MAX = pi * PROV_R_BIN2_MAX ** 2   # 0.0491 m^2
PROV_A_BIN3_MAX = pi * PROV_R_BIN3_MAX ** 2   # 0.1963 m^2

RANDOM_SEED = 42


def load_gt(path):
    print(f"loading {os.path.basename(path)} ...")
    gt = gpd.read_file(path)
    print(f"  -> {len(gt)} GT seeps loaded")
    required = {"area_m2", "solidity", "geometry"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"gt_seeps.gpkg missing columns: {missing}")
    return gt


def assign_strata(gt):
    """Adds `area_bin` (0..3) and `sol_bin` (0..2) columns in place."""
    print("\nassigning strata (area x solidity) ...")
    a = gt["area_m2"].to_numpy()
    area_bins = [0, PROV_A_BIN1_MAX, PROV_A_BIN2_MAX, PROV_A_BIN3_MAX, a.max() + 1e-9]
    gt["area_bin"] = np.digitize(a, area_bins[1:-1])  # 0..3 internally; labels are 1..4
    gt["area_label"] = pd.Categorical.from_codes(
        gt["area_bin"],
        categories=[
            f"1 (<{PROV_A_BIN1_MAX:.3f} m^2)",
            "2",
            "3",
            f"4 (>{PROV_A_BIN3_MAX:.3f} m^2)",
        ],
    )

    s = gt["solidity"].to_numpy()
    sol_q25, sol_q75 = np.percentile(s, [25, 75])
    sol_bins = [0, sol_q25, sol_q75, 1.01]
    gt["sol_bin"] = np.digitize(s, sol_bins[1:-1])  # 0..2
    gt["sol_label"] = pd.Categorical.from_codes(
        gt["sol_bin"],
        categories=[f"low (<{sol_q25:.3f})", "mid", f"high (>={sol_q75:.3f})"],
    )

    print(f"\nsolidity quartile cuts: [{sol_q25:.3f}, {sol_q75:.3f}]")
    print("\npopulation crosstab (area_bin x sol_bin):")
    print(gt.groupby(["area_bin", "sol_bin"]).size().unstack(fill_value=0))
    return gt


def _stratified_take(pool, n, area_bins, sol_bins, seed):
    """Sample `n` seeps from `pool` stratified equally across (area_bin x sol_bin),
    redistributing any per-cell shortfall to the densest non-empty cells.

    Returns the sampled GeoDataFrame (subset of `pool`).
    """
    rng = seed
    cells = [(ab, sb) for ab in area_bins for sb in sol_bins]
    cell_pops = {c: pool[(pool["area_bin"] == c[0]) & (pool["sol_bin"] == c[1])]
                 for c in cells}
    nonempty = [c for c in cells if len(cell_pops[c]) > 0]
    if not nonempty:
        return pool.iloc[0:0]
    per_cell = n // len(nonempty)

    picked = []
    for c in nonempty:
        take = min(per_cell, len(cell_pops[c]))
        if take > 0:
            picked.append(cell_pops[c].sample(n=take, random_state=rng))
            rng += 1

    out = pd.concat(picked) if picked else pool.iloc[0:0]
    deficit = n - len(out)
    if deficit > 0:
        # Fill remaining slots from the densest cells, skipping already-taken seeps.
        taken_ids = set(out["seep_id"]) if "seep_id" in out.columns else set(out.index)
        cell_order = sorted(nonempty, key=lambda c: -len(cell_pops[c]))
        for c in cell_order:
            if deficit == 0:
                break
            grp = cell_pops[c]
            extra = (grp[~grp["seep_id"].isin(taken_ids)]
                     if "seep_id" in grp.columns
                     else grp.drop(index=list(taken_ids), errors="ignore"))
            take = min(deficit, len(extra))
            if take == 0:
                continue
            chunk = extra.sample(n=take, random_state=rng)
            rng += 1
            picked.append(chunk)
            taken_ids.update(chunk["seep_id"] if "seep_id" in chunk.columns else chunk.index)
            deficit -= take
        out = pd.concat(picked)
    return out


def allocate_label_sets(gt, n_labelers, cal_n, unique_n, seed=RANDOM_SEED):
    """Return (calibration_gdf, [labeler_1_unique_gdf, labeler_2..., ...]).

    Calibration set takes ALL large-bin (area_bin==3) seeps first because they're
    rare and triple-labeling them is more valuable than splitting them across
    labelers. The remaining calibration slots are stratified-sampled from
    area_bin in {0,1,2}. Each labeler's unique set is then stratified-sampled
    from the residual (non-calibration, non-large) population, disjoint across
    labelers.
    """
    print("\nallocating calibration + per-labeler unique sets ...")
    # 1. Calibration: all large seeps + stratified fill from area_bin 0-2.
    large = gt[gt["area_bin"] == 3]
    print(f"  all {len(large)} bin-4 seeps -> calibration (rare, want triple labels)")
    cal_remaining = cal_n - len(large)
    if cal_remaining < 0:
        raise ValueError(
            f"More large seeps ({len(large)}) than calibration budget ({cal_n}). "
            "Increase CALIBRATION_N or change the large-bin boundary."
        )
    non_large = gt[gt["area_bin"] != 3]
    cal_fill = _stratified_take(
        non_large, cal_remaining,
        area_bins=[0, 1, 2], sol_bins=[0, 1, 2], seed=seed,
    )
    calibration = pd.concat([large, cal_fill])
    print(f"calibration set: {len(calibration)} seeps "
          f"({len(large)} large + {len(cal_fill)} stratified from non-large)")

    # 2. Per-labeler unique sets, disjoint from calibration and from each other.
    residual = non_large[~non_large["seep_id"].isin(cal_fill["seep_id"])] \
        if "seep_id" in non_large.columns \
        else non_large.drop(index=cal_fill.index, errors="ignore")

    unique_sets = []
    for i in tqdm(range(n_labelers), desc="  unique sets", unit="labeler"):
        sub = _stratified_take(
            residual, unique_n,
            area_bins=[0, 1, 2], sol_bins=[0, 1, 2], seed=seed + 100 * (i + 1),
        )
        unique_sets.append(sub)
        residual = (residual[~residual["seep_id"].isin(sub["seep_id"])]
                    if "seep_id" in residual.columns
                    else residual.drop(index=sub.index, errors="ignore"))
        tqdm.write(f"    labeler {i+1} unique set: {len(sub)} seeps")

    return calibration, unique_sets


def write_label_packs(calibration, unique_sets, out_dir):
    """Writes per-labeler GPKGs + a master GPKG with labeler/is_calibration tags."""
    front_cols = ["image", "seep_id", "cluster_id", "class",
                  "labeler", "is_calibration",
                  "area_bin", "area_label", "sol_bin", "sol_label"]

    def _prep(df, labeler_tag, is_cal):
        df = df.copy()
        df["class"] = ""
        df["labeler"] = labeler_tag
        df["is_calibration"] = is_cal
        front = [c for c in front_cols if c in df.columns]
        rest = [c for c in df.columns if c not in front and c != "geometry"]
        return df[front + rest + ["geometry"]]

    # Per-labeler GPKGs: calibration + unique, tagged with that labeler.
    # gpd.to_file is the slow step in this script -- progress bar helps.
    for i, uniq in tqdm(list(enumerate(unique_sets, start=1)),
                        desc="  writing labeler packs", unit="pack"):
        tag = f"labeler_{i}"
        pack = pd.concat([
            _prep(calibration, tag, True),
            _prep(uniq, tag, False),
        ])
        path = os.path.join(out_dir, f"gt_seeps_label_{tag}.gpkg")
        pack.to_file(path, driver="GPKG")
        tqdm.write(f"    -> {os.path.basename(path)}  ({len(pack)} seeps)")

    # Master GPKG: every (seep, labeler) row, for back-joining labels later.
    master_rows = []
    for i, uniq in enumerate(unique_sets, start=1):
        tag = f"labeler_{i}"
        master_rows.append(_prep(calibration, tag, True))
        master_rows.append(_prep(uniq, tag, False))
    master = pd.concat(master_rows)
    master_path = os.path.join(out_dir, "gt_seeps_label_master.gpkg")
    print(f"  writing master GPKG ({len(master)} rows) ...")
    master.to_file(master_path, driver="GPKG")
    print(f"    -> {os.path.basename(master_path)}  ({len(master)} rows = "
          f"{len(calibration)} cal x {len(unique_sets)} labelers + "
          f"{sum(len(u) for u in unique_sets)} unique)")


if __name__ == "__main__":
    gt = load_gt(GT_PATH)
    gt = assign_strata(gt)
    calibration, unique_sets = allocate_label_sets(
        gt, N_LABELERS, CALIBRATION_N, UNIQUE_PER_LABELER, seed=RANDOM_SEED,
    )

    print("\nwriting label packs:")
    write_label_packs(calibration, unique_sets, OUT_DIR)

    print("\nfinal stratum coverage per labeler pack (area x sol):")
    for i, uniq in enumerate(unique_sets, start=1):
        full = pd.concat([calibration, uniq])
        print(f"\n  labeler_{i} (n={len(full)}):")
        print(full.groupby(["area_bin", "sol_bin"]).size().unstack(fill_value=0))