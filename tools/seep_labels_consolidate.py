"""Phase 5 (2026-05-28 plan): consolidate returned labeler packs into seeps.

Reads the three returned labeler GPKGs (each carrying the grouping columns added
by tools/seep_add_group_fields.py, now filled in by labelers), resolves a single
consensus grouping + class per physical seep, and writes two artifacts:

  * gt_seeps_grouped.gpkg  -- PRE-dissolve. One row per original polygon, tagged
    with the consensus (image, seep_group_id) and class. This is what Phase 6
    (tools/seep_fit_clustering_params.py) needs: it must see the *constituent*
    polygons of each seep to measure anchor area, intra-group distance, etc.

  * gt_seeps_labeled.gpkg  -- DISSOLVED. One row per (image, seep_group_id) seep
    with recomputed per-seep features. This is what Phase 7
    (tools/seep_fit_class_tree.py) and the rewritten seep_level_eval.py consume.

Consensus rules
---------------
Calibration polygons (is_calibration == True) are labeled by every labeler:
  * grouping: two calibration polygons are in the same consensus seep iff a
    MAJORITY of labelers put them in the same group (compared as a partition,
    not by integer value -- labelers pick arbitrary group integers). Consensus
    seeps are the connected components of the resulting majority co-membership
    graph, computed per image.
  * class: majority vote across labelers' per-polygon class.
  * is_pregrouped: majority vote.
Unique polygons (is_calibration == False) come from a single labeler; their
grouping / class / is_pregrouped are taken as-is.

Per-seep features on the dissolved geometry are derived from the per-polygon
features already stored in the packs -- NO chip raster is read:
  * area_m2 / perim_m: from the unioned shapely geometry (CRS is meters).
  * circularity = 4*pi*area / perim^2; solidity = area / convex-hull area.
  * mean_R/G/B: area-weighted mean of the constituent polygons' means. Because
    bubble polygons within a seep are disjoint, the area-weighted mean equals
    the pixel mean over the union exactly (uniform pixel size).

Inter-rater agreement (Fleiss' kappa) is reported on the calibration subset for
both grouping (pairwise same/different) and classification, per Phase 4.
"""

import os
import sys
import argparse
import itertools

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
LABELING_DIR = os.path.join(
    REPO_PATH, "data", "results", "SWIN", "AE",
    "20260428-1537_SWINxAE.weights", "labeling",
)
DEFAULT_PACKS = [
    os.path.join(LABELING_DIR, f"gt_seeps_label_labeler_{i}.gpkg") for i in (1, 2, 3)
]
PRED_DIR = os.path.dirname(LABELING_DIR)
GROUPED_OUT = os.path.join(PRED_DIR, "gt_seeps_grouped.gpkg")
LABELED_OUT = os.path.join(PRED_DIR, "gt_seeps_labeled.gpkg")

KEY = ["image", "seep_id"]   # globally-unique identifier of an original polygon


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------
def load_packs(paths):
    frames = []
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"pack not found: {p}")
        g = gpd.read_file(p)
        for col in ("seep_group_id", "is_pregrouped", "class", "is_calibration"):
            if col not in g.columns:
                raise SystemExit(f"{os.path.basename(p)} missing column '{col}' "
                                 f"(run tools/seep_add_group_fields.py first?)")
        if "labeler" not in g.columns:
            g["labeler"] = os.path.splitext(os.path.basename(p))[0]
        frames.append(g)
    gdf = pd.concat(frames, ignore_index=True)
    gdf["class"] = gdf["class"].fillna("").astype(str).str.strip()
    gdf["is_pregrouped"] = gdf["is_pregrouped"].fillna(0).astype(int)
    print(f"loaded {len(paths)} packs -> {len(gdf)} rows, "
          f"{gdf['labeler'].nunique()} labelers")
    return gdf


def validate_group_class_consistency(gdf, hard=False):
    """Every (labeler, image, seep_group_id) must share one non-empty class."""
    problems = []
    grp = gdf.groupby(["labeler", "image", "seep_group_id"])
    for key, g in grp:
        classes = set(g.loc[g["class"] != "", "class"])
        if len(classes) > 1:
            problems.append((key, f"multiple classes {classes}"))
        elif len(classes) == 0:
            problems.append((key, "no class assigned"))
    if problems:
        print(f"\n!! {len(problems)} group/class consistency problems:")
        for key, msg in problems[:25]:
            print(f"   labeler={key[0]} image={key[1]} group={key[2]}: {msg}")
        if len(problems) > 25:
            print(f"   ... and {len(problems) - 25} more")
        if hard:
            raise SystemExit("class-consistency validation failed (hard mode)")
    else:
        print("class-consistency check passed (one non-empty class per group)")
    return problems


# ---------------------------------------------------------------------------
# Fleiss' kappa (no statsmodels dependency)
# ---------------------------------------------------------------------------
def fleiss_kappa(counts):
    """counts: (n_items, n_categories) integer array of rater-vote counts.

    Each row should sum to the same number of raters. Returns Fleiss' kappa.
    """
    counts = np.asarray(counts, dtype=float)
    n_items, n_cat = counts.shape
    n_raters = counts.sum(axis=1)
    if not np.allclose(n_raters, n_raters[0]):
        # Unequal rater counts per item: use the per-item mean for P_i.
        pass
    N = n_raters.mean()
    if N <= 1:
        return float("nan")
    p_j = counts.sum(axis=0) / counts.sum()
    P_i = (((counts ** 2).sum(axis=1) - n_raters) /
           (n_raters * (n_raters - 1)))
    P_bar = P_i.mean()
    P_e = (p_j ** 2).sum()
    if np.isclose(P_e, 1.0):
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def grouping_kappa(cal):
    """Fleiss' kappa on the pairwise 'same group?' relation, per image then pooled.

    For each image, every unordered pair of calibration polygons is an item with
    a binary category (together / apart); each labeler votes. Pools items across
    images into one Fleiss table.
    """
    rows = []
    labelers = sorted(cal["labeler"].unique())
    for image, g in cal.groupby("image"):
        sids = sorted(g["seep_id"].unique())
        if len(sids) < 2:
            continue
        # per-labeler map seep_id -> group
        lab_map = {lab: dict(zip(sub["seep_id"], sub["seep_group_id"]))
                   for lab, sub in g.groupby("labeler")}
        for i, j in itertools.combinations(sids, 2):
            together = 0
            apart = 0
            for lab in labelers:
                m = lab_map.get(lab, {})
                if i in m and j in m:
                    if m[i] == m[j]:
                        together += 1
                    else:
                        apart += 1
            if together + apart >= 2:   # need >=2 raters for the pair to count
                rows.append([together, apart])
    if not rows:
        return float("nan"), 0
    return fleiss_kappa(np.array(rows)), len(rows)


def classification_kappa(cal, classes=("A", "B", "C")):
    """Fleiss' kappa on per-polygon class across labelers (calibration subset)."""
    cat_index = {c: k for k, c in enumerate(classes)}
    rows = []
    for (image, sid), g in cal.groupby(KEY):
        votes = [c for c in g["class"] if c in cat_index]
        if len(votes) < 2:
            continue
        row = [0] * len(classes)
        for c in votes:
            row[cat_index[c]] += 1
        rows.append(row)
    if not rows:
        return float("nan"), 0
    return fleiss_kappa(np.array(rows)), len(rows)


# ---------------------------------------------------------------------------
# Consensus resolution
# ---------------------------------------------------------------------------
def _majority(values):
    """Mode of a list; ties broken by sorted order. None if empty."""
    vals = [v for v in values if v not in (None, "")]
    if not vals:
        return None
    s = pd.Series(vals)
    counts = s.value_counts()
    top = counts[counts == counts.max()].index.tolist()
    return sorted(top)[0]


def consensus_calibration_groups(cal):
    """Return dict (image, seep_id) -> local consensus group int (per image)."""
    n_labelers = cal["labeler"].nunique()
    majority = n_labelers // 2 + 1
    assignment = {}
    for image, g in cal.groupby("image"):
        sids = sorted(g["seep_id"].unique())
        idx = {s: k for k, s in enumerate(sids)}
        lab_map = {lab: dict(zip(sub["seep_id"], sub["seep_group_id"]))
                   for lab, sub in g.groupby("labeler")}
        n = len(sids)
        rr, cc = [], []
        for i, j in itertools.combinations(sids, 2):
            votes = sum(1 for lab in lab_map
                        if i in lab_map[lab] and j in lab_map[lab]
                        and lab_map[lab][i] == lab_map[lab][j])
            if votes >= majority:
                rr += [idx[i], idx[j]]
                cc += [idx[j], idx[i]]
        if rr:
            graph = csr_matrix((np.ones(len(rr), np.int8), (rr, cc)), shape=(n, n))
            _, comp = connected_components(graph, directed=False)
        else:
            comp = np.arange(n)
        for s in sids:
            assignment[(image, s)] = int(comp[idx[s]])
    return assignment


def build_consolidated(gdf):
    """Return a pre-dissolve GeoDataFrame: one row per (image, seep_id) with a
    final per-image-unique `seep_group_id`, consensus `class`, `is_pregrouped`.
    """
    is_cal = gdf["is_calibration"].astype(bool)
    cal = gdf[is_cal].copy()
    uniq = gdf[~is_cal].copy()

    # --- calibration: consensus group + class + pregrouped, one row per polygon
    cal_groups = consensus_calibration_groups(cal) if len(cal) else {}
    cal_rows = []
    for (image, sid), g in cal.groupby(KEY):
        geom = g.geometry.iloc[0]
        cal_rows.append({
            "image": image, "seep_id": int(sid),
            "src_group": ("cal", cal_groups[(image, sid)]),
            "class": _majority(list(g["class"])) or "",
            "is_pregrouped": int(round(g["is_pregrouped"].mean())),
            "area_m2": float(g["area_m2"].iloc[0]),
            "perim_m": float(g["perim_m"].iloc[0]),
            "centroid_x_m": float(g["centroid_x_m"].iloc[0]),
            "centroid_y_m": float(g["centroid_y_m"].iloc[0]),
            "mean_R": float(g["mean_R"].iloc[0]),
            "mean_G": float(g["mean_G"].iloc[0]),
            "mean_B": float(g["mean_B"].iloc[0]),
            "geometry": geom,
        })

    # --- unique: take labeler's own assignment (one row per polygon)
    uniq_rows = []
    for (image, sid), g in uniq.groupby(KEY):
        r = g.iloc[0]
        uniq_rows.append({
            "image": image, "seep_id": int(sid),
            "src_group": ("uniq", r["labeler"], int(r["seep_group_id"])),
            "class": r["class"],
            "is_pregrouped": int(r["is_pregrouped"]),
            "area_m2": float(r["area_m2"]), "perim_m": float(r["perim_m"]),
            "centroid_x_m": float(r["centroid_x_m"]),
            "centroid_y_m": float(r["centroid_y_m"]),
            "mean_R": float(r["mean_R"]), "mean_G": float(r["mean_G"]),
            "mean_B": float(r["mean_B"]),
            "geometry": r.geometry,
        })

    cons = pd.DataFrame(cal_rows + uniq_rows)
    # Renumber src_group -> a per-image-unique integer seep_group_id so that
    # calibration-consensus groups and unique groups never collide in a chip.
    final_ids = []
    for image, g in cons.groupby("image", sort=False):
        mapping = {}
        nxt = itertools.count(1)
        for sg in g["src_group"]:
            if sg not in mapping:
                mapping[sg] = next(nxt)
        final_ids.append(g["src_group"].map(mapping).rename("seep_group_id"))
    cons["seep_group_id"] = pd.concat(final_ids).astype(int)
    cons = cons.drop(columns=["src_group"])

    gdf_out = gpd.GeoDataFrame(cons, geometry="geometry", crs=gdf.crs)
    n_groups = gdf_out.groupby(KEY[:1] + ["seep_group_id"]).ngroups
    print(f"consolidated: {len(gdf_out)} polygons -> "
          f"{gdf_out.drop_duplicates(['image','seep_group_id']).shape[0]} consensus seeps")
    return gdf_out


# ---------------------------------------------------------------------------
# Dissolve + per-seep features (no raster needed)
# ---------------------------------------------------------------------------
def dissolve_to_seeps(grouped):
    rows = []
    geoms = []
    for (image, gid), g in grouped.groupby(["image", "seep_group_id"], sort=True):
        union = unary_union(list(g.geometry.values))
        area = float(union.area)
        perim = max(float(union.length), 1e-9)
        hull = float(union.convex_hull.area)
        w = g["area_m2"].to_numpy(dtype=float)
        wsum = w.sum() if w.sum() > 0 else 1.0
        rows.append({
            "image": image,
            "seep_group_id": int(gid),
            "seep_id": int(gid),   # eval expects a `seep_id`; group id is the seep
            "class": _majority(list(g["class"])) or "",
            # A seep is pregrouped if ANY constituent polygon is an envelope:
            # one envelope contaminates the dissolved area/perim/solidity with
            # interstitial ice, so the seep's features are not a clean
            # bubble-union measurement and must be kept out of Phase-7 training.
            # (NOT count-majority: a single area-dominant envelope is usually
            # outnumbered by its satellites, which majority would wrongly admit.)
            "is_pregrouped": int((g["is_pregrouped"].to_numpy() == 1).any()),
            "n_polygons_in_group": int(len(g)),
            "area_m2": area,
            "perim_m": perim,
            # Footprint (convex-hull) area: the only size measure defined
            # consistently across pregrouped envelopes, labeler-grouped unions,
            # and pred clusters -- so it (unlike bubble-union area_m2) transfers
            # and lets pregrouped seeps stay in the Phase-7 training set.
            "hull_area_m2": hull,
            "circularity": 4 * np.pi * area / (perim ** 2),
            "solidity": (area / hull) if hull > 0 else 0.0,
            "mean_R": float((g["mean_R"].to_numpy() * w).sum() / wsum),
            "mean_G": float((g["mean_G"].to_numpy() * w).sum() / wsum),
            "mean_B": float((g["mean_B"].to_numpy() * w).sum() / wsum),
        })
        geoms.append(union)
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=geoms, crs=grouped.crs)
    multi = (out["n_polygons_in_group"] >= 2).sum()
    print(f"dissolved -> {len(out)} seeps ({multi} multi-polygon, "
          f"{len(out) - multi} singletons)")
    return out


def _write_gpkg(gdf, path, layer):
    if os.path.exists(path):
        os.remove(path)
    gdf.to_file(path, driver="GPKG", layer=layer)
    print(f"  wrote {os.path.basename(path)} ({len(gdf)} rows)")


def main():
    ap = argparse.ArgumentParser(description="Consolidate labeler packs into seeps.")
    ap.add_argument("--packs", nargs="*", default=DEFAULT_PACKS)
    ap.add_argument("--grouped-out", default=GROUPED_OUT)
    ap.add_argument("--labeled-out", default=LABELED_OUT)
    ap.add_argument("--classes", nargs="*", default=["A", "B", "C"])
    ap.add_argument("--hard-validate", action="store_true",
                    help="abort on any group/class inconsistency")
    args = ap.parse_args()

    gdf = load_packs(args.packs)
    validate_group_class_consistency(gdf, hard=args.hard_validate)

    cal = gdf[gdf["is_calibration"].astype(bool)].copy()
    if len(cal):
        gk, npairs = grouping_kappa(cal)
        ck, npoly = classification_kappa(cal, classes=tuple(args.classes))
        print(f"\ninter-rater agreement (calibration subset):")
        print(f"  grouping  Fleiss kappa = {gk:.3f}  (n={npairs} polygon pairs)")
        print(f"  class     Fleiss kappa = {ck:.3f}  (n={npoly} polygons)")
        print(f"  guidance: >=0.6 publishable, <0.4 rubric failed")
    else:
        print("\n(no calibration rows found -- skipping kappa)")

    grouped = build_consolidated(gdf)
    seeps = dissolve_to_seeps(grouped)

    print("\nwriting outputs:")
    _write_gpkg(grouped, args.grouped_out, "gt_seeps_grouped")
    _write_gpkg(seeps, args.labeled_out, "gt_seeps_labeled")


if __name__ == "__main__":
    main()