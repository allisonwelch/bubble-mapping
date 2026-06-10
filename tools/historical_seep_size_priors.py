"""Derive seep size priors + class size axis from the historical field dataset.

Reads the many per-lake .xlsx files in labeling/historical/ (one workbook per
lake-year, one sheet per transect) and distills them into the handful of
numbers the grouping + classification pipeline needs, written as a versioned
JSON artifact (instead of a one-off print).

The four target fields live in columns V-Y of every sheet:
    Type_flux, length_cm, width_cm, bubAreaCm2
Type_flux is a FLUX class (a/b/c), optionally suffixed:
    *p  -> partial seep (crossed the transect boundary; dimensions truncated)
    hs* -> hotspot (hso / hsop); handled separately from a/b/c
bubAreaCm2 is NOT a measured bubble area -- it is exactly the ellipse area
pi/4 * length * width of the L x W envelope, so size == the (L, W) footprint.

What this emits (labeling/historical_seep_size_priors.json):
  * grouping geometry: R_max (candidate radius) + major-axis agglomeration cap,
    from the upper tail of real seep dimensions -- replaces the PROV_* guesses.
  * per-class (a/b/c) size distributions (major/minor axis, ellipse area).
  * provisional 1-D area cut points (A|B, B|C) WITH the achieved accuracy, so
    the size/class overlap is quantified honestly rather than hidden -- size is
    a noisy proxy for a flux class and needs the imagery brightness axis too.

NOTE: full-lake deploy never truncates seeps, so partials are excluded from the
priors. Hotspots are tallied separately (their own upstream criterion).
"""

import os
import glob
import json
import warnings

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")

REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
PRED_DIR = os.path.join(REPO_PATH, "data", "results", "SWIN", "AE",
                        "20260428-1537_SWINxAE.weights")
HIST_DIR = os.path.join(PRED_DIR, "labeling", "historical")
OUT_JSON = os.path.join(PRED_DIR, "labeling", "historical_seep_size_priors.json")

TARGET = ["Type_flux", "length_cm", "width_cm", "bubAreaCm2"]
PCTS = [5, 25, 50, 75, 95, 99]


def load_all(hist_dir):
    """Concatenate the 4 target columns from every sheet that has them."""
    files = sorted(glob.glob(os.path.join(hist_dir, "*.xlsx")))
    if not files:
        raise SystemExit(f"no .xlsx under {hist_dir}")
    frames, used = [], []
    for f in files:
        xl = pd.ExcelFile(f)
        for sh in xl.sheet_names:
            df = xl.parse(sh, header=0)
            cols = {str(c).strip(): c for c in df.columns}
            if not all(t in cols for t in TARGET):
                continue          # non-data sheet (maps, notes, gps, ...)
            sub = df[[cols[t] for t in TARGET]].copy()
            sub.columns = TARGET
            sub["file"] = os.path.basename(f)
            frames.append(sub)
            used.append((os.path.basename(f), sh))
    allrows = pd.concat(frames, ignore_index=True)
    return allrows, files, used


def parse_class(raw):
    """raw lowercased code -> (base_class, is_partial, is_hotspot)."""
    s = str(raw).strip().lower()
    if not s or s == "nan":
        return None, False, False
    hot = s.startswith("h")
    partial = s.endswith("p")
    base = s[0]
    return base, partial, hot


def pcts(arr):
    arr = np.asarray(arr, float)
    arr = arr[np.isfinite(arr)]
    return {f"p{p}": round(float(np.percentile(arr, p)), 2) for p in PCTS}


def best_1d_split(x_lo, x_hi):
    """Optimal area threshold separating two classes (lo < hi), + balanced acc.

    Returns (threshold, balanced_accuracy). BALANCED accuracy = mean of the two
    per-class recalls, so a rare class (e.g. C, n=87 vs B n=867) can't be hidden
    by predicting the majority everywhere. Candidates = midpoints of sorted
    unique pooled values.
    """
    x_lo = np.asarray(x_lo, float)
    x_hi = np.asarray(x_hi, float)
    vals = np.unique(np.concatenate([x_lo, x_hi]))
    if len(vals) < 2 or len(x_lo) == 0 or len(x_hi) == 0:
        return float("nan"), float("nan")
    cands = (vals[:-1] + vals[1:]) / 2.0
    best_t, best_acc = float("nan"), -1.0
    for t in cands:
        rec_lo = float((x_lo < t).mean())     # lo correctly below
        rec_hi = float((x_hi >= t).mean())    # hi correctly at/above
        bacc = 0.5 * (rec_lo + rec_hi)
        if bacc > best_acc:
            best_acc, best_t = bacc, float(t)
    return round(best_t, 1), round(best_acc, 3)


def main():
    allrows, files, used = load_all(HIST_DIR)
    d = allrows.dropna(subset=["Type_flux"]).copy()
    parsed = d["Type_flux"].apply(parse_class)
    d["cls"] = [p[0] for p in parsed]
    d["partial"] = [p[1] for p in parsed]
    d["hotspot"] = [p[2] for p in parsed]
    # ellipse area from L x W (== bubAreaCm2; recompute so it never depends on it)
    d["area_cm2"] = np.pi / 4.0 * d["length_cm"] * d["width_cm"]
    d["major_cm"] = d[["length_cm", "width_cm"]].max(axis=1)
    d["minor_cm"] = d[["length_cm", "width_cm"]].min(axis=1)

    full = d[(~d["partial"]) & (~d["hotspot"]) & (d["cls"].isin(["a", "b", "c"]))]
    full = full.dropna(subset=["length_cm", "width_cm"]).copy()

    # per-class size distributions
    per_class = {}
    for c in ["a", "b", "c"]:
        g = full[full["cls"] == c]
        if not len(g):
            continue
        per_class[c] = {
            "n": int(len(g)),
            "major_axis_cm": pcts(g["major_cm"]),
            "minor_axis_cm": pcts(g["minor_cm"]),
            "ellipse_area_cm2": pcts(g["area_cm2"]),
        }

    # grouping geometry from the upper tail of real seep dimensions
    maj = full["major_cm"].to_numpy(float)
    grouping = {
        "candidate_radius_m": round(float(np.percentile(maj, 95)) / 100.0, 3),
        "agglomeration_major_cap_m": round(float(np.percentile(maj, 99)) / 100.0, 3),
        "major_axis_max_cm": round(float(maj.max()), 1),
        "rationale": ("candidate_radius = p95 of major axis; cap = p99; "
                      "a cluster exceeding the cap bridges >1 seep -> split."),
    }

    # provisional 1-D area cut points + honest accuracy (size is a noisy proxy)
    a_area = full.loc[full["cls"] == "a", "area_cm2"]
    b_area = full.loc[full["cls"] == "b", "area_cm2"]
    c_area = full.loc[full["cls"] == "c", "area_cm2"]
    ab_t, ab_acc = best_1d_split(a_area, b_area)
    bc_t, bc_acc = best_1d_split(b_area, c_area)
    class_size_axis = {
        "feature": "ellipse_area_cm2 (pi/4 * L * W)",
        "A_B_threshold_cm2": ab_t, "A_B_balanced_acc": ab_acc,
        "B_C_threshold_cm2": bc_t, "B_C_balanced_acc": bc_acc,
        "A_B_threshold_m2": round(ab_t / 1e4, 5),
        "B_C_threshold_m2": round(bc_t / 1e4, 5),
        "warning": ("size alone separates a/b/c only weakly (see accuracies); "
                    "Type_flux is a FLUX class. Use these as a size prior and "
                    "combine with the imagery brightness axis from the labeled "
                    "chips -- do NOT classify on size alone."),
    }

    hot = d[d["hotspot"]]
    out = {
        "source": {
            "dir": os.path.relpath(HIST_DIR, REPO_PATH).replace("\\", "/"),
            "n_files": len(files),
            "files": [os.path.basename(f) for f in files],
            "n_coded_rows": int(len(d)),
            "n_full_abc": int(len(full)),
            "n_partials_excluded": int(d["partial"].sum()),
            "n_hotspots": int(len(hot)),
            "note": ("bubAreaCm2 == pi/4*L*W (ellipse area of the envelope), "
                     "not a measured bubble area. Some Goldstream transects were "
                     "measured by multiple observers (replicate rows)."),
        },
        "grouping": grouping,
        "class_size_axis": class_size_axis,
        "per_class_size": per_class,
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- console summary ----
    print(f"files: {len(files)} | coded seeps: {len(d)} | full a/b/c: {len(full)} "
          f"| partials excluded: {int(d['partial'].sum())} | hotspots: {len(hot)}")
    print("\nper-class ellipse area cm2 (p25/p50/p75/p95):")
    for c in ["a", "b", "c"]:
        if c in per_class:
            a = per_class[c]["ellipse_area_cm2"]
            print(f"  {c}: n={per_class[c]['n']:5d}  "
                  f"{a['p25']:.0f} / {a['p50']:.0f} / {a['p75']:.0f} / {a['p95']:.0f}")
    print(f"\ngrouping: candidate_radius={grouping['candidate_radius_m']} m  "
          f"cap={grouping['agglomeration_major_cap_m']} m  "
          f"(max observed {grouping['major_axis_max_cm']/100:.2f} m)")
    print(f"class size cuts (area cm2): A|B={ab_t} (acc {ab_acc})  "
          f"B|C={bc_t} (acc {bc_acc})  <- size alone is a weak class signal")
    print(f"\nwrote {os.path.relpath(OUT_JSON, REPO_PATH)}")


if __name__ == "__main__":
    main()
