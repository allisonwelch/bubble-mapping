# tools/flux/field_reference.py
"""What the 2014 field crews measured on Octopus, as a benchmark for a run.

Derived 2026-09-15 from `labeling/historical/2014_OctopusNorth.xlsx` and
`2014_OctopusSouth_HS.xlsx` -- the same lake the runner deploys over. Six
shore-to-shore transects, 639 seeps typed a/b/c by hand.

These are NOT targets to tune against. They are an EXTERNAL check: nothing in
the detector, grouper or classifier ever saw them, so agreement is evidence and
disagreement is a lead. Fitting a parameter to make a run match them converts
the only independent validation available into a circular one -- the same trap
tools/deploy/deploy.py warns about for --thr and --cap.

THE DENSITY DEPENDS ON AN ASSUMPTION. The workbooks record seeps per transect
and mark sections every 10 m, but nowhere record the transect WIDTH. Allison's
estimate is ~1 m, at most 2 m, which sets the density to within a factor of 2:

    50 m x 1 m  ->  2.13 seeps/m2      50 m x 2 m  ->  1.06 seeps/m2

Carry both. The 2026-09-14 whole-lake run came out at 2.18 seeps/m2, which
lands on the 1 m reading almost exactly and 2x above the 2 m one.

The size medians come from bubArea = pi/4 * L * W, the envelope the field crew
measured around each seep. Note the field's resolution floor: the smallest L or
W ever recorded is 5 cm, so the field simply cannot report a seep our detector
routinely resolves. Expect our size distribution to sit low, especially for A;
that is a known, measured difference in the unit, not necessarily an error.
"""
from __future__ import annotations

SOURCE = ("2014_OctopusNorth.xlsx + 2014_OctopusSouth_HS.xlsx, "
          "6 shore-to-shore transects, 639 typed seeps")

# Share of field seeps by class, Octopus North + South pooled.
CLASS_MIX_PCT = {"A": 64.5, "B": 29.0, "C": 6.6}
CLASS_COUNTS = {"A": 412, "B": 185, "C": 42}

# Median seep envelope area, m2 (pi/4 * L * W), Octopus only.
MEDIAN_AREA_M2 = {"A": 0.0314, "B": 0.0707, "C": 0.0942}

# Seeps per m2, by the two transect-width readings. See the caveat above.
DENSITY_PER_M2 = {"transect_1m": 2.13, "transect_2m": 1.06}

# Largest seep envelope in ANY of the 8 workbooks (all lakes, n=2785):
# 1.02 m2, whose L/W diagonal is 1.14 m. Octopus's own maximum is 1.10 m.
MAX_SEEP_SPAN_M = 1.14
MAX_SEEP_AREA_M2 = 1.02

# The screening anchors, over the 2429 whole (non-partial, non-hotspot) a/b/c
# seeps that carry both an L and a W. These measure the SAME quantities the
# crack screen computes off a connected component's minimum rotated rectangle,
# which MAX_SEEP_SPAN_M does not: 1.14 m is a DIAGONAL, and a rectangle's
# rotated-rectangle major axis is max(L, W), never the diagonal. Screening a
# major axis against a diagonal is the stricter of the two comparisons by
# accident rather than by argument.
#
#   major axis   max(L, W). The largest ever recorded is 1.30 m; 7 seeps exceed
#                1.14 m, 0 exceed 1.30 m.
#   aspect       max(L, W) / min(L, W). The 99th percentile is 3.0 and the
#                99.5th is 3.5; 12 of 2429 seeps (0.5%) reach 4.0.
#
# The pair is what carries the argument: NO field seep is both longer than
# 1.30 m and more elongated than 4:1. A connected component that is both has no
# counterpart in 2429 hand-measured seeps.
MAX_SEEP_MAJOR_AXIS_M = 1.30
MAX_SEEP_ASPECT = 4.0
N_SIZED_FIELD_SEEPS = 2429

# Smallest L or W ever recorded, i.e. the field's own resolution floor.
MIN_RECORDED_DIM_M = 0.05


def compare(class_counts: dict, surveyed_area_m2: float | None = None,
            median_area_m2: dict | None = None):
    """A small DataFrame lining a run's class mix and density up against 2014.

    `median_area_m2` is optional; pass the run's per-class median hull area to
    get the size rows too.
    """
    import pandas as pd

    n = sum(int(class_counts.get(c, 0)) for c in ("A", "B", "C"))
    rows = []
    for c in ("A", "B", "C"):
        k = int(class_counts.get(c, 0))
        rows.append({
            "quantity": f"class {c} share, %",
            "this_run": round(100 * k / n, 1) if n else float("nan"),
            "field_2014": CLASS_MIX_PCT[c],
        })
    if median_area_m2:
        for c in ("A", "B", "C"):
            v = median_area_m2.get(c)
            rows.append({
                "quantity": f"class {c} median area, m2",
                "this_run": round(v, 4) if v is not None else float("nan"),
                "field_2014": MEDIAN_AREA_M2[c],
            })
    if surveyed_area_m2:
        rows.append({"quantity": "seeps per m2",
                     "this_run": round(n / surveyed_area_m2, 3),
                     "field_2014": DENSITY_PER_M2["transect_1m"]})
        rows.append({"quantity": "seeps per m2 (2 m transect reading)",
                     "this_run": round(n / surveyed_area_m2, 3),
                     "field_2014": DENSITY_PER_M2["transect_2m"]})
    return pd.DataFrame(rows)
