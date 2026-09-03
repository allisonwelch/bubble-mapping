"""Exploratory (area x solidity) sampling strata for the labeler packs.

Extracted 2026-09-03 from the retired `seep_classification_allocation.py`
(now tools/archive/). That script's job -- sampling the original 3x300-seep
labeler packs -- is done and will not be repeated; `assign_strata` was the one
piece still imported by the live pack builders, so it lives here on its own.

The bins are per-BUBBLE (one row per drawn polygon), matching the pack schema.

IMPORTANT: the PROV_* radii below are EXPLORATORY SAMPLING STRATA, not
classification thresholds. Their only job was to give the downstream
classifier a representative spread of sizes to fit on. Do not cite them as
class boundaries -- the real per-class size axis comes from the historical
field workbooks (tools/historical_size_priors.py) and the fitted classifier.
"""
from math import pi

import numpy as np
import pandas as pd

# Bins numbered 1..4 smallest to largest; each PROV_R_BIN{n}_MAX is the upper
# edge radius of bin n (bin 4 is unbounded). Numeric labels chosen deliberately
# so they cannot be confused with the A/B/C flux-class taxonomy.
PROV_R_BIN1_MAX = 0.10   # m -- bin 1 (smallest): r < 10 cm
PROV_R_BIN2_MAX = 0.125  # m -- bin 2:            10 <= r < 12.5 cm
PROV_R_BIN3_MAX = 0.25   # m -- bin 3:            12.5 <= r < 25 cm
                         #      bin 4 (largest):  r >= 25 cm
PROV_A_BIN1_MAX = pi * PROV_R_BIN1_MAX ** 2   # 0.0314 m^2
PROV_A_BIN2_MAX = pi * PROV_R_BIN2_MAX ** 2   # 0.0491 m^2
PROV_A_BIN3_MAX = pi * PROV_R_BIN3_MAX ** 2   # 0.1963 m^2


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
