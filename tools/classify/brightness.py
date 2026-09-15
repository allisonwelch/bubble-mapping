# tools/classify/brightness.py
"""Relative (local-percentile) brightness for the A/B/C classifier.

Absolute mean_R/G/B is NOT comparable across images.

Measure brightness relative to a 15 m neighborhood rather than the whole lake

The neighborhood is defined in METRES, not by the `image` column:

    cell_m=None   rank within the whole image. Correct for the labeled chips,
                  which are 15 m squares -- the image already IS the
                  neighborhood.
    cell_m=15.0   rank within a 15 m grid cell. Correct for a lake-scale image,
                  and 15 m is not arbitrary: it is the chip size the classifier
                  was trained at and the tile size the runner already
                  normalizes the detector over (see tools/deploy/tiles.py).


The reference population is BUBBLES, not seeps, and at deploy it is every
DETECTED bubble. Nothing here needs labels, so it is computable at serve time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REL_CHANNELS = ("mean_R", "mean_G", "mean_B")
BRIGHTNESS_MODES = ("abs", "rel")

# The chip size the classifier was trained at, and the runner's tile size.
DEFAULT_CELL_M = 15.0

# A cell holding fewer reference bubbles than this gives a rank quantised into
# coarse steps (with n=4 the only possible values are 0, .25, .5, .75), which is
# noise, not signal. Such cells fall back to the whole image. At deploy a full
# 15 m cell holds ~660 bubbles, so only partial edge cells ever hit this.
MIN_REF = 30


def _cell_keys(df: pd.DataFrame, cell_m: float | None):
    """Neighbourhood key per row: the image, optionally subdivided into cells."""
    if cell_m is None:
        return list(zip(df["image"].to_numpy()))
    ix = np.floor(df["centroid_x_m"].to_numpy(dtype=float) / cell_m).astype(int)
    iy = np.floor(df["centroid_y_m"].to_numpy(dtype=float) / cell_m).astype(int)
    return list(zip(df["image"].to_numpy(), ix, iy))


def build_reference(reference: pd.DataFrame, cell_m: float | None = None,
                    channels=REL_CHANNELS) -> dict:
    """Sorted brightness arrays per neighbourhood, for ranking against.

    Returns {"cells": {key: {channel: sorted array}}, "images": {image: ...},
             "cell_m": cell_m} -- the per-image arrays are the MIN_REF fallback.
    """
    keys = _cell_keys(reference, cell_m)
    ref = pd.DataFrame({c: reference[c].to_numpy(dtype=float) for c in channels})
    ref["_key"] = keys
    ref["_img"] = reference["image"].to_numpy()

    cells, images = {}, {}
    for key, sub in ref.groupby("_key", sort=False):
        cells[key] = {c: np.sort(sub[c].to_numpy()) for c in channels}
    for img, sub in ref.groupby("_img", sort=False):
        images[img] = {c: np.sort(sub[c].to_numpy()) for c in channels}
    return {"cells": cells, "images": images, "cell_m": cell_m,
            "channels": tuple(channels)}


def apply_relative(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Replace the absolute brightness columns with their local percentile rank.

    Returns a copy. Rows whose neighbourhood holds no reference bubbles at all
    get 0.5 (the neutral rank) rather than NaN, because a NaN here would be
    dropped by `dropna(subset=FEATURES)` and silently shrink the training set.
    """
    channels = ref["channels"]
    keys = _cell_keys(df, ref["cell_m"])
    imgs = df["image"].to_numpy()
    out = df.copy()

    for c in channels:
        vals = np.full(len(df), 0.5, dtype=float)
        col = df[c].to_numpy(dtype=float)
        for i, (key, img) in enumerate(zip(keys, imgs)):
            arr = ref["cells"].get(key)
            if arr is None or len(arr[c]) < MIN_REF:
                fallback = ref["images"].get(img)
                arr = fallback if fallback is not None else arr
            if arr is None or len(arr[c]) == 0:
                continue
            a = arr[c]
            vals[i] = np.searchsorted(a, col[i]) / len(a)
        out[c] = vals
    return out


def relativize(df: pd.DataFrame, reference: pd.DataFrame, mode: str = "rel",
               cell_m: float | None = None) -> pd.DataFrame:
    """One-shot: build the reference population and apply it.

    `mode="abs"` returns `df` untouched, so callers can thread the mode through
    without branching. Use this at BOTH fit and deploy time -- a model fit on
    ranks and applied to raw 0-255 values produces plausible nonsense, which is
    why the mode is recorded in artifacts.json and checked on load.
    """
    if mode not in BRIGHTNESS_MODES:
        raise ValueError(f"brightness mode must be one of {BRIGHTNESS_MODES}, "
                         f"got {mode!r}")
    if mode == "abs":
        return df
    return apply_relative(df, build_reference(reference, cell_m=cell_m))
