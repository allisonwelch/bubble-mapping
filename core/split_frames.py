# core/split_frames.py
# Train/val/test splitting utilities
# - cross_validation_split: K-fold indices (saved/loaded from JSON)
# - split_dataset: stratified-by-positive-rate (with safe fallbacks), saved/loaded from JSON
#
# WHY TRAIN/VAL/TEST SPLITS MATTER:
#   - TRAINING SET: Used to teach the model by adjusting its internal parameters.
#     Analogy: the flashcards you study from.
#   - VALIDATION SET: Used during training to check if the model is generalizing
#     well (not just memorizing). Helps tune hyperparameters like learning rate.
#     Analogy: practice tests while studying.
#   - TEST SET: Held completely separate, used only at the very end to get an
#     honest estimate of how the model will perform on real, unseen data.
#     Analogy: the final exam you take after studying.
#
# WHY STRATIFICATION MATTERS:
#   If your frames have highly imbalanced labels (e.g., 90% background, 10% bubbles),
#   random splitting might accidentally put all the bubbles in train and none in test,
#   leading to a test set that doesn't represent the real distribution.
#   Stratification ensures each split has roughly the same proportion of positive pixels.

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import KFold, train_test_split

# Console style similar to training banner output
try:
    from core.common.console import _C, _col
except Exception:  # fallback if console helpers aren't available
    class _C:  # noqa: N801 (kept to mirror existing helper names)
        RED = ""
        YELLOW = ""
        GREEN = ""
        RESET = ""

    def _col(s, _color):
        return s


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist (no-op for empty)."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def summarize_positive_rates(
    frames: Sequence[Any],
    sets: Dict[str, Sequence[int]],
) -> Dict[str, Dict[str, float]]:
    """
    Return summary stats (in percent) of per-frame positive rates for each set.

    Why:
        Helpful sanity check to see whether train/val/test are similarly
        distributed with respect to foreground prevalence.
    """
    # Get the positive rate (fraction of bubble pixels) for each frame, converted to percent
    # This tells us: what % of pixels in each frame are labeled as bubbles vs background?
    pr = _pos_rate(frames) * 100.0
    out: Dict[str, Dict[str, float]] = {}
    for name, idx in sets.items():
        # Extract the positive rates for just the frames in this set (train/val/test)
        idx = np.asarray(idx, dtype=int)
        vals = pr[idx] if idx.size > 0 else np.asarray([], dtype=float)
        if vals.size == 0:
            # If the set is empty, fill with zeros
            out[name] = {
                "n": 0,
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
        else:
            # Compute distribution statistics to verify stratification worked.
            # KEY INSIGHT: if stratification is working correctly, the mean and std
            # of positive rates should be similar across train/val/test. If one set
            # has much higher mean than others, it's skewed (e.g., all hard frames
            # in test, all easy frames in train).
            out[name] = {
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    return out


# ---------- K-fold CV helper (not used right now) ----------
#
# WHAT IS K-FOLD CROSS-VALIDATION?
#   K-fold CV is an alternative to train/val/test splitting used when you have a small
#   dataset and want to make maximum use of your data.
#   How it works:
#     1. Divide your data into K equal chunks (e.g., 10 chunks if k=10).
#     2. In iteration 1: use chunk 1 as test, the other 9 as training.
#     3. In iteration 2: use chunk 2 as test, the other 9 as training.
#     4. Repeat K times, each time with a different chunk as test.
#   Why: You train and evaluate K separate models, so you get K estimates of model
#   performance. This gives a more stable estimate than a single train/test split,
#   especially on small datasets. Trade-off: K times more computation.
#   Note: This code is currently not used; the main pipeline uses split_dataset instead.


def cross_validation_split(
    frames: Sequence[Any],
    frames_json: str,
    patch_dir: str,
    n: int = 10,
) -> List[List[List[int]]]:
    """
    n-times divide the frames into training and test (no val inside each fold).

    Args:
        frames: list of frames (FrameInfo or similar)
        frames_json: path to JSON file where splits are stored (and read from
                     if present)
        patch_dir: directory where the JSON is stored (created if missing)
        n: number of folds

    Returns:
        splits: list of [train_index_list, test_index_list] for each fold
    """
    # Check if we've already computed these splits and saved them
    if os.path.isfile(frames_json):
        print("[SPLIT][CV] Reading n-splits from file")
        with open(frames_json, "r") as file:
            fjson = json.load(file)
            splits = fjson.get("splits", [])
            # Load cached splits for reproducibility—every experiment will use
            # identical train/test divisions, making results comparable.
            return splits

    # First time: compute the splits from scratch
    print("[SPLIT][CV] Creating and writing n-splits to file")
    frames_list = list(range(len(frames)))
    # Initialize K-fold splitter with shuffle=True to randomize fold assignment.
    # random_state=1117 is the RNG seed: same seed guarantees identical shuffling
    # if you re-run this function. This is crucial for reproducibility.
    kf = KFold(n_splits=n, shuffle=True, random_state=1117)
    print(f"[SPLIT][CV] Number of splitting iterations: {kf.get_n_splits(frames_list)}")

    # Iterate through each fold, storing the train/test indices for that fold
    splits: List[List[List[int]]] = []
    for train_index, test_index in kf.split(frames_list):
        # In this fold: train_index frames are used to train, test_index frames
        # are held out for evaluation. Next fold rotates which frames are "test".
        # Convert numpy arrays to lists and store the pair (train_ids, test_ids)
        splits.append([train_index.tolist(), test_index.tolist()])

    # Save to JSON so we can load the same splits next time (reproducibility).
    # Without saving, if you re-run with a different random seed, you'd get different
    # folds, making it impossible to compare model performance across experiments.
    frame_split = {"splits": splits}
    _ensure_dir(patch_dir)
    with open(frames_json, "w") as f:
        json.dump(frame_split, f, indent=2)

    print(_col("[SPLIT][CV] Saved cross-validation splits.", _C.GREEN))
    return splits


# ---------- helpers for stratified splitting by positive-pixel rate ----------
#
# WHAT IS POSITIVE RATE AND WHY STRATIFY BY IT?
#   "Positive rate" = fraction of bubble pixels in a frame
#   Example: if a frame is 1000x1000 = 1M pixels, and 50k are labeled as bubbles,
#   the positive rate is 50k/1M = 0.05 (or 5%).
#
#   Stratifying by positive rate ensures that train/val/test each get a mix of
#   "easy" frames (mostly background) and "hard" frames (dense bubbles).
#   Without stratification, you might randomly put all high-density frames in
#   training and all sparse ones in test, making test artificially hard.


def _pos_rate(frames: Sequence[Any]) -> np.ndarray:
    """
    Compute fraction of positive pixels per frame (used to stratify).

    Expects each frame to have `.annotations` (HxW) with positives > 0.

    POSITIVE RATE DEFINITION:
      The fraction of pixels in a frame that are labeled as the target (bubbles).
      Example: if a frame is 1000x1000 pixels and 50,000 are labeled bubbles,
      positive_rate = 50,000 / 1,000,000 = 0.05 (5%).

      A frame with 0.5 (50% bubbles) is "harder" for the model to learn than
      a frame with 0.01 (1% bubbles). Stratification ensures both easy and hard
      frames are distributed evenly across train/val/test.
    """
    _pos_rate = []
    for fr in frames:
        # Safely get annotations, defaulting to None if missing
        label_annotation = getattr(fr, "annotations", None)
        if label_annotation is None:
            # Frame has no label, so 0% positive
            _pos_rate.append(0.0)
        else:
            # Convert to binary (anything > 0 is a positive pixel).
            # This treats all bubble labels (1, 2, 3, etc.) as equivalent.
            lab = (np.asarray(label_annotation) > 0).astype(np.uint8)
            # Compute fraction: count of positive pixels / total pixels.
            # Add 1e-6 to avoid division by zero if all pixels are zero.
            _pos_rate.append(float(lab.sum()) / float(lab.size + 1e-6))
    return np.asarray(_pos_rate, dtype=np.float32)

def _make_strata(pos_rates: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Quantile-bin positive rates into strata labels for stratification.

    Falls back to a single stratum if all rates are identical.

    STRATIFICATION & QUANTILE BINNING:
      Stratification ensures that each of train/val/test receives a representative
      sample of the data in terms of class balance. By grouping frames into strata
      (buckets) by positive rate and then splitting proportionally across strata,
      we avoid accidentally putting all "easy" frames (sparse bubbles) in one set
      and all "hard" frames (dense bubbles) in another.

      Quantile binning divides frames into n_bins equally-populated buckets based
      on their positive rates. With n_bins=5:
      - Stratum 0: frames with lowest 20% of positive rates (sparse bubbles)
      - Stratum 1: next 20% of positive rates
      - ... up to Stratum 4: highest 20% of positive rates (dense bubbles)

      Example: if you have 100 frames, each stratum gets 20 frames. When you split
      80-20 into train-test, you take ~16 frames from each stratum to train, ~4 to test.
      This preserves the distribution: train and test are balanced by difficulty.

      Why not use simple cutoffs like pos_rate < 0.1 vs pos_rate > 0.1?
      Because the actual distribution of your data might be skewed. Quantiles
      automatically adapt to your data, ensuring equal-sized strata regardless
      of the underlying distribution.
    """
    # Edge case: empty input
    if pos_rates.size == 0:
        return np.zeros(0, dtype=int)

    # Edge case: all frames have identical positive rate (nothing to stratify).
    # If every frame has, say, 5% bubbles, there's no variation to preserve.
    if np.allclose(pos_rates, pos_rates[0]):
        return np.zeros_like(pos_rates, dtype=int)

    # Compute quantile boundaries that divide frames into n_bins equal-sized buckets.
    # For n_bins=5, qs=[0, 0.2, 0.4, 0.6, 0.8, 1.0].
    # These are percentiles: 0th percentile (min), 20th, 40th, 60th, 80th, 100th (max).
    qs = np.linspace(0, 1, n_bins + 1)
    bins = np.quantile(pos_rates, qs)

    # Ensure strictly increasing bins to avoid empty/duplicate bins.
    # This handles edge cases where multiple frames have identical rates.
    # If bins[i] == bins[i-1] (tied values at the boundary), nudge one slightly
    # to the right so np.digitize assigns ties correctly.
    bins[0] = -1e-12
    for i in range(1, len(bins)):
        if bins[i] <= bins[i - 1]:
            bins[i] = bins[i - 1] + 1e-12

    # Assign each frame to a stratum (0, 1, 2, ..., n_bins-1) based on which bin
    # its positive rate falls into. np.digitize returns the index of the bin.
    strata = np.digitize(pos_rates, bins[1:-1], right=False)
    return strata.astype(int)


# ---------- main split (backward-compatible signature) ----------


def split_dataset(
    frames: Sequence[Any],
    frames_json: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    *,
    n_bins: int = 5,
    random_state: int = 1337,
    stratify_by_positives: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Divide frames into training, validation, and test. If `frames_json` exists,
    read the split from it; otherwise create a new split (optionally stratified).

    WHY THIS FUNCTION IS CRITICAL:
      Properly splitting your data is one of the most important steps in ML.
      A bad split can make a mediocre model look great, or a great model look mediocre.

      DATA LEAKAGE is the biggest risk:
        - Data leakage occurs when information from test/val frames accidentally
          influences training, violating the assumption that test is unseen.
        - Example: If you use all data to compute normalization statistics
          (mean, std) before splitting, the test set becomes "contaminated"
          by those statistics—it's no longer truly unseen.
        - Another example: If you keep shuffling and re-splitting data randomly
          in each run, you might accidentally include the same frame in train one day
          and test the next day, making metrics inconsistent and conclusions invalid.
        - Leakage can make validation accuracy look great but test accuracy tank,
          or make results unreproducible across runs.

      WHY WE SAVE THE SPLIT:
        - Saving to frames_json ensures the same frames go to train/val/test
          across all runs. This prevents data leakage from accidental re-shuffling.
        - It also ensures fair comparisons: if you try 10 model architectures
          and they all use the same test set, differences in test accuracy come
          from the model, not from variance in the test set.

      WHY WE USE RANDOM_STATE:
        - random_state seeds the random number generator. Same seed = identical shuffling.
        - Without specifying random_state, each run uses a different shuffle,
          leading to different train/val/test splits. Your "improvement" might just
          be due to a luckier test set, not a better model.

      WHY WE STRATIFY:
        - If your positive class is rare (e.g., only 5% of pixels are bubbles),
          random splitting might accidentally put all positives in train and none in test.
        - Then your test metrics would be misleading: the model could achieve 95% accuracy
          by simply predicting "no bubble" on everything.
        - Stratification ensures train/val/test all have similar positive rates,
          so each set is representative of the real problem.

    Args:
        frames: list of frames (FrameInfo or similar)
        frames_json: path to JSON file for storing the split
        test_size: fraction of total frames to allocate to test
        val_size: fraction of total frames to allocate to validation
                  (of the overall dataset; we convert to a fraction of the
                  remaining train+val pool)
        n_bins: number of quantile bins for stratification by positive rate
        random_state: RNG seed (for reproducibility; same seed = identical split)
        stratify_by_positives: if True, stratify by per-frame positive rate;
                               else random splits (not recommended for imbalanced data)

    Returns:
        (training_frames, validation_frames, testing_frames) as lists of indices
    """
    # Check if this split was already computed and saved.
    # This is the key to preventing data leakage and ensuring reproducibility.
    # Load the same split every time = no accidental train/test mixing.
    if os.path.isfile(frames_json):
        print("[SPLIT][DATA] Reading train/val/test split from file")
        with open(frames_json, "r") as file:
            fjson = json.load(file)
            training_frames = fjson["training_frames"]
            testing_frames = fjson["testing_frames"]
            validation_frames = fjson["validation_frames"]

        # Print summary to confirm we loaded the expected split
        print(f"[SPLIT][DATA] training_frames={len(training_frames)}")
        print(f"[SPLIT][DATA] validation_frames={len(validation_frames)}")
        print(f"[SPLIT][DATA] testing_frames={len(testing_frames)}")
        return training_frames, validation_frames, testing_frames

    # First time: compute the split from scratch
    print(
        "[SPLIT][DATA] Creating and writing train/val/test split "
        f"({'stratified' if stratify_by_positives else 'random'}) to file"
    )

    # Create array of frame indices (0, 1, 2, ..., n-1)
    idx = np.arange(len(frames))
    strat_labels = None

    # Optionally compute stratification labels based on positive rates.
    # Stratification is the ML technique that ensures each split (train/val/test)
    # receives a balanced mix of easy and hard frames, preventing data leakage.
    if stratify_by_positives:
        # Get the positive rate (fraction of bubble pixels) for each frame
        pos_rates = _pos_rate(frames)
        try:
            # Bin the positive rates into n_bins strata (0, 1, 2, ..., n_bins-1).
            # Each frame gets assigned a stratum label based on its positive rate.
            strat_labels = _make_strata(pos_rates, n_bins=n_bins)
        except Exception:
            # If stratification fails (e.g., too few frames for n_bins), fall back
            # to random split. This is safer than crashing.
            strat_labels = None

    # ========== STEP 1: Split off the test set ==========
    # REPRODUCIBILITY: random_state seeds the RNG. Same seed = identical splits.
    # This is critical: you want identical test sets across runs so you can
    # fairly compare different model architectures.
    try:
        # Use stratified split if strat_labels is available, otherwise random.
        # With stratify: sklearn ensures each stratum is split proportionally.
        # Example: if stratum 0 is 30 frames and test_size=0.2, ~6 go to test, ~24 to train_val.
        # This preserves the representation of easy/hard frames in both sets.
        train_val_idx, test_idx = train_test_split(
            idx,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_labels,
        )
    except ValueError:
        # Fallback if strata too imbalanced for sklearn to handle
        # (e.g., a stratum has fewer frames than the split size).
        train_val_idx, test_idx = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=None
        )

    # ========== STEP 2: Split the remaining (train+val) into train and val ==========
    # Important: val_size is specified as a fraction of the original dataset.
    # But we need to convert it to a fraction of what's left after removing test.
    # Example: if test_size=0.2 and val_size=0.2:
    #   remaining = 1.0 - 0.2 = 0.8 (80% of original data is left after removing test)
    #   val_share_in_tv = 0.2 / 0.8 = 0.25 (25% of the remaining train+val pool)
    # This gives us: train=60%, val=20%, test=20% of original dataset.
    remaining = max(1e-12, 1.0 - float(test_size))
    val_share_in_tv = float(val_size) / remaining

    # If stratification is enabled, create strata for just the train_val pool.
    # Why recalculate strata? The train_val pool is now smaller, so we may need
    # fewer bins to avoid empty strata (e.g., a stratum with only 1 frame).
    tv_strata = None
    if stratify_by_positives and strat_labels is not None:
        # Extract the strata labels for only the frames that made it to train_val.
        # This is a subset of the original strat_labels.
        tv_strata = strat_labels[train_val_idx]
        # Reduce number of bins if the sample is small (to avoid empty strata).
        # Heuristic: n_bins_tv = ceil(len(train_val_idx) / 5), capped at n_bins above.
        n_bins_tv = max(2, min(n_bins, len(train_val_idx) // 5))
        try:
            # Recompute strata for the train_val subset using its own positive rates.
            tv_strata = _make_strata(tv_strata.astype(np.float32), n_bins=n_bins_tv)
        except Exception:
            # If recomputation fails, give up on stratification and use random split.
            tv_strata = None

    # Split train_val into train and val using stratification if available.
    # Use random_state + 1 (not the same seed) to ensure different random choices
    # than the first split, adding robustness.
    try:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_share_in_tv,
            random_state=random_state + 1,
            stratify=tv_strata,
        )
    except ValueError:
        # Fallback if stratification fails on the train_val subset.
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_share_in_tv,
            random_state=random_state + 1,
            stratify=None,
        )

    # Convert numpy arrays to Python lists for JSON serialization.
    training_frames = train_idx.tolist()
    validation_frames = val_idx.tolist()
    testing_frames = test_idx.tolist()

    # Create the output dictionary and save to JSON.
    # Persisting the split is CRITICAL for reproducibility and preventing data leakage.
    # By saving the split, you ensure:
    # 1. Reproducibility: re-running the code loads the exact same train/val/test frames.
    # 2. Fair comparisons: all model experiments use the same test set, so differences
    #    in performance reflect model quality, not variance in the test set.
    # 3. Data leakage prevention: you cannot accidentally use test data in training
    #    if you always load the same split from disk.
    frame_split: Dict[str, Any] = {
        "training_frames": training_frames,
        "testing_frames": testing_frames,
        "validation_frames": validation_frames,
    }
    _ensure_dir(os.path.dirname(frames_json))
    with open(frames_json, "w") as f:
        json.dump(frame_split, f, indent=2)

    # Print summary of split sizes
    print(f"[SPLIT][DATA] training_frames={len(training_frames)}")
    print(f"[SPLIT][DATA] validation_frames={len(validation_frames)}")
    print(f"[SPLIT][DATA] testing_frames={len(testing_frames)}")
    return training_frames, validation_frames, testing_frames
