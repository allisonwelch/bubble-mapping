# core/common/data.py
from __future__ import annotations

import glob
import os
from typing import Any, Iterable, Iterator, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from core.dataset_generator import DataGenerator as Generator
from core.frame_info import FrameInfo
from core.split_frames import split_dataset, summarize_positive_rates

config: Any = None


def _meters_to_pixels(m: float, pixel_size_m: float) -> int:
    return int(round(float(m) / float(pixel_size_m)))


def audit_sampler(gen_obj, name: str, batches: int = 200, batch_size: int = 8):
    """
    Audit patch sampling distribution *before* DataLoader workers.

    Prints:
      - mean/median/p90 positive fraction across sampled patches
      - fraction of empty patches (no positive pixels)
    """
    fracs = []
    for _ in range(int(batches)):
        _, y = gen_obj.random_patch(int(batch_size))  # y: (B,H,W)
        y = (y > 0).astype(np.float32)
        fracs.extend(y.mean(axis=(1, 2)).tolist())

    fracs = np.asarray(fracs, dtype=np.float32)
    mean = float(fracs.mean()) if fracs.size else 0.0
    med = float(np.median(fracs)) if fracs.size else 0.0
    p90 = float(np.quantile(fracs, 0.90)) if fracs.size else 0.0
    empty = float((fracs == 0).mean()) if fracs.size else 1.0

    print(
        f"[AUDIT][{name}] patches={fracs.size}  "
        f"pos_frac mean={mean:.6f}  median={med:.6f}  p90={p90:.6f}  empty={empty:.3f}"
    )


def get_all_frames(conf: Optional[Any] = None):
    """
    Load all preprocessed frames from disk.

    Each .tif is expected to contain stacked channels with the last channel
    being a mask. Images are returned as HWC for compatibility with the
    downstream Generator which expects that shape.

    Args:
        conf: Optional config override to set this module's global config.

    Returns:
        list[FrameInfo]: One entry per frame with (image_channels, annotations).
    """
    global config
    if conf is not None:
        config = conf

    if config is None:
        raise RuntimeError("common.data.get_all_frames called without a config.")

    # Default to the most recent preprocessed directory under the base location.
    if config.preprocessed_dir is None:
        config.preprocessed_dir = os.path.join(
            config.preprocessed_base_dir,
            sorted(os.listdir(config.preprocessed_base_dir))[-1],
        )

    # Check that preprocessed_dir exists
    if not os.path.exists(config.preprocessed_dir):
        raise FileNotFoundError(
            f"Preprocessed directory {config.preprocessed_dir} does not exist."
        )

    # Files are numbered and sorted by integer stem, e.g., 69.tif
    image_paths = sorted(
        glob.glob(f"{config.preprocessed_dir}/*.tif"),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0]),
    )

    print(
        f"[DATA][LOAD] Found {len(image_paths)} input frames in "
        f"{config.preprocessed_dir}"
    )

    frames = []
    for im_path in tqdm(image_paths, desc="Processing frames"):
        # Read as CHW, then split inputs vs label channel.
        preprocessed = rasterio.open(im_path).read()  # C,H,W

        image_channels = preprocessed[:-1, ...]  # C,H,W
        image_channels = np.transpose(image_channels, (1, 2, 0))  # H,W,C

        annotations = preprocessed[-1, ...]  # H,W
        frames.append(FrameInfo(image_channels, annotations))

    return frames


class TorchGeneratorDataset(IterableDataset):
    """
    Wrap a DatasetGenerator and convert its batches to NCHW tensors.

    Stores the generator *factory* (gen_obj + batch_size) rather than a pre-built
    Python generator, so the dataset can be pickled by DataLoader worker processes
    on Windows (which uses the 'spawn' multiprocessing start method).
    Each worker calls random_generator() independently after being spawned.

    Notes:
        - x arrives as NHWC, y as NHW or NHWC1. Output is always NCHW for both.
        - Masks are auto-scaled from 0..255 to 0..1 when needed.
        - No shuffling here; it happens in the underlying generator.
    """

    def __init__(self, gen_obj, batch_size: int):
        super().__init__()
        self.gen_obj = gen_obj
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for x, y in self.gen_obj.random_generator(self.batch_size):
            # x: (N, H, W, C) -> (N, C, H, W)
            if isinstance(x, np.ndarray):
                x_t = torch.from_numpy(np.transpose(x, (0, 3, 1, 2))).to(
                    dtype=torch.float32
                )
            else:
                # Already a tensor (NHWC) in some flows.
                x_t = x.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

            # y: (N, H, W) or (N, H, W, 1) -> (N, 1, H, W)
            if isinstance(y, np.ndarray):
                if y.ndim == 3:
                    y = y[..., None]
                y_t = torch.from_numpy(np.transpose(y, (0, 3, 1, 2))).to(
                    dtype=torch.float32
                )
            else:
                if y.ndim == 3:
                    y = y.unsqueeze(-1)
                y_t = y.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)

            # Ensure masks are in [0, 1]. Auto-scale 0..255 to 0..1 if detected.
            maxv = float(y_t.max().item()) if y_t.numel() > 0 else 1.0
            if maxv > 1.5:
                y_t = y_t / 255.0

            yield x_t, y_t


def create_train_val_datasets(frames):
    """
    Create training, validation, and test iterables built on the project Generator.

      - Train uses albumentations and your chosen sampling strategy.
      - Val can be either "like train" or "natural" depending on config.val_like_train.
      - Optional audit prints patch positive fraction distribution.

    Returns:
        Tuple[Iterable, Iterable, Iterable]: train, val, test DataLoaders.
    """
    frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
    training_frames, validation_frames, test_frames = split_dataset(
        frames, frames_json, config.test_ratio, config.val_ratio
    )

    # Quick distribution summary
    stats = summarize_positive_rates(
        frames,
        {"train": training_frames, "val": validation_frames, "test": test_frames},
    )
    print(
        "\n[DATA][STATS] positive-rate % by frame - "
        "mean | median | std | min..max  (n)"
    )
    for k in ("train", "val", "test"):
        s = stats[k]
        print(
            f"  {k:>5}: {s['mean']:.3f} | {s['median']:.3f} | {s['std']:.3f} | "
            f"{s['min']:.3f}..{s['max']:.3f}  (n={s['n']})"
        )

    # Channel layout: inputs first, then label immediately after inputs.
    input_channels = list(range(len(config.channel_list)))
    label_channel = len(config.channel_list)

    # Patch size (optionally overridden by tuner fields).
    patch_h, patch_w = config.patch_size
    if (
        hasattr(config, "tune_patch_h")
        and hasattr(config, "tune_patch_w")
        and config.tune_patch_h
        and config.tune_patch_w
    ):
        patch_h, patch_w = int(config.tune_patch_h), int(config.tune_patch_w)

    patch_size = [patch_h, patch_w, len(config.channel_list) + 1]

    # Sampling and augmentation controls.
    aug_strength = float(getattr(config, "augmenter_strength", 1.0))
    min_pos_frac = float(getattr(config, "min_pos_frac", 0.0))

    pos_ratio = getattr(config, "pos_ratio", None)
    pos_ratio = None if pos_ratio is None else float(pos_ratio)

    stride = getattr(config, "patch_stride", None)

    # Frame weighting ("uniform" recommended if all tiles are same size)
    frame_weighting = str(getattr(config, "sampling_weighting", "uniform")).lower()

    # Optional: negative controls
    neg_max_pos_frac = float(getattr(config, "neg_max_pos_frac", 0.0))
    hard_neg_ratio = float(getattr(config, "hard_neg_ratio", 0.0))

    # Ring sampling config (optional)
    pixel_size_m = float(getattr(config, "pixel_size_m", 0.15))
    ring_inner_px = getattr(config, "ring_inner_px", None)
    ring_outer_px = getattr(config, "ring_outer_px", None)

    if ring_inner_px is None and getattr(config, "ring_inner_m", None) is not None:
        ring_inner_px = _meters_to_pixels(getattr(config, "ring_inner_m"), pixel_size_m)
    if ring_outer_px is None and getattr(config, "ring_outer_m", None) is not None:
        ring_outer_px = _meters_to_pixels(getattr(config, "ring_outer_m"), pixel_size_m)

    ring_neg_ratio = float(getattr(config, "ring_neg_ratio", 0.0))
    ring_weight_power = float(getattr(config, "ring_weight_power", 0.5))

    # --------- NEW: A) component-balanced positives + B) boundary negatives ----------
    pos_sampling = str(getattr(config, "pos_sampling", "pixel"))
    component_weight_power = float(getattr(config, "component_weight_power", 0.0))
    boundary_neg_ratio = float(getattr(config, "boundary_neg_ratio", 0.0))
    boundary_max_pos_frac = float(getattr(config, "boundary_max_pos_frac", 0.0))
    # -------------------------------------------------------------------------------

    # Seeds (optional but recommended)
    base_seed = getattr(config, "seed", None)
    train_seed = None if base_seed is None else int(base_seed) + 0
    val_seed = None if base_seed is None else int(base_seed) + 1
    test_seed = None if base_seed is None else int(base_seed) + 2

    # ----------------------------
    # Instantiate generator objects
    # ----------------------------
    train_gen_obj = Generator(
        input_channels,
        patch_size,
        training_frames,
        frames,
        label_channel,
        augmenter="alb",
        augmenter_strength=aug_strength,
        min_pos_frac=min_pos_frac,
        pos_ratio=pos_ratio,
        stride=stride,
        weighting=frame_weighting,
        seed=train_seed,
        neg_max_pos_frac=neg_max_pos_frac,
        hard_neg_ratio=hard_neg_ratio,
        ring_inner_px=ring_inner_px,
        ring_outer_px=ring_outer_px,
        ring_neg_ratio=ring_neg_ratio,
        ring_weight_power=ring_weight_power,
        # --------- NEW: A + B passed through to Generator ----------
        pos_sampling=pos_sampling,
        component_weight_power=component_weight_power,
        boundary_neg_ratio=boundary_neg_ratio,
        boundary_max_pos_frac=boundary_max_pos_frac,
        # ----------------------------------------------------------
    )

    # Validation: either like train or "natural"
    val_like_train = bool(getattr(config, "val_like_train", False))

    if val_like_train:
        val_min_pos_frac = min_pos_frac
        val_pos_ratio = pos_ratio
        val_ring_inner_px = ring_inner_px
        val_ring_outer_px = ring_outer_px
        val_ring_neg_ratio = ring_neg_ratio
        val_ring_weight_power = ring_weight_power
        val_neg_max_pos_frac = neg_max_pos_frac
        val_hard_neg_ratio = hard_neg_ratio
        val_weighting = frame_weighting
        # --------- NEW: A + B for val (like train) ----------
        val_pos_sampling = pos_sampling
        val_component_weight_power = component_weight_power
        val_boundary_neg_ratio = boundary_neg_ratio
        val_boundary_max_pos_frac = boundary_max_pos_frac
        # ----------------------------------------------------
    else:
        # Natural validation: no balancing, no ring bias, no augmentation
        val_min_pos_frac = 0.0
        val_pos_ratio = None
        val_ring_inner_px = None
        val_ring_outer_px = None
        val_ring_neg_ratio = 0.0
        val_ring_weight_power = 0.5
        val_neg_max_pos_frac = 0.0
        val_hard_neg_ratio = 0.0
        val_weighting = frame_weighting
        # --------- NEW: A + B for val (natural) ----------
        val_pos_sampling = pos_sampling
        val_component_weight_power = component_weight_power
        val_boundary_neg_ratio = 0.0
        val_boundary_max_pos_frac = 0.0
        # -------------------------------------------------

    val_gen_obj = Generator(
        input_channels,
        patch_size,
        validation_frames,
        frames,
        label_channel,
        augmenter=None,
        augmenter_strength=1.0,
        min_pos_frac=val_min_pos_frac,
        pos_ratio=val_pos_ratio,
        stride=stride,
        weighting=val_weighting,
        seed=val_seed,
        neg_max_pos_frac=val_neg_max_pos_frac,
        hard_neg_ratio=val_hard_neg_ratio,
        ring_inner_px=val_ring_inner_px,
        ring_outer_px=val_ring_outer_px,
        ring_neg_ratio=val_ring_neg_ratio,
        ring_weight_power=val_ring_weight_power,
        # --------- NEW: A + B passed through to Generator ----------
        pos_sampling=val_pos_sampling,
        component_weight_power=val_component_weight_power,
        boundary_neg_ratio=val_boundary_neg_ratio,
        boundary_max_pos_frac=val_boundary_max_pos_frac,
        # ----------------------------------------------------------
    )

    test_gen_obj = Generator(
        input_channels,
        patch_size,
        test_frames,
        frames,
        label_channel,
        augmenter=None,
        augmenter_strength=0.0,
        min_pos_frac=0.0,
        pos_ratio=None,
        stride=stride,
        weighting=frame_weighting,
        seed=test_seed,
        ring_neg_ratio=0.0,
        # --------- NEW: A + B (test is natural) ----------
        pos_sampling=pos_sampling,
        component_weight_power=component_weight_power,
        boundary_neg_ratio=0.0,
        boundary_max_pos_frac=0.0,
        # -------------------------------------------------
    )

    # ----------------------------
    # Optional audit (main process)
    # ----------------------------
    do_audit = bool(getattr(config, "audit_sampling", False))
    if do_audit:
        audit_batches = int(getattr(config, "audit_batches", 200))
        audit_bs = int(getattr(config, "audit_batch_size", config.train_batch_size))
        audit_sampler(train_gen_obj, "train", batches=audit_batches, batch_size=audit_bs)
        audit_sampler(val_gen_obj, "val", batches=max(50, audit_batches // 2), batch_size=audit_bs)
        audit_sampler(test_gen_obj, "test", batches=max(50, audit_batches // 2), batch_size=audit_bs)

    # Wrap with DataLoader-compatible datasets.
    # Pass gen_obj + batch_size (not a pre-built generator) so workers can be pickled on Windows.
    workers = int(getattr(config, "fit_workers", 8))
    train_ds = TorchGeneratorDataset(train_gen_obj, config.train_batch_size)
    val_ds = TorchGeneratorDataset(val_gen_obj, config.train_batch_size)
    test_ds = TorchGeneratorDataset(test_gen_obj, config.train_batch_size)

    base_kwargs = dict(
        batch_size=None,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    if workers > 0:
        base_kwargs_pf = {**base_kwargs, "prefetch_factor": 2}
    else:
        base_kwargs_pf = base_kwargs

    train_dl = DataLoader(train_ds, **base_kwargs_pf)
    val_dl = DataLoader(val_ds, **base_kwargs_pf)
    test_dl = DataLoader(test_ds, **base_kwargs_pf)
    return train_dl, val_dl, test_dl
