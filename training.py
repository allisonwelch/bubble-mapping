# training.py (PyTorch) - BF16 + channels-last, EMA, progress bars, better visuals (raw | mask), robust logits->probs
# This script implements end-to-end training for semantic segmentation of frozen bubbles in aerial images.
# Key technologies: Mixed Precision (BF16/FP16), Exponential Moving Average (EMA), learning rate scheduling,
# gradient accumulation, and multi-architecture support (UNet, SwinUNet, TerraMind).

import glob
import json
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple, List

# h5py: for backwards compatibility with old TensorFlow model checkpoints (likely not in use)
import h5py
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
# tqdm: progress bars for visualizing training loops
from tqdm import tqdm
import inspect
# Schedulers: control how learning rate changes during training
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

# ===== Fast execution defaults / mixed precision =====
# These settings enable faster tensor operations on NVIDIA GPUs by using TF32 (a reduced-precision
# format). They don't affect model accuracy significantly but can speed up training by 2-3x.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# benchmark=True: cuDNN automatically tests different algorithms and picks the fastest for your GPU
torch.backends.cudnn.benchmark = True

# ===== Mixed Precision Setup (AMP = Automatic Mixed Precision) =====
# Mixed precision training uses lower precision (FP16/BF16) for some operations and FP32 for others.
# This reduces memory usage and speeds up training while maintaining accuracy.
# BF16 (Brain Float 16): more stable than FP16, preferred on newer NVIDIA A100s and H100s
# FP16 (Float 16): older format, needs gradient scaling to avoid underflow
# _AMP_DTYPE: the precision type selected based on GPU capability
_AMP_DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else torch.float16
)

# ===== project imports =====
# Model architectures for bubble segmentation
from core.Swin_UNetPP import SwinUNet
from core.TerraMind import TerraMind
from core.UNet import UNet
from core.common.console import _C, _col, _fmt_seconds
# Data loading and frame utilities
from core.common.data import create_train_val_datasets, get_all_frames
# Model utilities: EMA (Exponential Moving Average), probability conversion, format normalization
from core.common.model_utils import (
    ModelEMA,
    _as_probs_from_terratorch,
    _as_probs_from_terratorch_logits_first,
    _ensure_nchw,
    _forward_with_autopad,
    _is_terramind_model,
    set_global_seed,
)
from core.common.vis import _log_triptych_and_optional_heatmap
from core.dataset_generator import DataGenerator as Generator
from core.frame_info import FrameInfo
# Loss functions and metrics: IoU, Dice, F1, sensitivity, specificity, etc.
from core.losses import (
    Hausdorff_distance,
    IoU,
    accuracy,
    boundary_intersection_over_union,
    dice_coef,
    dice_loss,
    f1_score,
    f_beta,
    get_loss,
    normalized_surface_distance,
    sensitivity,
    specificity,
)
from core.optimizers import get_optimizer
from core.split_frames import split_dataset, summarize_positive_rates

# ===== Global config holder =====
# This module-level variable holds training configuration (learning rate, batch size, epochs, etc.)
# It's set by the public entry point functions (train_UNet, train_SwinUNetPP, train_TerraMind)
config = None


# ===== Sanitizers: Input validation & normalization =====
# These functions prevent NaN/Inf from propagating through the training loop.
# NaNs in gradients or losses are catastrophic — they make all weights become NaN, destroying the model.

def _nan_to_num_torch(x: torch.Tensor, constant: float) -> torch.Tensor:
    """
    Replace NaN/Inf with a finite constant (like numpy.nan_to_num),
    but keeping gradients for finite values.
    This is safer than clamp because it preserves the gradient flow for valid values.

    Args:
        x: input tensor (may contain NaN or Inf)
        constant: what value to substitute for NaN/Inf (e.g., 0.0 or -1.0)

    Returns:
        tensor with NaN/Inf replaced by constant, ready for loss computation
    """
    return torch.where(
        torch.isfinite(x),
        x,
        torch.as_tensor(constant, dtype=x.dtype, device=x.device),
    )


def _sanitize_pair_xy(x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sanitize inputs and labels to prevent NaN disasters during training.
    This is called before computing loss for each batch.

    Inputs (x): Aerial imagery, typically z-scored (normalized to mean=0, std=1).
      z-scored values are negative, which is OK — we preserve them.

    Labels (y): Binary masks (0 = no bubble, 1 = bubble).
      Clamp strictly to [0, 1] since probabilities must be in this range.

    Why sanitize?
      - Data loading pipeline may produce NaN (bad pixels, preprocessing errors)
      - Division by zero in loss functions can create Inf
      - If NaN enters loss, all gradients become NaN → training collapses
    """
    # Cast to FP32 for numerically stable loss computation
    x = x.to(dtype=torch.float32)
    y = y.to(dtype=torch.float32)

    # Replace NaN/Inf in inputs with 0.0 (a neutral value that won't affect loss much)
    # Do NOT clamp z-scored inputs: negative values are meaningful (below-mean pixel intensities)
    x = _nan_to_num_torch(x, 0.0)

    # Optional: uncomment if you see extreme outliers (e.g., 100+ std deviations)
    # x = x.clamp_(-5.0, 5.0)

    # Replace NaN/Inf in labels, then clamp to [0, 1] (valid probability range)
    y = _nan_to_num_torch(y, 0.0).clamp_(0.0, 1.0)
    return x, y


def _force_mask_nchw(y: torch.Tensor) -> torch.Tensor:
    """
    Normalize mask tensor shape to (B, 1, H, W).

    PyTorch expects image tensors in NCHW format (channels_first):
      - N = batch size
      - C = channels (1 for binary mask)
      - H = height
      - W = width

    Different data loaders may produce different shapes. This function handles:
      - (B, H, W, 1) → (B, 1, H, W): reshape from channels_last to channels_first
      - (B, H, W)    → (B, 1, H, W): add missing channel dimension
      - (B, 1, H, W) → unchanged: already correct

    Why NCHW?
      - Convolutional operations are optimized for channels_first on GPUs
      - Channels_last format can be 20-30% slower (more cache misses)
    """
    if y.ndim == 3:
        # Add channel dimension: (B, H, W) → (B, 1, H, W)
        return y.unsqueeze(1)
    if y.ndim == 4 and y.shape[-1] == 1 and y.shape[1] != 1:
        # Permute from channels_last to channels_first: (B, H, W, 1) → (B, 1, H, W)
        # .contiguous() ensures the tensor is stored contiguously in memory (required for some ops)
        return y.permute(0, 3, 1, 2).contiguous()
    return y


# ===== Model builders: Instantiate architectures from config =====
# These functions construct model architectures (UNet, SwinUNet, TerraMind) using hyperparameters
# from the config object. They respect tuning results and ensure consistency.

def _build_model_unet() -> nn.Module:
    """
    Instantiate a UNet model using architecture knobs from config.

    UNet is a classic encoder-decoder architecture for segmentation:
      - Encoder: downsamples, captures global context
      - Skip connections: concatenate encoder outputs to decoder at each level
      - Decoder: upsamples, reconstructs spatial detail
      - Head: applies sigmoid to output probabilities in [0, 1]

    Important: UNet's head already outputs probabilities (sigmoid applied).
    This avoids confusion in the training loop: no need for extra conversion.

    Args: (None, uses global config)
    Returns: UNet model ready for training
    """
    # Count input channels (e.g., 4 bands: red, green, blue, NIR)
    in_ch = len(getattr(config, "channel_list", []))

    model = UNet(
        # Model input shape: [batch_size, height, width, channels]
        [config.train_batch_size, *config.patch_size, in_ch],
        # Model output shape: [batch_size, channels] = [B, 1] (one probability per pixel)
        [in_ch],
        # Dilation rate: spacing between convolution kernel elements
        # >1 expands receptive field without extra parameters
        dilation_rate=int(getattr(config, "dilation_rate", 1)),
        # Layer count: base number of filters (64, 128, 256, ... as we go deeper)
        layer_count=int(getattr(config, "layer_count", 64)),
        # L2 regularization weight: penalizes large weights to prevent overfitting
        l2_weight=float(getattr(config, "l2_weight", 1e-4)),
        # Dropout rate: randomly disable neurons during training for regularization
        dropout=float(getattr(config, "dropout", 0.0)),
    )
    # Tag model to indicate it returns probabilities (not raw logits)
    setattr(model, "_returns_probabilities", True)
    return model


def _build_model_swin() -> nn.Module:
    """
    Instantiate a SwinUNet model using architecture knobs from config.

    SwinUNet uses Swin Transformer blocks (attention-based, not convolutions):
      - Shifted window self-attention: efficient (O(n) vs O(n²) standard attention)
      - Hierarchical patches: similar encoder-decoder structure to UNet
      - Better long-range context than convolutions (attention sees entire image)

    Swin is more expressive but slower to train. Use UNet if speed is critical,
    SwinUNet for maximum accuracy.

    Args: (None, uses global config)
    Returns: SwinUNet model ready for training
    """
    # Base channel count: determines model width
    base_c = getattr(config, "swin_base_channels", 64)
    # Patch size: divide input into patches of patch_size × patch_size pixels
    # Larger patches → fewer tokens → faster but less detail
    swin_patch_size = getattr(config, "swin_patch_size", 16)
    # Window size: local attention window (7 means 7×7 patches attend to each other)
    swin_window = getattr(config, "swin_window", 7)
    # Drop path: stochastic depth regularization (randomly skip residual connections)
    drop_path = getattr(config, "drop_path", 0.0)

    # Use inspect.signature to dynamically check what parameters SwinUNet accepts
    # This makes the code robust to API changes in the SwinUNet class
    sig = inspect.signature(SwinUNet)
    kwargs = dict(
        h=config.patch_size[0],  # patch height
        w=config.patch_size[1],  # patch width
        # Number of input channels (e.g., 4 for RGBN)
        ch=len(getattr(config, "channels_used", config.channel_list)),
        c=base_c,  # base channel count
        patch_size=swin_patch_size,
        window_size=swin_window,
    )
    # Only pass drop_path if the model's __init__ supports it
    if "drop_path" in sig.parameters:
        kwargs["drop_path"] = float(drop_path)

    model = SwinUNet(**kwargs)
    return model


def _build_model_terramind() -> nn.Module:
    """
    Instantiate a TerraMind model using hyperparameters from config.

    TerraMind is a foundation model for satellite geospatial tasks. Unlike UNet/SwinUNet
    (built from scratch), TerraMind comes with a PRETRAINED backbone that has seen
    thousands of satellite images. This is a huge advantage:
      - Better initialization: backbone weights already encode visual features
      - Transfer learning: backbone learns generic spatial patterns, decoder learns task-specific features
      - Discriminative learning rates: backbone learns slowly (fine-tuning), head learns fast (new task)

    Architecture:
      - Backbone: pretrained feature extractor (e.g., ResNet, ViT)
      - Decoder: UperNet or similar, converts backbone features to segmentation mask
      - Head: final classification layer

    Args: (None, uses global config)
    Returns: TerraMind model ready for training
    """
    # ---- Derive channel counts ----
    # Number of input channels (e.g., red, green, blue, NIR = 4 channels)
    in_ch = len(
        getattr(config, "channels_used", getattr(config, "channel_list", []))
    )
    # Number of output classes (1 for binary segmentation: bubble vs background)
    num_classes = int(getattr(config, "num_classes", 1))
    # Satellite modality (e.g., "S2" for Sentinel-2, "L8" for Landsat-8)
    modality = getattr(config, "modality", "S2")

    # ---- Read TerraMind-specific config knobs ----
    # Which backbone to use (e.g., "terramind_v1_large", or size token "base"/"large")
    tm_backbone = getattr(config, "tm_backbone", None)
    # Decoder architecture (UperNetDecoder, etc.)
    tm_decoder = getattr(config, "tm_decoder", "UperNetDecoder")
    # Decoder feature channels (256, 512, etc.) — higher = more expressive but slower
    tm_dec_ch = getattr(config, "tm_decoder_channels", 256)
    # Dropout on the head (helps prevent overfitting)
    tm_head_do = getattr(config, "tm_head_dropout", None)
    # Band selection: which satellite bands to use (optional subset)
    tm_indices = getattr(config, "tm_select_indices", None)
    tm_bands = getattr(config, "tm_bands", None)
    # Path to pretrained backbone checkpoint (if using custom weights)
    tm_ckpt = getattr(config, "tm_backbone_ckpt_path", None)
    # How to merge multi-spectral bands (mean, sum, concat, etc.)
    tm_merge = getattr(config, "terramind_merge_method", "mean")
    # Fallback model size if tm_backbone doesn't specify
    tm_size_fallback = getattr(config, "terramind_size", "base")

    # ---- Parse backbone name to extract model size ----
    # TerraMind has multiple sizes (tiny, small, base, large) — infer from config
    def _parse_size_from_backbone(
        s: Optional[str], default_size: str = "base"
    ) -> Tuple[Optional[str], str]:
        """
        Extract model size (tiny/small/base/large) from backbone name.
        Examples:
          "terramind_v1_large" → ("terramind_v1_large", "large")
          "base" → (None, "base")
          None → (None, "base")
        """
        if not s:
            return None, default_size
        lower = s.lower()
        if lower.startswith("terramind"):
            # Extract size from backbone name
            size = (
                "large" if "large" in lower else
                "base" if "base" in lower else
                "small" if "small" in lower else
                "tiny" if "tiny" in lower else
                default_size
            )
            return s, size
        # If not a full backbone name, treat as size token
        if lower in {"tiny", "small", "base", "large"}:
            return None, lower
        return None, default_size

    backbone_override, tm_size = _parse_size_from_backbone(
        tm_backbone, tm_size_fallback
    )

    decoder_kwargs: Dict[str, Any] = {}

    # ---- Build kwargs dynamically using inspect ----
    # Use reflection to check what parameters TerraMind.__init__ accepts
    # This makes code robust to API changes (new TerraMind versions)
    sig = inspect.signature(TerraMind)
    kwargs: Dict[str, Any] = {}

    def _add_if_supported(name: str, value: Any):
        """Only add parameter if TerraMind.__init__ supports it and value is not None."""
        if name in sig.parameters and value is not None:
            kwargs[name] = value

    _add_if_supported("in_channels", in_ch)
    _add_if_supported("num_classes", num_classes)
    _add_if_supported("modality", modality)
    _add_if_supported("tm_size", tm_size)
    _add_if_supported("merge_method", tm_merge)
    # Load pretrained weights from official model zoo
    _add_if_supported("pretrained", True)
    _add_if_supported("ckpt_path", tm_ckpt)
    _add_if_supported("indices_override", tm_indices)
    _add_if_supported("bands_override", tm_bands)
    _add_if_supported("decoder", tm_decoder)
    _add_if_supported("decoder_channels", tm_dec_ch)
    _add_if_supported("decoder_kwargs", decoder_kwargs)
    _add_if_supported("backbone", backbone_override)
    _add_if_supported("rescale", True)
    _add_if_supported("head_dropout", tm_head_do)

    model = TerraMind(**kwargs)

    # Tag model for downstream detection (e.g., for special loss handling)
    setattr(model, "_is_terramind", True)

    # ---- Optional: Freeze backbone at initialization ----
    # BACKBONE FREEZING: Train only the decoder/head, keep backbone weights frozen.
    # Use this when:
    #   - You have limited training data (prevent overfitting)
    #   - Compute is limited (frozen backbone is faster)
    # Don't freeze if you want the model to adapt to your specific satellite bands/task.
    if bool(getattr(config, "tm_freeze_backbone", False)):
        for name, p in model.named_parameters():
            if "backbone" in name:
                # requires_grad=False means this parameter won't be updated during backprop
                p.requires_grad = False

    return model


# ===== TerraMind-specific optimizer helpers =====
# TerraMind uses DISCRIMINATIVE LEARNING RATES: the backbone (pretrained feature extractor)
# learns much more slowly than the head (task-specific classifier). This makes sense:
#   - Backbone: already learned good features, only needs fine-tuning
#   - Head: untrained, needs large updates to learn the new task

def _split_backbone_head_params(inner_model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    """
    Split TerraMind model parameters into backbone vs head groups.

    Why split?
      - Backbone LR: usually 1e-5 to 1e-4 (slow fine-tuning of pretrained weights)
      - Head LR: usually 1e-3 to 1e-2 (fast learning for new task)
      - Ratio: head LR ≈ 10× backbone LR

    Works even if the model is wrapped (e.g., if model is inside model.model).

    Returns:
      - backbone_params: list of parameters from the backbone
      - head_params: list of parameters from decoder + head layers
    """
    inner = inner_model
    # Handle wrapped models
    if hasattr(inner, "model") and isinstance(inner.model, nn.Module):
        inner = inner.model

    bb_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    # Extract backbone parameters if they exist
    if hasattr(inner, "backbone") and isinstance(inner.backbone, nn.Module):
        bb_params = list(inner.backbone.parameters())
        # Create set of parameter IDs to quickly check membership
        bb_ids = {id(p) for p in bb_params}
        # All parameters not in backbone → head
        for p in inner_model.parameters():
            if id(p) not in bb_ids:
                head_params.append(p)
    else:
        # No backbone attribute, treat all as head
        head_params = list(inner_model.parameters())

    return bb_params, head_params


def _make_tm_optimizer_from_config(model: nn.Module, opt_name: str) -> optim.Optimizer:
    """
    Build TerraMind optimizer with DISCRIMINATIVE LEARNING RATES.

    Discriminative LRs: different parameter groups learn at different rates.
    For TerraMind:
      - Backbone: learn slowly (fine-tuning pretrained weights)
      - Head: learn fast (new task)

    Config parameters:
      - tm_lr_backbone: backbone learning rate (e.g., 1e-4)
      - tm_lr_head_mult: multiplier for head LR (e.g., 10.0 → head_lr = backbone_lr × 10)
      - tm_weight_decay: L2 regularization strength (prevents large weights)

    Args:
        model: TerraMind model to optimize
        opt_name: "adamw" or "adam"

    Returns:
        Optimizer with param_groups for [head, backbone]
    """
    # Extract learning rates from config
    lr_bb = float(getattr(config, "tm_lr_backbone"))
    lr_head_mult = float(getattr(config, "tm_lr_head_mult", 10.0))
    wd = float(getattr(config, "tm_weight_decay", 1e-4))

    # Split model parameters
    bb_params, head_params = _split_backbone_head_params(model)

    # Create parameter groups with different LRs
    if len(bb_params) == 0:
        # No backbone found, just one group
        groups = [{"params": head_params, "lr": lr_bb * lr_head_mult, "weight_decay": wd}]
    else:
        # Standard: head at high LR, backbone at low LR
        groups = [
            {"params": head_params, "lr": lr_bb * lr_head_mult, "weight_decay": wd},
            {"params": bb_params, "lr": lr_bb, "weight_decay": wd},
        ]

    # Instantiate optimizer with parameter groups
    opt_name = (opt_name or "adamw").lower()
    if opt_name == "adamw":
        return optim.AdamW(groups)
    return optim.Adam(groups)


def _set_backbone_requires_grad(model: nn.Module, requires_grad: bool):
    """
    Enable/disable gradient computation for backbone parameters.

    Use this to dynamically freeze/unfreeze the backbone during training.
    Example: freeze backbone for first N epochs, then fine-tune later.
    Used with config.tm_freeze_backbone_epochs.

    Args:
        model: TerraMind model
        requires_grad: True to enable gradients, False to freeze
    """
    inner = model
    # Handle wrapped models
    if hasattr(inner, "model") and isinstance(inner.model, nn.Module):
        inner = inner.model
    # Set gradient requirement for all backbone parameters
    if hasattr(inner, "backbone") and isinstance(inner.backbone, nn.Module):
        for p in inner.backbone.parameters():
            p.requires_grad = requires_grad


# ===== Optimizer and Learning Rate Scheduler Builder =====
# Optimizers control how weights are updated: optimizer.step() applies gradients.
# Schedulers control how learning rate changes over time. Two common strategies:
#
# OneCycleLR (for small models, limited data):
#   - Warmup: gradually increase LR from small to max (better convergence)
#   - Decay: gradually decrease LR (fine-tune, smaller updates)
#   - Steps per batch: LR updates every batch (fast feedback loop)
#
# CosineAnnealingLR (for large models, big data):
#   - Smooth cosine decay: LR decreases from base_lr to eta_min
#   - Steps per epoch: LR updates once per epoch (less frequent)
#   - Often better for deep networks (smoother optimization landscape)

def _build_optimizer_and_scheduler(model: nn.Module) -> Tuple[optim.Optimizer, Optional[Any], bool]:
    """
    Build optimizer and optional learning rate scheduler.

    Returns: (optimizer, scheduler, step_per_batch)
      - optimizer: applies gradient updates
      - scheduler: modulates learning rate during training (or None)
      - step_per_batch: if True, call scheduler.step() after every batch
                        if False, call scheduler.step() after every epoch
    """
    # Read training hyperparameters
    total_epochs = int(getattr(config, "num_epochs", 1))
    steps_per_epoch = int(getattr(config, "num_training_steps", 1))

    opt_name = getattr(config, "optimizer_fn", "adam")
    lr_tuned = getattr(config, "learning_rate", None)
    wd_tuned = getattr(config, "weight_decay", None)
    sched_name = str(getattr(config, "scheduler", "none")).lower()

    # ---- Build optimizer ----
    # Check if this is a TerraMind model (needs discriminative learning rates)
    is_tm = _is_terramind_model(model) or isinstance(model, TerraMind)
    if is_tm and all(
        hasattr(config, a) for a in ("tm_lr_backbone", "tm_lr_head_mult", "tm_weight_decay")
    ):
        # TerraMind path: use discriminative LRs (backbone slow, head fast)
        optimizer = _make_tm_optimizer_from_config(model, opt_name)
        # Capture LRs for each param group (for scheduler initialization)
        base_lrs = [float(g.get("lr", 1e-3)) for g in optimizer.param_groups]
        base_lr = base_lrs[0]
    else:
        # Standard path: use project-level optimizer factory
        optimizer = get_optimizer(
            opt_name,
            getattr(config, "num_epochs", total_epochs),
            getattr(config, "num_training_steps", steps_per_epoch),
            model,
            lr=lr_tuned,
            weight_decay=wd_tuned,
            clipnorm=None,            # training loop already clips gradients
            internal_schedule=None,   # Don't use internal schedule; we'll use external schedulers
        )

        # Capture base learning rate(s) for scheduler initialization
        # Capture per-param-group base learning rates (important for models with discriminative LRs)
        base_lrs = [float(g.get("lr", 1e-3)) for g in optimizer.param_groups]
        base_lr = base_lrs[0]

        # Override with config values if provided (fine-tuning from previous run)
        if lr_tuned is not None or wd_tuned is not None:
            for g in optimizer.param_groups:
                if lr_tuned is not None:
                    g["lr"] = float(lr_tuned)
                if wd_tuned is not None and str(opt_name).lower() == "adamw":
                    g["weight_decay"] = float(wd_tuned)

        # Final base_lr used for scheduler initialization
        base_lr = float(
            lr_tuned if lr_tuned is not None
            else optimizer.param_groups[0].get("lr", 1e-3)
        )

    # ---- Build learning rate scheduler ----
    # Schedulers modulate LR during training. Two common strategies:
    scheduler = None
    step_per_batch = False

    if sched_name == "onecycle":
        # OneCycleLR: warmup, then decay. Updates every batch (fast feedback).
        # Best for small models and limited data (images, small datasets).
        # Cycle: LR goes low → max_lr → low (sawtooth pattern)
        max_lr = base_lrs if len(base_lrs) > 1 else base_lr
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs,
        )
        step_per_batch = True

    elif sched_name == "cosine":
        # CosineAnnealingLR: smooth cosine decay. Updates once per epoch.
        # Better for large models and big datasets (deep networks often converge better).
        # Cycle: LR decays smoothly from base_lr to eta_min (like half a cosine wave).
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,        # total epochs for full cosine decay
            eta_min=base_lr * 0.01,    # minimum LR (1% of base)
        )
        step_per_batch = False

    return optimizer, scheduler, step_per_batch


# ===== Logging and Checkpointing =====
# During training, we monitor progress with TensorBoard and save best model checkpoints.
# BestModelSaver: saves model weights when validation loss improves
# MetricsCSVLogger: logs all metrics to CSV for analysis
# HeavyMetricsEvaluator: computes detailed metrics (IoU, Dice, F1, etc.)

class BestModelSaver:
    """
    Saves best model weights when validation loss improves.

    Why save checkpoints?
      - Training can diverge (loss explodes) → reload best weights
      - Overfitting: best performance often at early stopping point, not final epoch
      - Reproducibility: save exact weights that produced best validation score

    What it saves:
      - model.state_dict(): all model weights (can be loaded later with model.load_state_dict())
      - best_val: tracks best validation loss seen so far
    """

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.best_val = float("inf")  # Initialize to infinity (all losses are better)

    def maybe_save(self, model: nn.Module, current_val: float) -> bool:
        """
        Save model if current_val (validation loss) is better than previous best.

        Returns: True if model was saved (improvement), False otherwise
        """
        prev = self.best_val
        if current_val < self.best_val:
            # New best validation loss!
            self.best_val = current_val
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

            # Save model weights (can be EMA weights or standard weights)
            torch.save(model.state_dict(), f"{self.model_path}.weights.pt")
            print(
                _col(
                    f"New best! val_loss improved {prev:.6f} -> {current_val:.6f}. "
                    f"Saved: {self.model_path}.weights.pt",
                    _C.GREEN,
                )
            )
            return True  # indicate improvement
        return False


def _create_logging(model_path: str, log_suffix: str = "") -> Tuple[BestModelSaver, str]:
    """
    Set up logging directories and checkpoint saver.

    Args:
        model_path: base path for saving model weights
        log_suffix: optional suffix for log directory

    Returns:
        - best_saver: BestModelSaver instance for checkpoint management
        - log_dir: directory where all logs (TensorBoard, CSV, images) are saved
    """
    log_dir = os.path.join(
        config.logs_dir, os.path.basename(model_path) + log_suffix
    )
    os.makedirs(log_dir, exist_ok=True)
    best_saver = BestModelSaver(model_path)
    return best_saver, log_dir


class MetricsCSVLogger:
    """
    Log training metrics to CSV file for offline analysis.

    Each epoch, append all metrics (loss, accuracy, IoU, etc.) as a row.
    This makes it easy to plot training curves, find best epoch, etc.

    Why CSV?
      - Human-readable (open in Excel, Pandas, R)
      - No special tools needed (unlike TensorBoard which requires event files)
      - Can be version-controlled (no binary files)
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.metrics_df: Optional[pd.DataFrame] = None

    def update(self, epoch: int, logs: Dict[str, Any]):
        """
        Append metrics for current epoch to CSV.

        Args:
            epoch: current epoch number
            logs: dict of metrics (e.g., {"loss": 0.25, "val_loss": 0.30, "acc": 0.95})
        """
        if self.metrics_df is None:
            self.metrics_df = pd.DataFrame()
        # Create new row with epoch as index
        new_row = pd.DataFrame(logs, index=[epoch])
        # Append to existing dataframe
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        # Write updated CSV
        self.metrics_df.to_csv(self.csv_path, index=False)


class HeavyMetricsEvaluator:
    """
    Compute expensive per-sample metrics on validation set at epoch end.

    Unlike training loss (computed every batch), these metrics are expensive:
      - IoU (Intersection over Union): requires thresholding + counting
      - Dice coefficient: per-sample boundary metric
      - Hausdorff distance: pairwise pixel distance
      - Sensitivity/Specificity: true positive rate, true negative rate

    Strategy:
      - Run on small subset (e.g., 50 steps) for speed
      - Compute once per epoch (not per batch)
      - Log to TensorBoard for visualization
      - Add to metrics dict for CSV logging

    Why threshold (default 0.5)?
      - Model outputs probabilities [0, 1]
      - Need binary prediction: threshold converts prob to 0/1
      - Threshold 0.5 is standard; can tune for precision/recall tradeoff
    """

    def __init__(
        self,
        val_iterable,
        log_dir: str,
        steps: int = 50,
        threshold: float = 0.5,
        device: Optional[torch.device] = None,
    ):
        self.val_iterable = val_iterable
        self.steps = steps  # Validation steps per epoch (to save time)
        self.threshold = threshold  # Threshold for converting probs to binary predictions
        # TensorBoard writer for visualization
        self.tb_writer = SummaryWriter(os.path.join(log_dir, "heavy_metrics"))
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Metric functions to compute
        self.metric_fns = {
            "specificity": specificity,      # True negative rate (low false positives)
            "sensitivity": sensitivity,      # True positive rate (high recall)
            "f_beta": f_beta,                # Weighted harmonic mean of precision/recall
            "f1_score": f1_score,            # Balanced precision-recall metric
            "IoU": IoU,                      # Intersection over Union
            "normalized_surface_distance": normalized_surface_distance,  # Boundary error
            "Hausdorff_distance": Hausdorff_distance,  # Max distance between predictions/targets
            "boundary_intersection_over_union": boundary_intersection_over_union,
            "dice_loss": dice_loss,          # Dice coefficient (similar to Jaccard/IoU)
        }

    @staticmethod
    def _as_float(v) -> float:
        try:
            if isinstance(v, torch.Tensor):
                return float(v.detach().mean().cpu().item())
            return float(v)
        except Exception:
            return float("nan")

    def run(self, model: nn.Module, epoch: int, logs: Dict[str, Any]) -> Dict[str, Any]:
        model.eval()
        accum: Dict[str, list] = {k: [] for k in self.metric_fns.keys()}

        with torch.no_grad():
            it = iter(self.val_iterable)
            for _ in range(self.steps):
                try:
                    x, y_true = next(it)
                except StopIteration:
                    break

                x = x.to(
                    self.device, non_blocking=True
                ).contiguous(memory_format=torch.channels_last)
                y_true = y_true.to(self.device, non_blocking=True)
                y_true = _force_mask_nchw(y_true)
                # sanitize inputs/labels like in tuning
                x, y_true = _sanitize_pair_xy(x, y_true)

                # TerraMind-aware / Swin-aware probabilities
                if _is_terramind_model(model):
                    y_pred_raw = _forward_with_autopad(model, x)
                else:
                    # Swin / UNet: direct forward
                    y_pred_raw = model(x)

                num_classes = int(getattr(config, "num_classes", 1))

                if getattr(model, "_returns_probabilities", False):
                    # e.g. SwinUNet already returns probs in [0,1]
                    y_prob_full = _ensure_nchw(y_pred_raw).float()
                elif _is_terramind_model(model):
                    y_prob_full = _as_probs_from_terratorch_logits_first(
                        y_pred_raw, num_classes=num_classes
                    )
                    y_prob_full = _ensure_nchw(y_prob_full).float()
                else:
                    y_prob_full = _as_probs_from_terratorch(
                        y_pred_raw, num_classes=num_classes
                    )
                    y_prob_full = _ensure_nchw(y_prob_full).float()

                if y_prob_full.shape[1] > 1:
                    cls_idx = int(getattr(config, "metrics_class", 1))
                    cls_idx = max(0, min(cls_idx, y_prob_full.shape[1] - 1))
                    y_prob = y_prob_full[:, cls_idx: cls_idx + 1]
                else:
                    y_prob = y_prob_full

                # sanitize predictions like in tuning
                y_prob = _nan_to_num_torch(y_prob.float(), 0.5)
                y_prob = y_prob.clamp(1e-6, 1.0 - 1.0e-6)

                y_bin = (y_prob >= self.threshold).float()

                for name, fn in self.metric_fns.items():
                    try:
                        val = fn(y_true, y_prob)
                    except Exception:
                        val = fn(y_true, y_bin)
                    accum[name].append(self._as_float(val))

        for name, values in accum.items():
            if values:
                mean_val = sum(values) / len(values)
                self.tb_writer.add_scalar(name, mean_val, epoch)
                logs[f"val_{name}"] = mean_val
        self.tb_writer.flush()
        return logs


def _print_run_banner(model_key: str, log_dir: str):
    """Pretty console banner like tuning, including the loss function. Waste of space, but I forget my confg sometimes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ema_on = bool(getattr(config, "use_ema", False))
    aug = float(getattr(config, "augmenter_strength", 1.0))
    minpos = float(getattr(config, "min_pos_frac", 0.0))
    posr = getattr(config, "pos_ratio", None)
    workers = int(getattr(config, "fit_workers", 8))
    stride = getattr(config, "patch_stride", None)
    loss_name = getattr(config, "loss_fn", "tversky")
    ab = getattr(config, "tversky_alphabeta", (0.5, 0.5))
    optimizer_name = getattr(config, "optimizer_fn", "adam")
    scheduler_name = getattr(config, "scheduler", "none")
    lr_val = getattr(config, "learning_rate", None)

    def _dtype_str(dt):
        if dt is torch.bfloat16:
            return "bf16"
        if dt is torch.float16:
            return "fp16"
        if dt is torch.float32:
            return "fp32"
        return str(dt)

    print("\n" + "=" * 90)
    print(
        f"[{model_key.upper()}][TRAIN]  run={getattr(config, 'run_name', 'run')}  "
        f"model_name={config.model_name}"
    )
    print(
        f"[{model_key.upper()}][TRAIN]  device={device}, "
        f"amp_dtype={_dtype_str(_AMP_DTYPE)}, channels_last=True"
    )
    print(
        f"[{model_key.upper()}][TRAIN]  ema={ema_on} "
        f"(decay={getattr(config, 'ema_decay', 0.999)})"
    )
    print(
        f"[{model_key.upper()}][TRAIN]  epochs={config.num_epochs}, "
        f"steps/epoch={config.num_training_steps}, "
        f"val_steps={config.num_validation_images}, "
        f"batch={config.train_batch_size}, workers={workers}"
    )
    print(
        f"[{model_key.upper()}][TRAIN]  patch={config.patch_size}, stride={stride}, "
        f"aug={aug}, min_pos_frac={minpos}, pos_ratio={posr}"
    )
    if hasattr(config, "swin_patch_size"):
        print(
            f"[{model_key.upper()}][TRAIN]  swin_patch="
            f"{getattr(config,'swin_patch_size',16)}, "
            f"window={getattr(config,'swin_window',4)}"
        )
    # Explicit loss + optimizer print
    if str(loss_name).lower().startswith("tversky"):
        extra_lr = "" if lr_val is None else f", lr={lr_val:.2e}"
        print(
            f"[{model_key.upper()}][TRAIN]  loss={loss_name} "
            f"(alpha={ab[0]:.2f}, beta={ab[1]:.2f}), optimizer={optimizer_name}, "
            f"scheduler={scheduler_name}{extra_lr}"
        )
    else:
        extra_lr = "" if lr_val is None else f", lr={lr_val:.2e}"
        print(
            f"[{model_key.upper()}][TRAIN]  loss={loss_name}, optimizer={optimizer_name}, "
            f"scheduler={scheduler_name}{extra_lr}"
        )
    print(f"[{model_key.upper()}][TRAIN]  logs_dir={log_dir}")
    print("=" * 90 + "\n")


# -----------------------------
# Fit loop
# -----------------------------
def _fit_model(
    model: nn.Module,
    train_iterable,
    val_iterable,
    model_path: str,
    starting_epoch: int = 0,
    log_name: str = "",
    optimizer: Optional[optim.Optimizer] = None,
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    scheduler_step_per_batch: bool = False,
) -> None:
    best_saver, log_dir = _create_logging(model_path, log_suffix=log_name)

    # ===== TensorBoard & Logging Setup =====
    # TensorBoard: visual monitoring of metrics (loss curves, images, etc.)
    tb = SummaryWriter(log_dir)

    # CSV logging: append metrics each epoch for offline analysis
    csv_path = os.path.join(
        config.logs_dir, f"{os.path.basename(model_path)}_metrics.csv"
    )
    csv_logger = MetricsCSVLogger(csv_path)

    # Heavy metrics evaluator: compute expensive metrics (IoU, Dice, F1, etc.) once per epoch
    val_eval_steps = int(getattr(config, "heavy_eval_steps", 50))
    heavy_eval = HeavyMetricsEvaluator(
        val_iterable,
        log_dir,
        steps=val_eval_steps,
        threshold=float(getattr(config, "eval_threshold", 0.5)),
    )

    # ===== Device & Memory Format Setup =====
    # Move model to GPU (or CPU if no GPU available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # channels_last memory format: organize tensor as (N, H, W, C) internally
    # This improves GPU memory efficiency and throughput vs channels_first (N, C, H, W)
    model = model.to(device).to(memory_format=torch.channels_last)

    # ===== Debug Mode: Overfit on One Batch =====
    # Use same batch every step to verify model can memorize and loss goes to zero.
    # Useful for debugging data loading, loss computation, backprop.
    # NEVER use in production!
    if getattr(config, "overfit_one_batch", False):
        print(_col("==== WARNING: Overfitting on one Batch ====", _C.YELLOW))
        print(
            _col(
                "If you are running a real training, exit and set config.overfit_one_batch to False!",
                _C.YELLOW,
            )
        )
        # Grab one batch
        first_it = iter(train_iterable)
        first_x, first_y = next(first_it)

        # Return same batch every step (clone to prevent gradient accumulation issues)
        def _infinite_one_batch():
            while True:
                yield first_x.clone(), first_y.clone()

        # Validation also uses same batch
        def _repeat_val(n):
            for _ in range(int(n)):
                yield first_x.clone(), first_y.clone()

        train_iterable = _infinite_one_batch()
        val_iterable = _repeat_val(getattr(config, "num_validation_images", 10))
        heavy_eval.val_iterable = val_iterable

    # ===== Exponential Moving Average (EMA) Setup =====
    # EMA: maintain smoothed copy of model weights that often generalizes better.
    # During training, update main model weights with large steps (noisy gradients).
    # EMA weights are updated slowly: ema_weight ← decay × ema_weight + (1 - decay) × current_weight
    # At evaluation, use EMA weights instead of main weights → better validation accuracy.
    # Analogy: like keeping a "trendline" while training on noisy data.
    use_ema = bool(getattr(config, "use_ema", False))
    ema_decay = float(getattr(config, "ema_decay", 0.999))  # 0.999 means very slow updates
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    eval_with_ema = bool(getattr(config, "eval_with_ema", False))  # Use EMA weights for validation

    # ===== Gradient Scaling (Mixed Precision) Setup =====
    # GradScaler: rescale gradients in FP16 to prevent underflow.
    # Why needed? FP16 has smaller range than FP32 → small gradients underflow to zero.
    # Solution: scale up loss before backprop, scale down gradients before optimizer.step()
    # BF16 doesn't need scaling (better range than FP16).
    use_fp16 = torch.cuda.is_available() and (_AMP_DTYPE is torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    assert optimizer is not None and criterion is not None, "optimizer/criterion are required"

    # ===== Training Hyperparameters & Tracking =====
    # Light metrics: fast to compute every batch (Dice, Accuracy)
    light_metric_fns = [dice_coef, accuracy]

    steps_per_epoch = int(config.num_training_steps)
    val_steps = int(getattr(config, "num_validation_images", 0))
    total_epochs = int(config.num_epochs)

    # Gradient accumulation: accumulate gradients over N steps, then optimizer.step()
    # Effect: simulates larger batch size without needing more GPU memory
    # Example: grad_accum=4 means update weights every 4 steps (4× batch_size effect)
    grad_accum = int(getattr(config, "steps_per_execution", 1))

    # Gradient clipping: cap gradient norm to prevent exploding gradients
    # Exploding gradients: large updates → weights diverge → loss becomes NaN
    # Solution: clip gradients to max_norm before optimizer.step()
    clip_norm = float(getattr(config, "clip_norm", 0.0))  # 0 = disabled

    global_step = 0  # Total steps across all epochs
    model_save_interval = getattr(config, "model_save_interval", None)  # Save snapshot every N epochs
    log_visuals_every = int(getattr(config, "log_visuals_every", 5))  # Log images every N epochs
    vis_rgb_idx = tuple(getattr(config, "vis_rgb_idx", (0, 1, 2)))  # Which bands to visualize as RGB

    # Track logged validation samples
    logged_val_patches = False

    # ===== Verbosity & Progress Controls =====
    verbose = bool(getattr(config, "train_verbose", True))
    epoch_log_every = int(getattr(config, "train_epoch_log_every", 1))
    print_heavy = bool(getattr(config, "train_print_heavy", True))
    show_progress = bool(getattr(config, "show_progress", True))

    # Print run banner (architecture, hyperparams)
    key = "unet" if log_name == "_unet" else ("tm" if log_name == "_tm" else "swin")
    _print_run_banner(key, log_dir)

    # Track best validation loss for early stopping & checkpointing
    best_val_loss = float("inf")

    # ===== TerraMind Backbone Freeze Schedule =====
    # Some configs freeze backbone for first N epochs, then unfreeze to fine-tune.
    # Useful when: limited data, want stable training.
    freeze_ep = 0
    was_frozen = None
    if _is_terramind_model(model):
        freeze_ep = int(getattr(config, "tm_freeze_backbone_epochs", 0))


    # ===== MAIN TRAINING LOOP (Epoch Level) =====
    for epoch in range(starting_epoch, total_epochs):
        t0 = time.time()  # Track epoch timing
        model.train()  # Set model to training mode (enables dropout, batch norm updates, etc.)

        # Dictionary to accumulate metrics for this epoch
        logs: Dict[str, Any] = {}
        train_loss_accum = 0.0  # Sum of training losses
        metric_accums = {fn.__name__: 0.0 for fn in light_metric_fns}  # Dice, Accuracy, etc.

        # ===== Dynamic Backbone Freezing (TerraMind only) =====
        # Unfreeze backbone after freeze_ep epochs to allow fine-tuning of pretrained weights
        if freeze_ep > 0 and _is_terramind_model(model):
            if epoch < freeze_ep and was_frozen is not True:
                # Still in freeze phase: disable backbone gradients
                _set_backbone_requires_grad(model, False)
                was_frozen = True
            elif epoch >= freeze_ep and was_frozen:
                # Freeze phase done: enable backbone gradients for fine-tuning
                _set_backbone_requires_grad(model, True)
                was_frozen = False

        # ===== Iterate over training batches with progress bar =====
        train_range = range(steps_per_epoch)
        if show_progress:
            train_range = tqdm(
                train_range, desc=f"Epoch {epoch+1}/{total_epochs} [train]", leave=False
            )

        train_it = iter(train_iterable)
        # ===== BATCH LEVEL TRAINING LOOP =====
        # Each step:
        #   1. Load batch of images (x) and masks (y)
        #   2. Forward pass: x → model → predictions
        #   3. Compute loss: loss(predictions, y)
        #   4. Backward pass: compute gradients via backpropagation
        #   5. Update weights: optimizer.step() applies gradients
        for step in train_range:
            # ===== Load Batch =====
            try:
                x, y = next(train_it)
            except StopIteration:
                # Restart iterator if exhausted (IterableDataset behavior)
                train_it = iter(train_iterable)
                x, y = next(train_it)

            # ===== Move to GPU + Memory Format Optimization =====
            # non_blocking=True: asynchronous GPU transfer while CPU prepares next batch
            # channels_last: reorder memory layout for better GPU cache locality
            x = x.to(
                device, non_blocking=True
            ).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            # Normalize mask shape to (B, 1, H, W)
            y = _force_mask_nchw(y)

            # ===== Sanitize Inputs & Labels =====
            # Replace NaN/Inf with safe values to prevent loss explosion
            x, y = _sanitize_pair_xy(x, y)

            # ===== Mixed Precision Forward Pass =====
            # AMP: compute in FP16/BF16 for speed, but preserve FP32 for numerical stability
            # torch.cuda.amp.autocast: automatically casts suitable ops to lower precision
            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available(), dtype=_AMP_DTYPE
            ):
                # ===== Forward Pass: Images → Model → Predictions =====
                # Different architectures need different handling:
                # - TerraMind: use autopad (accepts any size)
                # - UNet/SwinUNet: expect specific input size
                if _is_terramind_model(model):
                    y_pred_raw = _forward_with_autopad(model, x)
                else:
                    y_pred_raw = model(x)

                # ===== Convert Raw Outputs to Probabilities =====
                # Models output different formats: raw logits, probs, multi-class, etc.
                # Standardize to probabilities in [0, 1] for loss computation.
                num_classes = int(getattr(config, "num_classes", 1))

                # Check if model already outputs probabilities (e.g., UNet with sigmoid head)
                if getattr(model, "_returns_probabilities", False):
                    # Already probabilities, just format as NCHW
                    y_prob_full = _ensure_nchw(y_pred_raw).float()
                elif _is_terramind_model(model):
                    # TerraMind returns logits: convert to probabilities via softmax/sigmoid
                    y_prob_full = _as_probs_from_terratorch_logits_first(
                        y_pred_raw, num_classes=num_classes
                    )
                    y_prob_full = _ensure_nchw(y_prob_full).float()
                else:
                    # Generic terratorch-style model: convert to probabilities
                    y_prob_full = _as_probs_from_terratorch(
                        y_pred_raw, num_classes=num_classes
                    )
                    y_prob_full = _ensure_nchw(y_prob_full).float()

                # ===== Select Class for Binary Losses =====
                # If multi-class output, pick one class (e.g., class 1 = bubble)
                if y_prob_full.shape[1] > 1:
                    cls_idx = int(getattr(config, "metrics_class", 1))
                    cls_idx = max(0, min(cls_idx, y_prob_full.shape[1] - 1))
                    y_prob = y_prob_full[:, cls_idx : cls_idx + 1]
                else:
                    y_prob = y_prob_full

                # ===== Gradient Flow Check =====
                # Warn if gradients are severed (e.g., by argmax operation)
                # Gradients must flow through all learned parameters for backprop to work.
                if _is_terramind_model(model) and not y_prob.requires_grad:
                    print(
                        _col(
                            "WARNING: TerraMind y_prob has no grad - check decode path.",
                            _C.YELLOW,
                        )
                    )

                # ===== Final Sanitization of Predictions =====
                # Replace any remaining NaN/Inf (emergency backup)
                y_prob = _nan_to_num_torch(y_prob.float(), 0.5)
                # Clamp to safe range: avoid log(0) or log(1) in loss functions
                y_prob = y_prob.clamp(1e-6, 1.0 - 1.0e-6)

                # ===== Compute Loss =====
                # loss = criterion(targets, predictions)
                # For segmentation: typically Dice loss, BCE loss, or focal loss
                # Returns scalar loss value. If this is NaN, training will collapse.
                loss = criterion(y, y_prob)

            # ===== Emergency NaN Detection =====
            # If loss is NaN, something went wrong in forward pass.
            # Print debug info before crashing (helps diagnose data/model bugs).
            if not torch.isfinite(loss):
                print(">>> NaN loss detected")
                with torch.no_grad():
                    try:
                        print(f"  y unique: {torch.unique(y)}")
                    except Exception:
                        pass
                    try:
                        print(f"  y_prob min/max: {y_prob.min()} {y_prob.max()}")
                    except Exception:
                        pass
                    try:
                        print(f"  y_prob all_zero: {bool((y_prob == 0).all().item())}")
                    except Exception:
                        pass
                    try:
                        print(f"  y all_zero: {bool((y == 0).all().item())}")
                    except Exception:
                        pass
                raise RuntimeError("NaN in loss")

            # ===== BACKWARD PASS: Compute Gradients =====
            # Backpropagation: traverse computation graph backwards, compute dLoss/dWeight for all params
            # In mixed precision: scale loss first (scaler.scale) to prevent underflow in FP16
            # Divide by grad_accum to normalize when using gradient accumulation
            scaler.scale(loss / grad_accum).backward()

            # ===== Weight Update (Every grad_accum steps) =====
            # Gradient accumulation: wait N steps to update weights
            # Effect: simulates 4× larger batch size without needing 4× GPU memory
            if (step + 1) % grad_accum == 0:
                # ===== Gradient Clipping =====
                # Clip gradient norm to max_norm to prevent exploding gradients
                # Exploding gradients: weights diverge wildly, loss becomes NaN
                # Solution: clip ∥gradient∥ to max value before stepping
                if clip_norm > 0:
                    scaler.unscale_(optimizer)  # Undo FP16 scaling to get true gradient norm
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=clip_norm
                    )

                # ===== Optimizer Step: Update Weights =====
                # Apply scaled gradients to weights: weight = weight - lr × gradient
                # scaler.step: handles FP16 gradient scaling internally
                scaler.step(optimizer)
                scaler.update()  # Update scale factor for next iteration

                # ===== Zero Gradients =====
                # Clear gradients so they don't accumulate next step
                # set_to_none=True is faster than zero_() on memory
                optimizer.zero_grad(set_to_none=True)

                # ===== Update EMA Weights =====
                # Update exponential moving average: smooth copy of model weights
                # Better generalization: validation accuracy often higher with EMA weights
                if ema is not None:
                    ema.update()

                # ===== Scheduler Step (Per-Batch) =====
                # Some schedulers (OneCycleLR) update learning rate every batch
                # Others (CosineAnnealingLR) update every epoch
                if scheduler is not None and scheduler_step_per_batch:
                    try:
                        scheduler.step()
                    except Exception:
                        pass

            # ===== Accumulate Metrics for Epoch Summary =====
            # Track loss and light metrics (fast to compute)
            train_loss_accum += float(loss.detach().cpu().item())
            for fn in light_metric_fns:
                try:
                    # Try to compute metric on probabilities
                    val = fn(y, y_prob)
                except Exception:
                    # Fallback: threshold probabilities and try again
                    val = fn(y, (y_prob >= 0.5).float())
                metric_accums[fn.__name__] += float(
                    val.detach().mean().cpu().item()
                    if isinstance(val, torch.Tensor)
                    else float(val)
                )

            global_step += 1

            # Step-wise visuals: [ RGB | PRED | GT ]
            if log_visuals_every > 0 and ((global_step % log_visuals_every) == 0):
                try:
                    _log_triptych_and_optional_heatmap(
                        tb=tb,
                        tag_prefix="viz/train",
                        x=x,
                        y_prob=y_prob,
                        y_true=y,
                        step=global_step,
                        rgb_idx=vis_rgb_idx,
                        threshold=float(getattr(config, "eval_threshold", 0.5)),
                        cls_idx=getattr(config, "viz_class", 1),  # pick class 1 by default
                        add_heatmap=True,
                    )
                except Exception:
                    pass

            # Live postfix for progress
            if show_progress:
                avg_tr = train_loss_accum / (step + 1)
                postfix = {
                    "loss": f"{avg_tr:.4f}",
                    "dice": f"{(metric_accums['dice_coef']/(step+1)):.4f}",
                    "acc": f"{(metric_accums['accuracy']/(step+1)):.4f}",
                }
                try:
                    train_range.set_postfix(postfix)
                except Exception:
                    pass

        # ===== Epoch-Level Training Summary =====
        # Average loss and metrics over all training steps
        avg_train_loss = train_loss_accum / max(1, steps_per_epoch)
        logs["loss"] = avg_train_loss
        # Add light metrics (Dice, Accuracy) to logs
        for fn in light_metric_fns:
            logs[fn.__name__] = metric_accums[fn.__name__] / max(1, steps_per_epoch)

        # ===== VALIDATION PHASE =====
        # Why validate?
        #   - Check generalization: training loss ≠ test performance (overfitting)
        #   - Select best checkpoint: save model with lowest val loss
        #   - Early stopping: stop if val loss plateaus
        # NOTE: Validation is NOT used for gradient updates (model.eval(), torch.no_grad())

        model.eval()  # Set to evaluation mode (disable dropout, fix batch norm, etc.)
        val_loss_accum = 0.0
        val_metric_accums = {fn.__name__: 0.0 for fn in light_metric_fns}

        # ===== Conditional EMA Context =====
        # If eval_with_ema=True, use smoothed EMA weights for validation
        # Often gives slightly better validation metrics (better generalization)
        @contextmanager
        def maybe_ema_ctx():
            """Use EMA weights during validation if enabled."""
            if eval_with_ema and ema is not None:
                with ema.use_ema_weights(model):
                    yield
            else:
                yield

        best_improved = False
        with maybe_ema_ctx():
            # ===== Validation Loop =====
            val_range = range(max(1, val_steps)) if val_steps > 0 else range(0)
            if show_progress and val_steps > 0:
                val_range = tqdm(
                    val_range, desc=f"Epoch {epoch+1}/{total_epochs} [val]", leave=False
                )

            # torch.no_grad(): disable gradient tracking (saves memory, speeds up inference)
            # We don't need gradients during validation → don't compute them
            with torch.no_grad():
                val_it = iter(val_iterable)
                # Save first batch for visualization (RGB | prediction | ground truth)
                x_vis = None
                y_vis = None
                y_hat_vis = None
                val_count = 0

                for _ in val_range:
                    # Load validation batch
                    try:
                        x, y = next(val_it)
                    except StopIteration:
                        break
                    val_count += 1

                    # Move to GPU + format
                    x = x.to(
                        device, non_blocking=True
                    ).contiguous(memory_format=torch.channels_last)
                    y = y.to(device, non_blocking=True)
                    y = _force_mask_nchw(y)

                    # Sanitize
                    x, y = _sanitize_pair_xy(x, y)

                    # Forward pass (same as training, but without gradients)
                    if _is_terramind_model(model):
                        y_pred_raw = _forward_with_autopad(model, x)
                    else:
                        y_pred_raw = model(x)

                    num_classes = int(getattr(config, "num_classes", 1))

                    # Convert to probabilities
                    if getattr(model, "_returns_probabilities", False):
                        y_prob_full = _ensure_nchw(y_pred_raw).float()
                    elif _is_terramind_model(model):
                        y_prob_full = _as_probs_from_terratorch_logits_first(
                            y_pred_raw, num_classes=num_classes
                        )
                        y_prob_full = _ensure_nchw(y_prob_full).float()
                    else:
                        y_prob_full = _as_probs_from_terratorch(
                            y_pred_raw, num_classes=num_classes
                        )
                        y_prob_full = _ensure_nchw(y_prob_full).float()

                    # Select class if multi-class
                    if y_prob_full.shape[1] > 1:
                        cls_idx = int(getattr(config, "metrics_class", 1))
                        cls_idx = max(0, min(cls_idx, y_prob_full.shape[1] - 1))
                        y_prob = y_prob_full[:, cls_idx : cls_idx + 1]
                    else:
                        y_prob = y_prob_full

                    # Sanitize predictions
                    y_prob = _nan_to_num_torch(y_prob.float(), 0.5)
                    y_prob = y_prob.clamp(1e-6, 1.0 - 1.0e-6)

                    # Compute validation loss and metrics
                    loss = criterion(y, y_prob)
                    val_loss_accum += float(loss.detach().cpu().item())
                    for fn in light_metric_fns:
                        try:
                            v = fn(y, y_prob)
                        except Exception:
                            v = fn(y, (y_prob >= 0.5).float())
                        val_metric_accums[fn.__name__] += float(
                            v.detach().mean().cpu().item()
                            if isinstance(v, torch.Tensor)
                            else float(v)
                        )

                    # Keep first batch samples for visualization
                    if x_vis is None:
                        x_vis, y_vis, y_hat_vis = (
                            x[:8].clone(),
                            y[:8].clone(),
                            y_prob[:8].clone(),
                        )

                        # Log raw validation patches once per entire run
                        if not logged_val_patches:
                            try:
                                n_show = min(8, x_vis.size(0))

                                # Inputs as-is (NCHW)
                                tb.add_images(
                                    "data/val_input",
                                    x_vis[:n_show].detach().cpu(),
                                    epoch,
                                )

                                # Masks -> single channel float
                                y_for_vis = y_vis[:n_show].detach().float().cpu()
                                if y_for_vis.dim() == 3:
                                    y_for_vis = y_for_vis.unsqueeze(1)
                                elif y_for_vis.dim() == 4 and y_for_vis.size(1) > 1:
                                    y_for_vis = y_for_vis[:, :1]

                                tb.add_images(
                                    "data/val_mask",
                                    y_for_vis,
                                    epoch,
                                )
                                logged_val_patches = True
                            except Exception:
                                pass

                denom = max(1, val_count if val_count > 0 else val_steps)
                avg_val_loss = val_loss_accum / denom
                logs["val_loss"] = avg_val_loss
                for fn in light_metric_fns:
                    logs[f"val_{fn.__name__}"] = val_metric_accums[fn.__name__] / denom

                # Heavy metrics on subset of val
                logs = heavy_eval.run(model, epoch, logs)

                # Save best (EMA context if applied)
                improved = best_saver.maybe_save(model, avg_val_loss)
                best_improved = improved
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    logs["best_val_loss"] = best_val_loss

                # Visualisation: epoch-end [ RGB | PRED | GT ]
                if x_vis is not None:
                    try:
                        _log_triptych_and_optional_heatmap(
                            tb=tb,
                            tag_prefix="viz/val",
                            x=x_vis,
                            y_prob=y_hat_vis,
                            y_true=y_vis,
                            step=epoch + 1,
                            rgb_idx=vis_rgb_idx,
                            threshold=float(getattr(config, "eval_threshold", 0.5)),
                            cls_idx=getattr(config, "viz_class", 1),
                            add_heatmap=True,
                        )
                    except Exception:
                        pass

        # If we evaluated/saved EMA weights and there was an improvement,
        # also save RAW weights snapshot
        if best_improved and eval_with_ema and ema is not None:
            try:
                torch.save(model.state_dict(), f"{model_path}.raw.weights.pt")
                print(
                    _col(
                        f"==> Also saved raw weights {model_path}.raw.weights.pt",
                        _C.GREEN,
                    )
                )
            except Exception:
                pass

        # ===== TensorBoard scalars – same logic as old training.py =====
        if "loss" in logs or "val_loss" in logs:
            pair = {}
            if "loss" in logs:
                pair["train"] = logs["loss"]
            if "val_loss" in logs:
                pair["val"] = logs["val_loss"]
            tb.add_scalars("loss", pair, epoch)

        for fn in light_metric_fns:
            name = fn.__name__
            train_key = name
            val_key = f"val_{name}"
            if train_key in logs or val_key in logs:
                pair = {}
                if train_key in logs:
                    pair["train"] = logs[train_key]
                if val_key in logs:
                    pair["val"] = logs[val_key]
                tb.add_scalars(name, pair, epoch)

        heavy_names = [
            "val_specificity",
            "val_sensitivity",
            "val_f_beta",
            "val_f1_score",
            "val_IoU",
            "val_normalized_surface_distance",
            "val_Hausdorff_distance",
            "val_boundary_intersection_over_union",
            "val_dice_loss",
        ]
        for name in heavy_names:
            if name in logs:
                base = name[4:] if name.startswith("val_") else name
                tb.add_scalars(base, {"val": logs[name]}, epoch)
        tb.flush()

        # ===== Scheduler Step (Per-Epoch) =====
        # CosineAnnealingLR and similar schedulers update once per epoch (not per batch)
        if scheduler is not None and not scheduler_step_per_batch:
            try:
                scheduler.step()
            except Exception:
                pass

        # ===== Save Checkpoint if Best Validation Loss =====
        # Best model saver: only keep weights with lowest validation loss
        # Prevents overfitting: model at epoch 20 may be better than final epoch 50
        best_improved = best_saver.maybe_save(
            ema.ema_model if eval_with_ema and ema is not None else model,
            avg_val_loss,
        )
        logs["best_val_loss"] = best_saver.best_val

        # ===== Per-Epoch Metadata =====
        # Save training metadata (hyperparameters, metrics, elapsed time) as JSON
        # Useful for experiment tracking, reproducibility, debugging
        meta_data = {
            "name": config.model_name,
            "model_path": model_path,
            "patch_size": tuple(config.patch_size),
            "channels_used": getattr(
                config, "channels_used", getattr(config, "channel_list", [])
            ),
            "resample_factor": getattr(config, "resample_factor", None),
            "frames_dir": config.preprocessed_dir,
            "train_ratio": float(f"{1 - config.val_ratio - config.test_ratio:.2f}"),
            "val_ratio": config.val_ratio,
            "test_ratio": config.test_ratio,
            "loss": config.loss_fn,
            "optimizer": config.optimizer_fn,
            "tversky_alpha": getattr(config, "tversky_alphabeta", (None, None))[0],
            "tversky_beta": getattr(config, "tversky_alphabeta", (None, None))[1],
            "batch_size": config.train_batch_size,
            "epoch_steps": config.num_training_steps,
            "val_steps": config.num_validation_images,
            "epochs_trained": f"{epoch + 1}/{config.num_epochs}",
            "total_epochs": config.num_epochs,
            "last_sensitivity": logs.get("val_sensitivity"),
            "last_specificity": logs.get("val_specificity"),
            "last_dice_coef": logs.get("val_dice_coef"),
            "last_dice_loss": logs.get("val_dice_loss"),
            "last_accuracy": logs.get("val_accuracy"),
            "last_f_beta": logs.get("val_f_beta"),
            "last_f1_score": logs.get("val_f1_score"),
            "last_IoU": logs.get("val_IoU"),
            "last_normalized_surface_distance": logs.get(
                "val_normalized_surface_distance"
            ),
            "last_Hausdorff_distance": logs.get("val_Hausdorff_distance"),
            "last_boundary_intersection_over_union": logs.get(
                "val_boundary_intersection_over_union"
            ),
            "start_time": getattr(_fit_model, "_start_time_str", None)
            or datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "elapsed_time": None,
        }

        # Update elapsed time
        if not hasattr(_fit_model, "_start_time_dt"):
            _fit_model._start_time_dt = datetime.now()
            _fit_model._start_time_str = meta_data["start_time"]
        elapsed = datetime.now() - _fit_model._start_time_dt
        meta_data["elapsed_time"] = (
            datetime.utcfromtimestamp(0) + elapsed
        ).strftime("%H:%M:%S")

        meta_path = f"{model_path}.metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta_data, f, indent=4)

        # CSV log
        csv_logger.update(epoch, logs)

        # Optional periodic snapshot
        if model_save_interval and (epoch + 1) % int(model_save_interval) == 0:
            torch.save(model.state_dict(), f"{model_path}.epoch{epoch+1}.weights.pt")

        # ---- Verbose console print of all metrics at end of validation ----
        if verbose and (
            ((epoch + 1) % max(1, epoch_log_every) == 0)
            or (epoch == 0)
            or (epoch + 1 == total_epochs)
        ):
            lr_val_runtime = None
            try:
                lr_val_runtime = optimizer.param_groups[0].get("lr", None)
            except Exception:
                pass
            took = _fmt_seconds(time.time() - t0)
            head = f"\n Epoch {epoch+1}/{total_epochs} [{took}]"
            if lr_val_runtime is not None:
                head += f"  lr={lr_val_runtime:.2e}"
            print(head)

            # Split logs into train / val for organized printing
            train_logs = {
                k: v for k, v in logs.items()
                if not k.startswith("val_") and k != "best_val_loss"
            }
            val_logs = {
                k: v for k, v in logs.items()
                if k.startswith("val_") or k == "best_val_loss"
            }

            def _format_all(d: Dict[str, Any]) -> str:
                parts = []
                for k, v in d.items():
                    if isinstance(v, (float, int)):
                        parts.append(f"{k}={v:.4f}")
                    else:
                        parts.append(f"{k}={v}")
                return " | ".join(parts)

            if train_logs:
                print("  train: " + _format_all(train_logs))
            if val_logs:
                print("   val: " + _format_all(val_logs))

    # End of training: export full model once
    final_export_path = f"{model_path}.pt"
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(
            {"model_state": model.state_dict(), "config": getattr(config, "__dict__", {})},
            final_export_path,
        )
        print(_col(f"Saved final model to: {final_export_path}", _C.GREEN))
    except Exception as exc:
        print(
            _col(
                f"Warning: final model save failed ({exc}). Attempting to save weights-only.",
                _C.YELLOW,
            )
        )
        torch.save(model.state_dict(), f"{model_path}.final.weights.pt")

    print(_col("Training completed.\n", _C.GREEN))


def _prepare_model_and_logging(model_path: str) -> Tuple[Optional[str], int]:
    """
    Handle continue-from checkpoint and carry over logs.
    Returns (state_dict_path_to_load, starting_epoch).
    """
    starting_epoch = 0
    state_path = None

    if getattr(config, "continue_model_path", None):
        state_path = config.continue_model_path
        # Try to read starting epoch from the JSON meta written each epoch
        try:
            meta_json = f"{state_path}.metadata.json"
            if os.path.exists(meta_json):
                with open(meta_json, "r") as f:
                    custom_meta = json.load(f)
                starting_epoch = int(str(custom_meta["epochs_trained"]).split("/")[0])
        except Exception:
            pass

        # Copy logs forward so TB shows a continuous curve
        old_log_dir = os.path.join(
            config.logs_dir, os.path.basename(config.continue_model_path).split(".")[0]
        )
        new_log_dir = os.path.join(config.logs_dir, os.path.basename(model_path))
        if os.path.exists(old_log_dir) and not os.path.exists(new_log_dir):
            try:
                shutil.copytree(old_log_dir, new_log_dir)
            except Exception:
                pass

    return state_path, starting_epoch


# ===== PUBLIC TRAINING ENTRY POINTS =====
# These are the main functions users call to train models.
# Each architecture (UNet, SwinUNet, TerraMind) has its own entry point.
# They all follow the same pattern:
#   1. Load/create data
#   2. Build model architecture
#   3. Build optimizer + loss function
#   4. Load checkpoint if resuming
#   5. Call _fit_model for the main training loop

def train_UNet(conf):
    """
    Train a UNet model for bubble segmentation.

    UNet: lightweight encoder-decoder architecture, fast training.
    Good for: limited compute, small datasets, real-time inference.

    Args:
        conf: config object with all hyperparameters (learning rate, batch size, epochs, etc.)

    Returns: None (saves checkpoints and logs to disk)

    Main steps:
      1. Load data (aerial image patches + bubble masks)
      2. Create UNet model architecture
      3. Build optimizer (Adam/AdamW) and scheduler (OneCycleLR/Cosine)
      4. Build loss function (Dice, BCE, Focal, etc.)
      5. Resume from checkpoint if config.continue_model_path is set
      6. Run training loop with validation, checkpointing, TensorBoard logging
    """
    global config
    config = conf
    print("Starting training (UNet).")
    start = time.time()

    # ===== Reproducibility =====
    # Set random seeds for numpy, torch, cuda (if seed provided in config)
    set_global_seed(getattr(config, "seed", None))

    # ===== Load Data =====
    # Get all image frames from preprocessed directory
    frames = get_all_frames(config)
    # Split into train/val/test based on config ratios
    train_ds, val_ds, test_ds = create_train_val_datasets(frames)

    # ===== Model Path =====
    # Unique timestamp + model name for organization
    stamp = time.strftime("%Y%m%d-%H%M")
    model_path = os.path.join(config.saved_models_dir, f"{stamp}_{config.model_name}")

    # ===== Resume from Checkpoint (if requested) =====
    # If config.continue_model_path is set, resume training from that checkpoint
    state_path, starting_epoch = _prepare_model_and_logging(model_path)

    # ===== Build Model =====
    model = _build_model_unet()

    # ===== Optional PyTorch 2.0 Compilation =====
    # torch.compile: JIT-compile model for faster execution (PyTorch 2.0+)
    # Trades compile time for faster inference/training
    if getattr(config, "use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    # ===== Build Optimizer & Scheduler =====
    # Optimizer: controls how weights are updated (Adam, AdamW, SGD, etc.)
    # Scheduler: modulates learning rate during training (warmup, decay, etc.)
    optimizer, scheduler, scheduler_step_per_batch = _build_optimizer_and_scheduler(model)

    # ===== Build Loss Function =====
    # Loss function: quantifies prediction error (Dice loss, BCE, Focal, etc.)
    # Lower loss = better predictions. Used for backpropagation.
    criterion = get_loss(
        config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))
    )

    # ===== Load Checkpoint (if Resuming) =====
    # Resume training from previous checkpoint (e.g., after interruption)
    if state_path and os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            print(f"Loaded weights from: {state_path}")
        except Exception as exc:
            print(
                _col(
                    f"Could not load PyTorch weights from {state_path}: {exc}", _C.RED
                )
            )

    # ===== Main Training Loop =====
    # Call _fit_model to run epochs with validation, checkpointing, logging
    _fit_model(
        model,
        train_ds,
        val_ds,
        model_path,
        starting_epoch,
        log_name="_unet",
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scheduler_step_per_batch=scheduler_step_per_batch,
    )

    print(
        _col(
            f"Training completed in "
            f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n",
            _C.GREEN,
        )
    )


def train_SwinUNetPP(conf):
    """
    Train a Swin-UNetPP model for bubble segmentation.

    SwinUNet: Vision Transformer-based architecture with window-based self-attention.
    Good for: larger datasets, higher compute budget, maximum accuracy.
    Better than UNet: captures long-range dependencies (entire image context).
    Slower than UNet: quadratic complexity in attention (more compute).

    Note: Same training flow as train_UNet above (data, model, optimizer, loss, fit).
    See train_UNet docstring for detailed explanation of each step.
    """
    global config
    config = conf
    print("Starting training (SwinUNet).")
    start = time.time()

    # Reproducibility seeding
    set_global_seed(getattr(config, "seed", None))

    # Load data and split into train/val/test
    frames = get_all_frames(config)
    train_ds, val_ds, test_ds = create_train_val_datasets(frames)

    # Model checkpoint path (unique timestamp)
    stamp = time.strftime("%Y%m%d-%H%M")
    model_path = os.path.join(config.saved_models_dir, f"{stamp}_{config.model_name}")

    # Resume from checkpoint if requested
    state_path, starting_epoch = _prepare_model_and_logging(model_path)

    # Build SwinUNet model (uses Swin Transformer blocks instead of convolutions)
    model = _build_model_swin()

    # Optional PyTorch 2.0+ compilation
    if getattr(config, "use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Build optimizer, scheduler, and loss function
    optimizer, scheduler, scheduler_step_per_batch = _build_optimizer_and_scheduler(model)
    criterion = get_loss(
        config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))
    )

    # Load checkpoint weights if resuming
    if state_path and os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            print(f"Loaded weights from: {state_path}")
        except Exception as exc:
            print(
                _col(
                    f"Could not load PyTorch weights from {state_path}: {exc}", _C.RED
                )
            )

    # Run training loop (same as UNet, but with different model)
    _fit_model(
        model,
        train_ds,
        val_ds,
        model_path,
        starting_epoch,
        log_name="_swin",
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scheduler_step_per_batch=scheduler_step_per_batch,
    )

    print(
        _col(
            f"Training completed in "
            f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n",
            _C.GREEN,
        )
    )


def train_TerraMind(conf):
    """
    Train a TerraMind-based segmentation model.

    TerraMind: foundation model for geospatial tasks (pretrained on millions of satellite images).
    Good for: any dataset size, want best accuracy with minimal tuning.
    Key advantage: pretrained backbone transfer learning (fast convergence).
    Works with: Sentinel-2, Landsat, custom satellite bands.

    Features:
      - Discriminative learning rates: backbone (slow) vs head (fast)
      - Optional backbone freezing: train only head for stable results
      - Multi-modality support: S2 (4 bands), L8 (8 bands), custom band combinations
      - UperNet decoder: efficient feature fusion from multiple scales

    Note: Same training flow as train_UNet and train_SwinUNetPP.
    TerraMind-specific: uses _make_tm_optimizer_from_config() for discriminative LRs.
    See _build_model_terramind() for backbone freezing and pretrained weight loading.
    """
    global config
    config = conf
    print("Starting training (TerraMind).")
    start = time.time()

    # Reproducibility seeding
    set_global_seed(getattr(config, "seed", None))

    # Load data
    frames = get_all_frames(config)
    train_ds, val_ds, test_ds = create_train_val_datasets(frames)

    # Model checkpoint path
    stamp = time.strftime("%Y%m%d-%H%M")
    model_path = os.path.join(config.saved_models_dir, f"{stamp}_{config.model_name}")

    # Resume from checkpoint if requested
    state_path, starting_epoch = _prepare_model_and_logging(model_path)

    # Build TerraMind model (with pretrained backbone + UperNet decoder)
    model = _build_model_terramind()

    # Optional PyTorch 2.0+ compilation
    if getattr(config, "use_torch_compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Build optimizer with discriminative LRs (backbone slow, head fast)
    optimizer, scheduler, scheduler_step_per_batch = _build_optimizer_and_scheduler(model)

    # Build loss function
    criterion = get_loss(
        config.loss_fn, getattr(config, "tversky_alphabeta", (0.5, 0.5))
    )

    # Load checkpoint weights if resuming
    if state_path and os.path.exists(state_path):
        try:
            state = torch.load(state_path, map_location="cpu")
            if isinstance(state, dict) and "model_state" in state:
                model.load_state_dict(state["model_state"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
            print(f"Loaded weights from: {state_path}")
        except Exception as exc:
            print(
                _col(
                    f"Could not load PyTorch weights from {state_path}: {exc}", _C.RED
                )
            )

    # Run training loop (same as UNet/SwinUNet, but with TerraMind model)
    _fit_model(
        model,
        train_ds,
        val_ds,
        model_path,
        starting_epoch,
        log_name="_tm",
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scheduler_step_per_batch=scheduler_step_per_batch,
    )

    print(
        _col(
            f"Training completed in "
            f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n",
            _C.GREEN,
        )
    )
