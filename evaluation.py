# evaluation.py (PyTorch) - Full-image sliding-window evaluation with BF16 + channels-last
#
# PURPOSE:
# This script runs a trained semantic segmentation model on test images to evaluate performance.
# Unlike training (where the model learns), evaluation/inference means making predictions on
# NEW unseen data without updating weights. The model processes full aerial images using
# a "sliding window" approach: scanning across the image in overlapping patches, making
# predictions patch-by-patch, then blending them to avoid edge artifacts where tiles meet.
#
# Outputs:
#   1. Probability maps (0-1 values showing confidence) + binary masks (0 or 1 after thresholding)
#      saved as georeferenced GeoTIFF files (can be loaded in GIS software)
#   2. Optional uncertainty maps (epistemic + aleatoric) estimated via MC Dropout
#   3. CSV summary with metrics for each checkpoint tested
#
# KEY METRICS EXPLAINED:
#   - Dice: Overlap between predicted and true regions (0-1, higher=better)
#   - IoU: Jaccard index, stricter than Dice, only counts shared pixels (0-1, higher=better)
#   - Accuracy: Percent of correctly classified pixels (but misleading if classes imbalanced)
#   - Sensitivity: True Positive Rate = TP/(TP+FN), how many bubbles did we find? (0-1, higher=better)
#   - Specificity: True Negative Rate = TN/(TN+FP), how many non-bubbles did we correctly skip? (0-1)
#   - F1-score: Harmonic mean of precision & recall, balanced metric (0-1, higher=better)
#   - F-beta: Weighted F1 favoring recall (beta=2) or precision (beta=0.5)
#   - Hausdorff distance: Max pixel distance between predicted and true boundaries (lower=better)
#   - Normalized Surface Distance: Average distance between surfaces, normalized by shape size
#   - Boundary IoU: IoU computed only on boundary pixels, focuses on edge accuracy
#
# MICRO VS MACRO AVERAGING:
#   - MICRO: pool all pixels from all test images, compute metrics once on the global pool
#            (pixel-level metrics: Dice, IoU, accuracy, sensitivity, specificity, F1, F-beta)
#            Best when test images have different foreground prevalence (some bubble-dense, some sparse)
#   - MACRO: compute metrics per-image, then average across images
#            (geometric metrics: Hausdorff, surface distance, boundary IoU)
#            Used for metrics not well-defined on pixel pools (depend on shape, not just pixels)
#
# MC DROPOUT FOR UNCERTAINTY:
#   At test time, normally dropout is OFF (deterministic predictions). MC Dropout enables
#   dropout DURING inference and runs the model multiple times on the same patch. The variation
#   across runs estimates model uncertainty. Two types:
#   - Epistemic: "What if I had different training data?" Variation from model architecture/weights
#   - Aleatoric: "Even with perfect model, pixels are ambiguous" (e.g., blurry boundary)
#   Higher uncertainty = less confident in that prediction.
#
# UPDATE (micro-averaging + robust uncertainty):
# - Pixel-level metrics (Dice/IoU/Acc/Sens/Spec/F1/Fbeta) are now MICRO-AVERAGED by
#   accumulating a global confusion matrix over all test pixels (thresholded at eval_threshold).
# - Geometric metrics (Hausdorff, nominal surface distance, boundary IoU) remain MACRO-AVERAGED
#   across scenes, as micro-averaging is not well-defined for them.
# - MC uncertainty supports Swin stochastic depth by enabling DropPath/StochasticDepth layers
#   during inference. If drop_path=0.0 (or no stochastic layers exist), epistemic uncertainty
#   will correctly collapse towards ~0 while aleatoric remains defined.

from __future__ import annotations

import csv as csv_module
import glob
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio  # Reads/writes georeferenced image files (GeoTIFF), keeps spatial metadata
import torch
import torch.nn as nn
from tqdm import tqdm  # Progress bar for long loops

# ===== Project imports =====
from core.Swin_UNetPP import SwinUNet  # Swin Transformer backbone for segmentation
from core.TerraMind import TerraMind  # Geospatial pretrained backbone
from core.UNet import UNet  # Standard U-Net architecture for segmentation
from core.common.console import _C, _col, _fmt_seconds  # Colored console output helpers
from core.common.data import get_all_frames  # Load all frame data with channel ordering
from core.common.model_utils import (
    _as_probs_from_terratorch,  # Convert logits to probabilities (sigmoid/softmax)
    _as_probs_from_terratorch_logits_first,  # Variant: ensure logits come first in conversion
    _ensure_nchw,  # Reshape to [N,C,H,W] format (standard PyTorch)
    _forward_with_autopad,  # Forward pass with auto-padding for non-standard sizes
    _is_terramind_model,  # Check if model is TerraMind (special handling for logits)
)
from core.frame_info import image_normalize  # Normalize pixel values to mean=0, std=1
from core.losses import (
    Hausdorff_distance,  # Max distance between predicted and true boundaries
    IoU,  # Intersection over Union metric
    accuracy,  # Percent of correctly classified pixels
    boundary_intersection_over_union,  # IoU computed only on boundary pixels
    dice_coef,  # Dice coefficient (also called F1 in segmentation)
    dice_loss,  # 1 - dice_coef (loss function)
    f1_score,  # F1 metric (harmonic mean of precision and recall)
    f_beta,  # F-beta metric, weighted F1 (can favor recall or precision)
    normalized_surface_distance,  # Average distance between surfaces, normalized
    sensitivity,  # True Positive Rate (how many positives did we find?)
    specificity,  # True Negative Rate (how many negatives did we correctly skip?)
)
from core.split_frames import split_dataset  # Load train/val/test split indices

from config.configSwinUnet import *

# ===== Fast execution defaults / mixed precision =====
# Enable TensorFloat-32 (less precision, faster math) for matrix multiply & convolutions
# This trades a tiny bit of accuracy for significant GPU speedup
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # auto-tune GPU kernels for current hardware

# Use BF16 (bfloat16) for inference if GPU supports it (more numerically stable than FP16)
# BF16 keeps the same exponent range as FP32 but with lower mantissa precision
# Fallback to FP16 if BF16 not available
_AMP_DTYPE = (
    torch.bfloat16
    if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    else torch.float16
)


# =====================================================
# Metric accumulator (MICRO for pixel metrics, MACRO for geometric metrics)
# =====================================================
class MetricAccumulator:
    """
    Hybrid dataset-level accumulator for evaluation metrics.

    This class pools statistics across all test images using TWO different averaging strategies:

    MICRO-AVERAGE (pixel-level metrics):
        Pool ALL pixels from ALL test images into one global confusion matrix.
        Then compute: Dice, IoU, accuracy, sensitivity, specificity, F1, F-beta
        Why? When test images have different bubble densities (some sparse, some dense),
        micro-averaging prevents small images from being overweighted. The global confusion
        matrix reflects the true foreground/background balance in the entire test set.
        Think of it like: "If I randomly sampled a pixel from all test images, how often
        would my prediction be correct?" This is especially important when test images
        have imbalanced classes (many background, few bubbles).

    MACRO-AVERAGE (geometric metrics):
        Compute metrics PER-IMAGE, then average the per-image results.
        For: Hausdorff_distance, normalized_surface_distance, boundary_intersection_over_union
        Why? These metrics depend on SHAPE and TOPOLOGY (where objects are), not just
        pixel counts. Micro-averaging doesn't make sense here. Imagine two images:
        Image A has one huge bubble, Image B has 100 tiny bubbles. One large shape and many
        small shapes have very different Hausdorff distances. Macro-averaging (average per-image
        first) respects the fact that each image is an independent test case.
    """

    def __init__(
        self,
        device: torch.device,
        threshold: float = 0.5,  # Probability threshold: predictions >= 0.5 become "bubble" (1)
        fbeta_beta: float = 2.0,  # Beta for F-beta metric (2.0 means recall 4x more important than precision)
    ) -> None:
        self.device = device
        self.threshold = float(threshold)
        self.fbeta_beta = float(fbeta_beta)
        self.reset()

        # Macro-only (geometry/boundary) metrics using training implementations
        # These compute per-image, then will be averaged at the end
        self.macro_metric_fns = {
            "normalized_surface_distance": normalized_surface_distance,
            "Hausdorff_distance": Hausdorff_distance,
            "boundary_intersection_over_union": boundary_intersection_over_union,
        }

    def reset(self) -> None:
        # ===== MICRO: Global confusion matrix =====
        # Confusion matrix counts pixels across ALL test images pooled together
        # TP (True Positive): predicted bubble, actually bubble
        # FP (False Positive): predicted bubble, actually background (false alarm)
        # FN (False Negative): predicted background, actually bubble (missed detection)
        # TN (True Negative): predicted background, actually background (correct rejection)
        self.tp = 0  # Total TP pixels across all test images
        self.fp = 0  # Total FP pixels across all test images
        self.tn = 0  # Total TN pixels across all test images
        self.fn = 0  # Total FN pixels across all test images

        # ===== MACRO: Per-image geometry metrics =====
        self.n_macro = 0  # Count of images processed (for averaging)
        self.macro_sums: Dict[str, float] = {
            "normalized_surface_distance": 0.0,  # Sum of per-image surface distances
            "Hausdorff_distance": 0.0,  # Sum of per-image Hausdorff distances
            "boundary_intersection_over_union": 0.0,  # Sum of per-image boundary IoU values
        }

    def _to_tensors(
        self, y_true_np: np.ndarray, y_prob_np: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert numpy ground truth and probability maps to PyTorch tensors
        in [batch, channels, height, width] format for metric functions.

        Metric functions expect [B,C,H,W] tensors (batch, channels, height, width),
        but our predictions are [H,W] (single image, single channel). This reshapes them.

        Args:
            y_true_np: Ground truth binary mask, shape [H,W], values in [0,1]
            y_prob_np: Predicted probability map, shape [H,W], values in [0,1]

        Returns:
            (y_true, y_prob): Both as [1,1,H,W] torch tensors on GPU/CPU device
        """
        # Convert numpy float32 to torch tensors (CPU first, then move to device)
        y_true = torch.from_numpy(y_true_np.astype(np.float32))
        y_prob = torch.from_numpy(y_prob_np.astype(np.float32))

        # Reshape to [batch, channels, height, width] format
        if y_true.ndim == 2:
            # Input is [H,W], add batch (1) and channel (1) dims: [1,1,H,W]
            y_true = y_true.unsqueeze(0).unsqueeze(0)
            y_prob = y_prob.unsqueeze(0).unsqueeze(0)
        elif y_true.ndim == 3 and y_true.shape[0] == 1:
            # Input is [1,H,W], just add batch dim: [1,1,H,W]
            y_true = y_true.unsqueeze(0)
            y_prob = y_prob.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected GT shape for metrics: {y_true.shape}")

        # Move to GPU (or CPU) for metric computation
        y_true = y_true.to(self.device, non_blocking=True)
        y_prob = y_prob.to(self.device, non_blocking=True)
        return y_true, y_prob

    @staticmethod
    def _as_float(value: Any) -> float:
        """Convert torch tensor or scalar to Python float."""
        if isinstance(value, torch.Tensor):
            # If tensor (e.g., loss value), extract scalar: detach(), move to CPU, convert to item()
            return float(value.detach().mean().cpu().item())
        return float(value)

    def add(self, y_true_np: np.ndarray, y_prob_np: np.ndarray) -> None:
        """
        Accumulate one image's predictions for dataset-level evaluation.

        Args:
            y_true_np: Ground truth binary mask [H,W], values in [0,1]
            y_prob_np: Predicted probability map [H,W], values in [0,1]

        Updates:
            - MICRO confusion matrix: counts TP/FP/FN/TN pixels pooled globally
            - MACRO geometry metrics: stores per-image Hausdorff, surface distance, boundary IoU
        """
        # Ensure all values are in valid range [0,1]
        y_true_np = np.clip(y_true_np, 0.0, 1.0)
        y_prob_np = np.clip(y_prob_np, 0.0, 1.0)

        # ===== MICRO: Global confusion matrix (accumulate pixel counts) =====
        # Convert probabilities to hard binary predictions using threshold
        gt = (y_true_np >= 0.5)  # Ground truth: 1 if bubble, 0 if background
        pr = (y_prob_np >= self.threshold)  # Prediction: 1 if model confident (prob >= 0.5), 0 otherwise

        # Count each cell of the 2x2 confusion matrix
        self.tp += int(np.logical_and(pr, gt).sum())  # Model predicted 1, was correct (TP)
        self.fp += int(np.logical_and(pr, np.logical_not(gt)).sum())  # Model predicted 1, was wrong (FP)
        self.fn += int(np.logical_and(np.logical_not(pr), gt).sum())  # Model predicted 0, was wrong (FN)
        self.tn += int(np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum())  # Model predicted 0, was correct (TN)

        # ===== MACRO: Per-image geometric metrics (one value per image) =====
        # Convert to torch tensors in [1,1,H,W] format for metric function compatibility
        y_true, y_prob = self._to_tensors(y_true_np, y_prob_np)
        y_bin = (y_prob >= self.threshold).float()  # Thresholded binary prediction [1,1,H,W]

        # Compute each geometric metric for this image and add to running sum
        for name, fn in self.macro_metric_fns.items():
            try:
                # Some metrics (like Hausdorff) work with float probabilities
                val = fn(y_true, y_prob)
            except Exception:
                # Others need binary predictions; try with thresholded version
                val = fn(y_true, y_bin)
            # Add this image's metric value to the cumulative sum (will average later)
            self.macro_sums[name] += self._as_float(val)

        # Increment image counter for averaging
        self.n_macro += 1

    def finalize(self) -> Dict[str, float]:
        """
        Compute final dataset-level metrics from accumulated statistics.

        MICRO metrics: Computed from global confusion matrix (all pixels pooled)
        MACRO metrics: Average of per-image values

        Returns:
            Dictionary with keys matching CSV header: dice_coef, IoU, accuracy, etc.
        """
        eps = 1e-9  # Small constant to prevent division by zero

        # Convert confusion matrix counts to floats for arithmetic
        tp = float(self.tp)
        fp = float(self.fp)
        tn = float(self.tn)
        fn = float(self.fn)

        # ===== MICRO: Pixel-level metrics from global confusion matrix =====
        # These formulas use TP/FP/FN/TN counts pooled across all test pixels

        # Dice: How much overlap between prediction and ground truth?
        # Range [0,1], higher=better. Formula: 2*TP / (2*TP + FP + FN)
        # Intuition: If perfect prediction, all pixels match, dice=1. If no overlap, dice=0.
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + eps)

        # IoU (Intersection over Union / Jaccard index): Stricter than Dice
        # Range [0,1], higher=better. Formula: TP / (TP + FP + FN)
        # Intuition: Union is all pixels that are bubble OR predicted-bubble, intersection is both.
        # Penalizes false positives and false negatives equally.
        iou = tp / (tp + fp + fn + eps)

        # Accuracy: Fraction of all pixels predicted correctly
        # Range [0,1], higher=better. Formula: (TP + TN) / (TP + TN + FP + FN)
        # Warning: Misleading if classes imbalanced (e.g., 95% background makes accuracy=0.95 trivial)
        acc = (tp + tn) / (tp + tn + fp + fn + eps)

        # Sensitivity (Recall, True Positive Rate): Of all actual bubbles, how many did we find?
        # Range [0,1], higher=better. Formula: TP / (TP + FN)
        # Intuition: How sensitive is our detector? False negatives are missed bubbles (bad for detecting all bubbles)
        sens = tp / (tp + fn + eps)

        # Specificity (True Negative Rate): Of all actual background pixels, how many did we skip?
        # Range [0,1], higher=better. Formula: TN / (TN + FP)
        # Intuition: How specific? False positives are false alarms (bad for avoiding noise)
        spec = tn / (tn + fp + eps)

        # Precision: Of all predictions we made as "bubble", how many were correct?
        # Formula: TP / (TP + FP)
        # Intuition: High precision = confident in our positive predictions (few false alarms)
        prec = tp / (tp + fp + eps)
        rec = sens  # Recall = Sensitivity

        # F1-score: Harmonic mean of precision and recall
        # Range [0,1], higher=better. Formula: 2 * (prec * rec) / (prec + rec)
        # Intuition: Balanced metric that penalizes being bad at EITHER precision or recall.
        # If precision=1 but recall=0.5, F1 ≈ 0.67 (lower than average). Balanced focus.
        f1 = (2.0 * prec * rec) / (prec + rec + eps)

        # F-beta: Weighted F1, can favor recall (beta>1) or precision (beta<1)
        # beta=2 (here): 4x weight on recall vs precision. Cares more about finding all bubbles.
        # Formula: (1 + beta^2) * (prec * rec) / (beta^2 * prec + rec)
        # Use case: If missing bubbles is worse than false alarms, use high beta.
        b2 = self.fbeta_beta ** 2
        fbeta = (1.0 + b2) * (prec * rec) / (b2 * prec + rec + eps)

        # Build output dict with all MICRO-averaged metrics
        out: Dict[str, float] = {
            "dice_coef": float(dice),
            "dice_loss": float(1.0 - dice),  # Loss version: lower=better (opposite of coef)
            "IoU": float(iou),
            "accuracy": float(acc),
            "sensitivity": float(sens),
            "specificity": float(spec),
            "f1_score": float(f1),
            "f_beta": float(fbeta),
        }

        # ===== MACRO: Geometric metrics averaged per-image =====
        # These metrics depend on SHAPE/TOPOLOGY, not just pixel counts, so averaging per-image is correct
        div = float(max(1, self.n_macro))  # Avoid division by zero if no images processed
        out["normalized_surface_distance"] = float(
            self.macro_sums["normalized_surface_distance"] / div
        )
        out["Hausdorff_distance"] = float(self.macro_sums["Hausdorff_distance"] / div)
        out["boundary_intersection_over_union"] = float(
            self.macro_sums["boundary_intersection_over_union"] / div
        )

        return out


# =====================================================
# Data helpers (reuse training ordering)
# =====================================================
def _list_preprocessed_paths(preprocessed_dir: str) -> List[str]:
    """
    Discover all preprocessed .tif frame files and sort by integer filename.

    Assumes filenames are integers (e.g., "0.tif", "1.tif", ..., "123.tif").
    Sorting ensures consistent ordering during evaluation.
    """
    image_paths = sorted(
        glob.glob(os.path.join(preprocessed_dir, "*.tif")),
        key=lambda f: int(os.path.basename(f)[:-4]),  # Sort by numeric filename
    )
    return image_paths


def _gather_frames_and_test_indices(config) -> Tuple[list, list, list]:
    """
    Load all frame data and retrieve the train/val/test split indices.

    Frames are loaded once to avoid redundant I/O. The split indices (which frames
    go into train/val/test) are read from JSON to match the training split exactly.

    Returns:
        (frames, image_paths, test_idx)
        - frames: List of all Frame objects with images and annotations
        - image_paths: List of paths to preprocessed .tif files (same order as frames)
        - test_idx: List of indices into frames that belong to test set
    """
    # Load all frame objects (includes image data and labels from last band)
    frames = get_all_frames(config)
    # Get corresponding file paths for later use
    image_paths = _list_preprocessed_paths(config.preprocessed_dir)

    # Load the train/val/test split from JSON (created during training)
    print("Reading train-test split from file")
    frames_json = os.path.join(config.preprocessed_dir, "aa_frames_list.json")
    train_idx, val_idx, test_idx = split_dataset(
        frames, frames_json, config.test_ratio, config.val_ratio
    )

    # Print diagnostic info
    print(f"training_frames {len(train_idx)}")
    print(f"validation_frames {len(val_idx)}")
    print(f"testing_frames {len(test_idx)}\n")

    return frames, image_paths, test_idx


# =====================================================
# Model builders (match training constructors)
# =====================================================
def _build_unet(config) -> torch.nn.Module:
    """
    Construct a UNet model with architecture matching training.

    UNet is a classic encoder-decoder architecture for semantic segmentation:
      - Encoder: Downsampling path (reduces spatial size, increases features)
      - Decoder: Upsampling path (increases spatial size, reduces features)
      - Skip connections: Concatenate encoder features to decoder at each level
    This preserves fine details lost during downsampling.

    Args:
        config: Configuration object with channel_list, patch_size, dilation_rate, etc.

    Returns:
        torch.nn.Module: Instantiated UNet model (weights not yet loaded)
    """
    # Input channels = number of spectral bands (e.g., RGB=3, RGB+NIR=4, Sentinel-2=11)
    in_ch = len(getattr(config, "channel_list", []))
    num_classes = int(getattr(config, "num_classes", 1))  # 1 for binary segmentation (bubble/not)

    model = UNet(
        [config.train_batch_size, *config.patch_size, in_ch],  # [batch, height, width, channels]
        num_classes,
        dilation_rate=getattr(config, "dilation_rate", 1),  # Dilated convolutions widen receptive field
        layer_count=getattr(config, "layer_count", 64),  # Base number of filters in first layer
        l2_weight=getattr(config, "l2_weight", 1e-4),  # L2 regularization strength (weight decay)
        dropout=getattr(config, "dropout", 0.0),  # Dropout rate during training (0.0 = no dropout)
    )
    return model



def _build_swin(config) -> torch.nn.Module:
    """
    Construct a SwinUNet model for evaluation with training-matched architecture.

    Swin Transformer is a vision transformer backbone using "shifted windows" attention:
      - Divides the image into windows and applies self-attention within each window (efficient)
      - Shifts windows periodically to allow cross-window communication
      - Hierarchical: features get coarser as you go deeper (like traditional CNNs)
    SwinUNet adds a decoder (like regular UNet) to make it suitable for dense prediction tasks
    like segmentation.

    Args:
        config: Configuration object with swin_base_channels, swin_patch_size, etc.

    Returns:
        torch.nn.Module: Instantiated SwinUNet model (weights not yet loaded)
    """
    # Swin architecture hyperparameters
    base_c = getattr(config, "swin_base_channels", 64)  # Number of hidden channels in first layer
    swin_patch = getattr(config, "swin_patch_size", 16)  # Patch size for initial tokenization (e.g., 16x16 patches)
    swin_window = getattr(config, "swin_window", 7)  # Window size for attention (e.g., 7x7 local windows)

    # Number of input channels (bands)
    in_ch = len(getattr(config, "channels_used", getattr(config, "channel_list", [])))

    model = SwinUNet(
        h=config.patch_size[0],  # Patch height
        w=config.patch_size[1],  # Patch width
        ch=in_ch,  # Input channels
        c=base_c,  # Base channels
        patch_size=swin_patch,  # Swin patch size
        window_size=swin_window,  # Swin window size
    )
    return model


def _build_terramind(config) -> torch.nn.Module:
    """
    Construct a TerraMind model for evaluation using training-matched configuration.

    TerraMind is a geospatial-specific model that:
      - Uses pretrained backbones from satellite imagery datasets (e.g., Sentinel-2)
      - Supports multiple spectral bands (e.g., RGB, NIR, SWIR)
      - Can merge multi-scale or multi-modal predictions
    This leverages transfer learning: starting from weights trained on massive amounts of
    satellite data, we fine-tune on our bubble detection task.

    Args:
        config: Configuration object with TerraMind-specific settings

    Returns:
        torch.nn.Module: Instantiated TerraMind model (weights not yet loaded)
    """
    in_ch = len(getattr(config, "channels_used", getattr(config, "channel_list", [])))
    num_classes = int(getattr(config, "num_classes", 1))  # 1 for binary (bubble/not)
    modality = getattr(config, "modality", "S2")  # Data modality: S2 (Sentinel-2), NAIP, etc.

    # TerraMind backbone and decoder configuration
    tm_backbone = getattr(config, "tm_backbone", None)  # Backbone name (e.g., "ViT", "ResNet")
    tm_decoder = getattr(config, "tm_decoder", "UperNetDecoder")  # Decoder type (upsampling strategy)
    tm_dec_ch = getattr(config, "tm_decoder_channels", 256)  # Number of decoder channels
    tm_indices = getattr(config, "tm_select_indices", None)  # Which bands to select from full set
    tm_bands = getattr(config, "tm_bands", None)  # Band names/indices for this data
    tm_ckpt = getattr(config, "tm_backbone_ckpt_path", None)  # Path to pretrained backbone weights
    tm_merge = getattr(config, "terramind_merge_method", "mean")  # How to merge multi-scale outputs
    tm_size_fallback = getattr(config, "terramind_size", "base")  # Model size: tiny/small/base/large

    def _parse_size_from_backbone(
        s: Optional[str], default_size: str = "base"
    ) -> Tuple[Optional[str], str]:
        """
        Extract model size token (tiny/small/base/large) from backbone string.

        TerraMind can specify size in the backbone name (e.g., "terramind_base") or
        separately in config. This parses the backbone string to extract the size.
        """
        if not s:
            return None, default_size
        lower = s.lower()
        if lower.startswith("terramind"):
            # Backbone name contains size token, extract it
            size = (
                "large"
                if "large" in lower
                else (
                    "base"
                    if "base" in lower
                    else (
                        "small"
                        if "small" in lower
                        else ("tiny" if "tiny" in lower else default_size)
                    )
                )
            )
            return s, size
        if lower in {"tiny", "small", "base", "large"}:
            # String is just the size token itself
            return None, lower
        return None, default_size

    # Parse backbone name to extract or override model size
    backbone_override, tm_size = _parse_size_from_backbone(tm_backbone, tm_size_fallback)

    # Instantiate TerraMind with all configuration
    model = TerraMind(
        in_channels=in_ch,  # Number of input spectral bands
        num_classes=num_classes,  # Output classes (1 for bubble detection)
        modality=modality,  # Satellite data modality (determines channel order/meaning)
        tm_size=tm_size,  # Model size (affects width/depth)
        merge_method=tm_merge,  # Multi-scale merge strategy (e.g., "mean" averages outputs)
        pretrained=True,  # Load pretrained weights from satellite imagery
        ckpt_path=tm_ckpt,  # Optional override path to specific checkpoint
        indices_override=tm_indices,  # Band selection (e.g., use only RGB, skip thermal)
        bands_override=tm_bands,  # Band metadata override
        decoder=tm_decoder,  # Decoder type (upsampling module)
        decoder_channels=tm_dec_ch,  # Number of channels in decoder
        decoder_kwargs={},  # Additional decoder options
        backbone=backbone_override,  # Backbone override if specified
        rescale=True,  # Rescale outputs to match input size
    )
    # Mark as TerraMind so logits are decoded correctly later
    setattr(model, "_is_terramind", True)
    return model


# =====================================================
# Checkpoint loading
# =====================================================
def _load_model_from_checkpoint(
    model: torch.nn.Module, ckpt_path: str, device: torch.device
) -> torch.nn.Module:
    """
    Load trained weights from a checkpoint file into the model.

    A checkpoint is a saved snapshot of model weights after training. During evaluation,
    we load these weights to use the trained model for inference (prediction).

    Args:
        model: Freshly instantiated (unweighted) model architecture
        ckpt_path: Path to checkpoint file (usually *.pt)
        device: GPU or CPU device

    Returns:
        model: Same model with weights loaded, in eval mode (dropout disabled, batch norm frozen)
    """
    # Move model to device and set to evaluation mode
    # eval() disables layers like Dropout and uses fixed batch norm statistics (computed during training)
    model = model.to(device=device).eval()

    # Load checkpoint file (could be raw state dict or wrapped in a dict with "model_state" key)
    state = torch.load(ckpt_path, map_location="cpu")  # Load to CPU first (memory efficient)

    # Extract state dict (the actual weights) from checkpoint
    if isinstance(state, dict) and "model_state" in state:
        # Checkpoint is wrapped: {"model_state": {...}, "optimizer_state": {...}, ...}
        state_dict = state["model_state"]
    else:
        # Checkpoint is just the state dict directly
        state_dict = state

    # Load weights into model
    try:
        # Try strict=True first: all model keys must match checkpoint keys exactly
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        # If strict fails (e.g., missing keys), fall back to non-strict
        # non-strict allows checkpoint to have extra keys or model to have extra keys
        # (useful if checkpoint was from a slightly different model version)
        model.load_state_dict(state_dict, strict=False)
    return model


# =====================================================
# Inference helpers (sliding-window)
# =====================================================
def _select_eval_channel(prob_nchw: torch.Tensor, config) -> torch.Tensor:
    """
    Select which output channel to evaluate (for multi-class models).

    For binary segmentation (bubble/no-bubble), there's 1 output channel.
    For multi-class (e.g., bubble/water/land), we can have multiple channels.
    This function picks which channel to compute metrics on.

    Args:
        prob_nchw: Probability tensor, shape [B,C,H,W]
        config: Config object specifying metrics_class (which channel to evaluate)

    Returns:
        Selected channel as [B,1,H,W]
    """
    # Ensure tensor is in [batch, channels, height, width] format
    prob_nchw = _ensure_nchw(prob_nchw).float()

    if prob_nchw.shape[1] > 1:
        # Multiple channels: select the one specified in config (default=1, the "positive" class)
        # This is typically the foreground (bubble) channel, not the background channel
        cls_idx = int(getattr(config, "metrics_class", 1))
        cls_idx = max(0, min(cls_idx, prob_nchw.shape[1] - 1))  # Clamp to valid range
        return prob_nchw[:, cls_idx : cls_idx + 1]

    # Single channel: just return as-is
    return prob_nchw


def _enable_stochastic_inference(model: torch.nn.Module) -> int:
    """
    Enable stochastic layers (dropout, DropPath) for MC Dropout inference.

    MC Dropout = Monte Carlo Dropout: Run inference multiple times WITH dropout ON.
    Each run produces slightly different predictions due to random dropout.
    Variation across runs = model uncertainty (epistemic uncertainty).

    Normally at test time, dropout is OFF (deterministic predictions).
    This function selectively enables ONLY the stochastic/dropout layers,
    keeping batch norm and other layers in eval mode.

    This is important for:
      - Epistemic uncertainty: "How much does the model vary due to different samples?"
      - Calibration: Can estimate confidence in predictions

    Stochastic layers to enable:
      - nn.Dropout / Dropout2d / Dropout3d: Standard dropout
      - DropPath / StochasticDepth: Randomly drops residual connections (used in Vision Transformers)

    Args:
        model: Trained model to enable stochastic inference for

    Returns:
        Number of stochastic modules enabled
    """
    n_enabled = 0

    for m in model.modules():
        # Enable standard dropout layers
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()  # Set to train mode so dropout is active
            n_enabled += 1
            continue

        # Enable stochastic depth / DropPath layers (common in Vision Transformers like Swin)
        # These layers aren't in nn module, so we check class name
        cls_name = m.__class__.__name__
        if cls_name in {"DropPath", "StochasticDepth"}:
            m.train()  # Activate stochastic behavior
            n_enabled += 1

    return n_enabled


def _infer_full_image(
    model: torch.nn.Module, frame, device: torch.device, config
) -> np.ndarray | Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run sliding-window inference on a full aerial image.

    This is the core inference loop for evaluation. The image is typically too large
    to fit in GPU memory, so we:
      1. Divide it into overlapping patches (sliding window)
      2. Run model on each patch independently
      3. Blend predictions where patches overlap (weighted average to smooth edges)

    Why overlapping patches?
      When tiles meet exactly, the model sees different context at tile boundaries,
      producing artifacts (visible "seams"). Overlap + blending smooths these out.
      Think of it like: if a bubble straddles two tiles, both see it from slightly
      different angles; averaging the two predictions is more robust.

    MC Dropout (if enabled):
      Runs each patch N times with different dropout masks. Across runs, a pixel in
      an ambiguous area (e.g., blurry bubble boundary) gets varying predictions,
      indicating uncertainty.

    Args:
        model: Trained segmentation model in eval mode
        frame: Frame object with .img [H,W,C] and .annotations [H,W]
        device: GPU/CPU device
        config: Configuration object with patch size, stride, MC dropout settings

    Returns:
        If eval_mc_dropout=False: prob_map [H,W] (probabilities 0-1)
        If eval_mc_dropout=True: (prob_map, epistemic_unc, aleatoric_unc) all [H,W]
    """
    # ===== MC Dropout configuration =====
    # MC Dropout runs inference multiple times with dropout ON to estimate uncertainty
    use_mc_dropout = bool(getattr(config, "eval_mc_dropout", True))
    mc_samples = int(getattr(config, "eval_mc_samples", 8))  # Number of forward passes per patch
    mc_samples = max(1, mc_samples)

    # ===== Load and prepare image =====
    x_full = frame.img  # Image array [H,W,C] where C = number of spectral bands
    img_h, img_w = x_full.shape[:2]

    # Select the spectral bands to use (e.g., RGB only, or RGB+NIR)
    channels_used = getattr(config, "channels_used", getattr(config, "channel_list", []))
    k = len(channels_used)

    # Validate that frame has enough bands
    if x_full.shape[2] < k:
        raise RuntimeError(
            f"Frame has {x_full.shape[2]} channels but config.channels_used/channel_list "
            f"requires {k}."
        )

    # Crop to selected channels
    x_full = x_full[:, :, :k]

    # ===== Sliding window configuration =====
    patch_h, patch_w = int(config.patch_size[0]), int(config.patch_size[1])

    # Stride: How far to move window each step. If stride < patch_size, patches overlap.
    # Overlap = patch_size - stride. Larger overlap = more blending = smoother but slower.
    eval_stride = getattr(config, "eval_patch_stride", None)
    if eval_stride is None:
        # No overlap: stride = patch size (non-overlapping tiles)
        stride_h, stride_w = patch_h, patch_w
    else:
        # Parse stride from config (can be single int or [height, width])
        if isinstance(eval_stride, int):
            stride_h = stride_w = int(eval_stride)
        else:
            stride_h, stride_w = int(eval_stride[0]), int(eval_stride[1])

    # ===== Pad image to integer multiple of patch size =====
    # Why? The sliding window expects an image size that divides evenly by patch size.
    # If image is 256x256 and patch is 64x64, we get 4x4 patches (good).
    # If image is 250x250, we need to pad to 256x256 first.
    pad_h = (patch_h - (img_h % patch_h)) if (img_h % patch_h) != 0 else 0
    pad_w = (patch_w - (img_w % patch_w)) if (img_w % patch_w) != 0 else 0

    # Reflect padding: Copy edge pixels reflected (reduces boundary artifacts)
    # E.g., [..., A, B, C] -> [..., A, B, C, B, A] (reflect across C)
    x_padded = np.pad(
        x_full,
        ((0, pad_h), (0, pad_w), (0, 0)),  # Pad height, width, don't pad channels
        mode="reflect",
    )
    pad_h_total, pad_w_total = img_h + pad_h, img_w + pad_w

    # Optional normalization (disabled here to match training behavior)
    # Normalization: Subtract mean, divide by std, to center data around 0 with scale ~1
    x_padded = image_normalize(x_padded, axis=(0, 1))

    # ===== Prediction blending accumulators =====
    # These accumulate predictions from overlapping patches, weighted by overlap count
    prob_accum = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)  # Sum of prob * weight
    weight_accum = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)  # Sum of weights

    if use_mc_dropout:
        # Store uncertainty estimates (will be averaged later)
        epi_accum = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)  # Epistemic uncertainty
        alea_accum = np.zeros((pad_h_total, pad_w_total), dtype=np.float32)  # Aleatoric uncertainty
    else:
        epi_accum = alea_accum = None  # type: ignore[assignment]

    # ===== Batch configuration =====
    # Process patches in batches to maximize GPU throughput
    # Use training batch size to match how model was trained
    batch_size = int(getattr(config, "train_batch_size", 1))
    batch_size = max(1, batch_size)

    num_classes = int(getattr(config, "num_classes", 1))  # Number of output classes

    # ===== Set model to evaluation mode =====
    model.eval()  # Disable dropout, use frozen batch norm statistics

    if use_mc_dropout:
        # Enable stochastic layers only (keep batch norm frozen)
        n_stoch = _enable_stochastic_inference(model)
        # Warn if no stochastic layers found (can happen if drop_path=0.0)
        if n_stoch == 0 and not getattr(model, "_warned_no_stochastic", False):
            print(
                _col(
                    "[EVAL] MC dropout requested but no Dropout/DropPath layers were enabled. "
                    "Epistemic uncertainty may be ~0 (expected if drop_path=0.0).",
                    _C.YELLOW,
                )
            )
            setattr(model, "_warned_no_stochastic", True)

    def _forward_probs(x_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through model, return probabilities [B,C,H,W].

        Most models output LOGITS (raw values, not normalized). This function:
          1. Runs the model forward pass
          2. Converts logits to probabilities via sigmoid (binary) or softmax (multi-class)

        Logits vs Probabilities:
          - Logits: Raw network output (-inf to +inf), interpreted as confidence
          - Probabilities: Normalized to [0,1], sum to 1 across classes
          For segmentation (per-pixel classification), we apply sigmoid (binary)
          or softmax (multi-class) to get interpretable probabilities.

        Args:
            x_t: Input batch [B,C,H,W]

        Returns:
            Probabilities tensor [B,C,H,W], values in [0,1]
        """
        # Forward pass: run data through model
        if _is_terramind_model(model):
            # TerraMind with auto-padding for variable sizes
            y_raw = _forward_with_autopad(model, x_t)
        else:
            # Standard forward
            y_raw = model(x_t)

        # Convert to probabilities
        if getattr(model, "_returns_probabilities", False):
            # Model already outputs probabilities (rare)
            prob_full = _ensure_nchw(y_raw).float()
        elif _is_terramind_model(model):
            # TerraMind model: convert logits to probabilities
            # (uses special logits format, so use specific converter)
            prob_full = _as_probs_from_terratorch_logits_first(
                y_raw, num_classes=num_classes
            )
        else:
            # Standard model (UNet, Swin): convert logits to probabilities
            prob_full = _as_probs_from_terratorch(y_raw, num_classes=num_classes)

        return prob_full

    # ===== MAIN SLIDING WINDOW LOOP =====
    # torch.no_grad(): Disable gradient computation during inference
    # Why? Gradients are only needed during training (backward pass). Inference doesn't need them.
    # Disabling saves memory and speeds up computation.
    with torch.no_grad():
        # Iterate over patch positions (sliding window)
        # y0, x0 are top-left corner coordinates of patch
        for y0 in range(0, pad_h_total - patch_h + 1, stride_h):
            y1 = y0 + patch_h
            for x0 in range(0, pad_w_total - patch_w + 1, stride_w):
                x1 = x0 + patch_w

                # ===== Prepare patch =====
                # Extract patch from padded image
                patch_np = x_padded[y0:y1, x0:x1, :]  # [H,W,C] numpy array

                # Convert to torch tensor [1,C,H,W] (add batch dimension)
                # Transpose: [H,W,C] -> [C,H,W] (PyTorch expects channels first)
                patch_t = torch.from_numpy(
                    np.transpose(patch_np.astype(np.float32), (2, 0, 1))
                ).unsqueeze(0)  # [1,C,H,W]

                # Move to GPU and convert to "channels_last" memory format
                # Channels_last (NHWC) can be faster on some GPUs (especially modern ones)
                patch_t = patch_t.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )

                # ===== Batch repetition =====
                # If batch_size > 1, replicate patch to fill batch (for model compatibility)
                # Some models expect batch > 1; we replicate the same patch to fill it
                if batch_size > 1:
                    x_t = (
                        patch_t.repeat(batch_size, 1, 1, 1)
                        .contiguous(memory_format=torch.channels_last)
                    )
                else:
                    x_t = patch_t

                # ===== Inference (with mixed precision) =====
                # Autocast: Use lower precision (BF16) where safe, FP32 elsewhere
                # Trades tiny accuracy loss for major speed/memory gain (~2x speedup)
                with torch.cuda.amp.autocast(
                    enabled=torch.cuda.is_available(), dtype=_AMP_DTYPE
                ):
                    if not use_mc_dropout:
                        # ===== Standard inference (deterministic) =====
                        prob_full = _forward_probs(x_t)  # [1,C,H,W] probabilities
                        prob_sel = _select_eval_channel(prob_full, config)  # [1,1,H,W]

                        # Squeeze out batch and channel dims, convert to numpy [H,W]
                        prob_patch = (
                            prob_sel[0:1]  # Keep batch dim [1,1,H,W]
                            .squeeze(0)  # Remove batch [1,H,W]
                            .squeeze(0)  # Remove channel [H,W]
                            .clamp(0.0, 1.0)  # Ensure values in [0,1] (numerical safety)
                            .detach()  # Detach from computation graph (no grad)
                            .cpu()  # Move to CPU before converting to numpy
                            .numpy()  # Convert to numpy
                        )

                        # Accumulate prediction (will average later)
                        prob_accum[y0:y1, x0:x1] += prob_patch
                        weight_accum[y0:y1, x0:x1] += 1.0  # Count how many patches touched each pixel

                    else:
                        # ===== MC Dropout inference (stochastic) =====
                        # Run multiple forward passes with different dropout masks
                        mc_patches: List[np.ndarray] = []

                        for _ in range(mc_samples):
                            # Each forward pass produces slightly different predictions due to dropout
                            prob_full = _forward_probs(x_t)  # [1,C,H,W]
                            prob_sel = _select_eval_channel(prob_full, config)  # [1,1,H,W]

                            prob_patch_sample = (
                                prob_sel[0:1]
                                .squeeze(0)
                                .squeeze(0)
                                .clamp(0.0, 1.0)
                                .detach()
                                .cpu()
                                .numpy()
                            )
                            mc_patches.append(prob_patch_sample)

                        # Stack samples: [S,H,W] where S = mc_samples
                        mc_stack = np.stack(mc_patches, axis=0).astype(np.float32)

                        # ===== Uncertainty estimation =====
                        # From MC Dropout samples, compute two types of uncertainty:

                        # Mean prediction across MC samples
                        mean_patch = mc_stack.mean(axis=0).astype(np.float32)

                        # Aleatoric (data) uncertainty: How ambiguous is each pixel?
                        # Formula: E[p(1-p)] = expected variance of Bernoulli with mean p
                        # High when p near 0.5 (ambiguous), low when p near 0 or 1 (certain)
                        # Intuition: If the true label is "bubble 60% of time, background 40%",
                        # even a perfect model would have uncertainty. This captures that.
                        alea_patch = np.mean(mc_stack * (1.0 - mc_stack), axis=0).astype(
                            np.float32
                        )

                        # Epistemic (model) uncertainty: How much do MC samples disagree?
                        # Formula: Var[p] = E[p^2] - (E[p])^2
                        # High when samples vary (model unsure), low when consistent (model sure)
                        # Intuition: If MC dropout produces very different predictions across runs,
                        # the model is uncertain (needs more training data, or the input is inherently ambiguous)
                        second_moment = np.mean(mc_stack ** 2, axis=0)
                        epi_patch = (second_moment - mean_patch ** 2).astype(np.float32)

                        # Accumulate predictions and uncertainties
                        prob_accum[y0:y1, x0:x1] += mean_patch
                        epi_accum[y0:y1, x0:x1] += epi_patch
                        alea_accum[y0:y1, x0:x1] += alea_patch
                        weight_accum[y0:y1, x0:x1] += 1.0

    # ===== Blend overlapping patches =====
    # Normalize by weight to average where patches overlap
    weight_accum = np.maximum(weight_accum, 1e-6)  # Avoid division by zero
    prob_full_padded = np.clip(prob_accum / weight_accum, 0.0, 1.0)

    # Remove padding added earlier (crop back to original image size)
    prob_full = prob_full_padded[:img_h, :img_w]

    if not use_mc_dropout:
        # Return just the probability map
        return prob_full

    # ===== Finalize uncertainty maps =====
    # Average uncertainty estimates (weighted by overlap count)
    epi_full = (epi_accum / weight_accum)[:img_h, :img_w]
    alea_full = (alea_accum / weight_accum)[:img_h, :img_w]

    return prob_full, epi_full, alea_full


# =====================================================
# Batch processing helper
# =====================================================
def _process_image_batch(
    batch_frames: List[Any],
    batch_im_fps: List[str],
    model: nn.Module,
    device: torch.device,
    config: Any,
    thr: float,
    use_mc_dropout: bool,
    out_dir: str,
) -> List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Process a batch of test images: infer, threshold, save GeoTIFFs, return metrics data.

    For each image:
      1. Run sliding-window inference to get probability map
      2. Threshold at 0.5 to get binary ground truth (annotation is 0-255 or 0-1)
      3. Save GeoTIFF outputs: hard binary mask, probability map, optionally uncertainty maps
         GeoTIFF = georeferenced TIFF, preserves spatial coordinates from original raster
      4. Return (gt, prob, epistemic_unc, aleatoric_unc) for metric computation

    Args:
        batch_frames: List of Frame objects with .img and .annotations
        batch_im_fps: Paths to original .tif files (for reading metadata like CRS, transform)
        model: Trained segmentation model
        device: GPU/CPU
        config: Configuration
        thr: Probability threshold (e.g., 0.5) to convert probs to binary
        use_mc_dropout: Whether to compute uncertainty maps
        out_dir: Directory to save GeoTIFF outputs

    Returns:
        List of (gt, prob, epi_map, alea_map) tuples for each image
    """
    results = []

    for frame, im_fp in zip(batch_frames, batch_im_fps):
        # ===== Inference =====
        # Run sliding-window inference to get probability map [H,W]
        if use_mc_dropout:
            # Get probabilities + uncertainty maps
            prob, epi_map, alea_map = _infer_full_image(
                model, frame, device, config
            )
        else:
            # Get just probabilities
            prob = _infer_full_image(model, frame, device, config)
            epi_map = None
            alea_map = None

        # ===== Load ground truth =====
        # annotations: usually 0=background, 1=bubble, or 0-255 uint8
        gt = frame.annotations.astype(np.float32)
        if gt.max() > 1.5:
            # If values > 1.5, assume they're in 0-255 range, scale to [0,1]
            gt = gt / 255.0
        gt = np.clip(gt, 0.0, 1.0)  # Ensure [0,1] range

        # ===== Create binary prediction mask =====
        # Convert soft probabilities to hard 0/1 predictions using threshold
        pred_bin = (prob >= thr).astype(np.uint8)

        # ===== Save GeoTIFF outputs =====
        # GeoTIFF preserves spatial metadata (CRS, transform) from original image
        # This allows loading predictions in GIS software with correct coordinates

        with rasterio.open(im_fp) as src:
            # Read metadata from original image file
            base_profile = src.profile.copy()
            base_profile.update(
                count=1,  # Single-band outputs
                compress="LZW",  # LZW compression (lossless)
                crs=src.crs,  # Coordinate reference system (e.g., UTM, lat/lon)
                transform=src.transform,  # Affine transform (pixel to world coordinates)
                width=src.width,
                height=src.height,
            )

            # ===== Save hard binary mask =====
            # uint8: 0=background, 1=bubble
            mask_profile = base_profile.copy()
            mask_profile["dtype"] = "uint8"
            out_fp = os.path.join(out_dir, os.path.basename(im_fp))
            with rasterio.open(out_fp, "w", **mask_profile) as dst:
                dst.write(pred_bin, 1)  # Write band 1

            # ===== Save probability mask =====
            # float32: raw probabilities [0,1] (or [0,255] after scaling)
            # Some software expects [0,255], so scale probabilities
            prob_profile = base_profile.copy()
            prob_profile["dtype"] = "float32"
            stem, ext = os.path.splitext(os.path.basename(im_fp))
            prob_fp = os.path.join(out_dir, f"{stem}_prob{ext}")
            with rasterio.open(prob_fp, "w", **prob_profile) as dst:
                # Scale probs [0,1] to [0,255] for visualization in GIS tools
                dst.write((prob * 255.0).astype(np.float32), 1)

            # ===== Save uncertainty maps (if MC Dropout enabled) =====
            # Epistemic: Model uncertainty (disagreement across MC samples)
            # Aleatoric: Data uncertainty (inherent ambiguity)
            if use_mc_dropout and epi_map is not None and alea_map is not None:
                unc_profile = base_profile.copy()
                unc_profile["dtype"] = "float32"

                stem, ext = os.path.splitext(os.path.basename(im_fp))
                epi_fp = os.path.join(out_dir, f"{stem}_epistemic{ext}")
                alea_fp = os.path.join(out_dir, f"{stem}_aleatoric{ext}")

                with rasterio.open(epi_fp, "w", **unc_profile) as dst:
                    # Epistemic uncertainty: How much do MC samples disagree?
                    # Higher = model is more uncertain (needs more data, or input is hard)
                    dst.write(epi_map.astype(np.float32), 1)
                with rasterio.open(alea_fp, "w", **unc_profile) as dst:
                    # Aleatoric uncertainty: Inherent ambiguity in data
                    # Higher = pixel is inherently ambiguous (e.g., fuzzy boundary)
                    dst.write(alea_map.astype(np.float32), 1)

        # ===== Return results for metric computation =====
        results.append((gt, prob, epi_map, alea_map))

    return results


# =====================================================
# Checkpoint discovery
# =====================================================
def _find_all_checkpoints(folder: str) -> List[str]:
    """
    Find all checkpoint files in a folder and sort by modification time.

    A checkpoint is a saved snapshot of model weights (typically *.pt file).
    We discover and sort by mtime so we evaluate checkpoints in training order
    (oldest = earlier training iteration, newest = latest).

    Args:
        folder: Directory containing checkpoint files

    Returns:
        List of checkpoint file paths, sorted by modification time
    """
    # Search for common checkpoint filename patterns
    pats = [
        os.path.join(folder, "*.pt"),  # Standard PyTorch checkpoint
        os.path.join(folder, "*.weights.pt"),  # Named checkpoints
        os.path.join(folder, "*.raw.weights.pt"),  # Raw weights variant
    ]
    fps: List[str] = []
    for p in pats:
        fps.extend(glob.glob(p))

    # Remove duplicates (in case patterns overlap) and sort by modification time
    fps = sorted(list(set(fps)), key=lambda f: os.stat(f).st_mtime)
    return fps


# =====================================================
# CSV writing (append for all models)
# =====================================================
def _append_results_row(csv_path: str, header: List[str], row: List[str]) -> None:
    """
    Append one row of metrics to CSV file (create file/header if needed).

    The CSV file accumulates results from evaluating multiple checkpoints.
    Each row = metrics for one checkpoint. Appending allows resuming
    evaluation if interrupted.

    Args:
        csv_path: Path to CSV file (e.g., "results/evaluation_unet.csv")
        header: Column names (e.g., ["checkpoint_path", "dice_coef", "IoU", ...])
        row: Values for this row (one per column)
    """
    # Check if file is empty or doesn't exist (need to write header)
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "a") as f:
        if write_header:
            # Write header row first time
            f.write(",".join(header) + "\n")
        # Append data row
        f.write(",".join(map(str, row)) + "\n")


# =====================================================
# Core evaluation loop
# =====================================================
def _evaluate_arch(config, arch: str = "unet") -> None:
    """
    Master evaluation function: run inference on all test images for all checkpoints.

    This is the main evaluation routine:
      1. Load all test images and ground truth once
      2. Discover all checkpoints for the given architecture
      3. For each checkpoint:
         a. Load model weights
         b. Run sliding-window inference on all test images
         c. Compute dataset-level metrics (MICRO pixel-level, MACRO geometric)
         d. Save prediction GeoTIFFs (hard masks, probability maps, uncertainty)
         e. Record metrics to CSV
      4. Print summary table

    Metrics (brief recap):
      - MICRO pixel metrics: Dice, IoU, Acc, Sensitivity, Specificity, F1, F-beta
        (computed from global confusion matrix pooling all test pixels)
      - MACRO geometric metrics: Hausdorff distance, surface distance, boundary IoU
        (computed per-image, then averaged)
      - Uncertainty: mean epistemic + mean aleatoric (from MC Dropout, if enabled)

    Args:
        config: Configuration object with paths and hyperparameters
        arch: Architecture type ("unet", "swin", "tm" = TerraMind)
    """
    print(f"{_C.CYAN}Starting evaluation for architecture:{arch}{_C.RESET}")
    print(f"{_C.YELLOW}MAKE SURE TO USE FULL AREAS FROM PREPROCESSED DIR{_C.RESET}")
    print(
        f"{_C.YELLOW}CURRENTLY USING THIS PATH AS INPUT: {config.preprocessed_dir}{_C.RESET}\n"
    )

    # ===== Device setup =====
    # Use specified GPU, or fall back to CPU if not available
    selected_gpu = getattr(config, "selected_gpu", 0)
    if selected_gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{selected_gpu}")
        torch.cuda.set_device(device)

    # ===== Load test data once =====
    # All test images are loaded upfront so we don't reload them for each checkpoint
    frames, image_paths, test_idx = _gather_frames_and_test_indices(config)
    print(f"Testing frames: {len(test_idx)}")

    # ===== Discover all checkpoints =====
    # Find all saved model checkpoints to evaluate
    ckpts = _find_all_checkpoints(config.saved_models_dir)
    print(f"Found {len(ckpts)} checkpoint(s).\n")

    # ===== CSV output configuration =====
    # Results are written to CSV (one row per checkpoint)
    csv_path = os.path.join(config.results_dir, f"evaluation_{arch}.csv")
    header = [
        "run_name",  # Name of training run
        "checkpoint_path",  # Path to checkpoint file
        # ===== MICRO pixel metrics =====
        "dice_coef",  # Dice coefficient (overlap metric)
        "dice_loss",  # 1 - dice_coef
        "IoU",  # Intersection over Union (stricter than Dice)
        "accuracy",  # Overall correctness (can be misleading with imbalanced classes)
        "sensitivity",  # True Positive Rate (how many bubbles found?)
        "specificity",  # True Negative Rate (how many background skipped?)
        "f1_score",  # Harmonic mean of precision and recall
        "f_beta",  # F-beta metric (can favor recall or precision)
        # ===== MACRO geometric metrics =====
        "normalized_surface_distance",  # Average distance between predicted and true boundaries
        "Hausdorff_distance",  # Maximum distance between surfaces
        "boundary_intersection_over_union",  # IoU computed only on boundary pixels
        # ===== Uncertainty estimates =====
        "mean_epistemic_uncertainty",  # Avg model uncertainty (MC Dropout variance)
        "mean_aleatoric_uncertainty",  # Avg data uncertainty (inherent ambiguity)
        "elapsed",  # Wall-clock time for this checkpoint
    ]

    results_written = 0

    # ===== Evaluation configuration =====
    thr = float(getattr(config, "eval_threshold", 0.5))  # Probability threshold for binary prediction
    use_mc_dropout = bool(getattr(config, "eval_mc_dropout", True))  # Enable MC Dropout for uncertainty
    fbeta_beta = float(getattr(config, "eval_fbeta_beta", 2.0))  # Beta for F-beta metric (2.0 = favor recall)

    # ===== MAIN LOOP: Evaluate each checkpoint =====
    for ckpt_path in ckpts:
        try:
            print(f"\n{'='*80}")
            print(f"{_C.GREEN}[EVAL]{_C.RESET} model={arch}  ckpt={ckpt_path}")
            base = os.path.basename(ckpt_path)
            out_dir = os.path.join(config.results_dir, base.replace(".pt", ""))
            print(f"{_C.YELLOW}[EVAL]{_C.RESET} saving masks -> {out_dir}")
            os.makedirs(out_dir, exist_ok=True)

            # ===== Build and load model =====
            # Construct model architecture (unweighted)
            if arch == "unet":
                model = _build_unet(config)
            elif arch == "swin":
                model = _build_swin(config)
            elif arch == "tm":
                model = _build_terramind(config)
            else:
                raise NotImplementedError(f"Unknown arch: {arch}")

            # Load trained weights from checkpoint and set to eval mode
            model = _load_model_from_checkpoint(model, ckpt_path, device).eval()

            # ===== Inference and metric accumulation =====
            # Accumulator pools metrics across all test images
            accum = MetricAccumulator(
                device=device, threshold=thr, fbeta_beta=fbeta_beta
            )
            t0 = time.time()

            # Uncertainty aggregation
            sum_epistemic = 0.0
            sum_aleatoric = 0.0
            unc_pixel_count = 0

            # ===== Process test images in batches =====
            # Batch processing balances GPU utilization with memory usage
            batch_size_images = 8  # Process 8 images at a time
            n_test = len(test_idx)

            # Progress bar for long evaluation
            pbar = tqdm(total=n_test, desc=f"Predicting ({arch})")

            # Iterate over batches of test images
            for batch_start in range(0, n_test, batch_size_images):
                batch_end = min(batch_start + batch_size_images, n_test)

                # ===== Gather batch data =====
                # Collect frames and paths for this batch
                batch_frames = []
                batch_im_fps = []
                for i in range(batch_start, batch_end):
                    idx = test_idx[i]
                    batch_frames.append(frames[idx])
                    batch_im_fps.append(image_paths[idx])

                # ===== Process batch (inference, threshold, save) =====
                # Returns (gt, prob, epi_map, alea_map) for each image
                batch_results = _process_image_batch(
                    batch_frames=batch_frames,
                    batch_im_fps=batch_im_fps,
                    model=model,
                    device=device,
                    config=config,
                    thr=thr,
                    use_mc_dropout=use_mc_dropout,
                    out_dir=out_dir,
                )

                # ===== Accumulate metrics from batch =====
                for gt, prob, epi_map, alea_map in batch_results:
                    # Add this image's predictions to global confusion matrix
                    accum.add(gt, prob)

                    # Aggregate uncertainty estimates if available
                    if use_mc_dropout and epi_map is not None and alea_map is not None:
                        # Only sum finite values (skip NaNs)
                        epi_finite = np.isfinite(epi_map)
                        alea_finite = np.isfinite(alea_map)

                        sum_epistemic += float(np.nansum(epi_map[epi_finite]))
                        sum_aleatoric += float(np.nansum(alea_map[alea_finite]))
                        unc_pixel_count += int(epi_map.size)

                    pbar.update(1)

            pbar.close()

            # ===== Finalize metrics =====
            # Compute pixel-level and geometric metrics from accumulators
            # Compute mean uncertainty across all test pixels
            if use_mc_dropout and unc_pixel_count > 0:
                mean_epistemic = sum_epistemic / float(unc_pixel_count)
                mean_aleatoric = sum_aleatoric / float(unc_pixel_count)
            else:
                # No uncertainty data available
                mean_epistemic = float("nan")
                mean_aleatoric = float("nan")

            # Get final metrics dict (all keys formatted as float)
            metrics = accum.finalize()
            elapsed = _fmt_seconds(time.time() - t0)

            # ===== Write results to CSV =====
            row = [
                getattr(config, "run_name", "run"),  # Experiment name
                ckpt_path,  # Checkpoint path
                # Pixel metrics (MICRO-averaged)
                f"{metrics['dice_coef']:.6f}",
                f"{metrics['dice_loss']:.6f}",
                f"{metrics['IoU']:.6f}",
                f"{metrics['accuracy']:.6f}",
                f"{metrics['sensitivity']:.6f}",
                f"{metrics['specificity']:.6f}",
                f"{metrics['f1_score']:.6f}",
                f"{metrics['f_beta']:.6f}",
                # Geometric metrics (MACRO-averaged)
                f"{metrics['normalized_surface_distance']:.6f}",
                f"{metrics['Hausdorff_distance']:.6f}",
                f"{metrics['boundary_intersection_over_union']:.6f}",
                # Uncertainty
                f"{mean_epistemic:.6f}",
                f"{mean_aleatoric:.6f}",
                elapsed,
            ]
            _append_results_row(csv_path, header, row)
            results_written += 1

        except Exception as exc:
            print(_col(f"Evaluation failed for {ckpt_path}: {exc}", _C.RED))

    # ===== Print summary table =====
    # Display evaluation results in a formatted table
    if results_written == 0:
        print(_col("No results to write.", _C.YELLOW))
    else:
        print(_col(f"\nWrote {results_written} result rows to {csv_path}", _C.GREEN))

        print(f"\n{'='*80}")
        print(_col(f"EVALUATION SUMMARY ({arch.upper()})", _C.GREEN))
        print(f"{'='*80}\n")

        # Read CSV and display key metrics for each checkpoint
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)

                if rows:
                    # Print header row with key metrics
                    print(
                        f"{'Checkpoint':<50} {'Dice':<10} {'IoU':<10} "
                        f"{'Acc':<10} {'Hausd':<10} {'F1':<10} {'Time':<10}"
                    )
                    print("-" * 110)

                    # Print one row per checkpoint
                    for row in rows:
                        ckpt_name = os.path.basename(row.get("checkpoint_path", ""))[:45]
                        dice = row.get("dice_coef", "N/A")
                        iou = row.get("IoU", "N/A")
                        acc = row.get("accuracy", "N/A")
                        hausdorff_distance = row.get("Hausdorff_distance", "N/A")
                        f1 = row.get("f1_score", "N/A")
                        elapsed = row.get("elapsed", "N/A")

                        print(
                            f"{ckpt_name:<50} {dice:<10} {iou:<10} "
                            f"{acc:<10} {hausdorff_distance:<10} {f1:<10} {elapsed:<10}"
                        )

                    print(f"\nFull results saved to: {csv_path}")
        print(f"{'='*80}\n")


# =====================================================
# Public entrypoints (called from main training script)
# =====================================================
def evaluate_unet(conf) -> None:
    """
    Evaluate ALL UNet checkpoints found in config.saved_models_dir.

    Output:
      - Per-frame hard masks: config.results_dir/<checkpoint_basename>/*.tif
      - Probability maps: config.results_dir/<checkpoint_basename>/*_prob.tif
      - CSV summary: config.results_dir/evaluation_unet.csv
    """
    global config
    config = conf
    _evaluate_arch(config, arch="unet")


def evaluate_SwinUNetPP(conf) -> None:
    """
    Evaluate ALL SwinUNet checkpoints found in config.saved_models_dir.

    SwinUNet = Swin Transformer backbone + U-Net decoder
    Outputs same as evaluate_unet but saved to evaluation_swin.csv
    """
    global config
    config = conf
    _evaluate_arch(config, arch="swin")


def evaluate_TerraMind(conf) -> None:
    """
    Evaluate ALL TerraMind checkpoints found in config.saved_models_dir.

    TerraMind = Geospatial pretrained backbone + decoder
    Outputs same as evaluate_unet but saved to evaluation_tm.csv
    """
    global config
    config = conf
    _evaluate_arch(config, arch="tm")


if __name__ == "__main__":
    config = Configuration().validate()
    evaluate_SwinUNetPP(config)