# core/frame_info.py  (PyTorch)
#    Edited by Sizhuo Li, Carl Stadie
#    Author: Ankit Kariryaa, University of Bremen
#
# ============================================================================
# OVERVIEW: HANDLING GEOSPATIAL DATA FOR DEEP LEARNING
# ============================================================================
#
# WHAT IS RASTERIO AND WHY IS IT USED?
#   Regular images (e.g., JPEGs) are typically 8-bit RGB: 3 channels, 0-255 each.
#   Satellite imagery is different: it's stored in GeoTIFF format, which:
#   - Preserves geographic metadata (lat/lon coordinates, projection, etc.)
#   - Can store many bands (channels) with different bit depths (16-bit, 32-bit float)
#   - Maintains exact pixel alignment across multiple files (critical for overlays)
#
#   Rasterio is a Python library that reads/writes GeoTIFFs while preserving this
#   geographic information. It ensures that when you load multiple satellite images
#   for the same region, pixels align perfectly (geo-registration).
#
#   Why it matters for bubbles:
#   - Ground-truth bubble locations come from field surveys (GPS-tagged)
#   - We must align those GPS points with satellite pixels precisely
#   - Rasterio handles this geo-transformation automatically
#
# HOW DATA FLOWS:
#   1. Rasterio loads a GeoTIFF → multi-band array (H, W, C) in memory
#   2. FrameInfo wraps this array + corresponding label array (annotations)
#   3. getPatch() extracts small tiles for neural network training
#   4. Patches are normalized (image_normalize) so the network can learn efficiently
#
# ============================================================================
# WHAT NORMALIZATION DOES AND WHY IT'S IMPORTANT FOR ML:
#   Raw pixel values (e.g., 0-255 for images, 0-10000 for satellite bands) can be
#   on very different scales. Neural networks learn best when inputs are normalized
#   (rescaled) to have zero mean (centered at 0) and unit variance (std=1). This is
#   called "standardization" or "z-score normalization".
#
#   Why it matters:
#   - Without normalization, large pixel values can dominate learning, drowning
#     out the effect of other features. Example: if band A is 0-100 and band B is
#     0-10000, the network might ignore band A entirely.
#   - Normalized inputs help the model converge faster and avoid numerical instability
#     (large weights, exploding gradients).
#   - Analogy: comparing heights of people in centimeters vs millimeters—
#     normalizing puts them on the same scale so comparisons are fair.
#
#   Z-SCORE NORMALIZATION FORMULA:
#     normalized = (original - mean) / std
#   This shifts so mean=0 and std=1. The constant epsilon (c=1e-8) prevents
#   division-by-zero if std is very small.

import numpy as np


def image_normalize(im, axis=(0, 1), c=1e-8, nodata_val=None):
    """
    Normalize to zero mean and unit std along the given axis.

    If a nodata value is specified, normalise without nodata pixels and set
    them to nan.

    Args:
        im: input image array, typically (H, W, C) where C is channels
        axis: which axes to compute mean/std over (default (0,1) = spatial dims)
        c: small constant to avoid division by zero (1e-8)
        nodata_val: if set, pixels with this value are excluded from mean/std
                    calculation and set to NaN in output
    """
    # PER-CHANNEL NORMALIZATION:
    # When axis=(0, 1), we compute mean/std separately for each channel (band).
    # This is critical because satellite imagery often has multiple spectral bands
    # (e.g., Red, Green, Blue, Near-Infrared), each with different value ranges.
    #
    # Why per-channel matters:
    # - A Red band might have values 0-1000, while an Infrared band might be 0-5000.
    # - Without per-channel normalization, the model would treat them as different
    #   "importance" even though they represent equal physical information.
    # - Per-channel normalization ensures the neural network treats all bands fairly.
    #
    # The result: for each channel C, we subtract its mean and divide by its std.
    # This transforms each band independently to mean=0, std=1.

    if nodata_val is not None and np.sum(im == nodata_val) > 0:
        # If there are "no data" pixels (e.g., clouds, missing data) specified
        im = im.astype(np.float32)
        # Mark entire pixels as invalid (NaN) if any channel has nodata value
        # np.any(..., axis=2) checks each channel; if ANY channel is nodata,
        # the whole pixel is marked bad. This is conservative: one bad band
        # contaminates the entire pixel.
        im[np.any(im == nodata_val, axis=2), :] = np.nan
        # Compute mean/std using only valid (non-NaN) pixels via np.nanmean/nanstd.
        # This excludes clouds, missing data, etc. from the normalization stats.
        return (im - np.nanmean(im, axis)) / (np.nanstd(im, axis) + c)
    else:
        # Standard z-score normalization: (x - mean) / (std + epsilon)
        # Epsilon (c=1e-8) prevents division by zero if std is very small
        # (e.g., a constant-valued channel with std=0).
        return (im - im.mean(axis)) / (im.std(axis) + c)


class FrameInfo:
    """
    Defines a frame, includes its constituent images (inputs) and annotation.

    A FrameInfo object represents one full satellite image ("frame") along with
    its ground-truth label (which pixels are bubbles and which aren't).

    WHAT IS A "FRAME"?
    In this project, a frame is a preprocessed GeoTIFF file loaded into memory.
    It consists of:
    1. Multiple spectral bands (e.g., Red, Green, Blue, Near-Infrared) stacked together
    2. A label band indicating which pixels are bubbles (1) vs. background (0)

    The term "frame" comes from satellite imagery processing: each frame is one
    coherent tile of land captured at a specific time. It's preprocessed to have
    consistent spatial resolution, geographic alignment, and band ordering.

    During training, we don't use the entire frame at once (it might be huge).
    Instead, we extract small square patches and feed those to the neural network.
    """

    def __init__(self, img, annotations, dtype=np.float32):
        """
        Args:
            img: ndarray (H, W, C_in) — the satellite image with C_in spectral bands.
                 Shape is (Height, Width, Channels), e.g., (1024, 1024, 4) for RGBN.
                 Stored as a single array to maintain pixel alignment (rasterio ensures this).

            annotations: ndarray (H, W) or (H, W, 1) — binary label (0=background, 1+=bubble).
                        Must have same spatial dims (H, W) as img.
                        This is the "ground truth" created by human annotation or automated
                        detection, used to train and evaluate the model.

            dtype: np.float32, optional — data type for array operations.
                   float32 is standard in deep learning (good balance of precision & memory).
        """
        self.img = img
        self.annotations = annotations
        self.dtype = dtype

    def getPatch(
        self,
        top: int,
        left: int,
        patch_size,
        img_size,
        pad_mode: str = "reflect",
    ):
        """
        Return a composite patch (inputs + label as last channel), padded
        to `patch_size`.

        - top, left: top-left of the slice to take from the full image
        - patch_size: (H, W) or (H, W, C_out). If 2-D, C_out = C_in + 1 (label)
        - img_size: (h_slice, w_slice) actual slice size (clamped to image bounds)
        - pad_mode: 'reflect' (default) or 'constant' (zeros).
                    'reflect' looks best for aug.
        """
        # Parse patch_size: either (H, W) or (H, W, C_out).
        # If not explicitly given C_out, we compute it as C_in + 1 to account
        # for adding the label band at the end.
        if isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
            H, W = int(patch_size[0]), int(patch_size[1])
            C_out = int(self.img.shape[2]) + 1
        else:
            H, W, C_out = (
                int(patch_size[0]),
                int(patch_size[1]),
                int(patch_size[2]),
            )

        # Ensure slice dimensions don't exceed patch dimensions.
        # This handles edge cases where the frame is smaller than patch_size.
        h_slice = min(int(img_size[0]), H)
        w_slice = min(int(img_size[1]), W)

        # Grab slice from source (clamped to bounds).
        # img_patch has shape (h_slice, w_slice, C_in).
        img_patch = self.img[top : top + h_slice, left : left + w_slice, :]
        # NOTE: Per-patch normalization is commented out here.
        # If enabled, it would normalize each small patch independently.
        # The alternative is to normalize once at the dataset level (compute stats
        # from the entire frame or full training set, then apply consistently).
        #img_patch = image_normalize(img_patch, axis=(0, 1), nodata_val=0)

        # Extract the corresponding label patch.
        # Shape: (h_slice, w_slice) or (h_slice, w_slice, 1)
        lab_patch = self.annotations[top : top + h_slice, left : left + w_slice]
        # Ensure label is 3D (h_slice, w_slice, 1) so we can concatenate along axis=-1.
        if lab_patch.ndim == 2:
            lab_patch = lab_patch[..., None]
        # Stack spectral bands + label together: shape becomes (h_slice, w_slice, C_in + 1).
        # The label is now the last channel. Some architectures treat this specially
        # (e.g., using it as a mask for boundary-aware loss), while others just feed it
        # alongside the spectral data.
        #
        # BOUNDARY BAND (THE LABEL AS A PREDICTION TARGET):
        # The label band serves dual purposes:
        # 1. As the ground-truth target: during training, the model learns to predict
        #    this channel (1=bubble, 0=background).
        # 2. As a boundary indicator: some loss functions use the label to identify
        #    edge pixels (where label transitions from 0->1 or 1->0). Boundaries are
        #    often harder to predict correctly, so boundary-aware training weights
        #    these pixels more heavily.
        comb = np.concatenate([img_patch, lab_patch], axis=-1)

        # Center the slice inside the output canvas.
        # If the frame is smaller than patch_size (e.g., frame is 256x256 but patch_size=512x512),
        # we center the actual content and pad around it.
        # off_h, off_w are the pixel offsets from top-left.
        off_h = (H - h_slice) // 2
        off_w = (W - w_slice) // 2

        # Prepare output canvas: zeros-initialized tensor of shape (H, W, C_out).
        # This will hold the combined input+label patch.
        patch = np.zeros((H, W, C_out), dtype=self.img.dtype)
        # Place the combined patch in the center of the canvas.
        patch[
            off_h : off_h + h_slice,
            off_w : off_w + w_slice,
            : comb.shape[-1],
        ] = comb

        # PADDING STRATEGY: Reflect-pad the edges to fill empty space.
        # Why reflect-padding?
        # - After data augmentation (random rotations, scaling), the patch boundaries
        #   might be exposed. Pure zero-padding would create big black borders.
        # - Reflect-padding mirrors the nearest valid pixels, creating more natural
        #   content at the boundaries. This helps the network learn without being
        #   confused by artificial black borders.
        # - Analogy: if you crop a photo off-center and need to fill edges, mirroring
        #   the nearby content looks more natural than black.
        if pad_mode == "reflect":
            # Top padding: mirror the topmost row upward.
            if off_h > 0:
                patch[
                    :off_h,
                    off_w : off_w + w_slice,
                    : comb.shape[-1],
                ] = patch[
                    off_h : off_h + 1,
                    off_w : off_w + w_slice,
                    : comb.shape[-1],
                ][::-1, ...]  # [::-1, ...] reverses the first dimension (vertical flip)
            # Bottom padding: mirror the bottommost row downward.
            if H - (off_h + h_slice) > 0:
                patch[
                    off_h + h_slice :,
                    off_w : off_w + w_slice,
                    : comb.shape[-1],
                ] = patch[
                    off_h + h_slice - 1 : off_h + h_slice,
                    off_w : off_w + w_slice,
                    : comb.shape[-1],
                ][::-1, ...]
            # Left padding: mirror the leftmost column leftward.
            if off_w > 0:
                patch[:, :off_w, : comb.shape[-1]] = patch[
                    :, off_w : off_w + 1, : comb.shape[-1]
                ][:, ::-1, :]  # [:, ::-1, :] reverses the second dimension (horizontal flip)
            # Right padding: mirror the rightmost column rightward.
            if W - (off_w + w_slice) > 0:
                patch[:, off_w + w_slice :, : comb.shape[-1]] = patch[
                    :, off_w + w_slice - 1 : off_w + w_slice, : comb.shape[-1]
                ][:, ::-1, :]

        return patch

    def sequential_patches(self, patch_size, step_size):
        """
        Return all sequential patches in this frame by sliding a window.

        This method is used for inference (prediction) or systematic evaluation.
        Unlike random_patch(), this extracts every patch in a grid pattern,
        ensuring complete coverage of the frame without overlap or duplication.
        """
        img_shape = self.img.shape
        # Compute row indices: start at 0, step by step_size[0], stop before leaving room for a patch.
        # step_size is the stride (e.g., 256 pixels), so overlapping patches are created.
        x = range(0, img_shape[0] - patch_size[0], step_size[0])
        y = range(0, img_shape[1] - patch_size[1], step_size[1])
        # Edge case: if frame is smaller than patch size, extract just one patch starting at [0, 0].
        if img_shape[0] <= patch_size[0]:
            x = [0]
        if img_shape[1] <= patch_size[1]:
            y = [0]

        # ic = actual slice dimensions (clamped to frame bounds).
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        # Generate all (row, col) combinations to extract patches.
        xy = [(i, j) for i in x for j in y]
        img_patches = []
        # Extract each patch at the computed positions.
        for i, j in xy:
            img_patch = self.getPatch(i, j, patch_size, ic)
            img_patches.append(img_patch)
        return img_patches

    def random_patch(self, patch_size):
        """
        Extract a random patch from this frame for data augmentation.

        TRAINING VS. INFERENCE:
        - Training (random_patch): We extract random patches to increase dataset diversity.
          Each time we load a frame, we sample a different location, so the model sees
          varied views of the same frame. This acts as data augmentation without storing
          multiple copies of the frame.
        - Inference (sequential_patches): We systematically cover the entire frame in a
          grid to ensure complete predictions.

        Random cropping adds regularization: the model must learn to recognize bubbles
        at any location, not just at a memorized set of positions.
        """
        img_shape = self.img.shape
        # Randomly select top-left corner, ensuring the patch fits within bounds.
        x = (
            0
            if (img_shape[0] <= patch_size[0])
            else np.random.randint(0, img_shape[0] - patch_size[0])
        )
        y = (
            0
            if (img_shape[1] <= patch_size[1])
            else np.random.randint(0, img_shape[1] - patch_size[1])
        )
        # ic = actual slice size (frame dimensions clamped by patch_size).
        ic = (min(img_shape[0], patch_size[0]), min(img_shape[1], patch_size[1]))
        # Extract patch using reflection padding to create natural-looking edges.
        return self.getPatch(x, y, patch_size, ic, pad_mode="reflect")
