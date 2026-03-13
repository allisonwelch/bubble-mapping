from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

# Albumentations: A library for image augmentation (random flips, crops, brightness changes, etc.)
# We use it to artificially create training variations that prevent overfitting.
import albumentations as A
import numpy as np


# ============================= AUGMENTATION (ALBUMENTATIONS) =============================
# This section builds the data augmentation pipeline. Augmentation is crucial for deep learning:
# it takes one image and creates many variations, so the model learns to recognize patterns
# despite real-world variations (angle, lighting, zoom, etc.) rather than memorizing exact pixels.
# ============================================================================================

def alb_augmentation(
    patch_size: Tuple[int, int],
    strength: float = 1.0,
) -> A.Compose:
    """
    Build a data augmentation pipeline using Albumentations.

    DATA AUGMENTATION: Artificially modifying training images to create variations
    that don't exist in the original data. This teaches the model to recognize
    patterns that are robust to real-world variations (like rotation, lighting changes,
    viewpoint shifts, etc.), preventing the model from overfitting to the exact
    images in the training set.

    ANALOGY: Imagine teaching someone to recognize birds. Showing them one photo
    from one angle in one lighting won't work well. But showing them the same bird
    from many angles, lighting conditions, and distances teaches them invariance—
    the ability to recognize birds despite variations.

    WHY IT HELPS:
    - Overfitting prevention: Without augmentation, a model might memorize features
      of specific training images rather than learning generalizable patterns.
    - Invariance learning: The model learns that bubbles are still bubbles whether
      flipped, cropped, or slightly blurred—teaching it to be robust to variations.

    All shape-changing augmentations keep output size equal to the input `patch_size`.
    This ensures patches fed to the model are always the expected size.

    STRENGTH parameter: A scalar in [0, 1] that scales augmentation intensity.
    - strength=0.0: No augmentation (p=0 for all transforms, no modifications)
    - strength=1.0: Full augmentation (each transform runs at its defined probability)
    - strength=0.5: 50% intensity (probabilities and amounts halved)
    Think of it as a volume knob: you can dial augmentation up or down globally.
    """
    # Ensure strength is in [0.0, 1.0] range; this is a safety check and normalization.
    # E.g., if caller passes strength=1.5, clip it to 1.0; if -0.1, clip to 0.0.
    s = float(np.clip(strength, 0.0, 1.0))
    # Extract patch height and width; these define the size of random crops from the image.
    # All augmented patches will be resized to this size (so the model sees consistent input).
    h, w = int(patch_size[0]), int(patch_size[1])

    # ===== ALBUMENTATIONS PIPELINE =====
    # A.Compose chains multiple transformations together. Each transform is applied sequentially
    # to the same image-mask pair. The key benefit: geometric transforms (rotation, crop) are
    # automatically applied to both the image AND the label mask, keeping them aligned.
    # Without this, if we rotate the image but not the mask, the labels would be wrong.
    aug = A.Compose(
        [
            # ===== GEOMETRIC AUGMENTATIONS (MIRRORING) =====
            # Flips teach the model rotational invariance: bubbles look the same whether
            # viewed right-side-up or flipped. This prevents the model from memorizing
            # specific orientations in training data.

            # HorizontalFlip: Mirror image left-right with probability p = 0.5 * s.
            # The 'p' parameter means: with 50% chance (when strength=1.0), flip this image.
            # strength acts as a global dial: strength=0.5 makes p=0.25 (25% flip rate).
            A.HorizontalFlip(p=0.5 * s),

            # VerticalFlip: Mirror image top-bottom with probability p = 0.5 * s.
            # Teaches the model that a bubble in the top half looks the same as one in the bottom.
            # Both flips combined teach the model to recognize objects from any orientation.
            A.VerticalFlip(p=0.5 * s),

            # ===== SCALE AUGMENTATION (ZOOM IN/OUT) =====
            # RandomResizedCrop teaches SCALE INVARIANCE: the model learns to detect bubbles
            # at different sizes. In real data, bubbles may be far away (small) or close (large).
            #
            # HOW IT WORKS:
            #   1. Randomly crop a region from the image (e.g., 95% of original size)
            #   2. Resize that crop back to patch_size (h x w)
            #   Result: the image is zoomed in (large objects appear larger),
            #   forcing the model to recognize bubbles at multiple scales.
            #
            # PARAMETERS:
            #   scale: (min_crop_ratio, max_crop_ratio). E.g., scale=(0.9, 1.0) means
            #     crop 90-100% of the original image (i.e., slight zoom).
            #     With strength=1.0: scale=(0.9, 1.0) -> zoom up to 10%.
            #     With strength=0.0: scale=(1.0, 1.0) -> no zoom (full image each time).
            #   ratio: aspect ratio. (0.9, 1.1) means keep it near-square (allow 10% deviation).
            #   p: probability of applying this transform.
            A.RandomResizedCrop(
                height=h,
                width=w,
                scale=(max(0.0, 1.0 - 0.10 * s), 1.0),
                ratio=(0.9, 1.1),
                p=0.5 * s,
            ),

            # ===== PHOTOMETRIC AUGMENTATIONS (PIXEL VALUE CHANGES) =====
            # These transforms modify brightness, contrast, and blur WITHOUT changing
            # geometric structure (shape). They teach LIGHTING INVARIANCE: the model learns
            # bubbles look the same in bright sunlight, shade, or dim conditions.

            # GaussianBlur: Add slight blur with kernel size 3x3 to 7x7 pixels.
            # WHY: Real camera optics (lenses, focus) introduce slight blur. Training on
            # blurry and sharp versions teaches the model not to rely on razor-sharp edges
            # that might not be there in all acquisition conditions. Probability 0.30*s.
            A.GaussianBlur(blur_limit=(3, 7), p=0.30 * s),

            # RandomBrightnessContrast: Simulate different lighting and exposure.
            # brightness_limit=0.0: Keep brightness constant (don't brighten/darken).
            # contrast_limit=0.7*s: Stretch or compress the histogram by up to ±70%.
            #   With strength=1.0: contrast can change by ±70% (dramatic changes).
            #   With strength=0.5: contrast can change by ±35% (subtle changes).
            # WHY: Real-world images vary in lighting. A bubble in direct sunlight looks
            # different from one in shadow. This augmentation prevents the model from
            # learning to recognize bubbles ONLY in bright or dark conditions—it learns
            # the bubble pattern regardless of brightness. Probability 0.30*s.
            A.RandomBrightnessContrast(
                brightness_limit=0.0,
                contrast_limit=0.7 * s,
                p=0.30 * s,
            ),

            # ===== ADVANCED GEOMETRIC WARPS (CURRENTLY DISABLED) =====
            # The transforms below are COMMENTED OUT but available if needed in future.
            # They provide extreme geometric invariance but can distort labels in ways
            # that confuse training, so they're disabled by default.
            #
            # A.PiecewiseAffine: Local bending/warping (like poking a rubber sheet).
            # A.ElasticTransform: Smooth, organic deformations (like rubber being stretched).
            # A.Perspective: View images from a tilted angle (3D perspective).
            # These are powerful but risky: if labels get warped incorrectly, the model
            # learns bad associations. They're best used with careful tuning.
            # A.PiecewiseAffine(scale=0.05 * s, p=0.30 * s),
            #A.ElasticTransform(
            #    alpha=1.0 * s,
            #    sigma=50.0 * s,
            #    alpha_affine=50.0 * s,
            #    p=0.30 * s,
            #),
            #A.Perspective(scale=(0.0, 0.01 * s), keep_size=True, p=0.10 * s),
        ]
    )
    # Return the composed pipeline. This object is callable: aug(image=img, mask=lbl)
    # applies all transforms in sequence, returning augmented image and mask together.
    return aug


# ============================== DATA GENERATOR CLASS ==============================
# The DataGenerator is the engine that feeds training data to a deep learning model.
# Instead of loading all images into memory at once (which may exceed RAM), it
# generates random batches on-the-fly during training. This is memory-efficient
# and enables infinite training data through augmentation.
# ==================================================================================
class DataGenerator:
    """Generate random or sequential patches from frames.

    WHAT IS A DATA GENERATOR?
    A data generator automatically feeds training data (image patches + labels)
    to a machine learning model in batches. Instead of loading all data into
    memory at once, it generates random batches on-the-fly during training,
    making training efficient and enabling augmentation (different transforms
    each epoch).

    KEY CAPABILITIES:
    * Frame sampling weighted by image area: Larger images get sampled more often,
      ensuring no single small image dominates the training distribution.
    * Returns numpy arrays; training wraps them into PyTorch tensors.
    * Applies data augmentation (Albumentations) if enabled.
    * Controls positive/negative patch sampling: You can require the model to see
      more positive (labeled) or negative (unlabeled) patches by setting pos_ratio.

    POSITIVE VS NEGATIVE PATCHES:
    - Positive patch: Contains at least some pixels labeled as bubbles (or meets
      a minimum threshold, controlled by min_pos_frac).
    - Negative patch: Contains no labeled pixels (or below the threshold).
    By controlling the pos_ratio, you balance the dataset. If you always feed
    patches with 99% unlabeled background, the model might learn to just say "no
    bubbles" everywhere. Balancing positive/negative teaches the model to be
    sensitive to actual bubble presence.

    NOTES
    -----
    * frame sampling is weighted by image area.
    * Returns numpy arrays; training wraps them into PyTorch tensors.
    * If `augmenter` is 'alb'/'albumentations', apply Albumentations;
      else no aug.
    * `min_pos_frac` and `pos_ratio` control positive/negative patch sampling:
        - `min_pos_frac`: minimum positive fraction of a patch to treat it
          as "positive" (0.0 -> any positive pixel makes it positive).
        - `pos_ratio`: target fraction of positive patches in random batches.
    * `stride`, `weighting` are accepted for compatibility with newer callers.
    """

    def __init__(
        self,
        input_image_channel: Sequence[int],
        patch_size: Tuple[int, int],
        frame_list: Sequence[int],
        frames: Sequence[Any] | Dict[int, Any],
        annotation_channel: int,
        augmenter: Optional[str] = "alb",  # 'alb'/'albumentations' or None
        augmenter_strength: float = 1.0,
        min_pos_frac: float = 0.0,
        pos_ratio: Optional[float] = None,
        stride: Optional[Tuple[int, int]] = None,
        weighting: str = "area",  # accepted; only 'area' supported
        **_: Any,
    ) -> None:
        # ===== INITIALIZATION: STORE CONFIGURATION =====
        # The __init__ method saves all parameters needed for generating batches.
        # These become instance attributes (self.X) so other methods can use them later.

        # PATCHES vs FULL IMAGES:
        # Deep learning models work on PATCHES (small crops), not full-size images.
        # WHY? Full-size images may be 1000x1000 pixels or larger, which wastes memory
        # and makes training slow. Small patches (e.g., 128x128) are faster and let the
        # model see different regions across different batches. We extract random patches
        # and feed them to the model in batches during training.

        # input_image_channel: Which channels to use as model input.
        # E.g., if your image has RGB (channels 0,1,2) + label (channel 3),
        # input_image_channel=[0,1,2] means feed only RGB to the model.
        self.input_image_channel = list(input_image_channel)

        # PATCH SIZE (height, width): All patches extracted will be this size.
        # Larger patches: more context, but slower and more memory.
        # Smaller patches: faster, less memory, but less context (the model sees smaller regions).
        # Typical: 128x128, 256x256, or 512x512 depending on bubble size and available memory.
        self.patch_size = (int(patch_size[0]), int(patch_size[1]))

        # frame_list: List of which frames to sample from (e.g., [0, 1, 2, 3]).
        # You might have 100 frames total but only want to train on frames [0, 5, 10, 20].
        self.frame_list = list(frame_list)

        # frames: The actual frame data (images + labels combined).
        # Can be a list [frame0, frame1, ...] or a dict {0: frame0, 5: frame5, ...}.
        # This is flexible because different data sources organize data differently.
        self.frames = frames

        # annotation_channel: Index of the label/mask channel in the frame data.
        # E.g., if frame.shape = (height, width, 3) and channel 2 is the label mask,
        # annotation_channel=2. We use this channel to determine if a patch is positive/negative.
        self.annotation_channel = int(annotation_channel)

        # augmenter: Which augmentation library to use ('alb', 'albumentations', or None).
        # If 'alb', apply Albumentations transforms. Otherwise, no augmentation.
        self.augmenter = augmenter

        # augmenter_strength: Scale factor for augmentation intensity (0.0 to 1.0).
        # Controls how aggressive the transforms are: strength=0.0 means no augmentation,
        # strength=1.0 means full augmentation (all transforms at defined probability).
        # Acts as a global "volume knob" for augmentation.
        self.augmenter_strength = float(augmenter_strength)

        # ===== POSITIVE/NEGATIVE PATCH CLASSIFICATION =====
        # POSITIVE PATCH: Contains enough labeled pixels (bubbles) to count as "has object".
        # NEGATIVE PATCH: Contains no (or very few) labeled pixels.
        # These definitions affect class balance during training.

        # min_pos_frac: Minimum fraction of labeled pixels to call a patch "positive".
        # E.g., min_pos_frac=0.0 means ANY patch with ≥1 labeled pixel is positive.
        #       min_pos_frac=0.1 means patch must have ≥10% labeled pixels to be positive.
        #       min_pos_frac=0.5 means patch must be ≥50% labeled (strong presence only).
        # Use low values to catch faint/small bubbles; high values to focus on obvious ones.
        self.min_pos_frac = float(min_pos_frac)

        # ===== WEIGHTED SAMPLING: BALANCING POSITIVE/NEGATIVE PATCHES =====
        # pos_ratio: Target fraction of positive patches in sampled batches.
        # WHY BALANCE? Imagine dataset with 1% bubbles, 99% empty space. If you train on
        # raw data, the model learns "say no bubble everywhere" and gets 99% accuracy
        # without learning anything useful. By forcing 50%/50% pos/neg patches, you teach
        # the model to actually distinguish between classes.
        #
        # EXAMPLE:
        # - pos_ratio=0.5: Aim for 50% positive, 50% negative patches in each batch.
        # - pos_ratio=0.3: Aim for 30% positive, 70% negative (if data is imbalanced).
        # - pos_ratio=None: Don't force balance; just sample proportionally by frame area.
        #
        # This is called WEIGHTED SAMPLING: larger frames are sampled more, and within
        # sampling, the ratio of pos/neg patches is controlled.
        self.pos_ratio = None if pos_ratio is None else float(pos_ratio)

        # ===== BUILD AUGMENTATION PIPELINE ONCE =====
        # Create the Albumentations pipeline NOW (at initialization), not later.
        # WHY? Building it once is efficient. If we rebuilt it every time we sampled
        # a patch, we'd waste CPU time. Instead, we create it once and reuse it.
        # IMPORTANT: The pipeline is deterministic in structure (same transforms every time)
        # but STOCHASTIC in application (each transform has a probability p, and random
        # parameters like rotation angle are sampled fresh each time it's called).
        self._alb = alb_augmentation(self.patch_size, self.augmenter_strength)

        # ===== AREA-WEIGHTED FRAME SAMPLING PROBABILITIES =====
        # PROBLEM: You have 5 frames. Frame 0 is 1000x1000 pixels (1M pixels).
        # Frame 1 is 100x100 pixels (10k pixels). Should they be sampled equally?
        # If you sample them equally (20% each), frame 0 dominates the training despite
        # being just 1 of 5 frames. Its pixels are seen 100x more often than frame 1's.
        #
        # SOLUTION: Area-weighted sampling. Sample frame 0 with 99% probability,
        # frame 1 with 1% probability. This ensures EACH PIXEL in the dataset has
        # equal probability of being seen. Now training data is balanced by pixel count,
        # not frame count—which is fairer and more representative.
        #
        # ANALOGY: Imagine surveying people from 2 towns. Town A has 1M people,
        # town B has 1k people. If you interview 50 people from each (equal sampling),
        # you bias toward town B. Instead, interview proportionally: ~50 from town A,
        # ~0.05 from town B. This represents the true population distribution.

        total_area = 0.0
        areas: List[float] = []

        # Loop through each frame index and compute its pixel area.
        for i in self.frame_list:
            fr = self._frame(i)
            # Get frame dimensions (height, width) from the image data.
            h, w = fr.img.shape[:2]
            # Compute total pixels in this frame (area = h * w).
            a = float(h * w)
            areas.append(a)
            total_area += a

        # Safety check: prevent division-by-zero if all frames are somehow empty.
        # If total_area is 0, use 1e-6 as a fallback to avoid crashing.
        total_area = max(total_area, 1e-6)

        # ===== COMPUTE SAMPLING PROBABILITIES =====
        # Normalize each frame's area to a probability.
        # E.g., if frame 0 is 1M pixels out of 10M total:
        #   weight = 1M / 10M = 0.1 (10% sampling probability).
        # These weights are used by np.random.choice() to select frames randomly
        # but proportionally to their size. Larger frames are picked more often.
        self.frame_list_weights = [a / total_area for a in areas]

    # ====== PUBLIC METHODS: CALLED BY TRAINING SCRIPTS ======

    def all_sequential_patches(self, step_size: Tuple[int, int]):
        """Return all sequential patches and labels given a step size.

        SEQUENTIAL PATCHING: Extract patches in a GRID PATTERN (non-random, systematic)
        across all frames. Used for INFERENCE/EVALUATION, not training.
        During inference, we want to scan the ENTIRE image to make predictions everywhere,
        so we tile the image with patches in a grid. This is deterministic (no randomness).

        WHY SEQUENTIAL FOR INFERENCE?
        - Training uses random patches to save memory and avoid overfitting.
        - Inference needs COMPLETE coverage: we want predictions for every pixel.
        - So we grid-scan the image with a fixed stride, extracting every possible patch.

        EXAMPLE: If image is 256x256, step_size=(64,64), patch_size=(128,128):
        - Patch 1: rows[0:128], cols[0:128] (top-left)
        - Patch 2: rows[0:128], cols[64:192] (top-center, slides right by 64)
        - Patch 3: rows[0:128], cols[128:256] (top-right)
        - Patch 4: rows[64:192], cols[0:128] (middle-left, slides down by 64)
        - ... continue tiling the entire image.
        """
        patches = []
        # Iterate through all frames in the dataset
        for fn in self.frame_list:
            frame = self._frame(fn)
            # Ask the frame object to extract all sequential patches for this frame.
            # The frame knows how to grid-tile itself.
            ps = frame.sequential_patches(self.patch_size, step_size)
            # Accumulate all patches from this frame
            patches.extend(ps)
        # Stack all patches into a single numpy array: (num_patches, H, W, channels)
        data = np.asarray(patches)
        # Extract input channels (e.g., RGB bands [0,1,2]) from each patch
        img = data[..., self.input_image_channel]
        # Extract annotation/label channel (usually the last channel)
        ann = data[..., -1]
        return img, ann

    def random_patch(self, batch_size: int):
        """Return a single random batch (X, y).

        This method samples a batch of random patches (typically used for training).
        Each patch is sampled independently via _sample_one_patch(), which respects
        area-weighting and pos_ratio constraints if configured.
        """
        # Sample batch_size random patches independently
        patches = [self._sample_one_patch() for _ in range(int(batch_size))]
        # Stack them into a single array (B, H, W, C)
        data = np.asarray(patches)
        # Extract input channels
        img = data[..., self.input_image_channel]
        # Extract annotation/label channel
        ann = data[..., -1]
        return img, ann

    def random_generator(self, batch_size: int):
        """Yield endless batches (X, y) as (B,H,W,C_in) and (B,H,W,1).

        INFINITE GENERATOR FOR TRAINING:
        This is a Python generator (uses 'yield') that produces batches indefinitely.
        During training, the model calls this repeatedly to get batches:
        - Epoch 1: generator produces batches 1, 2, 3, ...
        - Epoch 2: generator produces more batches, possibly with different patches
          due to randomness in sampling and augmentation.

        ANALOGY: Think of it like an assembly line that never stops. Each time the
        trainer asks for a batch, the generator produces one on-the-fly, applying
        different augmentations, sampling different patches, etc. This is memory-
        efficient and enables truly infinite training data (via augmentation).

        WHAT HAPPENS IN EACH ITERATION:
        1. Sample a random batch of patches (respecting area-weighting and pos_ratio)
        2. Normalize label shapes and convert to binary (0/1)
        3. Apply augmentation (if enabled)
        4. Yield (images, labels) to the trainer
        """
        # Check if augmentation is enabled
        use_alb = str(self.augmenter).lower() in {"alb", "albumentations"}

        # INFINITE LOOP: 'while True' means this generator never stops naturally.
        # The trainer controls iteration by calling next() until it's done with an epoch.
        while True:
            # Sample a batch of random patches
            X, y = self.random_patch(batch_size)  # X: (B,H,W,C_in), y: (B,H,W) initially

            # ========== LABEL SHAPE NORMALIZATION ==========
            # Patches may come from different sources with different label formats.
            # Normalize to (B, H, W, 1) with binary values 0 and 1.
            m = y
            if m.ndim == 4:
                # Labels are 4D. Could be:
                #   (B, H, W, C): Channels-last format
                #   (B, C, H, W): Channels-first format (common in PyTorch)
                if m.shape[-1] == 1:
                    # Already (B, H, W, 1), no change needed
                    pass
                elif m.shape[1] == 1 and m.shape[-1] != 1:
                    # Detect (B, 1, H, W) format: second dimension is 1, last is not
                    # Transpose to (B, H, W, 1)
                    m = np.transpose(m, (0, 2, 3, 1))
                else:
                    # Multiple channels; take only the first one (first class label)
                    m = m[..., :1]
            elif m.ndim == 3:
                # Labels are (B, H, W), no channel dimension. Add one.
                m = m[..., np.newaxis]  # Now (B, H, W, 1)
            else:
                # Unexpected format; raise an error to catch data issues early
                raise ValueError(
                    f"Unexpected mask shape {m.shape}; expected 3D or 4D."
                )

            # BINARIZATION: Convert labels to binary 0/1.
            # (m > 0) creates a boolean array: True where labels > 0 (bubble), False elsewhere.
            # .astype(np.uint8) converts True->1, False->0.
            # This ensures labels are simple binary regardless of their original values.
            m = (m > 0).astype(np.uint8)

            if use_alb:
                # ========== AUGMENTATION: Albumentations ==========
                # Apply geometric and photometric transforms to each patch in the batch.
                # Augmentation is applied AFTER sampling, meaning each call to
                # random_generator() produces differently augmented patches, even if
                # sampling the same underlying patch multiple times.

                # Pre-allocate output arrays for augmented images and masks
                X_aug = np.empty_like(X, dtype=np.float32)
                y_aug = np.empty(m.shape[:3] + (1,), dtype=np.float32)

                # Apply augmentation to each image-mask pair in the batch
                for i in range(X.shape[0]):
                    # Call the Albumentations pipeline on image and mask.
                    # The pipeline applies the same random transforms to both,
                    # ensuring they stay aligned (e.g., if image is cropped, mask is too).
                    res = self._alb(
                        image=X[i].astype(np.float32), mask=m[i, ..., 0]
                    )
                    # Store augmented image (shape H, W, C_in)
                    X_aug[i] = res["image"].astype(np.float32)
                    # Store augmented mask, binarizing to ensure 0/1 (blur in interpolation might create intermediate values)
                    y_aug[i, ..., 0] = (res["mask"] > 0).astype(np.float32)

                # Use augmented batch
                X, ann = X_aug, y_aug
            else:
                # NO AUGMENTATION: Just cast to float32
                ann = m.astype(np.float32)
                X = X.astype(np.float32)

            # YIELD THE BATCH: Send (images, labels) to the trainer.
            # X: (B, H, W, C_in) where C_in is the number of input channels
            # ann: (B, H, W, 1) with binary labels (0.0 or 1.0)
            # After yielding, execution pauses. On next next() call, loop continues
            # and a new batch is sampled and yielded.
            yield X, ann

    # -------- internal --------
    def _frame(self, idx: int):
        """Return frame by index for list/dict-backed storage.

        Abstraction layer to handle both list and dict storage of frames.
        Some callers pass a list of frames, others a dict mapping IDs to frames.
        """
        # If frames is a list/sequence, index directly
        if not isinstance(self.frames, dict):
            return self.frames[idx]
        # If frames is a dict, use idx as key
        return self.frames[int(idx)]

    def _random_frame_patch(self) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
        """Sample a random patch from a randomly chosen frame (area-weighted).

        Returns the padded patch along with the unpadded (h, w) size if padding was needed.
        The unpadded size is used to evaluate pos_ratio against real pixels only.

        AREA-WEIGHTED FRAME SELECTION:
        Uses np.random.choice() with probabilities pre-computed in __init__.
        Larger frames are selected more often, ensuring the model sees data
        distributed fairly across the dataset (proportional to image sizes).

        WHY TRACK UNPADDED SIZE?
        If a frame is smaller than patch_size, the frame object pads it with zeros.
        But when checking if a patch is "positive" (contains labels), we should only
        count real pixels, not padded zeros. The unpadded_slice tells us the region
        containing real data, so _is_positive_patch() can exclude padding.
        """
        # Randomly choose a frame index using area-weighted probabilities
        fn = int(np.random.choice(self.frame_list, p=self.frame_list_weights))
        frame = self._frame(fn)

        # Ask the frame object to generate a random patch of target size.
        # The frame handles padding if it's smaller than target patch_size.
        patch = frame.random_patch(self.patch_size)

        # Get frame dimensions to check if padding occurred
        h_frame, w_frame = frame.img.shape[:2]
        h_target, w_target = self.patch_size

        unpadded_slice = None
        if h_frame < h_target or w_frame < w_target:
            # Frame is smaller than target patch, so padding happened.
            # Record the real (unpadded) dimensions for later filtering in _is_positive_patch()
            unpadded_slice = (min(h_frame, h_target), min(w_frame, w_target))

        return patch, unpadded_slice

    def _is_positive_patch(
        self,
        ann: np.ndarray,
        unpadded_slice: Optional[Tuple[int, int]] = None,
    ) -> bool:
        """Return True if patch is considered 'positive' given the annotation mask.

        A patch is "positive" if it contains enough labeled pixels to meet the
        threshold set by min_pos_frac. This controls what counts as training data
        for the model to learn bubble presence.

        POSITIVE PATCH DEFINITION:
        - min_pos_frac=0.0: Any patch with >= 1 labeled pixel is positive.
        - min_pos_frac=0.1: Patch must have >= 10% of pixels labeled to be positive.
        - min_pos_frac=0.5: Patch must be >= 50% labeled (rich labels only).

        WHY VARY THIS?
        - Low threshold: Catch patches with faint/small bubbles, but may include
          patches with only edges or artifacts.
        - High threshold: Focus on patches with clear, obvious bubble regions,
          harder to overfit but may miss small/faint cases.

        Args:
            ann: annotation array (HxW or HxWx1)
            unpadded_slice: optional (slice_h, slice_w) tuple indicating the real (unpadded) region.
                           If provided, only counts positives in the unpadded region.
        """
        # ========== EXTRACT UNPADDED REGION IF NEEDED ==========
        # If the frame was smaller than patch_size and got padded with zeros,
        # we want to evaluate positivity only on real data, not padding.
        if unpadded_slice is not None:
            h_slice, w_slice = unpadded_slice
            # Get patch dimensions
            H, W = ann.shape[:2]
            # Calculate offsets to center-crop the real data region
            off_h = (H - h_slice) // 2
            off_w = (W - w_slice) // 2
            # Extract the unpadded region and binarize (True where labeled)
            mask = ann[off_h : off_h + h_slice, off_w : off_w + w_slice] > 0
        else:
            # No padding; use the entire patch
            mask = ann > 0

        # ========== POSITIVITY THRESHOLD CHECK ==========
        if self.min_pos_frac <= 0.0:
            # Lenient threshold: Any labeled pixel makes this positive
            # np.any(mask) returns True if any element in mask is True
            return np.any(mask)
        else:
            # Stricter threshold: Require min_pos_frac fraction of pixels to be labeled
            # mask.mean() = (number of True values) / (total pixels)
            # E.g., mask.mean() = 0.15 means 15% are labeled.
            return mask.mean() >= self.min_pos_frac

    def _sample_one_patch(self):
        """Sample a single random patch, optionally enforcing a pos/neg ratio.

        POSITIVE/NEGATIVE SAMPLING BALANCE:
        If pos_ratio is set (e.g., 0.5), this method ensures that ~50% of sampled
        patches are positive and ~50% are negative. This prevents the model from
        being biased towards one class due to imbalanced data.

        ANALOGY: Imagine a disease dataset where only 1% of patients have the
        disease. If you train a model on raw data, it learns "say no to everyone"
        and achieves 99% accuracy without learning anything useful. By balancing
        pos/neg patches (e.g., 50%/50%), you force the model to actually learn
        to distinguish between the classes.

        HOW IT WORKS:
        1. Randomly decide if this patch should be positive or negative (based on pos_ratio)
        2. Sample patches until finding one matching the desired class
        3. Fallback to any patch if we exceed max_tries (avoid infinite loops)

        WHY MAX_TRIES?
        If the dataset has very few positive samples (< 2% of pixels), sampling
        until finding a positive patch could take forever. max_tries=50 means we
        try 50 times; if no match, return the last sampled patch anyway. This
        prevents the generator from hanging while still attempting the balance.
        """
        # ========== CHECK IF BALANCE IS REQUESTED ==========
        # If pos_ratio is None or invalid (outside 0-1), skip balancing.
        # This maintains backward compatibility: if pos_ratio is not set,
        # patches are sampled purely via area-weighted frame selection
        # without any positive/negative filtering.
        if self.pos_ratio is None or not (0.0 < self.pos_ratio < 1.0):
            patch, _unpadded_slice = self._random_frame_patch()
            return patch

        # ========== BALANCED SAMPLING WITH POS_RATIO ==========
        # Decide whether this draw should be positive or negative.
        # np.random.rand() returns a uniform float in [0, 1).
        # E.g., if pos_ratio=0.5, roughly 50% of calls return want_pos=True.
        want_pos = np.random.rand() < float(self.pos_ratio)

        # Maximum attempts to find a patch matching the desired class.
        # Prevents infinite loops if the dataset is extremely imbalanced.
        max_tries = 50
        last_patch = None

        # Retry loop: keep sampling until finding a match or hitting max_tries
        for _ in range(max_tries):
            # Sample a random patch from the dataset
            patch, unpadded_slice = self._random_frame_patch()
            last_patch = patch
            # Extract the annotation/label channel (last channel)
            ann = patch[..., -1]
            # Check if this patch is positive according to min_pos_frac threshold
            is_pos = self._is_positive_patch(ann, unpadded_slice)

            # Accept if the patch matches our desired class
            if want_pos and is_pos:
                # Wanted positive, got positive
                return patch
            if (not want_pos) and (not is_pos):
                # Wanted negative, got negative
                return patch
            # Otherwise, loop and try again

        # ========== FALLBACK HANDLING ==========
        # If we exhausted max_tries without finding a match:
        # Return the last sampled patch (even if it doesn't match the desired class).
        # This ensures the generator keeps flowing without hanging.
        if last_patch is not None:
            return last_patch

        # Defensive fallback: If something went very wrong (no patches sampled),
        # sample one final patch and return it.
        fallback_patch, _ = self._random_frame_patch()
        return fallback_patch


# ========== BACKWARDS-COMPATIBLE ALIAS ==========
# Newer training code may import 'Generator' instead of 'DataGenerator'.
# This alias ensures old code doesn't break.
Generator = DataGenerator