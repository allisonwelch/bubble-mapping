# configUnet.py
# Configuration class for UNet semantic segmentation model
# This file controls all hyperparameters, data paths, and training settings for the neural network

import os
import warnings
import numpy as np
from osgeo import gdal  # GDAL is used for reading/writing geospatial raster data (satellite imagery)

REPO_PATH = r"C:\Users\amwelch3\git_repos\bubble-mapping"


class Configuration:
    """
    Configuration used by preprocessing.py, training.py, tuning.py and evaluation.py (UNet).
    Only includes parameters actually referenced by the UNet workflows.
    """

    def __init__(self):
        # --------- RUN NAME ---------
        # "Modality" refers to the type of satellite imagery being analyzed.
        # AE = Airbus Econos, PS = PlanetScope, S2 = Sentinel-2
        # Each has different number of spectral bands (color channels in the imagery)
        self.modality = "AE"

        # The run name is used to identify saved models, logs, and results uniquely
        # This way you can run multiple experiments and keep them organized
        self.run_name = f"UNETx{self.modality}"

        # ---------- PATHS -----------
        # NOTE: Update these paths to match your own project structure

        # Training data and imagery
        # These folders contain the raw satellite imagery and ground truth labels (training labels are called "geopackages" or .gpkg files)
        # geopackages are expected to be inside training_data_dir
        self.training_data_dir = (
            f"{REPO_PATH}/data/training/{self.modality}"
        )
        self.training_area_fn = "training_areas.gpkg"  # Geopackage defining where training data exists
        self.training_polygon_fn = f"labels_full_dataset_{self.modality}.gpkg"  # The actual labeled polygons (bubbles, non-bubbles) for training
        # Set to a filename (e.g. f"focus_areas_{self.modality}.gpkg") if you have a focus-areas
        # geopackage; leave as None to skip focus-area chip generation entirely.
        self.focus_areas = None  # Regions of special interest (optional)

        # Directory containing the raw satellite images (.tif files) for the chosen modality
        self.training_image_dir = (
            f"{REPO_PATH}/data/training_images/{self.modality}"
        )

        # Preprocessed data roots
        # After the raw imagery is converted into 256x256 patches, they're saved here
        self.preprocessed_base_dir = (
            f"{REPO_PATH}/data/preprocessed/"
        )

        # Alternative base directory for processed training data
        self.training_data_base_dir = (
            f"{REPO_PATH}/data/training_data/"
        )
        # The specific folder where preprocessed patches will be saved for THIS experiment
        # Update the timestamp to run a new preprocessing
        self.preprocessed_dir = (
            f"{REPO_PATH}/data/preprocessed/"
            f"2026-03-26_UNETxAE"
        )



        # Checkpointing / logs / results
        # Resuming training: if you have a partially trained model, point to it here to resume training
        # Set to None to train from scratch
        self.continue_model_path = None
        # Where the best trained model weights will be saved after training completes
        self.saved_models_dir = f"{REPO_PATH}/data/models/UNET/{self.modality}"
        # Training logs (loss curves, validation metrics) go here for later visualization
        self.logs_dir = f"{REPO_PATH}/data/logs/UNET/{self.modality}"
        # Final predictions on test set go here
        self.results_dir = f"{REPO_PATH}data/results/UNET/{self.modality}"

        # -------- IMAGE / CHANNELS --------
        # Satellite imagery is stored in GeoTIFF format (.tif), a standard format for geo-referenced raster data
        self.image_file_type = ".tif"

        # If you want to downsample images (e.g., resample_factor=2 means half resolution, faster training but less detail)
        # Most commonly kept at 1 (no resampling)
        self.resample_factor = 1

        # Each satellite image has multiple spectral "bands" (different wavelengths of light)
        # AE/PS have 4 bands: Red, Green, Blue, Near-Infrared (RGBN)
        # Sentinel-2 has 12 bands (includes more specialized bands like coastal aerosol, SWIR, etc.)
        # Setting channels_used to True/False lets you pick which bands to feed to the model
        if self.modality != "S2":
            self.channels_used = [True, True, True]  # Use 3 RGB bands
        else:
            self.channels_used = [True, True, True, True, True, True, True, True, True, True, True, True]  # Use all 12 S2 bands

        # Convert the boolean list above into indices (e.g., [0, 1, 2, 3] means use bands 0-3)
        self.preprocessing_bands = np.where(self.channels_used)[0]
        # Keep a copy for reference throughout the code
        self.channel_list = self.preprocessing_bands

        # Whether to include border pixels from labels (useful if polygon boundaries matter for your task)
        self.rasterize_borders = True
        # Deprecated/internal setting
        self.get_json = False

        # -------- DATA SPLIT --------
        # How to divide your data into training, validation, and test sets
        # The model learns from training data, validation data is used to tune hyperparameters,
        # and test data is held out completely to measure final performance
        # test_ratio = 0.2 means 20% of your patches go to testing
        self.test_ratio = 0.2
        # val_ratio = 0.2 means 20% go to validation
        self.val_ratio = 0.2
        # The remaining 60% (1 - 0.2 - 0.2) are used for actual training
        # Changing these ratios affects model generalization: smaller val/test = more training data but less reliable evaluation

        # If you already have a train/val/test split you want to reuse, point to the JSON file here
        # Otherwise leave as None and the code will randomly split based on the ratios above
        self.split_list_path = None  # Optional path to predefined train/val/test split lists
        # "/isipd/projects/p_planetdw/data/methods_test/preprocessed/20251226-0433_UNETxAE/aa_frames_list.json"

        # Manual test set — prefix-based override (currently disabled)
        # Uncomment the block below to re-enable forced prefix-based test splitting.
        #
        # # If set, any source image whose FILENAME starts with this prefix will have ALL of its
        # # training areas forced into the test split. Those areas are excluded from the random
        # # train/val/test split entirely, so they are guaranteed never to appear in training.
        # #
        # # Use this when you have separate ground-truth ("field-labeled") imagery that should
        # # serve as your held-out test set, independent of the automatic random split.
        # #
        # # Example:
        # #   self.manual_test_image_prefix = "FIELD_"
        # #   -> any image file named "FIELD_*.tif" in training_image_dir will be test-only.
        # #
        # # Leave as None to use the standard random split (or split_list_path if set).
        # self.manual_test_image_prefix = None

        # -------- TRAINING (CORE) --------
        # The model processes satellite images as 256x256 pixel patches
        # Larger patches capture more context but use more memory and training time
        # Smaller patches are faster but miss larger spatial patterns
        # 256x256 is a good balance for typical geospatial tasks
        self.patch_size = (256, 256)
        # These are redundant with patch_size above, but kept for compatibility
        self.tune_patch_h = 256
        self.tune_patch_w = 256

        # Tversky loss is a variant of the Dice loss, a common loss function for segmentation
        # Alpha and beta control how much the model penalizes different types of errors:
        # - Alpha = 0.55: penalty for false negatives (missing bubbles) - higher = care more about finding all bubbles
        # - Beta = 0.45: penalty for false positives (predicting bubbles where there are none) - higher = care more about precision
        # 0.55/0.45 means you care slightly more about not missing bubbles than about false alarms
        self.tversky_alphabeta = (0.55, 0.45)

        # Dilation rate controls how large a receptive field each neuron sees
        # dilation_rate=2 means skip every other pixel when applying convolution filters
        # Larger dilation = sees bigger spatial context but is faster; smaller = sees fine details but slower
        self.dilation_rate = 2

        # Dropout is a regularization technique: randomly "turn off" neurons during training
        # This prevents the model from memorizing and forces it to learn robust features
        # Higher dropout = more regularization = simpler model, less overfitting, but may underfit
        # 0.1 (10% dropout) is mild regularization, good for well-sized datasets
        self.dropout = 0.1

        # These are UNet architecture hyperparameters tuned for this specific task
        # layer_count = 96: number of convolutional filters/features the model learns (controls model capacity)
        # Higher = more expressive model but more parameters and slower; lower = simpler model, faster
        self.layer_count = 96

        # L2 regularization (weight decay): adds a penalty based on how large the weights are
        # Encourages smaller weights = simpler model = less overfitting
        # Typical values are 1e-5 to 1e-3; larger values = stronger regularization
        self.l2_weight = 1e-5

        # Name to save the trained model as
        self.model_name = self.run_name

        # ------ OPTIM / SCHED / EPOCHS ------
        # Loss function measures how "wrong" the model's predictions are
        # "tversky" is good for imbalanced classes (e.g., bubbles might be rare in the image)
        self.loss_fn = "tversky"

        # Optimizer controls how the model updates its weights based on errors
        # "adamw" is AdamW (Adam with weight decay), a popular modern optimizer
        # It adapts learning rates per parameter and helps convergence
        self.optimizer_fn = "adamw"

        # Learning rate controls step size when updating weights
        # Too high = unstable training, too low = slow training
        # 0.0004 is a reasonable starting point; you'd typically experiment to find the best value
        self.learning_rate = 0.0004

        # Weight decay applied by AdamW optimizer (slightly different from L2 regularization above)
        # Works alongside l2_weight to prevent overfitting
        self.weight_decay = 4.8e-6

        # Scheduler adjusts learning rate during training
        # "onecycle" starts low, ramps up to a peak, then decreases (helps avoid local minima)
        self.scheduler = "onecycle"

        # Number of patches to process in each batch
        # Larger batches = more stable gradients but require more GPU memory
        # Smaller batches = noisier training but work on limited memory
        # 32 is a common compromise
        self.train_batch_size = 8  # Windows smoke-test (main uses 32)

        # Total number of training epochs (full passes through the training set)
        # 100 is typical; too few = underfitting, too many = overfitting (but early stopping can help)
        self.num_epochs = 10  # Windows smoke-test (main uses 100)

        # Number of optimization steps per epoch
        # 500 steps with batch_size=32 means processing ~16,000 patches per epoch
        # Controls how many batches are shown before validation
        self.num_training_steps = 50  # Windows smoke-test (main uses 500)

        # How many validation patches to evaluate on during training
        # Validation provides feedback on generalization without overfitting to training data
        # More samples = more reliable validation but slower per epoch
        self.num_validation_images = 50

        # ------ EMA (Exponential Moving Average) ------
        # EMA keeps a smoothed copy of the model weights that often generalizes better
        # Useful for difficult tasks or when you have limited data
        self.use_ema = False  # Disabled here, but set to True if validation plateaus
        # How much to weight historical parameters vs. new ones (0.999 = favor history, very stable)
        self.ema_decay = 0.999
        # Whether to use the EMA weights or regular weights for evaluation
        self.eval_with_ema = False

        # ------ CHECKPOINTING / LOGGING ------
        # Save model weights every N epochs (None = only save the best model)
        # Set to an integer (e.g., 5) if you want periodic checkpoints for long experiments
        self.model_save_interval = None
        # Debugging feature: train only on one batch to verify the model can memorize it (sanity check)
        self.overfit_one_batch = False
        # Print detailed loss information during training
        self.train_verbose = True
        # Log validation metrics every N epochs (1 = after every epoch)
        self.train_epoch_log_every = 1
        # Print extra details like per-class accuracies
        self.train_print_heavy = True
        # Show a progress bar during training
        self.show_progress = True
        # Every N epochs, save predicted images so you can visually inspect model output
        # Set to larger value (e.g., 20) to reduce storage if disk space is limited
        self.log_visuals_every = 5
        # When saving visual predictions, which bands to use for the RGB display
        # (0, 1, 2) = Red, Green, Blue (standard RGB ordering)
        self.vis_rgb_idx = (0, 1, 2)

        # ------ AUG / SAMPLING / DATALOADER ------
        # Data augmentation creates variations of training images (rotations, flips, brightness changes, etc.)
        # This helps the model generalize and avoid overfitting to the exact training imagery
        # augmenter_strength = 0.7 means aggressive augmentation (0.0 = none, 1.0 = maximum)
        # Higher values = more variety but may distort labels; lower = safer but less diverse
        self.augmenter_strength = 0.7

        # Minimum fraction of positive class (bubble) pixels in a patch for it to be used for training
        # min_pos_frac = 0.001 means patches with at least 0.1% bubble pixels are included
        # If bubbles are very rare, set this lower; if common, set higher to balance classes
        self.min_pos_frac = 0.001

        # Alternatively, specify desired ratio of positive to negative patches (e.g., 0.5 = 50/50 balance)
        # Set to None to use min_pos_frac instead
        self.pos_ratio = None

        # Stride when sliding window across the image to extract patches
        # None = use patch_size (non-overlapping), smaller = overlapping patches (more training data, slower)
        self.patch_stride = None

        # Number of CPU workers for loading data in parallel during training
        # Higher = faster data loading but uses more CPU
        # 8 is typical for multi-core machines; reduce if you have CPU issues
        # Windows: must be 0 — spawn multiprocessing cannot pickle large numpy arrays via IPC pipe
        # On Linux/HPC use 8 (fork-based workers inherit memory without pickling)
        self.fit_workers = 0

        # How many batches to accumulate before updating weights
        # steps_per_execution = 1 means update after every batch
        # Larger values (e.g., 4) can reduce overhead but require more memory
        self.steps_per_execution = 1

        # ------ EVALUATION ------
        # Threshold for converting model probabilities to binary predictions
        # Model outputs a probability (0-1) for each pixel being a bubble
        # eval_threshold = 0.5 means: if probability > 0.5, predict bubble; otherwise, not a bubble
        # Lower threshold = more bubbles detected (higher recall, more false positives)
        # Higher threshold = fewer bubbles but more confident (higher precision, more false negatives)
        self.eval_threshold = 0.5

        # Detailed evaluation is expensive; do it every N steps during validation
        # heavy_eval_steps = 50 means compute F1, precision, recall every 50 validation steps
        self.heavy_eval_steps = 10  # Windows smoke-test (main uses 50)

        # Print statistics specifically about the positive class (bubbles)
        # Useful for imbalanced datasets where you care more about one class
        self.print_pos_stats = True

        # Monte Carlo Dropout: evaluate model with dropout enabled to estimate prediction uncertainty
        # Useful to know which predictions the model is unsure about
        # NOTE: set False for smoke-tests — MC dropout runs 8+ forward passes per patch (very slow)
        self.eval_mc_dropout = False  # Windows smoke-test (main uses True)
        # Number of forward passes with different dropout masks to average predictions
        # More samples = better uncertainty estimate but slower; 20 is reasonable
        # NOTE: eval code reads "eval_mc_samples" not "mc_dropout_samples" — fix the name
        self.eval_mc_samples = 8  # used by evaluation.py; mc_dropout_samples is a dead alias

        # ------ MIXED PRECISION / COMPILE / REPRO ------
        # PyTorch compilation: converts model to optimized machine code for faster inference
        # Good for production but can make debugging harder; usually not needed for training
        self.use_torch_compile = False

        # Random seed for reproducibility (None = non-deterministic, results vary)
        # Set to an integer (e.g., 42) if you need identical results across runs
        # Useful for publishing results or debugging; slower than non-deterministic mode
        self.seed = 123

        # Gradient clipping: if gradients exceed this norm, scale them down
        # Prevents "exploding gradients" which cause training to diverge
        # 0.0 = disabled; typical values are 1.0 to 10.0 if you have gradient issues
        self.clip_norm = 0.0

        # --- POSTPROCESSING (kept for downstream scripts) ----
        # After the model predicts pixels, convert them to polygons (vector geometries)
        # Useful for exporting results in GIS-compatible format
        self.create_polygons = True
        # Number of parallel workers for polygon creation
        self.postproc_workers = 12

        # Prediction outputs (for completeness with your tools)
        # Format to use for prediction images (usually same as input: .tif)
        self.train_image_type = self.image_file_type
        # Prefix added to training image filenames (if any)
        self.train_image_prefix = ""
        # Attribute field in training_area_fn whose value is the .tif basename (no extension)
        # that each training area belongs to. When set, areas are matched to images by this
        # field rather than by spatial overlap — required when .tif files overlap spatially.
        # Set to None to fall back to the legacy spatial-overlap matching.
        self.image_link_field = "image_link"
        # File format for saved predictions
        self.predict_images_file_type = self.image_file_type
        # Prefix added to prediction filenames (if any)
        self.predict_images_prefix = ""
        # Whether to overwrite existing prediction files
        self.overwrite_analysed_files = False
        # Name for the prediction output (used in filenames)
        self.prediction_name = self.run_name
        # Directory to save predictions (None = use results_dir)
        self.prediction_output_dir = None
        # Patch size for prediction (None = use patch_size from above)
        self.prediction_patch_size = None
        # How to combine overlapping patch predictions: "MAX" = take highest probability
        # Other options might include "MEAN" (average) or "VOTE" (majority)
        self.prediction_operator = "MAX"
        # Data type for output predictions
        # "bool" = binary (0/1), "uint8" = 0-255, "float32" = probabilities (0.0-1.0)
        self.output_dtype = "bool"

        # ------ GPU / ENV ------
        # Which GPU to use for training
        # 7 = GPU #7 (useful if you have multiple GPUs and want to avoid conflicts with other jobs)
        # -1 = use CPU (much slower but useful for debugging)
        self.selected_GPU = 1

        # GDAL configuration for geospatial data handling
        gdal.UseExceptions()  # Make GDAL report errors as exceptions instead of silently failing
        gdal.SetCacheMax(32000000000)  # Cache size for GDAL operations (32 GB)
        gdal.SetConfigOption("CPL_LOG", "NUL")  # Suppress GDAL log messages
        # Suppress Python warnings to reduce clutter in training output
        warnings.filterwarnings("ignore")

        # Tell PyTorch/TensorFlow which GPU(s) to use
        if int(self.selected_GPU) == -1:
            # CPU mode
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            # GPU mode: make only the selected GPU visible to the framework
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.selected_GPU)

    def validate(self):
        # Verify that all required input paths exist before training begins
        # This catches configuration errors early instead of failing mid-training
        if not os.path.exists(self.training_data_dir):
            raise ConfigError(f"Invalid path: config.training_data_dir = {self.training_data_dir}")
        if not os.path.exists(os.path.join(self.training_data_dir, self.training_area_fn)):
            raise ConfigError(f"File not found: {os.path.join(self.training_data_dir, self.training_area_fn)}")
        if not os.path.exists(os.path.join(self.training_data_dir, self.training_polygon_fn)):
            raise ConfigError(f"File not found: {os.path.join(self.training_data_dir, self.training_polygon_fn)}")
        if not os.path.exists(self.training_image_dir):
            raise ConfigError(f"Invalid path: config.training_image_dir = {self.training_image_dir}")

        # Create output directories if they don't exist (for storing models, logs, and results)
        for cfg_dir in ["preprocessed_base_dir", "saved_models_dir", "logs_dir"]:
            target = getattr(self, cfg_dir)
            if not os.path.exists(target):
                try:
                    os.mkdir(target)
                except OSError as exc:
                    raise ConfigError(f"Unable to create folder config.{cfg_dir} = {target}") from exc

        # Verify output formats are supported
        if self.predict_images_file_type not in [".tif", ".jp2"]:
            raise ConfigError("Invalid format for config.predict_images_file_type. Supported: .tif, .jp2")
        if self.output_dtype not in ["bool", "uint8", "float32"]:
            raise ConfigError("Invalid config.output_dtype: choose 'bool', 'uint8' or 'float32'")
        return self


# Custom exception class for configuration errors
class ConfigError(Exception):
    pass