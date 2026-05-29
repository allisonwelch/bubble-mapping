# configSwinUnet.py

import os
import warnings
import numpy as np
from osgeo import gdal

REPO_PATH = os.path.expanduser("~/bubble-mapping")

class Configuration:
    """
    Configuration used by preprocessing.py, training.py, tuning.py and evaluation.py (Swin-UNet).
    Only includes parameters referenced by the Swin workflows.
    """

    def __init__(self):
        # --------- RUN NAME ---------
        # Modality to be run can be AE, PS or S2
        self.modality = "AE"

        self.run_name = f"SWINx{self.modality}"

        # ---------- PATHS -----------

        # Training data and imagery
        self.training_data_dir = (
            f"{REPO_PATH}/data/training/{self.modality}/2026-04-16_UNETxAE"
        )
        self.training_area_fn = "training_areas_no_toolik.gpkg"
        self.training_polygon_fn = f"labels_full_dataset_no_toolik_{self.modality}.gpkg"
        self.training_image_dir = (
            f"{REPO_PATH}/data/training_images/{self.modality}"
        )

        # Preprocessed data roots
        self.preprocessed_base_dir = (
            f"{REPO_PATH}/data/preprocessed/"
        )
        # Reuse UNETxAE chips (full dataset)
        self.preprocessed_dir = (
            f"{REPO_PATH}/data/preprocessed/"
            f"2026-04-22_SWINxAE"
        )

        # Checkpointing / logs / results (model + modality subfolders)
        # Continue from run #5's best raw weights (best seep-F1 of the three saved files
        # per CLAUDE.md 2026-04-28). starting_epoch will fall back to 0 because the
        # metadata.json lookup uses {state_path}.metadata.json which won't match
        # `.raw.weights.pt`'s naming — that's fine, num_epochs below counts fresh epochs.
        self.continue_model_path = (
            f"{REPO_PATH}/data/models/SWIN/{self.modality}/20260423-1056_SWINxAE/"
            f"20260423-1056_SWINxAE.raw.weights.pt"
        )
        self.saved_models_dir = (
            f"{REPO_PATH}/data/models/SWIN/{self.modality}/20260428_SWINxAE_continued"
        )
        self.logs_dir = (
            f"{REPO_PATH}/data/logs/SWIN/{self.modality}"
        )
        self.results_dir = (
            f"{REPO_PATH}/data/results/SWIN/{self.modality}"
        )

        # -------- IMAGE / CHANNELS --------
        self.image_file_type = ".tif"
        self.resample_factor = 1
        
        if self.modality != "S2":
            self.channels_used = [True, True, True]
        else:
            self.channels_used = [True, True, True, True, True, True, True, True, True, True, True, True]
            
        self.preprocessing_bands = np.where(self.channels_used)[0]
        self.channel_list = list(self.preprocessing_bands)

        # change add rasterize_borders called by preprocessing
        self.rasterize_borders = True
        # Deprecated/internal setting
        self.get_json = False

        # -------- DATA SPLIT --------
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        # train is 1 - test_ratio - val_ratio

        # If you already have a train/val/test split you want to reuse, point to the JSON file here
        # Otherwise leave as None and the code will randomly split based on the ratios above
        self.split_list_path = None  # Optional path to predefined train/val/test split lists

        # -------- TRAINING (CORE) --------
        self.patch_size = (448, 448)
        self.tune_patch_h = None
        self.tune_patch_w = None
        # flip alpha and beta (previously (0.7, 0.3) to penalize FN over FP
        self.tversky_alphabeta = (0.3, 0.7)
        self.model_name = self.run_name

        # ------ OPTIM / SCHED / EPOCHS ------
        self.loss_fn = "tversky"
        self.optimizer_fn = "adamw"

        # These three are used directly in training.py and match tuner names   # NEW
        self.learning_rate = 0.0001       # NEW: tuned "learning_rate" goes here
        self.weight_decay = 0.0063         # NEW: tuned "weight_decay" (for AdamW)
        # change from none to onecycle
        self.scheduler = "onecycle"      # NEW: tuned "scheduler" ("none"|"cosine"|"onecycle")

        self.train_batch_size = 8
        # Continuation run from run #5: 50 fresh epochs from warm-started weights.
        # Early stopping (val_dice_coef, patience=15) will gate this if it plateaus.
        self.num_epochs = 50
        self.num_training_steps = 500
        #change num_validation_images from 50 to 500 to address undersampling in low quantity training areas
        self.num_validation_images = 500

        # ------ EARLY STOPPING ------
        # Stop training if a monitored metric stops improving for `patience` epochs.
        # Saves compute and auto-picks the best checkpoint (best_saver still runs).
        # Set patience=0 to disable.
        # self.early_stopping_patience = 15
        # Which key in the per-epoch logs dict to monitor.
        # "val_dice_coef" (higher=better) is the most informative for sparse segmentation.
        # Use "val_loss" (lower=better) as a safer fallback.
        # self.early_stopping_metric = "val_dice_coef"
        # "max" if higher metric is better (Dice, F1, IoU), "min" if lower is better (loss).
        # self.early_stopping_mode = "max"
        # Minimum improvement (absolute) to count as progress; prevents tiny fluctuations from
        # resetting the patience counter. 0.001 = require ≥0.001 Dice improvement.
        # self.early_stopping_min_delta = 0.001

        # ------ EMA ------
        self.use_ema = False
        self.ema_decay = 0.999
        self.eval_with_ema = False

        # ------ CHECKPOINTING / LOGGING ------
        self.model_save_interval = None
        self.overfit_one_batch = False
        self.train_verbose = True
        self.train_epoch_log_every = 1
        self.train_print_heavy = True
        self.show_progress = True
        self.log_visuals_every = 5
        self.vis_rgb_idx = (0, 1, 2)

        # ------ AUG / SAMPLING / DATALOADER ------
        self.augmenter_strength = 0.7
        # change min_pos_frac from 0.001 to 0.01 to address undersampling in low quantity training areas
        self.min_pos_frac = 0.01
        self.pos_ratio = 0.0
        #self.patch_stride = 0.3
        # change fit_workers from 0 to 8
        self.fit_workers = 8
        self.steps_per_execution = 1

        # ------ EVALUATION ------
        self.eval_threshold = 0.5
        self.heavy_eval_steps = 50
        self.print_pos_stats = True
        self.eval_mc_dropout = True
        # change mc_dropout_samples to eval_mc_samples, which is called in evaluation (mc_dropout_samples is not)
        self.eval_mc_samples = 20

        # ------ MIXED PRECISION / COMPILE / REPRO ------
        self.use_torch_compile = False
        self.seed = 123
        #change to 1
        self.clip_norm = 1.0

        # ------ SWIN-UNET (PP) ------
        self.swin_patch_size = 4
        self.swin_window = 7
        self.swin_levels = 3
        self.swin_base_channels = 96
        self.use_imagenet_weights = True
        # Tuned Swin stochastic depth rate (used in training._build_model_swin)  # NEW
        self.drop_path = 0.2   # NEW: tuned "drop_path" goes here

        # --- POSTPROCESSING (kept for downstream scripts) ---
        self.create_polygons = True
        self.postproc_workers = 12

        # --- SEEP FEATURE TABLE / ANCHOR-CONDITIONAL CLUSTERING ---
        # Read by tools/seep_feature_table.py and tools/seep_level_eval.py.
        # A predicted bubble with area >= seep_anchor_area_m2 is an "anchor" (always
        # a cluster head). Non-anchor bubbles within seep_cluster_radius_m of an
        # anchor centroid join that anchor's cluster. Others form singleton clusters.
        # Default anchor area = pi * (25 cm)^2; provisional until Walter's threshold.
        self.seep_anchor_area_m2 = float(np.pi * (0.125 ** 2))
        self.seep_cluster_radius_m = 0.45
        # Satellite area cap: a non-anchor bubble must be at most this large to
        # be eligible as a satellite. Set to None to disable (every non-anchor
        # eligible). Default = pi * (7.5 cm)^2 — provisional; tune from
        # cluster-size histogram and n_medium count in the script printout.
        self.seep_satellite_max_area_m2 = float(np.pi * (0.1 ** 2))
        # Phase-2 ("lonely") clustering: after the anchor pass, group Phase-1
        # singletons that sit close together AND have a sparse halo around the
        # candidate cluster's centroid. Catches isolated clusters of small
        # bubbles that have no anchor among them.
        #   lonely_cluster_radius_m   inner grouping radius for singletons
        #   lonely_halo_radius_m      radius around the candidate centroid in
        #                             which other bubbles are counted
        #   lonely_max_halo_neighbors strict-less-than; <= this many neighbors
        #                             outside the candidate accepts the cluster
        # Set lonely_cluster_radius_m=0 OR lonely_max_halo_neighbors=0 to disable.
        self.seep_lonely_cluster_radius_m = 0.4
        self.seep_lonely_halo_radius_m = 1.5
        self.seep_lonely_max_halo_neighbors = 5
        self.write_seep_cluster_rasters = False

        # --- SEEP-LEVEL EVAL: GROUND-TRUTH GROUPING MODE ---
        # Read by tools/seep_level_eval.py. Controls how the GT side is grouped
        # into seeps for the cluster-level matcher (the headline cluster_f1):
        #   "auto"   (headline) rule-group the FULL per-chip GT with the fitted
        #            seep_anchor_area_m2 / seep_cluster_radius_m /
        #            seep_satellite_max_area_m2 above (the same rule applied to
        #            predictions), so cluster_f1 covers every test-chip seep and
        #            is a seep-level DETECTION metric. ** Set the fitted Phase-6
        #            values above BEFORE running, or GT gets grouped with the
        #            provisional placeholders. **
        #   "manual" match against the labeler-grouped 750-seep sample
        #            (gt_seeps_labeled.gpkg) — human-vs-pred SANITY CHECK only,
        #            NOT the headline (the 750 are a deliberately
        #            non-representative stratified sample; wrong denominator).
        # Grouping-rule-vs-human fidelity is reported separately by the Phase-6
        # cross-validation kappa, not by this eval. See CLAUDE.md 2026-05-29.
        self.gt_grouping_mode = "auto"
        # Optional override of the manual-mode labeled-seeps file. Leave this
        # line commented to auto-discover gt_seeps_labeled.gpkg in the
        # checkpoint's pred_dir (do NOT set it to None — that disables discovery).
        # self.gt_labeled_seeps_path = "/path/to/gt_seeps_labeled.gpkg"

        # --- HSV SNOW MASK (predictions only; GT is never masked) ---
        # Read by tools/seep_level_eval.py. Snow heuristic per pixel:
        #   V (= max(R,G,B) / dtype_max)            >= snow_v_thresh
        #   S (= (max-min) / max, == 0 for grayscale) <= snow_s_thresh
        # When `snow_mask_enabled`, the snow mask drives a CC-LEVEL filter:
        # the smoothed prediction is labeled into connected components, and
        # any CC whose pixel-overlap with the snow mask exceeds
        # `snow_cc_drop_frac` is dropped whole (set to 0). This is whole-or-
        # nothing — unlike pixel-level zeroing it never carves holes inside
        # a CC, which avoids fragmenting one snow-FP into several smaller
        # FPs. A {stem}_snow.tif is also written per chip when
        # write_snow_rasters=True so the mask can be overlaid on the chip in
        # QGIS to verify it isn't eating real bubbles. Tune by toggling
        # snow_mask_enabled and inspecting seep_level_summary.csv columns
        # `snow_pct_masked` (how much of the image the mask hits) and
        # `snow_ccs_dropped` (how many CCs were filtered out).
        #
        # snow_cc_drop_frac sweep guide:
        #   0.3 = aggressive (drop CCs with any substantial snow overlap)
        #   0.5 = majority   (default — drop CCs that are mostly snow)
        #   0.7 = conservative (drop only CCs that are almost fully snow)
        # Set to 0 to disable the CC filter while still writing _snow.tif
        # rasters for diagnostic overlay.
        # snow_mask_close_px: morphological closing (binary_closing with
        # disk(r)) on the snow mask BEFORE dilation. Fills holes in the mask
        # smaller than the disk — designed to catch dark features embedded
        # in snow (animal tracks, mud spots, shadowed patches) that the V/S
        # threshold rejects on a per-pixel basis but that sit inside obvious
        # snow regions. 5px fills narrow tracks; 10–15px fills larger
        # embedded features at the cost of bridging nearby snow patches.
        # Set to 0 to disable.
        self.snow_mask_enabled = False
        self.snow_v_thresh = 0.85
        self.snow_s_thresh = 0.15
        self.snow_mask_close_px = 0
        self.snow_mask_dilate_px = 0
        self.snow_cc_drop_frac = 0.5
        self.write_snow_rasters = False

        # All artifacts from a tools/seep_level_eval.py run (per-chip rasters,
        # CSVs, GPKGs) land in {pred_dir}/{seep_eval_out_subdir}/ when this is
        # set. Lets A/B runs (e.g. snow-mask on vs off) write to parallel
        # subdirectories without overwriting the canonical baseline artifacts
        # directly in pred_dir. Set to None to write to pred_dir as before.
        self.seep_eval_out_subdir = None

        # Prediction outputs (for completeness with tools)
        # Attribute field in training_area_fn whose value is the .tif basename (no extension)
        # that each training area belongs to. When set, areas are matched to images by this
        # field rather than by spatial overlap — required when .tif files overlap spatially.
        # Set to None to fall back to the legacy spatial-overlap matching.
        self.image_link_field = "image_link"
        # change from train_image_file_type to train_image_type called by preprocessing
        self.train_image_type = self.image_file_type
        # change from train_images_prefix to train_image_prefix
        self.train_image_prefix = ""
        self.predict_images_file_type = self.image_file_type
        self.predict_images_prefix = ""
        self.overwrite_analysed_files = False
        self.prediction_name = self.run_name
        self.prediction_output_dir = None
        self.prediction_patch_size = None
        self.prediction_operator = "MAX"
        self.output_prefix = "det_" + self.prediction_name + "_"
        self.output_dtype = "bool"

        # ------ GPU / ENV ------
        self.selected_GPU = 3
        gdal.UseExceptions()
        gdal.SetCacheMax(32000000000)
        gdal.SetConfigOption("CPL_LOG", "/dev/null")
        warnings.filterwarnings("ignore")

        if int(self.selected_GPU) == -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.selected_GPU)

    def validate(self):
        # Basic path checks
        if not os.path.exists(self.training_data_dir):
            raise ConfigError(f"Invalid path: config.training_data_dir = {self.training_data_dir}")
        if not os.path.exists(os.path.join(self.training_data_dir, self.training_area_fn)):
            raise ConfigError(f"File not found: {os.path.join(self.training_data_dir, self.training_area_fn)}")
        if not os.path.exists(os.path.join(self.training_data_dir, self.training_polygon_fn)):
            raise ConfigError(f"File not found: {os.path.join(self.training_data_dir, self.training_polygon_fn)}")
        if not os.path.exists(self.training_image_dir):
            raise ConfigError(f"Invalid path: config.training_image_dir = {self.training_image_dir}")

        for cfg_dir in ["preprocessed_base_dir", "saved_models_dir", "logs_dir"]:
            target = getattr(self, cfg_dir)
            if not os.path.exists(target):
                try:
                    os.mkdir(target)
                except OSError as exc:
                    raise ConfigError(f"Unable to create folder config.{cfg_dir} = {target}") from exc

        if self.predict_images_file_type not in [".tif", ".jp2"]:
            raise ConfigError("Invalid format for config.predict_images_file_type. Supported: .tif, .jp2")
        if self.output_dtype not in ["bool", "uint8", "float32"]:
            raise ConfigError("Invalid config.output_dtype: choose 'bool', 'uint8' or 'float32'")
        return self


class ConfigError(Exception):
    pass