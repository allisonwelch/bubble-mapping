# configTerraMindxPS.py

import os
import warnings
import numpy as np
from osgeo import gdal


class Configuration:
    """
    Configuration used by preprocessing.py, training.py, tuning.py and evaluation.py (TerraMind).
    Only parameters referenced by the TerraMind workflows are included.
    """

    def __init__(self):
        # --------- RUN NAME ---------
        # Modality to be run can be AE, PS or S2
        self.modality = "PS"

        self.run_name = f"TERRAMINDx{self.modality}"

        # ---------- PATHS -----------

        # Training data and imagery
        self.training_data_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/training/{self.modality}"
        )
        self.training_area_fn = "training_areas.gpkg"
        self.training_polygon_fn = f"labels_{self.modality}.gpkg"
        self.focus_areas = f"focus_areas_{self.modality}.gpkg"

        self.training_image_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/training_images/{self.modality}"
        )

        # Preprocessed data roots
        self.preprocessed_base_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/training_data/{self.modality}"
        )
        self.preprocessed_dir = (
            "/isipd/projects/p_planetdw/data/methods_test/preprocessed/"
            "20260109-1434_UNETxPS"
        )

        # Checkpointing / logs / results (model + modality subfolders)
        self.continue_model_path = None
        self.saved_models_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/models/TERRAMIND/{self.modality}"
        )
        self.logs_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/logs/TERRAMIND/{self.modality}"
        )
        self.results_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/results/TERRAMIND/{self.modality}"
        )

        # -------- IMAGE / CHANNELS --------
        self.image_file_type = ".jp2"
        self.resample_factor = 3

        if self.modality != "S2":
            self.channels_used = [True, True, True, True]
        else:
            self.channels_used = [True, True, True, True, True, True, True, True, True, True, True, True]
            
        self.preprocessing_bands = np.where(self.channels_used)[0]
        self.channel_list = self.preprocessing_bands
        self.num_classes = 1  # TerraMind head classes (1 for binary)

        # -------- DATA SPLIT --------
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        # train is 1 - test_ratio - val_ratio

        # -------- TRAINING (CORE) --------
        self.patch_size = (128, 128)
        self.tversky_alphabeta = (0.59, 0.41)
        self.train_batch_size = 16
        self.num_epochs = 100
        self.num_training_steps = 500
        self.num_validation_images = 50

        # ------ LOSS / OPTIM / SCHED ------
        self.loss_fn = "tversky"
        self.optimizer_fn = "adam"
        self.learning_rate = 1e-3          # used if you don't use tm_* LRs
        self.weight_decay = 3e-6           # NEW: global weight decay (for AdamW / tuner "weight_decay")
        self.scheduler = "cosine"            # NEW: "none" | "cosine" | "onecycle"

        self.dilation_rate = 1  # unused for TerraMind itself, but kept for compatibility
        self.model_name = self.run_name
        self.boundary_weight = 5
        self.model_save_interval = None

        # ------ EMA ------
        self.use_ema = False
        self.ema_decay = 0.999
        self.eval_with_ema = False

        # ------ CHECKPOINTING / LOGGING ------
        self.overfit_one_batch = False
        self.train_verbose = True
        self.train_epoch_log_every = 1
        self.train_print_heavy = True
        self.show_progress = True
        self.log_visuals_every = 1
        self.vis_rgb_idx = (0, 1, 2)

        # ------ AUG / SAMPLING / DATALOADER ------
        self.augmenter_strength = 1.0
        self.min_pos_frac = 0.0005
        self.pos_ratio = 0.25
        self.patch_stride = None
        self.fit_workers = 0
        self.steps_per_execution = 1

        # ------ EVALUATION ------
        self.eval_threshold = 0.5
        self.heavy_eval_steps = 50
        self.print_pos_stats = True
        self.metrics_class = 1
        self.viz_class = 1
        self.eval_mc_dropout = True
        self.eval_mc_samples  = 20

        # ------ MIXED PRECISION / COMPILE / REPRO ------
        self.use_torch_compile = False
        self.seed = None
        self.clip_norm = 0.0

        # -------- TERRAMIND ARCH KNOBS --------
        self.tm_backbone = "terramind_v1_base"   # or size token: 'tiny','small','base','large'
        self.tm_decoder = "UNetDecoder"          # tuned: 'UNetDecoder' | 'UperNetDecoder'
        self.tm_decoder_channels = [384, 192, 96, 48]  # list for UNetDecoder, int for UperNetDecoder
        self.tm_head_dropout = 0.18               # NEW: tuned "tm_head_dropout"
        self.tm_select_indices = None
        self.tm_bands = None
        self.tm_backbone_ckpt_path = None
        self.terramind_merge_method = "mean"
        self.terramind_size = "base"
        self.tm_freeze_backbone = False          # full freeze at init

        # --- TerraMind optimizer / schedule knobs (from tuner) ---       # NEW block
        # These are used in training._build_optimizer_and_scheduler()
        # when training a TerraMind model.
        self.tm_lr_backbone = 1.9e-05               # NEW: tuned "tm_lr_backbone"
        self.tm_lr_head_mult = 10.0              # NEW: tuned "tm_lr_head_mult"
        self.tm_weight_decay = 1.0e-06              # NEW: tuned "tm_weight_decay"
        self.tm_freeze_backbone_epochs = 5       # NEW: freeze backbone for first N epochs

        # (Optionally) keep tversky_alpha around for direct copy from tuner JSON  # NEW (optional)
        self.tversky_alpha = self.tversky_alphabeta[0]

        # --- POSTPROCESSING (kept for downstream scripts) ---
        self.create_polygons = True
        self.postproc_workers = 12

        # Prediction outputs (for later inference scripts)
        self.train_image_file_type = self.image_file_type
        self.train_images_prefix = ""
        self.predict_images_file_type = self.image_file_type
        self.predict_images_prefix = ""
        self.overwrite_analysed_files = False
        self.prediction_name = self.run_name
        self.prediction_output_dir = None
        self.prediction_patch_size = None
        self.prediction_operator = "MAX"
        self.output_prefix = "INF_" + self.prediction_name + "_"
        self.output_dtype = "bool"

        # ------ GPU / ENV ------
        self.selected_GPU = 4  # CUDA device index, -1 for CPU
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
