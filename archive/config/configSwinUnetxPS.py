# configSwinUnet.py

import os
import warnings
import numpy as np
from osgeo import gdal


class Configuration:
    """
    Configuration used by preprocessing.py, training.py, tuning.py and evaluation.py (Swin-UNet).
    Only includes parameters referenced by the Swin workflows.
    """

    def __init__(self):
        # --------- RUN NAME ---------
        # Modality to be run can be AE, PS or S2
        self.modality = "PS"

        self.run_name = f"SWINx{self.modality}"

        # ---------- PATHS -----------

        # Training data and imagery
        self.training_data_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/training/{self.modality}"
        )
        self.training_area_fn = "training_areas.gpkg"
        self.training_polygon_fn = f"labels_{self.modality}.gpkg"
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
            f"/isipd/projects/p_planetdw/data/methods_test/models/SWIN/{self.modality}"
        )
        self.logs_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/logs/SWIN/{self.modality}"
        )
        self.results_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/results/SWIN/{self.modality}"
        )

        # -------- IMAGE / CHANNELS --------
        self.image_file_type = ".tif"
        self.resample_factor = 1
        
        if self.modality != "S2":
            self.channels_used = [True, True, True, True]
        else:
            self.channels_used = [True, True, True, True, True, True, True, True, True, True, True, True]
            
        self.preprocessing_bands = np.where(self.channels_used)[0]
        self.channel_list = list(self.preprocessing_bands)

        # -------- DATA SPLIT --------
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        # train is 1 - test_ratio - val_ratio

        # -------- TRAINING (CORE) --------
        self.patch_size = (448, 448)
        self.tune_patch_h = None
        self.tune_patch_w = None
        self.tversky_alphabeta = (0.7, 0.3)
        self.model_name = self.run_name

        # ------ OPTIM / SCHED / EPOCHS ------
        self.loss_fn = "tversky"
        self.optimizer_fn = "adamw"

        # These three are used directly in training.py and match tuner names   # NEW
        self.learning_rate = 0.0001       # NEW: tuned "learning_rate" goes here
        self.weight_decay = 0.00016         # NEW: tuned "weight_decay" (for AdamW)
        self.scheduler = "onecycle"      # NEW: tuned "scheduler" ("none"|"cosine"|"onecycle")

        self.train_batch_size = 8
        self.num_epochs = 100
        self.num_training_steps = 500
        self.num_validation_images = 50

        # ------ EMA ------
        self.use_ema = True
        self.ema_decay = 0.999
        self.eval_with_ema = True

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
        self.min_pos_frac = 0.0001
        self.pos_ratio = 0.0
        #self.patch_stride = 0.3
        self.fit_workers = 0
        self.steps_per_execution = 1

        # ------ EVALUATION ------
        self.eval_threshold = 0.5
        self.heavy_eval_steps = 50
        self.print_pos_stats = True
        self.eval_mc_dropout = True
        self.mc_dropout_samples = 20

        # ------ MIXED PRECISION / COMPILE / REPRO ------
        self.use_torch_compile = False
        self.seed = None
        self.clip_norm = 0.0

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

        # Prediction outputs (for completeness with tools)
        self.train_image_file_type = self.image_file_type
        self.train_images_prefix = ""
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
        self.selected_GPU = 4
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