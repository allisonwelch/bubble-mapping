# configUnet.py

import os
import warnings
import numpy as np
from osgeo import gdal


class Configuration:
    """
    Configuration used by preprocessing.py, training.py, tuning.py and evaluation.py (UNet).
    Only includes parameters actually referenced by the UNet workflows.
    """

    def __init__(self):
        # --------- RUN NAME ---------
        # Modality to be run can be AE, PS or S2
        self.modality = "S2"
        
        self.run_name = f"UNETx{self.modality}"

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

        self.split_list_path = "/isipd/projects/p_planetdw/data/methods_test/preprocessed/20251226-0433_UNETxAE/aa_frames_list.json"


        # Preprocessed data roots
        self.preprocessed_base_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/preprocessed"
        )

        self.training_data_base_dir = (
            f"/isipd/projects/p_planetdw/data/methods_test/training_data/"
        )
        # Keep explicit override exactly as provided
        self.preprocessed_dir = (
            "/isipd/projects/p_planetdw/data/methods_test/preprocessed/"
            "20260108-1335_UNETxS2"
        )



        # Checkpointing / logs / results
        self.continue_model_path = None
        self.saved_models_dir = f"/isipd/projects/p_planetdw/data/methods_test/models/UNET/{self.modality}"
        self.logs_dir = f"/isipd/projects/p_planetdw/data/methods_test/logs/UNET/{self.modality}"
        self.results_dir = f"/isipd/projects/p_planetdw/data/methods_test/results/UNET/{self.modality}"

        # -------- IMAGE / CHANNELS --------
        self.image_file_type = ".tif"
        self.resample_factor = 1
        
        if self.modality != "S2":
            self.channels_used = [True, True, True, True]
        else:
            self.channels_used = [True, True, True, True, True, True, True, True, True, True, True, True]

        self.preprocessing_bands = np.where(self.channels_used)[0]
        self.channel_list = self.preprocessing_bands
        self.rasterize_borders = True
        self.get_json = False

        # -------- DATA SPLIT --------
        self.test_ratio = 0.2
        self.val_ratio = 0.2
        # train is 1 - test_ratio - val_ratio

        self.split_list_path = None  # Optional path to predefined train/val/test split lists

        # -------- TRAINING (CORE) --------
        self.patch_size = (256, 256)
        self.tune_patch_h = 256
        self.tune_patch_w = 256
        self.tversky_alphabeta = (0.63, 0.37) #alpha controls penalty for false negatives, beta for false positives
        self.dilation_rate = 2
        self.dropout = 0.1
        # Tuned UNet architecture / regularization params 
        self.layer_count = 96     
        self.l2_weight = 1e-5       
        self.model_name = self.run_name

        # ------ OPTIM / SCHED / EPOCHS ------
        self.loss_fn = "tversky"
        self.optimizer_fn = "adam"
   
        self.learning_rate = 0.00042
        self.weight_decay = 4.8e-6
        self.scheduler = "onecycle"
        
        self.train_batch_size = 1
        self.num_epochs = 100
        self.num_training_steps = 500
        self.num_validation_images = 50

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
        self.min_pos_frac = 1e-5
        self.pos_ratio = 0.5
        self.patch_stride = None
        self.fit_workers = 8
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

        # --- POSTPROCESSING (kept for downstream scripts) ----
        self.create_polygons = True
        self.postproc_workers = 12

        # Prediction outputs (for completeness with your tools)
        self.train_image_type = self.image_file_type
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
        self.selected_GPU = 7
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