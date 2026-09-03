# DriftwoodMappingBenchmark

A complete pipeline to preprocess imagery, tune hyperparameters, train deep models, and evaluate segmentation performance for driftwood mapping across multiple modalities (AE, PS, S2). Entry points are provided for UNet, Swin-UNet, and the TerraMind foundation model.


## Purpose

- Compare architectures (UNet, Swin-UNet, TerraMind) across image modalities (Aerial/MACS, PlanetScope, Sentinel-2).
- Provide a reproducible end-to-end workflow: preprocessing → tuning → multi-run training → evaluation.
- Produce logs, saved models, prediction rasters, and CSV summaries that can be evaluated using Bayesian hirarchical models (not yet included here).

---

## Quickstart


### Configuration

Pick the config for your model and set the `modality` and paths:

- `config/configUnet.py`
- `config/configSwinUnet.py`
- `config/configTerraMind.py`

Each config:
- Uses `modality` (e.g., "AE", "PS", "S2").
- Writes model artifacts to per-model/per-modality subfolders, for example:
  - `.../models/UNET/<MODALITY>`
  - `.../logs/SWIN/<MODALITY>`
  - `.../results/TERRAMIND/<MODALITY>`

### Run

Using pixi tasks (from your `pixi.toml`):

```toml
[tasks]
unet  = "python ./mainUnet.py"
swin  = "python ./mainSwinUnet.py"
terra = "python ./mainTerramind.py"
```

Execute:

```bash

# full pipelines
pixi run unet
pixi run swin
pixi run terra
```

Each main script typically runs:
1) `preprocessing.preprocess_all(config)`
2) `tuning.tune_<Model>(config)`  (hyperparameter search)
3) Loop of training runs (e.g., 10)
4) `evaluation.evaluate_<Model>(config)` (saves masks + CSV rows)

### TerraTorch Backbone Sweep (Experimental)

The `terratorch_benchmark.py` script allows testing multiple TerraTorch foundation model backbones:

```bash
python terratorch_benchmark.py --epochs 10 --steps-per-epoch 200 --out backbone_results.csv
```

Options:
- `--epochs`: Number of training epochs per backbone (default: 8)
- `--steps-per-epoch`: Training steps per epoch (default: 200)
- `--val-steps`: Validation steps per epoch (default: 50)
- `--lr`: Learning rate (default: 1e-4)
- `--weight-decay`: Weight decay (default: 0.05)
- `--include`: Filter backbones by substring (e.g., `--include prithvi --include terramind`)
- `--max-models`: Limit number of backbones to test
- `--no-pretrained`: Skip loading pretrained weights
- `--no-amp`: Disable automatic mixed precision

The script will:
1. Load config from `config.configTerraMind` (or `configUnet`/`configSwinUnet`)
2. Use the project's data pipeline to load preprocessed frames
3. Test each backbone from the TerraTorch registry
4. Save results (validation loss, parameters, training time) to CSV

---

## Project layout

```
.
├─ mainUnet.py             # UNet pipeline entry point
├─ mainSwinUnet.py         # Swin-UNet pipeline entry point
├─ mainTerraMind.py        # TerraMind pipeline entry point
├─ terratorch_benchmark.py # TerraTorch backbone sweep (experimental)
|
├─ preprocessing.py        # builds frame dataset (GeoTIFFs with [bands | label | boundary])
├─ tuning.py               # hyperparameter tuning (model-specific entrypoints)
├─ training.py             # shared PyTorch training loop (BF16/FP16 AMP, channels_last, EMA)
├─ evaluation.py           # batch evaluation: predicts masks + writes metrics CSV
|
├─ config/
│  ├─ configUnet.py        # UNet config (model+modality-aware paths)
│  ├─ configSwinUnet.py    # Swin config (model+modality-aware paths)
│  └─ configTerraMind.py   # TerraMind config (model+modality-aware paths)
|
├─ core/
│  ├─ UNet.py              # UNet model
│  ├─ Swin_UNetPP.py       # Swin-UNet model
│  ├─ TerraMind.py         # TerraMind wrapper
|  |
│  ├─ common/
│  │  ├─ console.py        # colored banners, pretty printing
│  │  ├─ model_utils.py    # AMP helpers, EMA, logits→probs, autopad forward
│  │  ├─ data.py           # frame discovery, dataset creation
│  │  └─ vis.py            # TensorBoard image logging
|  |
│  ├─ losses.py            # dice/tversky + helpers
│  ├─ optimizers.py        # get_optimizer()
│  ├─ split_frames.py      # split_dataset(), stats
│  ├─ dataset_generator.py # iterable patch generator
│  └─ frame_info.py        # FrameInfo container, normalization utils
└─ tools/                  # post-inference pipeline (see tools/README.md)
   ├─ eval/                # pixel -> BUBBLE detection metrics
   ├─ labeling/            # build + maintain QGIS labeler packs
   ├─ grouping/            # bubble -> SEEP (learned random-forest grouper)
   ├─ classify/            # per-seep A/B/C class + count-based flux
   ├─ viz/                 # plotting notebooks
   └─ archive/             # superseded / rejected, kept for provenance
```

Run any tool as a module from the repo root, e.g.

```bash
python -m tools.eval.bubble_level_eval
python -m tools.grouping.deploy_grouper PACK.gpkg
```

---

## Results

- Logs: TensorBoard scalars + images  
  `logs/<MODEL>/<MODALITY>/<run>/...`
- Models: best and periodic snapshots  
  `models/<MODEL>/<MODALITY>/<timestamp>_name.*`
- Evaluation outputs:
  - GeoTIFF masks per test frame per checkpoint  
    `results/<MODEL>/<MODALITY>/<checkpoint_basename>/...`
  - CSV summary per architecture:  
    `evaluation_unet.csv`, `evaluation_swin.csv`, `evaluation_tm.csv`

Open TensorBoard:

```bash
tensorboard --logdir <path-to-logs> --port 6006
```

---

## Data assumptions

- Preprocessing expects AOI and polygon files (`training_area_fn`, `training_polygon_fn`) and raw imagery under `training_image_dir`.
- Preprocessed frames are GeoTIFFs with bands:
  ```
  [input_bands..., label_band, boundary_band]
  ```
- `config.preprocessed_dir` can be set to reuse a prepared dataset.


