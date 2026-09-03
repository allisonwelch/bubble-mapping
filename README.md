# bubble-mapping

Mapping methane ebullition seeps in frozen Arctic lake ice from 1 cm/px aerial
imagery, and turning them into a lake-total methane flux.

The work splits into **two pipelines that meet at a raster of predicted pixels**:

| | Pipeline | What it produces | Where it lives |
|---|---|---|---|
| **A** | **Detection** — Swin-UNet++ segmentation | a binary bubble mask per chip | repo root + `core/` |
| **B** | **Interpretation** — group, classify, integrate | seeps, their A/B/C class, and flux | `tools/` |

Pipeline A is a fairly ordinary supervised segmentation setup inherited from the
DriftwoodMappingBenchmark codebase (its original README is kept locally at
`archive/README_DriftwoodMappingBenchmark.md`).
Pipeline B is the part specific to this project. **Flux is count-based**, so both Swin-UNet bubble detection as well as grouping detected bubbles into seeps and classifying them based on a established classification system (Walter Anthony et al., 2010) matters for flux estimation.

---

## Naming Conventions


| Tier | One unit is… | Exists after | Code |
|---|---|---|---|
| **pixel** | one pixel | model inference | `evaluation.py` |
| **bubble** | one connected component, or one drawn polygon | connected components / labeling | `tools/eval/`, `tools/labeling/` |
| **seep** | one **group** of bubbles | the grouper runs | `tools/grouping/`, `tools/classify/` |

In the GeoPackages this means **`bubble_id`** is the per-polygon id
and **`seep_group_id`** is the seep id; keys are always the tuple
`(image, seep_group_id)`, never the group id alone, because ids restart per chip.

---

## Pipeline A — detection (Swin-UNet++)

```
 training_areas.gpkg          preprocessing.py     one GeoTIFF per training area:
 labels_*.gpkg          ──►   preprocess_all()  ─► [R, G, B, label, boundary]
 raw aerial .tif                                   + aa_frames_list.json (the split)
                                                          │
                                       training.py        ▼
                                  train_SwinUNetPP()  ── checkpoints ──►  models/SWIN/AE/
                                                          │
                                       evaluation.py      ▼
                                 evaluate_SwinUNetPP() ─► per-chip masks + PIXEL metrics
```

**Run** `mainSwinUnet.py`; its stages are commented in and
out by hand. On the cluster each stage has its own Slurm script:

```bash
sbatch preprocessing.slurm     # python preprocessing.py --config config.configSwinUnet --arch swin
sbatch training.slurm          # 24 h, h100
sbatch evaluation.slurm        # 2 h, h100
sbatch check_config.slurm      # 5 min sanity check that the config validates
```

**Config.** `config/configSwinUnet.py` is the live one (`configUnetxAE.py` is the
UNet baseline kept for comparison; everything non-AE is in `archive/config/`).
Paths are model- and modality-aware: `models/SWIN/AE`, `logs/SWIN/AE`,
`results/SWIN/AE`.

**Outputs**, per checkpoint, under `results/SWIN/AE/<checkpoint>/`:

| File | Contents |
|---|---|
| `<chip>.tif` | hard binary mask |
| `<chip>_prob.tif` | probability map |
| `<chip>_epistemic.tif` / `_aleatoric.tif` | MC-dropout uncertainty |
| `../evaluation_swin.csv` | one row per checkpoint: Dice, IoU, F1, F-beta, sensitivity, specificity, Hausdorff, boundary IoU, uncertainty |

`evaluation.py` then calls Pipeline B's bubble-level eval automatically, so a
single evaluation run yields both pixel and bubble metrics.

---

## Pipeline B — interpretation (bubbles → seeps → flux)

```
 predicted mask
       │
       ▼
 tools/eval/write_bubble_rasters.py     morphological closing+opening (disk 1),
       │                                then connected components
       ▼                                -> <chip>_smoothed.tif, <chip>_cc.tif
 tools/eval/bubble_features.py          per-bubble area, perimeter, circularity,
       │                                solidity, eccentricity, mean RGB
       ▼
 tools/eval/bubble_level_eval.py  ────► BUBBLE metrics: precision / recall / F1
       │
       ▼   ◄── a bubble becomes part of a SEEP here ──────────────────────────
 tools/grouping/train_grouper.py        learned pairwise "same-seep?" random forest
 tools/grouping/deploy_grouper.py       candidate pairs <= 0.5 m -> P(same) -> threshold
       │                                -> diameter-capped agglomeration (1.0 m)
       ▼                                -> seep_group_id
 tools/classify/fit_classifier.py       per-seep A / B / C
       │
       ▼
 count-based flux = SUM over class (n_seeps * per-class rate)
```


### The labeling loop

Labelers work in QGIS on per-bubble packs to train the grouper (deploy_grouper.py) and the classifier (fit_classifier.py), correcting the model's proposed
grouping and filling the class:

```bash
python -m tools.labeling.build_eval_chip_pack        # build a pack
python -m tools.grouping.deploy_grouper PACK.gpkg    # pre-group it + inject the QGIS style
python -m tools.grouping.build_hull_layer PACK.gpkg  # static hulls (fixes the render stall)
#   ... labeling happens in QGIS ...
python -m tools.grouping.grouper_corrections PACK_grouped.gpkg   # edits -> training pairs
python -m tools.classify.fit_classifier                          # refit + inter-labeler kappa
```

`deploy_grouper` freezes its proposal in an immutable `seep_group_id_pred`
column, so a returned pack can be diffed back into grouper training data.

---

## Layout

```
.
├─ mainSwinUnet.py          # Pipeline A entry point (stages commented in/out)
├─ mainUnet.ipynb           # notebook variant
├─ preprocessing.py         # AOIs + labels + imagery -> training chips
├─ training.py              # shared training loop (AMP, channels_last, EMA, early stopping)
├─ tuning.py                # Optuna hyperparameter search
├─ evaluation.py            # per-checkpoint inference -> masks + PIXEL metrics
├─ *.slurm                  # cluster job scripts
│
├─ config/
│  ├─ configSwinUnet.py     # LIVE config (SWIN x AE)
│  └─ configUnetxAE.py      # UNet baseline, kept for comparison
│
├─ core/
│  ├─ Swin_UNetPP.py        # Swin-UNet++ model
│  ├─ UNet.py               # UNet model
│  ├─ TerraMind.py          # TerraMind wrapper (unused here)
│  ├─ losses.py             # dice / tversky
│  ├─ optimizers.py, split_frames.py, dataset_generator.py, frame_info.py
│  └─ common/               # console, model utils, data discovery, TensorBoard vis
│
├─ tools/                   # Pipeline B  (see tools/README.md for per-script detail)
│  ├─ eval/                 # pixel -> BUBBLE detection metrics
│  ├─ labeling/             # build + maintain QGIS labeler packs
│  ├─ grouping/             # bubble -> SEEP (learned RF grouper)
│  ├─ classify/             # per-seep A/B/C class + count-based flux
│  ├─ viz/                  # plotting notebooks
│  └─ archive/              # superseded / rejected  (untracked, local only)
│
├─ analysis/                # Bayesian benchmark stats (architecture comparison)
└─ archive/                 # non-AE entry points, unused configs, upstream README
                           #   (untracked, local only — see git history to recover)
```

Every tool runs as a module from the repo root:

```bash
python -m tools.eval.bubble_level_eval
python -m tools.grouping.deploy_grouper PACK.gpkg --thr 0.6 --cap 1.0
```


---

## Environment

Local Windows development uses `venv-bubble/` (Python 3.14, `requirements-win.txt`).
`pixi.toml` defines the linux-64 / CUDA 12.1 cluster environment. `example_env.yml`
is the conda equivalent.

```bash
venv-bubble/Scripts/tensorboard --logdir logs/SWIN/AE --port 6006
```

Detailed history, decisions and negative results are in `CLAUDE.md`; per-script
documentation is in `tools/README.md`.
