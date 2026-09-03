# bubble-mapping

Mapping methane ebullition seeps in frozen Arctic lake ice from 1 cm/px aerial
imagery, and turning them into a lake-total methane flux.

The work splits into **two pipelines that meet at a raster of predicted pixels**:

| | Pipeline | What it produces | Where it lives |
|---|---|---|---|
| **A** | **Detection** — Swin-UNet++ segmentation | a binary bubble mask per chip | repo root + `core/` |
| **B** | **Interpretation** — group, classify, integrate | seeps, their A/B/C class, and flux | `tools/` |

Pipeline A is a fairly ordinary supervised segmentation setup inherited from the
[DriftwoodMappingBenchmark](archive/README_DriftwoodMappingBenchmark.md) codebase.
Pipeline B is the part specific to this project, and it is where most of the
research risk lives: **flux is count-based**, so how bubbles are grouped into
seeps and how those seeps are classified drives the headline number far more
than detector quality does.

---

## The naming convention (read this first)

Three tiers, defined by **when the unit comes into existence**. The rule that
matters: **nothing is a "seep" until the grouper has run.**

| Tier | One unit is… | Exists after | Code |
|---|---|---|---|
| **pixel** | one pixel | model inference | `evaluation.py` |
| **bubble** | one connected component, or one drawn polygon | connected components / labeling | `tools/eval/`, `tools/labeling/` |
| **seep** | one **group** of bubbles | the grouper runs | `tools/grouping/`, `tools/classify/` |

A file, column or variable named `seep_*` that operates *before* the grouper is
a naming bug. In the GeoPackages this means **`bubble_id`** is the per-polygon id
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

**Run it.** `mainSwinUnet.py` is the entry point; its stages are commented in and
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

> ⚠️ **The chip dir in the config is not the one that reproduces the canonical
> metrics.** Use `data/preprocessed/2026-04-22_SWINxAE` — it is the only SWIN
> preprocessed dir whose GT label band matches the canonical per-image `n_gt` on
> all 9 labeled chips. `2026-04-27_SWINxAE` (the config default) differs on 7 of
> 9 and yields bubble F1 = 0.6135 instead of 0.6449. Nothing errors; you just get
> a quietly wrong number.

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
 tools/eval/bubble_level_eval.py  ────► BUBBLE metrics: P / R / F1 = 0.645
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

### Why the grouper is a learned model and not a rule

A hand-tuned anchor/radius rule cannot hold big spread-out seeps together
*and* avoid bridging dense fields of small ones with a single threshold. The
learned pairwise model gets seep counts to within ~2% of hand delineation
(360 vs 365) where the old rule over-fragmented by +119 seeps (+38% flux).
Inter-bubble distance dominates the model; it is essentially a learned
distance-join, but it is the right framework as labels grow.

The old rule is **deleted** from the pipeline, and the seep-level metric it used
to produce (`cluster_f1 = 0.672`) is **retired** — it grouped the ground-truth
*and* predicted sides with the same rule, so grouping error cancelled and it was
a detection metric wearing a seep-level label. See `tools/archive/polygon_matcher.py`
for the recipe for an honest replacement.

### Why the classifier matters more than the grouper

Per-seep field flux rates are **A = 16, B = 131, C = 971 mg CH₄/day**. C is ~4%
of seeps but **~52% of the methane**, so cross-chip C recall is the dominant
error term. Swapping the grouper moves flux a few points; getting the classes
wrong moves it by hundreds. The ±14.2% uncertainty on the rates themselves is a
floor no classifier can beat — always report it alongside.

### The labeling loop

Labelers work in QGIS on per-bubble packs, correcting the model's proposed
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
│  └─ archive/              # superseded / rejected, kept for provenance
│
├─ analysis/                # Bayesian benchmark stats (architecture comparison)
└─ archive/                 # non-AE entry points, unused configs, upstream README
```

Every tool runs as a module from the repo root:

```bash
python -m tools.eval.bubble_level_eval
python -m tools.grouping.deploy_grouper PACK.gpkg --thr 0.6 --cap 1.0
```

---

## Current numbers

| Stage | Metric | Value |
|---|---|---|
| pixel | Dice / IoU | 0.68 / 0.52 |
| **bubble** | **precision / recall / F1** | **0.571 / 0.741 / 0.645** |
| seep (grouping) | seep count vs hand delineation | within ~2% (360 vs 365) |
| seep (class) | Cohen's κ | 0.62, vs human-human 0.52–0.73 |

**Caveats that must travel with these.** Each stage is validated *in isolation* —
the grouper assumes oracle classes, the classifier assumes correct grouping,
neither includes detector error, and **there is no composed end-to-end flux
number**. There is currently **no seep-level detection metric** at all. Always
quote the cross-chip (leave-one-image-out) regime, never grouped CV, which reads
optimistic because it shares chips between train and test.

The classifier κ and class-balance figures recorded in `CLAUDE.md` for 2026-07-29
are **stale** — the labeler packs were completed substantially in August (one went
from 289 to 1138 classified rows) — and should be recomputed before being cited.

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
