# tools/

Everything that happens **after** the segmentation model has run. The model
emits pixels; these scripts turn pixels into bubbles, bubbles into seeps, and
seeps into a methane flux.

Run any of them as a module from the repo root:

```bash
python -m tools.eval.bubble_level_eval
python -m tools.grouping.deploy_grouper PACK.gpkg --thr 0.6
```

---

## The naming convention

Three tiers, defined by **when the unit comes into existence**. The rule that
matters: **nothing is a "seep" until the grouper has run.**

| Tier | One unit is… | Exists after | Lives in |
|---|---|---|---|
| **pixel** | one pixel | model inference | `evaluation.py` (repo root) |
| **bubble** | one connected component, or one drawn polygon | connected components / labeling | `eval/`, `labeling/` |
| **seep** | one *group* of bubbles | the grouper runs | `grouping/`, `classify/` |

A file or column named `seep_*` that runs before `grouping/` is a naming bug.

---

## Pipeline order

```
        training.py ──► evaluation.py ──────────────► PIXEL metrics (Dice, IoU, F1)
                              │
                              ▼
        eval/write_bubble_rasters.py    smoothing + connected components
                              │
                              ▼
        eval/bubble_features.py         per-bubble morphology + brightness
                              │
                              ▼
        eval/bubble_level_eval.py ─────────────────► BUBBLE metrics (precision/recall/F1)
                              │
                              ▼
        grouping/train_grouper.py       learn pairwise "same-seep?"
        grouping/deploy_grouper.py      apply it, assign seep_group_id
                              │
                              ▼  ← a bubble becomes part of a SEEP here
        classify/fit_classifier.py      per-seep A/B/C
                              │
                              ▼
        flux/rates.py                   count-based flux + uncertainty
```

---

## The runner: `deploy.py` — orthomosaic to lake flux

**Training and deploying are separate.** The runner doesn't train: both
post-hoc models (grouper and classifier) are built once on disk with set seed. 

### Build once, per (checkpoint, label set)

```
  labeling/final_labeler_packs/*.gpkg ─┐
  labeling/gt_seeps_label_all_chips.gpkg ─┤
                                        ▼
                   deploy/build_artifacts.py
                     ├── grouping/deploy_grouper.py::train_model   ──► grouper_rf.joblib
                     ├── classify/fit_classifier.py::fit_deploy_model ──► classifier_rf.joblib
                     └── sklearn version + seed + sha256 of each pack ──► artifacts.json
                                        │
                                        ▼  written beside the model checkpoint
```

`python -m tools.deploy.build_artifacts`

### Then run

```
  ortho .tif ── lake .gpkg ── checkpoint .pt
        │
        ▼
  [1] DETECT          deploy/detect.py            GPU
        │             └─ deploy/tiles.py          15 m tiles inside the lake polygon
        │                                          (the tile IS the z-score window)
        │             └─ evaluation.py::_infer_full_image
        │             └─ eval/write_bubble_rasters.py   smooth + connected components
        ▼
     bubbles.gpkg     ← stage boundary: torch needed above, never below
        │
        ▼
  [2] FEATURES        deploy/postproc.py::load_bubbles
        │             └─ eval/bubble_features.py::polygonize_labels
        ▼
  [3] GROUP           deploy/postproc.py::group_bubbles       grouper_rf.joblib
        │             └─ grouping/deploy_grouper.py::_pair_features
        │             └─ grouping/deploy_grouper.py::constrained_cluster  cap 1.0 m
        │             └─ grouping/train_grouper.py            FEATURES, CAND_RADIUS
        ▼  ← bubbles become SEEPS here
  [4] DISSOLVE        deploy/postproc.py::dissolve_bubbles_to_seeps
        │             mirrors classify/fit_classifier.py::dissolve_to_seeps
        │             term for term -- change one, change both
        ▼
  [5] CLASSIFY        deploy/postproc.py::classify_seeps      classifier_rf.joblib
        │             └─ posterior, then the DECISION RULE:
        │                argmax (default) | conservative (classify/fit_classifier.py
        │                ::decide_with_cost, --overcall-penalty)
        ▼
  [6] FLUX            deploy/postproc.py::attach_flux / flux_report
        │             └─ flux/rates.py::lake_total, flux_table
        ▼
     seeps.gpkg  seeps.csv  flux_summary.csv  flux_per_image.csv
     lake_flux_totals_{run_id}.csv     run_info_postproc.json
```

`python deploy.py --image ORTHO.tif --lake LAKE.gpkg --out-dir OUT`
(`--stage detect` / `--stage postproc` to split across machines;
`--from-pred-dir` to exercise 2–6 against existing chips; `--refit` to fit from
the packs instead of the artifacts, which makes the run unreproducible.)

### Copy onto HPC

| File | What it is |
|---|---|
| `<ortho>.tif` | The whole-lake orthomosaic. `--image` |
| `<lake>.gpkg` | The lake polygon. Crops the ortho to the shoreline, so shore and snow-covered bank are never inferred over. `--lake` |
| `20260428-1537_SWINxAE.weights.pt` | The detector checkpoint. |
| `grouper_rf.joblib` | Frozen pairwise "same-seep?" forest. |
| `classifier_rf.joblib` | Frozen A/B/C forest. |
| `artifacts.json` | sklearn version, seed, and the sha256 of every input pack behind those two forests. |

The three artifact files live beside the checkpoint. `deploy_lake.slurm`
preflights all six before claiming the GPU.

---

## Run a new orthomosaic, start to finish

The whole procedure is three SLURM jobs: one GPU job for the flux number, then
two CPU jobs for the error bar. Each takes its inputs from environment
variables, so you override what changed and leave the rest at the defaults
recorded in the script.

### 1. Digitize the lake polygon

Inference is cropped to this polygon, and the polygon also defines
`surveyed_area_m2`. Without it,
the run covers the whole raster footprint including shore and snow-covered
land (skewing results high).

Save it as a GeoPackage in the ortho's own CRS. Pass `PATH` or `PATH:LAYER`.

### 2. Build the frozen artifacts, if they are missing or stale

```bash
python -m tools.deploy.build_artifacts
```

Rebuild only when the checkpoint changes or a labeler pack changes. The run
writes `grouper_rf.joblib`, `classifier_rf.joblib` and `artifacts.json` beside
the checkpoint; `artifacts.json` carries the sklearn version, the seed and the
sha256 of every input pack, which is the triple that makes a flux number
reproducible. A deploy run never trains — `--refit` exists for experiments and
makes the run unreproducible.

### 3. Smoke-test the detector on a few tiles

```bash
IMAGE=/path/ORTHO.tif LAKE=/path/LAKE.gpkg \
  OUT_DIR=/path/deploy_out/SMOKE EXTRA="--limit-tiles 20" \
  sbatch deploy_lake.slurm
```

Open `tiles.gpkg` and `bubbles.gpkg` in QGIS over the ortho. Confirm that the
tile grid sits inside the shoreline and that detections land on bubbles rather
than on snow or on the bank. Fixing a bad polygon here costs 20 tiles instead
of a whole lake.

### 4. Run the full lake

```bash
IMAGE=/path/ORTHO.tif LAKE=/path/LAKE.gpkg \
  RUN_NAME=<lake>_<YYYYmmdd> sbatch deploy_lake.slurm
```

`OUT_DIR` defaults to `$DEPLOY_ROOT/$RUN_NAME`, and `RUN_NAME` defaults to the
lake tag parsed out of the image filename plus today's date. The script
preflights the ortho, the polygon, the checkpoint and the three artifact files
before it claims the GPU.

To split the stages across machines, run `--stage detect` on the GPU node and
`--stage postproc` anywhere afterwards against the same `--out-dir`. Stage B
needs neither torch nor a GPU.

### 5. Check that the chain is the one on disk

```bash
python -m tools.deploy.test_chain OUT_DIR
```

Run this after the deploy job and after any edit to `chain.py`, `postproc.py`
or either uncertainty module. A SKIP means the run's `run_info_postproc.json`
predates the current screen and needs `deploy.py --stage postproc` re-run — a
skip is not a pass, so say so when you report it.

### 6. Read the outputs

`seeps.gpkg` carries the hulls, the A/B/C label, `p_A` / `p_B` / `p_C` and the
per-seep rate. `flux_summary.csv` carries the per-season totals,
`lake_flux_totals_<timestamp>.csv` the long per-season × class table, and
`run_info_postproc.json` the operating point every later step must reproduce.
`surveyed_area_m2` travels inside the GeoPackage, so the count never gets
separated from the area it was counted over.

Every flux column in every CSV is
written three ways:

| Column | Unit |
|---|---|
| `*_mg_CH4_per_day` | mg CH₄/day |
| `*_mg_CH4_per_m2_per_day` | mg CH₄ m⁻² day⁻¹ |
| `*_g_CH4_per_m2_per_year` | g CH₄ m⁻² year⁻¹ |

The conversion is linear, so it applies to every percentile and standard error
exactly as it applies to the central estimate. `tools/flux/rates.py::add_per_area_columns` does it in
one place for all of them.

The per-year column is **blank for the summer and winter rows**.

---

## Two uncertainty forward propagation methods

Both methods run against the deploy run's `bubbles.gpkg`, both are CPU-only,
and both call the same `tools.deploy.chain.run_chain` the deploy run calls. So
each one's point estimate reproduces the deploy total by construction, and each
fails loudly if its ensemble does not sit on that point estimate.

| | `uncertainty_labels` (method 1) | `uncertainty_proba` (method 2) |
|---|---|---|
| The total counts | seeps labeled *c* | Σ p_c over seeps |
| Sampled | both forests' trees, + the three published rates | each seep's class from its own posterior, + the three published rates |
| Held fixed | the P(same) threshold, the decision rule | the grouping, the threshold |
| Cost | re-groups every draw: hours, sharded across 8 workers | groups once: seconds |
| Variance decomposition | `--terms` one at a time | printed in closed form |
| Writes | `point.json`, `draws_*.csv`, `meta_*.json`, `summary.csv` | `point.json`, `draws.csv`, `summary.csv`, `threshold_sweep.csv` |

Method 2's total sits above method 1's because a map made by taking the most
likely class at each location systematically loses rare classes, and C carries
the most methane per seep (Olofsson et al. 2014, *Remote Sensing of
Environment* 148, 42–57). Method 2's total will differ from maps of classified seep occurance.
### Run method 1 — trees resampled, label-based total

```bash
RUN_DIR=/path/deploy_out/<run> DRAWS=500 sbatch uncertainty_labels.slurm
```

The job writes `point.json` first, forks `WORKERS` shards over the draws, then
summarizes. `DRAWS` is the total across every shard, and shard *i/N* takes
every *N*th draw, so a shard that dies still leaves a uniform sample behind.
Draw *i* is seeded `base_seed + i` and nothing else.

To pilot it, submit with `DRAWS=16 WORKERS=4` to `t1small` rather than to the
debug partition. To run a stage by hand:

```bash
python -m tools.deploy.uncertainty_labels --bubbles RUN/bubbles.gpkg \
    --out-dir RUN/uncertainty_labels --point
python -m tools.deploy.uncertainty_labels --bubbles RUN/bubbles.gpkg \
    --out-dir RUN/uncertainty_labels --draws 500 --shard 0/8
python -m tools.deploy.uncertainty_labels --summarize RUN/uncertainty_labels
```

Run `--point` before the shards. The centering check needs it, and without it
nothing can tell a valid interval from a miscenterd one. If the check fails,
re-run `--terms` one term at a time to find which term moved the center; do not
rescale the interval back onto the point estimate.

### Run method 2 — posterior sampled, closed form cross-checked

```bash
RUN_DIR=/path/deploy_out/<run> SWEEP=0.5,0.6,0.7 sbatch uncertainty_proba.slurm
```

One command does the point estimate, the draws, the threshold sweep and the
summary, because nothing is re-grouped between draws. By hand:

```bash
python -m tools.deploy.uncertainty_proba --bubbles RUN/bubbles.gpkg \
    --out-dir RUN/uncertainty_proba --draws 500 --threshold-sweep 0.5,0.6,0.7
```

The classifier term also has a closed form — the class count is a
Poisson-binomial, so `sd(N_c) = sqrt(Σ p_c(1−p_c))` — and `summarize` checks
that the simulation reproduces it. A failure there means the sampling loop is
wrong, not that the interval is wide.

`--threshold-sweep` re-runs the whole chain at each grouping threshold and
always includes the artifact's own operating point, flagged in the output.
Report that spread as its own line **beside** the interval, never inside it: the
threshold is a chosen operating point, not an unknown.

### What the interval does not contain

These should be included in a separate bias percentage:

- **Detector false positives.** At a precision below 1 the seep count is
  systematically high. Quote precision and recall beside the
  interval, with the domain named.
- **The grouping threshold.** Reported as the sweep, separately.
- **The correlated part of classifier error,** which the Poisson-binomial
  treats as a floor. The held-out-chip confusion matrix measures it as a bias.

The published per-class rates dominate the variance, and they are external to
the pipeline.

---

## `eval/` — detection metrics

| Script | Does |
|---|---|
| `write_bubble_rasters.py` | Morphological closing+opening (disk 1) then connected components. Writes `{stem}_smoothed.tif`, `{stem}_cc.tif`, optional `{stem}_snow.tif`. |
| `bubble_features.py` | Per-bubble area / perimeter / circularity / solidity / eccentricity / mean RGB. Also rebuilds GT bubble polygons from the **original drawn shapes** (`build_gt_bubbles_from_source`) rather than re-polygonizing a raster, which used to merge adjacent polygons. |
| `bubble_level_eval.py` | **The canonical detection metric.** Matches predicted CCs to GT CCs one-to-one. Writes the summary / per-image / pairs CSVs. |
| `gt_bubbles_export.py` | The single producer of `gt_bubbles.gpkg`, the per-bubble ground-truth layer every labeler pack is built from. |

There is deliberately **no seep-level eval**. The retired one grouped the GT
and predicted sides with the *same* hand-tuned rule, so grouping
error cancelled and it measured detection twice. See
`archive/polygon_matcher.py` for the honest RF-grouper version and what it
needs, and CLAUDE.md for the numbers.

## `labeling/` — QGIS labeler packs

Packs are **per-bubble**: one row per drawn polygon. Labelers correct the
grouping (`seep_group_id`) and fill the class (`A`/`B`/`C`).

| Script | Does |
|---|---|
| `strata.py` | `assign_strata` — exploratory (area × solidity) sampling bins. **Not** class thresholds. |
| `build_chip_pack.py` | Every polygon on one chip. |
| `build_quarter_packs.py` | Per-labeler quarter-chip packs with a shared calibration set and context rings. |
| `build_eval_chip_pack.py` | Every polygon on the 9 evaluation chips. |
| `apply_labeler_classes.py` | Merge a returned pack's `class` into a master pack, then promote each class to its seep group (max wins: C > B > A). |
| `salvage_chip39_north.py` | Applied one-shot (2026-09-11). Folded the 200 still-unique rows of the old chip-39 pack into allison's pack as `unit='39-N-extra'` and dropped the 206 her `39-NW` quarter supersedes. Refuses to run twice. |

**`final_labeler_packs/` is the only directory the grouper and the classifier
read.** Everything named after a person elsewhere under `labeling/` is in
`labeling/archive/`, whose README maps each file to what superseded it.

## `grouping/` — bubble → seep

| Script | Does |
|---|---|
| `train_grouper.py` | Trains and scores the pairwise "same-seep?" random forest on **all three final packs**. Candidate pairs ≤ 0.5 m; diameter-capped agglomeration at 1.0 m. Pairs are partitioned by `(labeler, image)` and weighted `1/(distinct labelers)`, so the shared calibration units do not carry 3× the influence. `--seed`; metrics CSVs land in `labeling/train_grouper/metrics/`. |
| `deploy_grouper.py` | Applies it to a pack, writes `seep_group_id` in place via sqlite, freezes the proposal in `seep_group_id_pred`, injects the `group_hull_by_class` QGIS style. |
| `group_predictions.py` | Same, but on **detected** bubbles instead of GT — the serve-time view. |
| `grouper_corrections.py` | Diffs a labeler-edited pack against the frozen proposal into accept/split/merge training records. |
| `build_hull_layer.py` | Precomputes per-seep hulls into a static layer. Fixes the QGIS render stall on multi-chip packs. |

## `classify/` — seep → class → flux

| Script | Does |
|---|---|
| `fit_classifier.py` | Fits and validates the A/B/C classifier on the three labeler packs; reports inter-labeler κ. Every number is weighted `1/(distinct labelers per phys_id)`, so metrics describe physical seeps rather than labeled rows. Also owns `decide_with_cost`, the conservative decision rule the runner can switch to. Writes `classifier_results_{date}_{time}.xlsx`. |
| `model_comparison.py` | Features × model families × training objective. **Not dead** — it is the harness for the two open questions (per-image brightness normalization, and re-testing ExtraTrees/GradientBoosting under it), and it owns `RICH`/`enrich`, which `c_augment_eval` imports. |
| `c_augment_eval.py` | Does adding hand-hunted C seeps help? C is the rare class that dominates the flux budget. The only measurement of cross-chip C recall on chips the model never saw. |

Both comparison scripts build their own tables and were **not** migrated to the
duplicate weighting, so their numbers are not on the same basis as
`fit_classifier`'s. Don't put them in one table.

## `flux/` — class → methane

| Script | Does |
|---|---|
| `rates.py` | **The single owner of the per-class flux rates** (annual / summer / winter, Walter Anthony et al. 2010). `lake_total` turns per-class seep counts into mg CH₄/day; `flux_table` / `write_flux_table` emit the per-season summary spreadsheet. Nothing else in the repo should define a rate. |

The ± figures are
**standard errors on each class mean**, so they are shared by every seep of a
class and do not average down. And `lake_total` takes an optional
`rng`, which draws each class rate from a moment-matched lognormal; that is the
hook both `deploy/uncertainty_*.py` modules draw the rate term through, and it
is the dominant term in either interval.

## `deploy/` — orthomosaic → lake total

The composed end-to-end runner. Everything above is a stage validated in
isolation; this is the only thing that chains them.

| Script | Does |
|---|---|
| `tiles.py` | Grids a whole-lake ortho into **15 m tiles inside the lake polygon**. Not a chunking convenience: the tile is the z-score window, and every canonical number in this repo was produced with that window at chip size. `tile_m` changes the detections. |
| `detect.py` | **Stage A, needs a GPU.** Runs `evaluation.py::_infer_full_image` per tile, masks to the lake polygon and the ortho's alpha band, smooths, labels connected components, and writes `bubbles.gpkg` + `run_info.json`. Refuses to run a checkpoint whose keys don't match the config, rather than loading it partially the way `evaluation.py` would. |
| `postproc.py` | **Stage B, CPU only.** `bubbles.gpkg` → grouper → dissolve + hull → classifier → decision rule → count-based flux. `--from-pred-dir` runs it against an existing prediction directory instead, which is how the chain gets exercised without a lake-scale run. |
| `chain.py` | Steps 1–4 live here **once**: `run_chain` is what `postproc.run` and both uncertainty modules call, so a no-sampling draw reproduces the deploy number by construction. Do not re-implement the chain in a new script. |
| `uncertainty_labels.py` | **Method 1.** Resamples both forests' trees and draws the rates; the headline stays the label-based total. Holds the threshold and the decision rule, and refuses to report an interval whose centre moved. |
| `uncertainty_proba.py` | **Method 2.** Draws each seep's class from its posterior and makes the headline the posterior sum, with the Poisson-binomial closed form as a cross-check and a grouping-threshold sweep beside the interval. |
| `test_chain.py` | Acceptance tests for the single-implementation chain. Run after any edit to `chain.py`, `postproc.py` or either uncertainty module. It SKIPS when its inputs are absent, and a skip is not a pass. |
| `build_artifacts.py` | Fits both post-hoc forests **once** and freezes them beside the checkpoint as `grouper_rf.joblib` / `classifier_rf.joblib` / `artifacts.json`. The manifest carries the sklearn version, the seed, and the sha256 of every input pack — the triple that makes a flux number reproducible. `load_artifacts` warns loudly on a sklearn mismatch rather than failing silently. |
| `runinfo.py` | Reads/writes the run metadata **inside** each output GeoPackage, as a registered `attributes` table. QGIS lists it, GDAL ignores it when reading the spatial layer. |
| — | **The entry point is `deploy.py` at the repo root**, not a module here. `deploy_lake.slurm` submits it. Each stage module keeps its own `main()` so it can be run alone when needed. |

The stage boundary is `bubbles.gpkg` on purpose: stage A needs torch, stage B
needs the geo + sklearn stack, so the slow half runs on HPC and the half you
actually read runs anywhere.

**`surveyed_area_m2` travels with every total**, and it travels *inside* the
GeoPackage rather than in a sidecar, so copying `seeps.gpkg` somewhere can't
separate the count from the area it was counted over. It is measured: valid pixels (alpha band ∩ lake polygon) accumulated per tile core,
including tiles where nothing was detected.

## Other

| Path | Does |
|---|---|
| `historical_size_priors.py` | Distills the historical field workbooks into grouping geometry + per-class size priors. |
| `viz/` | Plotting notebooks: loss curves, example frames, prediction overlays. |
| `archive/` | Superseded, rejected, or already-applied. **Untracked** (local only; recover from git history). Nothing here is imported by live code — see `archive/README.md`. |
