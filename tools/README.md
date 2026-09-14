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

Two things to know before quoting a number from here. The ± figures are
**standard errors on each class mean**, so they are shared by every seep of a
class and do not average down — that is why the rate term is a floor rather
than something more mapping can shrink. And `lake_total` takes an optional
`rng`, which draws each class rate from a moment-matched lognormal; that is the
hook for the Monte Carlo error propagation, and it is the only part of the
uncertainty story that covers detector / grouper / classifier error.

## `deploy/` — orthomosaic → lake total

The composed end-to-end runner. Everything above is a stage validated in
isolation; this is the only thing that chains them.

| Script | Does |
|---|---|
| `tiles.py` | Grids a whole-lake ortho into **15 m tiles inside the lake polygon**. Not a chunking convenience: the tile is the z-score window, and every canonical number in this repo was produced with that window at chip size. `tile_m` changes the detections. |
| `detect.py` | **Stage A, needs a GPU.** Runs `evaluation.py::_infer_full_image` per tile, masks to the lake polygon and the ortho's alpha band, smooths, labels connected components, and writes `bubbles.gpkg` + `run_info.json`. Refuses to run a checkpoint whose keys don't match the config, rather than loading it partially the way `evaluation.py` would. |
| `postproc.py` | **Stage B, CPU only.** `bubbles.gpkg` → grouper → dissolve + hull → classifier → decision rule → count-based flux. `--from-pred-dir` runs it against an existing prediction directory instead, which is how the chain gets exercised without a lake-scale run. |
| `build_artifacts.py` | Fits both post-hoc forests **once** and freezes them beside the checkpoint as `grouper_rf.joblib` / `classifier_rf.joblib` / `artifacts.json`. The manifest carries the sklearn version, the seed, and the sha256 of every input pack — the triple that makes a flux number reproducible. `load_artifacts` warns loudly on a sklearn mismatch rather than failing silently. |
| `runinfo.py` | Reads/writes the run metadata **inside** each output GeoPackage, as a registered `attributes` table. QGIS lists it, GDAL ignores it when reading the spatial layer. |
| — | **The entry point is `deploy.py` at the repo root**, not a module here. `deploy_lake.slurm` submits it. Each stage module keeps its own `main()` so it can be run alone when needed. |

The stage boundary is `bubbles.gpkg` on purpose: stage A needs torch, stage B
needs the geo + sklearn stack, so the slow half runs on HPC and the half you
actually read runs anywhere.

**`surveyed_area_m2` travels with every total**, and it travels *inside* the
GeoPackage rather than in a sidecar, so copying `seeps.gpkg` somewhere can't
separate the count from the area it was counted over. It is measured, never
configured: valid pixels (alpha band ∩ lake polygon) accumulated per tile core,
including tiles where nothing was detected. A count-based flux figure without
that denominator is not comparable to a field campaign, to another lake, or to
itself on another date.

## Other

| Path | Does |
|---|---|
| `historical_size_priors.py` | Distills the historical field workbooks into grouping geometry + per-class size priors. |
| `viz/` | Plotting notebooks: loss curves, example frames, prediction overlays. |
| `archive/` | Superseded, rejected, or already-applied. **Untracked** (local only; recover from git history). Nothing here is imported by live code — see `archive/README.md`. |
