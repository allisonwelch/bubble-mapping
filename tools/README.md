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
        eval/bubble_level_eval.py ─────────────────► BUBBLE metrics (P/R/F1 = 0.645)
                              │
                              ▼
        grouping/train_grouper.py       learn pairwise "same-seep?"
        grouping/deploy_grouper.py      apply it, assign seep_group_id
                              │
                              ▼  ← a bubble becomes part of a SEEP here
        classify/fit_classifier.py      per-seep A/B/C
                              │
                              ▼
                        count-based flux
```

---

## `eval/` — detection metrics

| Script | Does |
|---|---|
| `write_bubble_rasters.py` | Morphological closing+opening (disk 1) then connected components. Writes `{stem}_smoothed.tif`, `{stem}_cc.tif`, optional `{stem}_snow.tif`. |
| `bubble_features.py` | Per-bubble area / perimeter / circularity / solidity / eccentricity / mean RGB. Also rebuilds GT bubble polygons from the **original drawn shapes** (`build_gt_bubbles_from_source`) rather than re-polygonizing a raster, which used to merge adjacent polygons. |
| `bubble_level_eval.py` | **The canonical detection metric.** Matches predicted CCs to GT CCs one-to-one. Writes the summary / per-image / pairs CSVs. |
| `gt_bubbles_export.py` | The single producer of `gt_bubbles.gpkg`, the per-bubble ground-truth layer every labeler pack is built from. |

There is deliberately **no seep-level eval**. The old `cluster_f1 = 0.672`
grouped the GT and predicted sides with the *same* hand-tuned rule, so grouping
error cancelled and it measured detection twice. See
`archive/polygon_matcher.py` for the honest RF-grouper version and what it needs.

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

## `grouping/` — bubble → seep

| Script | Does |
|---|---|
| `train_grouper.py` | Trains the pairwise "same-seep?" random forest on the combined multi-image labels. Candidate pairs ≤ 0.5 m; diameter-capped agglomeration at 1.0 m. |
| `deploy_grouper.py` | Applies it to a pack, writes `seep_group_id` in place via sqlite, freezes the proposal in `seep_group_id_pred`, injects the `group_hull_by_class` QGIS style. |
| `group_predictions.py` | Same, but on **detected** bubbles instead of GT — the serve-time view. |
| `grouper_corrections.py` | Diffs a labeler-edited pack against the frozen proposal into accept/split/merge training records. |
| `build_hull_layer.py` | Precomputes per-seep hulls into a static layer. Fixes the QGIS render stall on multi-chip packs. |

## `classify/` — seep → class → flux

| Script | Does |
|---|---|
| `fit_classifier.py` | Fits and validates the A/B/C classifier on the three labeler packs; reports inter-labeler κ. Holds `FLUX_RATE` / `FLUX_SIGMA`. |
| `model_comparison.py` | Features × model families × training objective. |
| `c_augment_eval.py` | Does adding hand-hunted C seeps help? (C is ~4% of seeps but ~52% of the flux.) |

## Other

| Path | Does |
|---|---|
| `historical_size_priors.py` | Distills the historical field workbooks into grouping geometry + per-class size priors. |
| `viz/` | Plotting notebooks: loss curves, example frames, prediction overlays. |
| `archive/` | Superseded, rejected, or already-applied. Nothing here is imported by live code — see `archive/README.md`. |

---

## Leakage trap — read before fitting anything on labeler data

The three labelers overlap on a shared calibration set, so the **same physical
seep appears up to three times** in the pooled data (1,508 seeps over 1,148
distinct physical seeps). Naive k-fold puts one labeler's copy in train and
another's in test and reports a badly inflated score.

Every seep gets a `phys_id` (union-find over "shares any member bubble") and
**all cross-validation is `GroupKFold` on `phys_id`**. Leave-one-image-out is
reported alongside as the cross-chip test — and LOIO is the regime that counts,
because deployment is always an unseen chip.

`assign_phys_id` must run **before** any trainable-row filter: group links can
run *through* a row the filter drops, so filtering first silently splits groups.
