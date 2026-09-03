# Pre-cleanup manifest (2026-09-03)

Snapshot taken at tag `pre-cleanup` (= commit `e114a27`), immediately before the
tools/ restructure + pixel/bubble/seep naming migration.

| File | What it pins |
|---|---|
| `gpkg_schemas.json` | Every GeoPackage under `data/` — per layer: row count, column list, and a SHA-256 of the concatenated geometry blobs in rowid order. The geometry hash is the check that the Phase-4 column rename touched *only* column names. |
| `seep_level_summary.csv` | The canonical bubble-level metrics row this cleanup must not move. |
| `seep_level_per_image.csv` | Per-chip `n_gt` — used to identify which preprocessed chip dir reproduces the canonical run. |

## The acceptance test

Bubble-level metrics from the canonical run `20260514-2`:

    precision = 0.570838   recall = 0.740996   f1 = 0.644882

Reproduced locally at `pre-cleanup` with:

    pred_dir = data/results/SWIN/AE/20260428-1537_SWINxAE.weights
    chip_dir = data/preprocessed/2026-04-22_SWINxAE

**`2026-04-22_SWINxAE` is the correct chip dir** — it is the only SWIN
preprocessed dir whose GT label band reproduces the canonical per-image `n_gt`
on all 9 labeled chips. `2026-04-27_SWINxAE` (the value in `configSwinUnet.py`)
does NOT: it gives 21.tif n_gt=40 vs the canonical 54 (2/9 chips match), which
works out to bubble F1 = 0.6135 instead of 0.6449.
Anyone re-running the bubble eval must use the 04-22 dir to compare against the
recorded metric history.
