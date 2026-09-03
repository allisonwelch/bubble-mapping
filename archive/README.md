# archive/

Root-level code that is not on the bubble-mapping path. Moved here with
`git mv`, so `git log --follow` still resolves the full history of every file.
Recover anything with `git checkout pre-cleanup -- <path>`.

## Model entry points for other modalities

This project is **aerial (AE) only**. These drive Planet, Sentinel-2 and
TerraMind variants inherited from the upstream DriftwoodMappingBenchmark
codebase, and none of them has been run here.

`mainUnetxPS.py` · `mainUnetxPS2.py` · `mainSwinUnetxPS.py` ·
`mainTerraMind.py` · `mainTerraMindPS.py` · `terratorch_benchmark.py`

Still live at the repo root: `mainSwinUnet.py` (the canonical model),
`mainUnet.ipynb`, `tuning.py`.

## `config/` — unused configs

Same reason. The two live configs stay in `config/`: `configSwinUnet.py` (the
canonical SWIN×AE config the checkpoint was trained under) and
`configUnetxAE.py` (the UNet baseline referenced throughout CLAUDE.md).

`configUnetxPS.py` · `configUnetxPS2.py` · `configUnetxS2.py` ·
`configTerraMind.py` · `configTerraMindxPS.py` · `configTerraMindxS2.py` ·
`configSwinUnetxPS.py` · `configSwinUnetxS2.py`

## One-off scripts

| File | Why |
|---|---|
| `check.py` | Ad-hoc histogram of GT area/solidity, used once to pick sampling strata. Reads `gt_seeps.gpkg` and calls columns that the anchor+lonely removal deleted. |
| `seep_classification_allocation.py` | Sampled the original 3 × 300-seep labeler packs. That allocation is done and will not be repeated. Its one still-used function, `assign_strata`, now lives on its own at `tools/labeling/strata.py`. |
