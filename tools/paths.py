# tools/paths.py
"""Single source of truth for the canonical prediction directory.

The canonical checkpoint is `20260428-1537_SWINxAE.weights` -- the run whose
bubble F1 = 0.644882 is the repo's canonical detection metric (CLAUDE.md
2026-09-03). Its outputs (rasters, feature CSVs, GeoPackages, and the
`labeling/` packs) all live under one directory, and before this module the
name was copy-pasted into 13 places across 12 modules.

Switching checkpoints is now a one-line edit here. Note that
`config.preprocessed_dir` and `config.saved_models_dir` must move WITH it --
the preprocessed dir supplies the train/val/test split as well as the chips,
so a mismatch silently evaluates a model on chips from its own training split.

Deliberately dependency-free (stdlib `os` only, no config import): most callers
avoid `configSwinUnet` because its REPO_PATH assumes the HPC layout
(`~/bubble-mapping`) and `validate()` raises on machines that don't match.
Callers that DO load config should join against `config.results_dir`:

    os.path.join(config.results_dir, CANONICAL_PRED_SUBDIR)

Callers that don't can use the repo-relative or absolute forms below.
"""
import os

# The checkpoint directory name, relative to config.results_dir.
CANONICAL_PRED_SUBDIR = "20260428-1537_SWINxAE.weights"

# Repo-relative path, for tools run from the repo root.
CANONICAL_PRED_RELDIR = os.path.join(
    "data", "results", "SWIN", "AE", CANONICAL_PRED_SUBDIR)

# Absolute path, for tools that resolve against the user's checkout.
REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
CANONICAL_PRED_DIR = os.path.join(REPO_PATH, CANONICAL_PRED_RELDIR)
