# tools/archive/

Superseded, rejected, or already-applied code. Kept (moved with `git mv`, not
deleted) so `git log --follow` still resolves their history. **Nothing here is
imported by live code.** Several files reference functions that no longer
exist — that is expected; they are a record, not a dependency.

## Rejected grouper prototypes
Explored on chip 39, June 2026, and all beaten by the learned pairwise
random-forest grouper that is now deployed. See CLAUDE.md 2026-06-10.

| File | Why it lost |
|---|---|
| `seep_anchor_fix_prototype.py` | local-max anchor + size-scaled reach; couldn't hit both a clean A count and an accurate B count |
| `seep_regime_grouper_prototype.py` | explicit two-regime (anchor-gather + density-merge) rule; lost to the learned model |
| `seep_pairwise_grouper_prototype.py` | chip-39-only ancestor of the deployed grouper; superseded by the multi-image `seep_pairwise_grouper_quarters.py` |

## Superseded by the anchor+lonely removal (2026-09-03)
| File | Note |
|---|---|
| `seep_fit_clustering_params.py` | fitted the hand-tuned clustering thresholds. Its last importer was a baseline row in the pairwise grouper, removed with the rule. |
| `polygon_matcher.py` | the seep-level polygon IoU matcher, lifted out of `seep_level_eval.py`. **Not stale** — it is what a future RF-grouper seep-level eval needs. Header carries the recipe. |

## Applied one-shots
Schema migrations and scans that have already run against the live data;
re-running them would be a no-op at best and destructive at worst.

## Upstream Planet / Sentinel-2 tooling
Inherited from the DriftwoodMappingBenchmark codebase. This project is
aerial (AE) only.

---
Recover anything with `git checkout pre-cleanup -- <path>`.
