# tools/deploy/
"""Whole-lake deployment: the composed end-to-end runner.

    ortho + lake polygon
      -> tiles      (tools.deploy.tiles)
      -> detector   (tools.deploy.detect)     GPU / HPC
      -> CC + per-bubble features             ]
      -> learned pairwise grouper -> seeps    ] tools.deploy.postproc, CPU
      -> A/B/C classifier                     ]
      -> count-based flux total               ]

The stage boundary is `bubbles.gpkg`. Everything upstream of it needs torch and
a GPU; everything downstream needs only the geo + sklearn stack, so the slow
half runs on HPC and the analysis half runs anywhere against the same file --
which also carries the run metadata, `surveyed_area_m2` included.

The entry point is `deploy.py` at the repo root.
"""