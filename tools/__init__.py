"""Post-inference tooling for the bubble-mapping pipeline.

Stage packages, in pipeline order:
    eval/      pixel -> bubble detection metrics
    labeling/  build and maintain labeler packs
    grouping/  bubble -> seep (the learned RF grouper)
    classify/  per-seep A/B/C class and flux
    viz/       plotting notebooks

Run any script as a module from the repo root, e.g.
    python -m tools.eval.bubble_level_eval
"""
