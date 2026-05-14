# tools/gt_seeps_export.py
"""
Export per-chip ground-truth seeps to a single labelable vector layer,
PRESERVING the original drawn polygon shapes.

Walks chip_dir; for each chip reads its CRS+extent and clips the original
GT polygons (config.training_polygon_fn) to that extent. Per-polygon
features (area, perim, circularity, solidity, eccentricity, mean_R/G/B)
are computed by build_gt_seeps_from_source so adjacent polygons never get
merged into a single CC the way the rasterize -> CC -> repolygonize path
used to merge them.

QGIS workflow:
  - Load gt_seeps.gpkg in QGIS, chip imagery as basemap underneath.
  - Toggle editing on the layer -> open attribute table -> fill `class`.
  - Save edits -> ship the .gpkg back.

Output path defaults to pred_dir (so all eval artifacts live together);
override via --out_dir or out_dir=. CLI passes through to main(...).
"""
import os
import sys
import glob
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seep_feature_table import (
    build_gt_seeps_from_source,
    write_seeps_gpkg,
    _AUX_SUFFIXES,
)

try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    gpd = None
    _HAS_GPD = False


def _load_source_polygons(source_polygons_fp):
    if not _HAS_GPD:
        raise RuntimeError("geopandas is required")
    print(f"loading source polygons: {source_polygons_fp}")
    src = gpd.read_file(source_polygons_fp)
    print(f"  -> {len(src)} polygons (crs={src.crs})")
    return src


def main(chip_dir, out_dir, source_polygons_fp,
         pred_dir=None, glob_pattern="*.tif"):
    """Build gt_seeps.gpkg from the ORIGINAL drawn polygons.

    Scope of which chips to include is controlled by `pred_dir`:
      - If pred_dir is given, only chips whose basename matches a prediction
        .tif in pred_dir are processed. This is the canonical convention
        (matches what seep_level_eval.py writes) -- gt_seeps reflects the
        TEST SET that was evaluated.
      - If pred_dir is None, walks every chip in chip_dir. Use only when you
        intentionally want GT polygons for all chips (train+val+test).
    """
    if not _HAS_GPD:
        raise RuntimeError("geopandas is required")
    src_gdf = _load_source_polygons(source_polygons_fp)

    if pred_dir is not None:
        pred_fps = [
            fp for fp in sorted(glob.glob(os.path.join(pred_dir, glob_pattern)))
            if not fp.endswith(_AUX_SUFFIXES)
            and not fp.endswith(("_seep_cluster.tif",))
            and "_r125_r45_r10_lonely_seep_cluster" not in fp
        ]
        chip_fps = []
        for pfp in pred_fps:
            cfp = os.path.join(chip_dir, os.path.basename(pfp))
            if os.path.exists(cfp):
                chip_fps.append(cfp)
        scope_label = f"{len(chip_fps)} chips matching predictions in {pred_dir}"
    else:
        chip_fps = [
            fp for fp in sorted(glob.glob(os.path.join(chip_dir, glob_pattern)))
            if not fp.endswith(_AUX_SUFFIXES)
            and not fp.endswith(("_seep_cluster.tif",))
            and "_r125_r45_r10_lonely_seep_cluster" not in fp
        ]
        scope_label = f"all {len(chip_fps)} chips in {chip_dir}"

    print(f"processing {scope_label}")
    gdfs = []
    for fp in tqdm(chip_fps, desc="GT seep polygons"):
        g = build_gt_seeps_from_source(fp, src_gdf, id_name="seep_id")
        if g is None or g.empty:
            continue
        g.insert(0, "image", os.path.basename(fp))
        gdfs.append(g)
    if not gdfs:
        raise RuntimeError(
            f"No GT polygons extracted.\n"
            f"  scope: {scope_label}\n"
            f"  source polygons: {source_polygons_fp}\n"
            f"Check that source_polygons_fp covers the same area as the chips."
        )
    os.makedirs(out_dir, exist_ok=True)
    out_fp = os.path.join(out_dir, "gt_seeps.gpkg")
    write_seeps_gpkg(out_fp, gdfs, class_column=True)
    total = sum(len(g) for g in gdfs)
    print(f"\nWrote {total} GT seeps from {len(gdfs)} chips to {out_fp}")
    print("Open in QGIS, toggle editing, fill the `class` column per row, save.")
    return out_fp


def _parse_args():
    p = argparse.ArgumentParser(description="Export GT seep polygons to GPKG")
    p.add_argument("--chip_dir", default=None,
                   help="Directory of chip .tif files (default: config.preprocessed_dir)")
    p.add_argument("--out_dir", default=None,
                   help="Where to write gt_seeps.gpkg (default: pred_dir)")
    p.add_argument("--source_polygons", default=None,
                   help="Path to source polygon GPKG "
                        "(default: config.training_data_dir/config.training_polygon_fn)")
    p.add_argument("--pred_dir", default=None,
                   help="Restrict scope to chips matching predictions in this dir "
                        "(default: same as out_dir, which is the canonical pred_dir). "
                        "Pass --pred_dir '' to walk every chip in chip_dir instead.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # Only load config when at least one CLI arg is missing -- skipping config
    # avoids validate() failures on machines whose REPO_PATH doesn't match.
    if not (args.chip_dir and args.out_dir and args.source_polygons):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from config import configSwinUnet
        config = configSwinUnet.Configuration().validate()
        chip_dir = args.chip_dir or config.preprocessed_dir
        out_dir = args.out_dir or os.path.join(
            config.results_dir, "20260428-1537_SWINxAE.weights"
        )
        source_polygons_fp = args.source_polygons or os.path.join(
            config.training_data_dir, config.training_polygon_fn
        )
    else:
        chip_dir = args.chip_dir
        out_dir = args.out_dir
        source_polygons_fp = args.source_polygons

    # pred_dir scope:
    #   None (default)  -> use out_dir (canonical: out_dir IS pred_dir)
    #   ""              -> disable; walk every chip in chip_dir
    #   <path>          -> use the given path
    if args.pred_dir is None:
        pred_dir = out_dir
    elif args.pred_dir == "":
        pred_dir = None
    else:
        pred_dir = args.pred_dir

    main(chip_dir=chip_dir, out_dir=out_dir,
         source_polygons_fp=source_polygons_fp,
         pred_dir=pred_dir)