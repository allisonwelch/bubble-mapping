import json
import math
import os
import time
from datetime import timedelta

# Core data science libraries
import geopandas as gpd  # GIS library: reads/manipulates vector data (shapefiles, geopackages) with geometry awareness
import numpy as np  # Numerical computing
import pandas as pd  # Data frames and tabular data
import rasterio  # Geospatial library: reads/writes raster images (GeoTIFFs) with spatial metadata
import scipy  # Scientific computing (here used for image processing, connected components labeling)
import skimage.transform  # Image resizing/rescaling operations
from osgeo import gdal  # GDAL: geospatial I/O library for rasterization and format conversion
from shapely.geometry import box  # Creates rectangular bounding boxes from coordinates
from shapely.ops import unary_union  # Merges overlapping geometries
from tqdm import tqdm  # Progress bars for loops

from core.frame_info import image_normalize
from core.util import raster_copy

# Reuse the same split logic as training (these functions handle train/val/test splitting)
from core.split_frames import split_dataset, summarize_positive_rates

from config.configUnetxAE import *


def get_areas_and_polygons():
    """Read training rectangles and polygon shapefiles and pre-index polygons.

    The spatial join assigns rectangle ids to polygons in a column "index_right".

    CONTEXT:
    - 'Training areas' are hand-drawn rectangular regions on satellite imagery
      that you want to train a model on. Think of them as "focus windows."
    - 'Polygons' are hand-drawn outlines of bubbles (the objects you're detecting).
      Each polygon marks one bubble's boundary.
    - This function links each polygon to its parent training area, so we know
      which area(s) to process and where the training targets are.
    """
    print("Reading training data shapefiles.. ", end="")
    start = time.time()

    # Load the training area rectangles from a shapefile or geopackage.
    # Each row is one rectangular region; the 'geometry' column holds the
    # Shapely polygon objects that define each rectangle's spatial bounds.
    areas = gpd.read_file(os.path.join(config.training_data_dir, config.training_area_fn))
    # Keep geometry plus image_link_field (if configured) for image matching.
    # Drop all other attribute columns — we don't need them downstream.
    keep_cols = {"geometry"}
    if getattr(config, "image_link_field", None) and config.image_link_field in areas.columns:
        keep_cols.add(config.image_link_field)
    areas = areas.drop(columns=[c for c in areas.columns if c not in keep_cols])

    # Load the hand-annotated bubble polygons from a shapefile/geopackage.
    # Each row is one bubble's boundary.
    polygons = gpd.read_file(
        os.path.join(config.training_data_dir, config.training_polygon_fn)
    )
    # Drop attribute columns; we only need the geometry
    polygons = polygons.drop(columns=[c for c in polygons.columns if c != "geometry"])

    print(
        f"Done in {time.time()-start:.2f} seconds. Found {len(polygons)} polygons in "
        f"{len(areas)} areas.\nAssigning polygons to areas..      ",
        end="",
    )
    start = time.time()

    # Spatial join: For each polygon, find which training area(s) it overlaps.
    # sjoin = spatial join (a GIS operation that matches geometries by location).
    # predicate="intersects" means: keep polygon-area pairs that touch or overlap.
    # how="inner" means: only keep polygons that intersect an area (drop orphaned ones).
    # Result: a new column "index_right" added to polygons, containing the area's ID.
    polygons = gpd.sjoin(polygons, areas, predicate="intersects", how="inner")

    print(f"Done in {time.time()-start:.2f} seconds.")
    return areas, polygons


def get_images_with_training_areas(areas):
    """Return list of (image_path, [area_ids]) for images overlapping training areas.

    PURPOSE:
    Satellite imagery files are often large regional scenes. We don't process the
    entire image—only the rectangular training areas we defined. This function
    figures out which satellite images contain which training areas, so we can
    later extract just the relevant portions.
    """
    print("Assigning areas to input images..  ", end="")
    start = time.time()

    # Collect all candidate image file paths by walking the directory tree
    image_paths = []
    for root, dirs, files in os.walk(config.training_image_dir):
        for fname in files:
            # Filter by naming convention (e.g., "S2_*.tif") and file type
            if (
                fname.startswith(config.train_image_prefix)
                and fname.lower().endswith(config.train_image_type.lower())
            ):
                image_paths.append(os.path.join(root, fname))

    # --- image_link matching (preferred when .tif files overlap spatially) ---
    # When config.image_link_field is set, each training area carries a field whose
    # value is the .tif basename (no extension) it belongs to. We build a dict from
    # basename -> full path, then group area indices by their linked image.
    image_link_field = getattr(config, "image_link_field", None)

    if image_link_field and image_link_field in areas.columns:
        from pathlib import Path

        # Map .tif basename (no extension) -> full path for every discovered image
        stem_to_path = {Path(p).stem: p for p in image_paths}

        # Group area row-indices by the image they belong to
        link_to_area_ids = {}
        for idx, row in areas.iterrows():
            stem = row[image_link_field]
            if stem not in stem_to_path:
                print(f"\n  WARNING: image_link '{stem}' (area {idx}) has no matching .tif — skipped.")
                continue
            link_to_area_ids.setdefault(stem, []).append(int(idx))

        images_with_areas = [
            (stem_to_path[stem], ids) for stem, ids in link_to_area_ids.items()
        ]

    else:
        # --- Legacy spatial-overlap matching ---
        # For each image, find which training areas fall inside its geographic bounds.
        images_with_areas = []
        for im in image_paths:
            # Open the raster (satellite image) with rasterio to read metadata
            with rasterio.open(im) as raster:
                # raster.bounds = (left, bottom, right, top) in native coordinates.
                # Convert to a rectangular Shapely geometry for spatial comparison.
                im_bounds = box(*raster.bounds)
                # CRS = Coordinate Reference System. A CRS defines how lat/lon maps to
                # a 2D coordinate system. Examples: EPSG:4326 (WGS84, lat/lon degrees),
                # EPSG:6933 (equal-area meters). All geometries must be in the same CRS
                # to meaningfully compare their spatial overlap.
                image_crs = raster.crs

            # Sanity check: skip images in a different CRS than our training areas
            # (they can't be compared spatially without reprojection)
            if image_crs != areas.crs:
                continue

            # Find which training area rectangles overlap this satellite image.
            # areas.envelope returns the bounding box of each area (as a rectangle).
            # .intersects(im_bounds) returns a boolean array: True if area overlaps image.
            # np.where(...)[0] converts booleans to indices of matching areas.
            areas_in_image = np.where(areas.envelope.intersects(im_bounds))[0]
            if len(areas_in_image) > 0:
                # Store (image_path, list_of_area_ids) for later processing
                images_with_areas.append((im, [int(x) for x in list(areas_in_image)]))

    print(
        f"Done in {time.time()-start:.2f} seconds. Found {len(image_paths)} training "
        f"images of which {len(images_with_areas)} contain training areas."
    )
    return images_with_areas


def calculate_boundary_weights(polygons, scale):
    """Return boundary polygons between close polygons using scaled intersections.

    PURPOSE:
    In some deep learning models, it's helpful to give higher weight to pixels
    at the *boundaries* between bubbles (where the model is most uncertain).
    This function creates boundary masks by:
    1. Expanding each polygon outward by a factor (scale)
    2. Finding where the expanded versions overlap
    3. These overlaps represent the boundary regions

    This is used to create boundary weights for loss calculations, which helps
    the model learn to separate adjacent bubbles more precisely.
    """
    # Scale (expand/shrink) each polygon by a factor about its center.
    # If scale > 1.0, polygons grow larger (expansion).
    # If scale < 1.0, they shrink.
    # Think of this like drawing a buffer zone around each bubble.
    scaled_polys = gpd.GeoDataFrame(
        {"geometry": polygons.geometry.scale(xfact=scale, yfact=scale, origin="center")}
    )

    # Find all pairwise intersections of the scaled polygons.
    # Where two scaled bubbles overlap = the boundary region between originals.
    boundaries = []
    for i in range(len(scaled_polys)):
        # Find all scaled polygons that touch or overlap this one
        nearby_polys = scaled_polys[
            scaled_polys.geometry.intersects(scaled_polys.iloc[i].geometry)
        ]
        for j in range(len(nearby_polys)):
            # Skip self-intersections (a polygon always intersects itself)
            if nearby_polys.iloc[j].name != scaled_polys.iloc[i].name:
                # Compute the intersection geometry (the region where both overlap)
                boundaries.append(
                    scaled_polys.iloc[i].geometry.intersection(
                        nearby_polys.iloc[j].geometry
                    )
                )

    # After intersection, results may include LineString or Point geoms; keep only Polygons
    boundaries = gpd.GeoDataFrame({"geometry": gpd.GeoSeries(boundaries)}).explode()
    boundaries = boundaries[boundaries.type == "Polygon"]

    # Remove any boundary regions that overlap the original bubble polygons.
    # This leaves only the true boundary regions (buffer zones, not bubble interiors).
    if len(boundaries) > 0:
        boundaries = gpd.overlay(boundaries, polygons, how="difference")
    # Ensure we have at least one row (even if empty) to avoid downstream errors
    if len(boundaries) == 0:
        boundaries = boundaries.append({"geometry": box(0, 0, 0, 0)}, ignore_index=True)

    return boundaries


def get_vectorized_annotation(polygons, areas, area_id, xsize, ysize):
    """Return list of per-polygon dicts (center, areas, pseudo-radii, geometry).

    PURPOSE:
    Convert hand-drawn bubble polygons to a structured format suitable for training.
    Each polygon is converted to:
    - Its physical size (in meters and pixels)
    - Its center point (in pixel coordinates)
    - Its boundary points (for later use in loss calculations, e.g., boundary weighting)

    This creates a JSON-serializable annotation file for each training area.
    """
    # Filter to polygons that fall within this specific training area rectangle
    isinarea = polygons[polygons.within(box(*areas.bounds.iloc[area_id]))]
    # If a polygon geometry is a multi-part geometry, split it into single parts.
    # (Simplifies downstream processing.)
    isinarea = isinarea.explode(column="geometry", index_parts=False)

    # CRITICAL: Convert to equal-area projection (EPSG:6933 = Sphere Equal-Area).
    # Why? Because we need accurate area measurements in square meters.
    # The original CRS (often UTM or lat/lon) distorts area, especially away from
    # the projection's central meridian. Equal-area projections preserve area everywhere.
    isinarea_ea = isinarea.to_crs(epsg=6933)
    # Compute each polygon's area in square meters (from equal-area projection)
    isinarea.loc[:, "area(m)"] = isinarea_ea.area
    # Pseudo-radius: assume each bubble is roughly circular; compute equivalent radius.
    # Formula: area = pi*r^2, so r = sqrt(area/pi)
    # This is useful for training—gives model a sense of bubble size.
    isinarea.loc[:, "pseudo_radius(m)"] = isinarea_ea.area.apply(
        lambda x: np.sqrt(x / np.pi)
    )

    # Create a geospatial transform that maps between:
    #   - Ground coordinates (meters in EPSG:6933)
    #   - Pixel coordinates (0..xsize-1, 0..ysize-1 in the raster image)
    # The bounds define the real-world extent; xsize/ysize is the pixel grid size.
    bounds = areas.iloc[[area_id]].to_crs(epsg=6933).bounds.iloc[0]
    trsfrm = rasterio.transform.from_bounds(*bounds, xsize, ysize)

    # Ground resolution: how many meters per pixel.
    # trsfrm.a = pixel width in meters (X direction)
    # trsfrm.e = pixel height in meters (Y direction, typically negative)
    gr = np.mean([np.abs(trsfrm.a), np.abs(trsfrm.e)])
    # Convert polygon area from square meters to square pixels
    isinarea.loc[:, "area(px)"] = isinarea["area(m)"] / (gr**2)
    # Convert pseudo-radius from meters to pixels
    isinarea.loc[:, "pseudo_radius(px)"] = isinarea["pseudo_radius(m)"] / gr

    # Transform polygon coordinates from world (meters) to pixel space.
    # ~trsfrm = inverse transform (world coords -> pixel coords).
    # trsfrm.column_vectors flattens the transform matrix for affine_transform().
    trsfrm = ~trsfrm
    trsfrm = [element for tupl in trsfrm.column_vectors for element in tupl]
    # Apply the inverse transform to convert polygon geometries to pixel coordinates
    isinarea.loc[:, "geometry"] = isinarea_ea["geometry"].affine_transform(trsfrm[:6])
    # Compute the center (centroid) of each polygon in pixel space
    isinarea.loc[:, "center"] = isinarea.centroid
    # Convert centroid Point objects to (x, y) tuples for JSON serialization
    isinarea.loc[:, "center"] = isinarea["center"].apply(lambda p: (p.x, p.y))
    # Extract the boundary vertices (exterior ring coordinates) as a list of tuples
    # This defines the polygon's outline for later use (e.g., rendering, loss weighting)
    isinarea.loc[:, "geometry"] = isinarea["geometry"].apply(
        lambda x: list(x.exterior.coords)
    )

    # Clean up and convert to a list of dictionaries (one per polygon)
    # Drop the "index_right" column (it's no longer needed)
    isinarea.drop(labels=["index_right"], inplace=True, axis=1)
    isinarea = pd.DataFrame(isinarea)
    # Convert each row to a dict: {"area(m)": ..., "pseudo_radius(m)": ..., ...}
    dic = isinarea.to_dict(orient="records")
    return dic


def resolution_degrees2metres(xres_degrees, yres_degrees, latitude):
    """Convert resolution in degrees to approximate meters at given latitude.

    CONTEXT:
    Lat/lon (EPSG:4326) measures distance in degrees. But a degree of longitude
    is shorter near the poles and longer at the equator. This function converts
    a raster's resolution (in degrees) to its approximate resolution in meters,
    accounting for the location's latitude.

    Constants: 111320 m/degree (longitude, at equator), 110540 m/degree (latitude).
    """
    # Longitude (X direction) resolution depends on latitude due to Earth's curvature.
    # Multiply by cos(latitude) to adjust for the meridian's convergence toward poles.
    xres_metres = xres_degrees * (111320 * math.cos(math.radians(abs(latitude))))
    # Latitude (Y direction) is roughly constant everywhere (~110.54 km/degree).
    yres_metres = yres_degrees * 110540
    return xres_metres, yres_metres


def add_additional_band(image_fp, image_bounds, out_fp, new_band, pbar_pos=0):
    """Add an auxiliary band to a raster by sampling a source file over given bounds.

    PURPOSE:
    Enrich satellite imagery by adding extra data bands (e.g., elevation, land-use mask).
    The new band may come from a different source file, at a different resolution,
    and with different value ranges. This function resamples and aligns it to match
    the input image's spatial grid.

    Example: Add elevation (DEM) as a 4th band to RGB satellite imagery.
    """
    pbar = tqdm(
        total=5,
        desc=f"{'Adding coverband...':<25}",
        leave=False,
        position=pbar_pos,
        disable=True,
    )

    # Read the original image (e.g., RGB satellite data) and crop to the specified bounds
    with rasterio.open(image_fp) as image_ds:
        # Convert geographic bounds (minx, miny, maxx, maxy) to pixel coordinates (Window)
        # This tells rasterio which rectangular region to read from the file.
        image_window = rasterio.windows.from_bounds(*image_bounds, image_ds.transform)
        # Read all bands within the window. img.shape = (bands, height, width)
        img = image_ds.read(window=image_window)
        pbar.update()

        # Read the new band data from its source file, cropped to the same geographic bounds
        with rasterio.open(new_band["source_fp"]) as src:
            # Which band index to read? (1-indexed in rasterio; default is 1 = first band)
            band_index = new_band["source_band"] if "source_band" in new_band else 1
            new_band_img = src.read(
                band_index,
                window=rasterio.windows.from_bounds(*image_bounds, src.transform),
            )
        pbar.update()

        # Optionally mask out invalid values (e.g., water, clouds) by setting them to 0
        if "maskvals" in new_band and len(new_band["maskvals"]) > 0:
            # Create a boolean mask where True = value is in the "maskvals" list
            mask = np.isin(new_band_img, new_band["maskvals"])
            # Set masked pixels to 0 (or another sentinel value)
            new_band_img[mask] = 0

        # Optionally scale the band values (e.g., normalize to [0, 1], or invert a mask)
        if "scale_factor" in new_band and new_band["scale_factor"] is not None:
            new_band_img = new_band_img.astype(np.float32) * new_band["scale_factor"]
        pbar.update()

        # Optionally resample the new band to a coarser resolution.
        # Example: downsample a 10m DEM to 30m to match satellite imagery.
        if (
            "average_to_resolution_m" in new_band
            and new_band["average_to_resolution_m"] is not None
        ):
            # Compute the scale factor: (current resolution) / (target resolution)
            # If rescaling from 10m to 30m, scale = 10/30 ≈ 0.33 (downsampling by 3x)
            scale = resolution_degrees2metres(*image_ds.res, 0)[1] / new_band[
                "average_to_resolution_m"
            ]
            # Rescale using nearest-neighbor (order=0) to avoid creating artifact values
            new_band_img = skimage.transform.rescale(
                new_band_img, scale=scale, order=0, mode="reflect"
            )
        pbar.update()

        # Resize the new band to exactly match the original image's pixel dimensions.
        # (After rescaling, dimensions might be slightly off due to rounding.)
        # order=0 = nearest-neighbor (preserves discrete values like class IDs).
        new_band_img = skimage.transform.resize(
            new_band_img, img.shape[1:], order=0, mode="reflect"
        )

        # Stack the new band onto the original bands.
        # img.shape[0] = number of original bands (e.g., 3 for RGB)
        # new_band_img.shape = (height, width), so wrap in [..] to add a dimension
        # Result: merged_img.shape = (original_bands + 1, height, width)
        merged_img = np.concatenate([img, [new_band_img]], axis=0)

        # Write the merged image (all original bands + new band) to disk
        profile = image_ds.profile  # Copy metadata (CRS, transform, dtype, etc.)
        profile["count"] = profile["count"] + 1  # Increment band count
        # Update the geospatial transform to match the cropped window
        profile["transform"] = image_ds.window_transform(image_window)
        profile["width"] = img.shape[2]  # Pixel width
        profile["height"] = img.shape[1]  # Pixel height
        with rasterio.open(out_fp, "w", **profile) as dst:
            # Convert to the same dtype as the original (often uint8 or uint16)
            dst.write(merged_img.astype(profile["dtype"]))

    pbar.update()
    return out_fp


# ============================
# helpers (split + chips)
# ============================
# The following functions handle:
# 1. Train/val/test splitting: dividing training data into disjoint sets for model evaluation
# 2. "Chips": small image tiles extracted from large training areas, used for neural network training

def _ensure_dir(path: str) -> None:
    """Create a directory if it doesn't exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _load_split_indices_from_aalist(
    split_list_path: str,
    stems_sorted: list,
) -> tuple[list, list, list]:
    """
    Load a predefined train/val/test split from a JSON file and map it
    onto the *current* stems_sorted ordering.

    PURPOSE:
    Sometimes you want to reuse a specific split from a previous run (so results
    are reproducible and comparable). This function reads a saved split file and
    translates its indices to match the current set of preprocessed frames.

    The JSON file has keys like "training_frames", "validation_frames",
    "testing_frames" — each is a list of integer indices.

    Supports optional explicit id list in the JSON (area_ids/frame_ids/stems/ids).
    If absent, assumes indices already refer to the current ordering.
    """
    print("Loading split...")
    # Read the JSON file containing the previously saved train/val/test split
    with open(split_list_path, "r") as f:
        obj = json.load(f)

    # Extract the lists of frame indices for each split (try multiple key names for flexibility)
    train_idx = obj.get("training_frames", obj.get("train", []))
    val_idx = obj.get("validation_frames", obj.get("val", []))
    test_idx = obj.get("testing_frames", obj.get("test", []))

    # Optional explicit mapping list: if the JSON contains a master list of area IDs,
    # then the indices in train/val/test refer to *positions within that list* rather
    # than direct frame indices. This handles cases where the set of frames has changed
    # between runs (some areas added/removed), so the indices need re-mapping.
    id_list = None
    for k in ("area_ids", "frame_ids", "stems", "ids"):
        if k in obj and isinstance(obj[k], list) and len(obj[k]) > 0:
            id_list = [int(x) for x in obj[k]]
            break

    # If there's an id_list, translate: saved_index -> area_id -> current_position
    if id_list is not None:
        # Build a lookup: area_id -> position in the current sorted list of stems
        stem_to_pos = {int(s): int(i) for i, s in enumerate(stems_sorted)}

        def _map(idxs):
            out = []
            for i in idxs:
                i = int(i)
                if i < 0 or i >= len(id_list):
                    continue
                area_id = int(id_list[i])
                if area_id in stem_to_pos:
                    out.append(stem_to_pos[area_id])
            return out

        tr = _map(train_idx)
        va = _map(val_idx)
        te = _map(test_idx)
    else:
        # Otherwise assume indices already match current ordering; just clip to range
        n = len(stems_sorted)

        def _clip(idxs):
            out = []
            for i in idxs:
                i = int(i)
                if 0 <= i < n:
                    out.append(i)
            return out

        tr = _clip(train_idx)
        va = _clip(val_idx)
        te = _clip(test_idx)

    print(
        f"[SPLIT][LOAD] Using predefined split from {split_list_path} | "
        f"train={len(tr)} val={len(va)} test={len(te)}"
    )
    return tr, va, te


def _write_split_json(frames_json: str, train_idx: list, val_idx: list, test_idx: list) -> None:
    """Save the train/val/test split to a JSON file for reproducibility.

    This file records exactly which frames went into which split, so you can:
    1. Reuse the same split in future runs (for fair comparison)
    2. Verify which frames the model trained on vs. was tested on
    """
    frame_split = {
        "training_frames": list(map(int, train_idx)),
        "validation_frames": list(map(int, val_idx)),
        "testing_frames": list(map(int, test_idx)),
    }
    _ensure_dir(os.path.dirname(frames_json))
    with open(frames_json, "w") as f:
        json.dump(frame_split, f, indent=2)


def _chip_params_from_config():
    """Read chip-creation parameters from the config object.

    WHAT ARE CHIPS?
    "Chips" are small image tiles cropped from the larger training-area rasters.
    Neural networks train on fixed-size patches (e.g., 256x256 pixels), so we
    pre-cut the large rasters into smaller pieces that fit the model's input size.

    This function calculates physical chip dimensions in meters:
    - chip_size_m: the default chip side length in meters
    - chip_pad_m: padding added around each chip (so bubbles at edges aren't cut off)
    - cluster_buffer_m: buffer for grouping nearby bubbles into one chip
    - chip_max_size_m: maximum allowed chip size (prevents giant chips)
    - neg_chips_per_area: how many "negative" (no-bubble) chips to create per training area

    WHY NEGATIVE CHIPS?
    The model needs to learn what "not a bubble" looks like. If you only show it
    bubble-containing patches, it might learn to predict "bubble" everywhere.
    Negative chips are patches with zero bubbles, teaching the model to output
    low probabilities for empty background.
    """
    # Pixel size in meters (e.g., 0.15 m/pixel for high-res aerial imagery)
    px = float(getattr(config, "pixel_size_m", 0.15))
    # Patch size in pixels (the model's input dimensions)
    ps = (16,16) #getattr(config, "patch_size", (256, 256))
    p = int(ps[0]) if isinstance(ps, (tuple, list)) else int(ps)

    # Default chip size: patch_size * pixel_size = physical extent in meters
    chip_size_m = float(getattr(config, "chip_size_m", p * px))
    # Padding: 25% of chip size around each chip (avoids cutting bubbles at edges)
    chip_pad_m = float(getattr(config, "chip_pad_m", 0.25 * chip_size_m))
    # Buffer for clustering nearby bubble groups into single chips
    cluster_buffer_m = float(getattr(config, "chip_cluster_buffer_m", 0.50 * chip_size_m))
    # Maximum chip size (caps very large bubble clusters)
    chip_max_size_m = float(getattr(config, "chip_max_size_m", 2.00 * chip_size_m))
    # How many empty (no-bubble) chips to generate per training area
    neg_chips_per_area = int(getattr(config, "neg_chips_per_area", 1))

    return chip_size_m, chip_pad_m, cluster_buffer_m, chip_max_size_m, neg_chips_per_area


def _random_negative_boxes_in_area_ea(area_ea_geom, polys_ea: gpd.GeoDataFrame, chip_size_m: float, n: int, rng):
    """
    Sample up to n random NEGATIVE chip boxes inside a training area (in equal-area CRS).

    NEGATIVE CHIPS: Image tiles that contain NO bubbles at all. These teach the model
    what "empty background" looks like. Without negative examples, the model might learn
    to predict "bubble" everywhere (since all its training examples contain bubbles).

    CONSTRAINTS:
    - Each box must be fully within the training area (no "leakage" outside the labeled region)
    - Each box must NOT overlap any bubble polygon (that's what makes it "negative")
    - Uses rejection sampling: randomly propose boxes, keep only valid ones, retry if invalid

    WHY EQUAL-AREA CRS (EPSG:6933)?
    All distance and area calculations are done in an equal-area projection where 1 unit = 1 meter.
    This ensures chip_size_m actually means "chip_size_m meters on the ground" regardless of
    where on Earth your study site is. In lat/lon (EPSG:4326), 1 degree of longitude varies
    from ~111 km at the equator to 0 km at the poles, so distances would be meaningless.
    """
    if n <= 0:
        return []

    # Get the bounding box of the training area in meters
    minx, miny, maxx, maxy = area_ea_geom.bounds
    half = 0.5 * chip_size_m  # Half the chip size (for centering)

    # Merge all bubble polygons into one geometry for fast intersection checks
    # unary_union combines many polygons into one multi-polygon (much faster to check against)
    polys_union = unary_union(list(polys_ea.geometry)) if len(polys_ea) > 0 else None

    out = []
    tries = 0
    max_tries = 200 * max(1, n)  # Give up after many failed attempts (area may be too crowded)

    # REJECTION SAMPLING: Randomly propose chip locations, keep valid ones
    # This is simpler than computing exact valid regions, and works well when
    # bubbles don't cover most of the training area
    while len(out) < n and tries < max_tries:
        tries += 1

        # Random center point within the training area bounds (inset by half-chip to avoid edges)
        if (maxx - minx) > chip_size_m:
            cx = float(rng.uniform(minx + half, maxx - half))
        else:
            cx = float((minx + maxx) / 2)  # Area too small; center it

        if (maxy - miny) > chip_size_m:
            cy = float(rng.uniform(miny + half, maxy - half))
        else:
            cy = float((miny + maxy) / 2)

        # Create a square box centered at (cx, cy) with side = chip_size_m
        b = box(cx - half, cy - half, cx + half, cy + half)

        # CONSTRAINT 1: Box must be fully inside the training area (no leakage)
        if not b.within(area_ea_geom):
            continue

        # CONSTRAINT 2: Box must not touch any bubble polygon (must be truly "negative")
        if polys_union is not None and b.intersects(polys_union):
            continue

        out.append(b)

    return out


def _merge_overlapping_boxes_ea(boxes_ea):
    """Iteratively merge any overlapping boxes into union bounding boxes.

    WHY MERGE? After padding chips to include context around bubbles, some chips
    may now overlap. Overlapping chips waste computation and could cause data leakage
    (the same pixel appearing in multiple training chips). This function merges any
    overlapping boxes into a single larger box that covers both, repeating until no
    overlaps remain. Think of it like merging overlapping sticky notes into one big note.
    """
    boxes = list(boxes_ea)
    changed = True
    while changed and len(boxes) > 1:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            merged = False
            for j in range(len(boxes)):
                b = boxes[j]
                inter = a.intersection(b)
                if not inter.is_empty and float(getattr(inter, "area", 0.0)) > 0.0:
                    u = a.union(b)
                    a = box(*u.bounds)
                    boxes.pop(j)
                    changed = True
                    merged = True
                    break
            if merged:
                boxes.append(a)
            else:
                out.append(a)
        boxes = out
    return boxes


def _safe_box_within_area_ea(b_ea, area_ea):
    """Ensure a chip box is fully within the training area; shrink it if needed.

    DATA LEAKAGE PREVENTION: If a chip extends beyond the training area boundary,
    it would include unlabeled pixels (we don't know if those pixels contain bubbles
    or not). This function clips the box to the training area's boundary, preventing
    the model from learning on ambiguous/unlabeled data.
    """
    if b_ea.within(area_ea):
        return b_ea
    clipped = b_ea.intersection(area_ea)
    if clipped.is_empty:
        return None
    cand = box(*clipped.bounds)
    if cand.within(area_ea):
        return cand
    return None


def _pos_boxes_from_label_components(
    src: rasterio.io.DatasetReader,
    area_geom_crs,
    *,
    chip_pad_m: float,
    chip_max_size_m: float,
    chip_cc_dilate_m: float,
):
    """
    Build non-overlapping POSITIVE chip boxes from connected components of the label band.

    WHAT ARE CONNECTED COMPONENTS?
    Imagine the label band as a black-and-white image where white pixels = bubbles.
    "Connected components" groups adjacent white pixels into blobs. Each blob is one
    connected region of bubbles. If two bubbles are touching (or nearby after dilation),
    they become one connected component.

    ALGORITHM:
    1. Read the label band (last band of the GeoTIFF) and find all bubble pixels
    2. Optionally DILATE (expand) bubble pixels so nearby clusters merge into one component
       (this prevents cutting a bubble cluster across two chips)
    3. Label connected components: each isolated blob gets a unique integer ID
    4. Find the bounding box of each component (smallest rectangle containing that blob)
    5. Convert bounding boxes to equal-area CRS (meters), add padding, merge overlaps
    6. Return the final non-overlapping chip boxes

    WHY DILATION?
    If two bubbles are 5 pixels apart, they're separate components. But cutting a chip
    between them would split context. Dilation expands each bubble by a few pixels,
    causing nearby bubbles to merge into one component (and thus one chip).

    Returns a list of boxes in EPSG:6933 (meters), guaranteed non-overlapping and within area geometry.
    """
    # Read the label band (last band in the multi-band GeoTIFF)
    # This is the binary mask: 1 = bubble pixel, 0 = background
    lab = src.read(src.count)
    pos = (np.asarray(lab) > 0)  # Convert to boolean: True where there are bubbles

    # No bubbles at all in this area? Return empty list
    if not np.any(pos):
        return []

    # Convert dilation distance from meters to pixels
    px_m = float(getattr(config, "pixel_size_m", 0.15)) * float(getattr(config, "resample_factor", 1))
    dilate_px = int(max(0, round(float(chip_cc_dilate_m) / max(px_m, 1e-12))))

    # BINARY DILATION: Expand each bubble pixel outward by dilate_px pixels.
    # This merges nearby bubble clusters into single connected components.
    # Think of it like inflating each bubble slightly so nearby ones touch.
    if dilate_px > 0:
        pos = scipy.ndimage.binary_dilation(pos, iterations=dilate_px)

    # CONNECTED COMPONENT LABELING: Assign a unique integer ID to each isolated blob.
    # lbl = array where each pixel's value is its component ID (0 = background, 1..n = components)
    # n = total number of connected components found
    lbl, n = scipy.ndimage.label(pos)
    if n <= 0:
        return []

    # Find the bounding box (slice) for each connected component
    # objs = list of (row_slice, col_slice) tuples, one per component
    objs = scipy.ndimage.find_objects(lbl)
    bboxes_crs = []
    for sl in objs:
        if sl is None:
            continue
        # Extract pixel row/col bounds of this component's bounding box
        r0, r1 = int(sl[0].start), int(sl[0].stop)
        c0, c1 = int(sl[1].start), int(sl[1].stop)
        if r1 <= r0 or c1 <= c0:
            continue  # Skip degenerate (zero-area) bounding boxes

        # Convert pixel coordinates to a rasterio Window object
        win = rasterio.windows.Window(col_off=c0, row_off=r0, width=c1 - c0, height=r1 - r0)
        # Convert pixel window to geographic coordinates using the raster's transform
        # (transform maps between pixel space and CRS coordinates)
        b = rasterio.windows.bounds(win, transform=src.transform)
        bboxes_crs.append(box(*b))  # Create a Shapely rectangle from (minx, miny, maxx, maxy)

    if len(bboxes_crs) == 0:
        return []

    # Convert all bounding boxes to EPSG:6933 (equal-area projection, units=meters)
    # This ensures padding and size calculations are in real-world meters, not degrees
    bboxes_ea = gpd.GeoSeries(bboxes_crs, crs=src.crs).to_crs(epsg=6933).tolist()
    area_ea = gpd.GeoSeries([area_geom_crs], crs=src.crs).to_crs(epsg=6933).iloc[0]

    # Add padding around each bounding box (so the chip includes context around bubbles)
    # .buffer(chip_pad_m) expands the box outward by chip_pad_m meters on all sides
    padded = []
    for b in bboxes_ea:
        bb = box(*b.bounds).buffer(float(chip_pad_m))
        # Clip the padded box to the training area (prevent leakage outside labeled region)
        inter = bb.intersection(area_ea)
        if inter.is_empty:
            continue
        cand = box(*inter.bounds)
        cand = _safe_box_within_area_ea(cand, area_ea)
        if cand is None:
            continue
        padded.append(cand)

    if len(padded) == 0:
        return []

    # Merge any boxes that now overlap after padding (avoids duplicate pixels in training)
    merged = _merge_overlapping_boxes_ea(padded)

    # Return the final non-overlapping chip boxes
    out = []
    for m in merged:
        out.append(m)

    return out


def _write_chip_from_area_raster(src: rasterio.io.DatasetReader, bounds_crs, out_fp: str):
    """Crop a chip from the full-area raster and write it to a new GeoTIFF file.

    The chip includes ALL bands (image bands + label band), so each chip is a
    self-contained training example. Returns the label band array so the caller
    can check whether this chip actually contains any bubbles (positive vs negative).

    WHAT IS A GeoTIFF?
    A GeoTIFF is a standard TIFF image file augmented with geographic metadata:
    - CRS (Coordinate Reference System): what projection the coordinates are in
    - Transform: a 6-number affine matrix mapping pixel (row, col) to geographic (x, y)
    - Band descriptions: what each band represents (e.g., Red, Green, Blue, Label)
    This metadata means you can overlay the chip on a map and it aligns perfectly.
    """
    # Clamp the requested bounds to the raster's actual extent (no reading outside the image)
    rb = src.bounds
    minx = max(float(bounds_crs[0]), float(rb.left))
    miny = max(float(bounds_crs[1]), float(rb.bottom))
    maxx = min(float(bounds_crs[2]), float(rb.right))
    maxy = min(float(bounds_crs[3]), float(rb.top))
    if minx >= maxx or miny >= maxy:
        return None

    win = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=src.transform)
    win = win.round_offsets().round_lengths()
    if win.width <= 0 or win.height <= 0:
        return None

    data = src.read(window=win)
    if data.size == 0 or data.shape[1] <= 0 or data.shape[2] <= 0:
        return None

    prof = src.profile.copy()
    prof.update(
        {
            "height": int(data.shape[1]),
            "width": int(data.shape[2]),
            "transform": src.window_transform(win),
        }
    )

    with rasterio.open(out_fp, "w", **prof) as dst:
        dst.write(data)

    return data[-1]


def preprocess_all(conf):
    """
    MAIN PREPROCESSING FUNCTION: Orchestrates the entire data preparation pipeline.

    This is the entry point called from mainUnet.py. It performs these major steps:
    1. Read training area rectangles and bubble polygon labels from shapefiles
    2. Find which satellite images contain training areas
    3. For each training area: crop the image, rasterize bubble labels onto it,
       and save as a multi-band GeoTIFF
    4. Split frames into train/val/test sets (stratified by bubble density)
    5. Create small "chips" (sub-tiles) from the large area rasters

    WHAT "RASTERIZATION" MEANS:
    Your bubble labels start as vector polygons (hand-drawn shapes with coordinates).
    The model needs pixel grids (rasters). "Rasterization" converts vector polygons
    into a pixel grid: each pixel that falls inside a bubble polygon gets value 1,
    all others get 0. This creates the binary label band that the model learns from.
    Think of it like coloring inside the lines of a coloring book, but on a pixel grid.
    """
    # Store config in module-level variable so helper functions can access it
    global config
    config = conf

    print("Starting preprocessing.")
    start = time.time()

    # Create a unique timestamp for this preprocessing run (e.g., "20260312-1430_AE_run1")
    # This prevents different runs from overwriting each other's outputs
    # Output directory for the large rasterized training-area GeoTIFFs
    # Use config.preprocessed_dir directly so training can find the output without manual path updates
    output_dir = config.preprocessed_dir
    _ensure_dir(output_dir)

    # Output directory for the smaller chips (sub-tiles of the training areas)
    chips_dir = os.path.join(config.training_data_base_dir, os.path.basename(config.preprocessed_dir))
    _ensure_dir(chips_dir)

    # STEP 1: Load the hand-drawn training areas and bubble polygon labels
    areas, polygons = get_areas_and_polygons()

    # STEP 2: Figure out which satellite images contain which training areas
    images_with_areas = get_images_with_training_areas(areas)

    # Remember which source image was used for each area (needed later for chips)
    area_to_im = {}

    # Track which area rasters actually got written successfully
    written_area_ids = set()

    # STEP 3: For each satellite image, extract and rasterize each overlapping training area.
    # This is the core loop that creates the labeled GeoTIFF files the model trains on.
    for im_path, area_ids in tqdm(
        images_with_areas, "Processing images with training areas", position=1
    ):
        for area_id in tqdm(
            area_ids, f"Extracting areas for {os.path.basename(im_path)}", position=0
        ):
            out_fp = os.path.join(output_dir, f"{area_id}.tif")

            # Skip if already processed (an area might overlap multiple images)
            if os.path.exists(out_fp):
                written_area_ids.add(int(area_id))
                if int(area_id) not in area_to_im:
                    area_to_im[int(area_id)] = im_path
                continue

            # CROP: Extract just the training area rectangle from the large satellite image.
            # raster_copy uses GDAL's virtual filesystem (/vsimem/) to do this in memory.
            # - mode="translate" = crop to specified bounds
            # - resample = downsampling factor (1 = keep original resolution)
            # - bands = which bands to extract (config.preprocessing_bands + 1 for 1-indexed GDAL)
            extract_ds = raster_copy(
                "/vsimem/extracted",
                im_path,
                mode="translate",
                bounds=areas.bounds.iloc[area_id],
                resample=config.resample_factor,
                bands=list(config.preprocessing_bands + 1),
            )

            # Create an in-memory GDAL dataset with space for image bands PLUS one label band.
            # The "+1" in (n_bands + 1) allocates an extra band for the rasterized bubble labels.
            # GDT_Float32 = 32-bit floating point pixel values (needed for normalized imagery).
            n_bands = len(config.preprocessing_bands)
            mem_ds = gdal.GetDriverByName("MEM").Create(
                "",
                xsize=extract_ds.RasterXSize,
                ysize=extract_ds.RasterYSize,
                bands=n_bands + 1,  # image bands + 1 label band
                eType=gdal.GDT_Float32,
            )
            # Copy the geographic metadata (projection + transform) from the cropped image
            mem_ds.SetProjection(extract_ds.GetProjection())
            mem_ds.SetGeoTransform(extract_ds.GetGeoTransform())

            # Copy the image bands (R, G, B, NIR, etc.) into the in-memory dataset
            for i in range(1, n_bands + 1):
                mem_ds.GetRasterBand(i).WriteArray(
                    extract_ds.GetRasterBand(i).ReadAsArray()
                )

            # RASTERIZE: "Burn" the vector bubble polygons into the last band (n_bands + 1).
            # gdal.Rasterize reads the polygon shapefile and sets pixels inside polygons to 1.
            # - burnValues=[1] means pixels inside polygons get value 1 (bubble)
            # - allTouched: if True, any pixel touched by a polygon edge gets value 1
            #   (if False, only pixels whose CENTER falls inside the polygon are labeled)
            polygons_fp = os.path.join(config.training_data_dir, config.training_polygon_fn)
            gdal.Rasterize(
                mem_ds,
                polygons_fp,
                bands=[n_bands + 1],     # Write to the last band (label band)
                burnValues=[1],           # Bubble = 1, background stays 0
                allTouched=config.rasterize_borders,
            )

            # Optionally save vectorized annotation info as JSON (polygon centers, sizes, etc.)
            if config.get_json:
                dic = get_vectorized_annotation(
                    polygons, areas, area_id, extract_ds.RasterXSize, extract_ds.RasterYSize
                )
                json_fp = os.path.join(output_dir, f"{area_id}.json")
                with open(json_fp, "w") as fp:
                    json.dump(dic, fp)

            # Write the in-memory dataset to a GeoTIFF file on disk.
            # The result is a multi-band .tif: [R, G, B, NIR, ..., Label]
            gdal.GetDriverByName("GTiff").CreateCopy(out_fp, mem_ds, 0)

            written_area_ids.add(int(area_id))
            if int(area_id) not in area_to_im:
                area_to_im[int(area_id)] = im_path

            # Release GDAL datasets to free memory
            mem_ds = None
            extract_ds = None

    if len(areas) > len([f for f in os.listdir(output_dir) if f.lower().endswith(".tif")]):
        print(
            f"WARNING: Training images not found for "
            f"{len(areas)-len([f for f in os.listdir(output_dir) if f.lower().endswith('.tif')])} areas!"
        )

    # ---- STEP 4: TRAIN/VAL/TEST SPLIT ----
    # Now that all area rasters are written, split them into train/val/test sets.
    # The split is STRATIFIED: it ensures each split has roughly the same proportion
    # of bubble-containing pixels. Without stratification, all bubble-rich frames might
    # end up in training, leaving validation/test with only empty frames (unfair evaluation).
    written_area_stems = sorted([int(x) for x in written_area_ids])

    # Create lightweight "frame stub" objects that hold just the label band.
    # The split algorithm uses these to calculate positive rates (bubble density per frame).
    class _FrameStub:
        def __init__(self, annotations):
            self.annotations = annotations

    frames = []
    for sid in written_area_stems:
        fp = os.path.join(output_dir, f"{sid}.tif")
        if not os.path.exists(fp):
            continue
        with rasterio.open(fp) as ds:
            lab = ds.read(ds.count)  # Read the last band (the label/mask band)
        frames.append(_FrameStub(lab))

    # Path where the split indices will be saved as JSON
    frames_json = os.path.join(output_dir, "aa_frames_list.json")

    split_list_path = getattr(config, "split_list_path", None)

    # -----------------------------------------------------------------------
    # PREFIX-BASED FORCED TEST SET (currently disabled)
    # To re-enable, uncomment the block below and change the following
    # `if split_list_path` back to `elif split_list_path`.
    # -----------------------------------------------------------------------
    # manual_test_prefix = getattr(config, "manual_test_image_prefix", None)
    # forced_test_pos = set()
    # if manual_test_prefix:
    #     prefix_str = str(manual_test_prefix)
    #     for pos, stem in enumerate(written_area_stems):
    #         im_path = area_to_im.get(int(stem))
    #         if im_path and os.path.basename(im_path).startswith(prefix_str):
    #             forced_test_pos.add(pos)
    #     print(
    #         f"[SPLIT] manual_test_image_prefix='{prefix_str}': "
    #         f"{len(forced_test_pos)} area(s) forced to test set."
    #     )
    #
    # if forced_test_pos:
    #     splittable_pos = [i for i in range(len(frames)) if i not in forced_test_pos]
    #     splittable_frames = [frames[i] for i in splittable_pos]
    #     test_ratio = float(getattr(config, "test_ratio", 0.2))
    #     val_ratio  = float(getattr(config, "val_ratio", 0.2))
    #     val_share  = val_ratio / max(1e-9, 1.0 - test_ratio)
    #     strata = [
    #         1 if (getattr(fr, "annotations", None) is not None
    #               and (np.asarray(fr.annotations) > 0).any())
    #         else 0
    #         for fr in splittable_frames
    #     ]
    #     use_strata = strata if len(set(strata)) > 1 else None
    #     splittable_idx = list(range(len(splittable_frames)))
    #     from sklearn.model_selection import train_test_split as _tts
    #     try:
    #         sub_tr, sub_va = _tts(
    #             splittable_idx,
    #             test_size=val_share,
    #             random_state=int(getattr(config, "split_random_state", 1337)),
    #             stratify=use_strata,
    #         )
    #     except ValueError:
    #         sub_tr, sub_va = _tts(
    #             splittable_idx,
    #             test_size=val_share,
    #             random_state=int(getattr(config, "split_random_state", 1337)),
    #             stratify=None,
    #         )
    #     tr_idx = [splittable_pos[i] for i in sub_tr]
    #     va_idx = [splittable_pos[i] for i in sub_va]
    #     te_idx = sorted(forced_test_pos)
    #     _write_split_json(frames_json, tr_idx, va_idx, te_idx)
    #
    # elif split_list_path is not None ...:  <- restore this elif when re-enabling

    if split_list_path is not None and str(split_list_path).strip() != "" and os.path.exists(str(split_list_path)):
        # Load a previously saved split (ensures same frames in same splits across experiments)
        tr_idx, va_idx, te_idx = _load_split_indices_from_aalist(str(split_list_path), written_area_stems)
        _write_split_json(frames_json, tr_idx, va_idx, te_idx)
    else:
        # Create a new stratified split: 60% train, 20% val, 20% test (by default)
        # stratify_by_positives=True ensures balanced bubble density across splits
        tr_idx, va_idx, te_idx = split_dataset(
            frames,
            frames_json,
            test_size=float(getattr(config, "test_ratio", 0.2)),
            val_size=float(getattr(config, "val_ratio", 0.2)),
            n_bins=int(getattr(config, "split_n_bins", 5)),
            random_state=int(getattr(config, "split_random_state", 1337)),
            stratify_by_positives=True,
        )

    # Print training-like distribution stats
    stats = summarize_positive_rates(
        frames,
        {"train": tr_idx, "val": va_idx, "test": te_idx},
    )

    def _fmt(s):
        return (
            f"{s['mean']:.3f} | {s['median']:.3f} | {s['std']:.3f} | "
            f"{s['min']:.3f}..{s['max']:.3f}  (n={s['n']})"
        )

    print("\n[DATA][STATS] positive-rate % by frame - mean | median | std | min..max  (n)")
    print(f"  train: {_fmt(stats['train'])}")
    print(f"    val: {_fmt(stats['val'])}")
    print(f"   test: {_fmt(stats['test'])}")

    # Build a lookup: area_id -> which split it belongs to ("train", "val", or "test").
    # Chips inherit their parent area's split assignment to prevent DATA LEAKAGE:
    # if a chip from a "test" area ended up in training, the model would be tested
    # on data it has already seen, giving inflated performance numbers.
    split_of_area = {}
    for i in tr_idx:
        if 0 <= int(i) < len(written_area_stems):
            split_of_area[int(written_area_stems[int(i)])] = "train"
    for i in va_idx:
        if 0 <= int(i) < len(written_area_stems):
            split_of_area[int(written_area_stems[int(i)])] = "val"
    for i in te_idx:
        if 0 <= int(i) < len(written_area_stems):
            split_of_area[int(written_area_stems[int(i)])] = "test"
    for aid in written_area_stems:
        if int(aid) not in split_of_area:
            split_of_area[int(aid)] = "train"

    # ============================
    # STEP 5: CREATE CHIPS (small sub-tiles for model training)
    # ============================
    # WHY CHIPS?
    # Training areas can be huge (thousands of pixels wide). Neural networks train on
    # small fixed-size patches (e.g., 256x256). We could randomly sample patches during
    # training (which this pipeline also does), but pre-cutting "chips" has advantages:
    # - Ensures all bubble regions are covered (random sampling might miss sparse bubbles)
    # - Controls the balance of positive (bubble) vs negative (empty) training examples
    # - Chips inherit their parent area's train/val/test assignment (prevents data leakage)
    make_chips = bool(getattr(config, "make_chip_dataset", True))
    if make_chips:
        chip_size_m, chip_pad_m, cluster_buffer_m, chip_max_size_m, neg_chips_per_area = _chip_params_from_config()
        # Seeded random number generator for reproducible negative chip placement
        rng = np.random.default_rng(int(getattr(config, "split_random_state", 1337)) + 17)

        # .buffer(0) is a GIS trick: it "repairs" geometries with self-intersections
        # (e.g., a polygon that crosses itself, which would cause errors in spatial operations)
        polys = polygons.copy()
        try:
            polys["geometry"] = polys["geometry"].buffer(0)
        except Exception:
            pass

        chip_id = 0       # Auto-incrementing chip ID (0, 1, 2, ...)
        chip_split = {}   # Maps chip_id -> "train"/"val"/"test"

        # How far to expand (dilate) bubble pixels when finding connected components.
        # Larger dilation merges nearby bubbles into one chip; smaller keeps them separate.
        chip_cc_dilate_m = float(getattr(config, "chip_cc_dilate_m", min(10.0, 0.25 * chip_size_m)))

        # FOCUS AREAS: A separate shapefile defining where to cut chips.
        # Think of these as "regions of interest" within each training area.
        # Each focus polygon becomes one positive chip (cropped to its bounding box).
        # Skipped when config.focus_areas is None (no focus-area file available).
        focus_by_area = {}
        if getattr(config, "focus_areas", None) is not None:
            focus_fp = os.path.join(config.training_data_dir, config.focus_areas)
            if not os.path.exists(focus_fp):
                raise FileNotFoundError(f"Focus areas file not found: {focus_fp}")

            # Load and align focus area polygons to the same CRS as training areas
            focus_areas = gpd.read_file(focus_fp)
            focus_areas = focus_areas.drop(columns=[c for c in focus_areas.columns if c != "geometry"])
            if focus_areas.crs is None:
                focus_areas = focus_areas.set_crs(areas.crs)
            if focus_areas.crs != areas.crs:
                focus_areas = focus_areas.to_crs(areas.crs)

            # Spatial join: link each focus polygon to the training area(s) it overlaps
            focus_areas = gpd.sjoin(focus_areas, areas, predicate="intersects", how="inner")

            # Pre-group focus geometries by area for fast lookup in the chip loop
            for _rid, _row in focus_areas.iterrows():
                _aid = int(_row["index_right"])
                _geom = _row.geometry
                if _geom is None or _geom.is_empty:
                    continue
                focus_by_area.setdefault(_aid, []).append(_geom)

        for area_id in tqdm(written_area_stems, desc="Building chips (no-overlap CC)", position=0):
            # Keep original guard (but we no longer use src_im; we crop from the area raster)
            if int(area_id) not in area_to_im:
                continue

            split_name = split_of_area.get(int(area_id), "train")

            area_raster_fp = os.path.join(output_dir, f"{area_id}.tif")
            if not os.path.exists(area_raster_fp):
                continue

            # Training area geometry in CRS (same as rasters)
            area_geom_crs = areas.iloc[int(area_id)].geometry

            # POS + NEG chips are made disjoint by tracking occupied EA boxes
            occupied_ea = []

            with rasterio.open(area_raster_fp) as src:
                # --- POS chips: direct crop of focus areas (vector polygons) ---
                # Focus polygons are in areas CRS; we clip them to the training area to avoid leakage,
                # then crop their *bounding boxes* from the already-rasterized area TIFF.
                focus_geoms = focus_by_area.get(int(area_id), [])

                # Convert area geometry to equal-area once for safe box-within-area checks
                try:
                    area_geom_src = gpd.GeoSeries([area_geom_crs], crs=areas.crs).to_crs(src.crs).iloc[0]
                except Exception:
                    area_geom_src = area_geom_crs
                area_ea = gpd.GeoSeries([area_geom_src], crs=src.crs).to_crs(epsg=6933).iloc[0]

                pos_boxes_ea = []
                for fg in focus_geoms:
                    if fg is None or fg.is_empty:
                        continue

                    # Clip focus polygon by training area in vector domain (areas CRS)
                    fg_clip = fg.intersection(area_geom_crs)
                    if fg_clip is None or fg_clip.is_empty:
                        continue

                    # Project to raster CRS for bounds extraction
                    fg_src = gpd.GeoSeries([fg_clip], crs=areas.crs).to_crs(src.crs).iloc[0]
                    b_src = box(*fg_src.bounds)

                    # Convert bbox to equal-area and ensure it is fully within the training area (no leakage)
                    b_ea = gpd.GeoSeries([b_src], crs=src.crs).to_crs(epsg=6933).iloc[0]
                    b_ea = _safe_box_within_area_ea(b_ea, area_ea)
                    if b_ea is None:
                        continue

                    pos_boxes_ea.append(b_ea)

                # Merge overlaps so final chips are non-overlapping within each training area
                if len(pos_boxes_ea) > 1:
                    pos_boxes_ea = _merge_overlapping_boxes_ea(pos_boxes_ea)

                # Write POS chips (crop from area raster)
                for b_ea in pos_boxes_ea:
                    b_crs = gpd.GeoSeries([b_ea], crs="EPSG:6933").to_crs(src.crs).iloc[0]
                    out_fp = os.path.join(chips_dir, f"{chip_id}.tif")
                    _ = _write_chip_from_area_raster(src, b_crs.bounds, out_fp)
                    chip_split[int(chip_id)] = split_name
                    chip_id += 1
                    occupied_ea.append(b_ea)

                occupied_union = unary_union(occupied_ea) if len(occupied_ea) > 0 else None

                # --- NEG chips: random, fully within training area, label-free, and non-overlapping with POS/other NEG ---
                if neg_chips_per_area > 0:
                    # We'll sample in equal-area (meters) for size correctness
                    area_ea = gpd.GeoSeries([area_geom_crs], crs=src.crs).to_crs(epsg=6933).iloc[0]
                    minx, miny, maxx, maxy = area_ea.bounds

                    # Enforce square NEG chips of fixed pixel size (default 512x512)
                    # Compute pixel size (m) from area raster transform
                    try:
                        px_m_x = abs(float(src.transform.a))
                        px_m_y = abs(float(src.transform.e))
                        px_m = float((px_m_x + px_m_y) / 2.0)
                    except Exception:
                        px_m = float(getattr(config, "pixel_size_m", 0.15))

                    neg_chip_px = int(getattr(config, "neg_chip_pixels", 512))
                    half = 0.5 * neg_chip_px * px_m

                    n_done = 0
                    tries = 0
                    max_tries = 400 * max(1, int(neg_chips_per_area))

                    while n_done < int(neg_chips_per_area) and tries < max_tries:
                        tries += 1

                        width = maxx - minx
                        height = maxy - miny

                        # Need enough room for a full square of side 2*half
                        if width <= 2 * half or height <= 2 * half:
                            cx = 0.5 * (minx + maxx)
                            cy = 0.5 * (miny + maxy)
                        else:
                            cx = float(rng.uniform(minx + half, maxx - half))
                            cy = float(rng.uniform(miny + half, maxy - half))


                        # Square NEG chip box sized to produce ~neg_chip_px pixels per side
                        nb = box(cx - half, cy - half, cx + half, cy + half)

                        # Must be fully within training area to prevent leakage
                        if not nb.within(area_ea):
                            continue

                        # No overlap with any existing chip (POS or previously accepted NEG)
                        if occupied_union is not None:
                            inter = nb.intersection(occupied_union)
                            if not inter.is_empty and float(getattr(inter, "area", 0.0)) > 0.0:
                                continue

                        # Convert to raster CRS and crop; then verify label-free
                        nb_crs = gpd.GeoSeries([nb], crs="EPSG:6933").to_crs(src.crs).iloc[0]
                        out_fp = os.path.join(chips_dir, f"{chip_id}.tif")
                        lab = _write_chip_from_area_raster(src, nb_crs.bounds, out_fp)
                        if lab is None:
                            try:
                                os.remove(out_fp)
                            except Exception:
                                pass
                            continue
                        if np.any(np.asarray(lab) > 0):
                            # Not truly negative; remove and retry
                            try:
                                os.remove(out_fp)
                            except Exception:
                                pass
                            continue

                        chip_split[int(chip_id)] = split_name
                        chip_id += 1
                        n_done += 1

                        # Update occupied set (keeps all chips disjoint)
                        occupied_ea.append(nb)
                        occupied_union = unary_union(occupied_ea)

        # Write aa_frames_list.json for chips (inherit split -> index lists)
        train_idx = [i for i in range(chip_id) if chip_split.get(i, "train") == "train"]
        val_idx = [i for i in range(chip_id) if chip_split.get(i, "train") == "val"]
        test_idx = [i for i in range(chip_id) if chip_split.get(i, "train") == "test"]
        _write_split_json(os.path.join(chips_dir, "aa_frames_list.json"), train_idx, val_idx, test_idx)

    print(
        f"Preprocessing completed in "
        f"{str(timedelta(seconds=time.time() - start)).split('.')[0]}.\n"
    )


# Global config holder
config = None
