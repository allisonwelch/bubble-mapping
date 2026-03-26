# core/util.py

"""
Utility helpers around GDAL and memory management.

Printing follows the bracketed tag style used in the training script to keep
logs consistent and easy to scan.
"""

import sys
import platform
from osgeo import gdal

if platform.system() != "Windows":
    import resource


def gdal_progress_callback(complete, message, data):
    """
    Progress callback for GDAL operations such as gdal.Warp or gdal.Translate.

    Expects a tqdm progress bar in 'data', passed via GDAL's 'callback_data'.
    GDAL supplies 'complete' as a float in [0, 1].
    """
    if data:
        data.update(int(complete * 100) - data.n)
        if complete == 1:
            data.close()
    return 1


def raster_copy(
    output_fp,
    input_fp,
    mode="warp",
    resample=1,
    out_crs=None,
    bands=None,
    bounds=None,
    bounds_crs=None,
    multi_core=False,
    pbar=None,
    compress=False,
    cutline_fp=None,
    resample_alg=gdal.GRA_Bilinear,
):
    """
    Copy a raster using GDAL Warp or GDAL Translate, with common options.

    Notes:
        - Use 'mode' to select Warp (multi-core capable) or Translate
          (allows band selection).
        - A specific window to copy can be specified via 'bounds' and
          'bounds_crs'.
        - Optional resampling uses bilinear interpolation when resample != 1.
    """
    # Common options
    base_options = dict(
        creationOptions=[
            "TILED=YES",
            "BLOCKXSIZE=256",
            "BLOCKYSIZE=256",
            "BIGTIFF=IF_SAFER",
            "NUM_THREADS=ALL_CPUS",
        ],
        callback=gdal_progress_callback,
        callback_data=pbar,
    )
    if compress:
        base_options["creationOptions"].append("COMPRESS=LZW")
    if resample != 1:
        # Get input pixel sizes
        raster = gdal.Open(input_fp)
        gt = raster.GetGeoTransform()
        x_res, y_res = gt[1], -gt[5]
        base_options["xRes"] = x_res / resample
        base_options["yRes"] = y_res / resample
        base_options["resampleAlg"] = resample_alg

    # Use GDAL Warp
    if mode.lower() == "warp":
        warp_options = dict(
            dstSRS=out_crs,
            cutlineDSName=cutline_fp,
            outputBounds=bounds,
            outputBoundsSRS=bounds_crs,
            multithread=multi_core,
            warpOptions=["NUM_THREADS=ALL_CPUS"] if multi_core else [],
            # Processing chunk size; around 1-4GB tends to work well
            warpMemoryLimit=1000000000,
        )
        return gdal.Warp(output_fp, input_fp, **base_options, **warp_options)

    # Use GDAL Translate
    if mode.lower() == "translate":
        translate_options = dict(
            bandList=bands,
            outputSRS=out_crs,
            projWin=[bounds.iloc[0], bounds.iloc[3], bounds.iloc[2], bounds.iloc[1]] if bounds is not None else None,
            projWinSRS=bounds_crs,
        )
        return gdal.Translate(output_fp, input_fp, **base_options, **translate_options)

    raise Exception("Invalid mode argument, supported modes are 'warp' or 'translate'.")


def get_driver_name(extension):
    """Map file extension to GDAL/OGR driver name."""
    ext = extension.lower()
    if ext.endswith("tif"):
        return "GTiff"
    if ext.endswith("jp2"):
        return "JP2OpenJPEG"
    if ext.endswith("shp"):
        return "ESRI Shapefile"
    if ext.endswith("gpkg"):
        return "GPKG"
    raise Exception(f"Unable to find driver for unsupported extension {extension}")


def memory_limit(percentage: float):
    """Set soft memory limit to a percentage of total available memory."""
    resource.setrlimit(resource.RLIMIT_AS, (int(get_memory() * 1024 * percentage), -1))
    print(f"[UTIL][MEM] limit={int(percentage*100)}% ({get_memory() * percentage/1024/1024:.2f} GiB)")


def get_memory():
    """
    Get available memory (kB) from Linux system.

    Note:
        Including 'SwapFree:' also counts cache as available memory. This can
        still cause OOM crashes for a memory-heavy single thread.
    """
    with open("/proc/meminfo", "r") as mem_info:
        free_memory = 0
        for line in mem_info:
            if str(line.split()[0]) in ("MemFree:", "Buffers:", "Cached:", "SwapFree:"):
                free_memory += int(line.split()[1])
    return free_memory


def memory(percentage):
    """
    Decorator to limit memory of a function to a percentage of available RAM.

    Printing follows the training script style when errors occur.
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            memory_limit(percentage)
            try:
                return function(*args, **kwargs)
            except MemoryError:
                mem = get_memory() / 1024 / 1024
                print(f"[UTIL][MEM] Available memory: {mem:.2f} GB")
                sys.stderr.write("\n\n[UTIL][MEM][ERROR] Memory Exception\n")
                sys.exit(1)

        return wrapper

    return decorator
