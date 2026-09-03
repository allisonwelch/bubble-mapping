import json
from pathlib import Path
import re
from typing import Optional, List

import geopandas as gpd
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FuncFormatter

import seaborn as sns


# ----------------------------
# USER OPTIONS (paper version)
# ----------------------------
# Leave empty => draw an empty PS subplot (but keep the 3-panel layout)
#LABELS_PS = ""

# Boxplot settings (can adjust showfliers, width, etc.)
# BW_ADJUST = 1.0  # Not used for boxplots

TEXTOFF = 0.925

# ----------------------------
# PALETTE OPTIONS (CHOOSABLE CONSTANTS)
# ----------------------------
# Panel order (requested): AE, PS, S2
PANEL_ORDER = ["AE", "PS", "S2"]

# Matplotlib colormaps used per modality (change these to anything in mpl colormap list)
# Examples: "Blues", "Greens", "Oranges", "Purples", "viridis", "magma", "cividis", etc.
MODALITY_CMAPS = {
    "AE": "Blues",
    "PS": "Greens",
    "S2": "Oranges",
}

# Shade range sampled from each colormap for train/val/test (0..1)
# Lower -> lighter, higher -> darker
PALETTE_SHADE_MIN = 0.15
PALETTE_SHADE_MAX = 0.65

# Plot split order (keep as-is unless you change your splits)
SPLIT_ORDER = ["training", "validation", "testing"]

# ----------------------------
# Paths
# ----------------------------
TRAINING_AREAS = r"/isipd/projects/p_planetdw/data/methods_test/training/AE/training_areas.gpkg"

LABELS_S2 = r"/isipd/projects/p_planetdw/data/methods_test/training/S2/labels_S2_exp.gpkg"
LABELS_AE = r"/isipd/projects/p_planetdw/data/methods_test/training/AE/labels_AE.gpkg"
LABELS_PS = r"/isipd/projects/p_planetdw/data/methods_test/training/PS/labels_PS.gpkg"
DATA_SPLIT_AA = r"/isipd/projects/p_planetdw/data/methods_test/preprocessed/20251226-0433_UNETxAE/aa_frames_list.json"

FIG_DIR = Path("figs")
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = FIG_DIR / "area_boxplots_m2ticks.pdf"
OUT_PNG = FIG_DIR / "area_boxplots_m2ticks.png"

OUT_TRAINING_AREAS_SPLIT = (
    r"/isipd/projects/p_planetdw/data/methods_test/training/AE/training_areas_with_split.gpkg"
)


# ----------------------------
# Helpers
# ----------------------------
def safe_read_labels(path: str, source: str) -> Optional[gpd.GeoDataFrame]:
    """Read a label file if path is non-empty. Return None if empty or read fails."""
    if path is None or str(path).strip() == "":
        return None
    try:
        gdf = gpd.read_file(path)
        if gdf is None or len(gdf) == 0:
            return None
        gdf = gdf.copy()
        gdf["source"] = source
        return gdf
    except Exception as e:
        print(f"[WARN] Could not read {source} labels from: {path}\n       {e}")
        return None


def make_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Best-effort geometry validity fix (works well with shapely>=2)."""
    gdf = gdf.copy()
    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        pass
    return gdf


def sci_compact(x: float) -> str:
    """
    Compact scientific formatting for tick labels:
      1e+03 -> 1e3
      1e+00 -> 1
    """
    if not np.isfinite(x) or x <= 0:
        return ""
    s = f"{x:.0e}"  # e.g. 1e+03
    s = s.replace("e+0", "e").replace("e+", "e").replace("e0", "")
    if re.fullmatch(r"-?\d+e0", s):
        s = s.replace("e0", "")
    return s


def empty_panel(ax, title: str, ns: List[int], ylabel: Optional[str] = None):
    ax.set_title(title, loc="left", pad=14)
    if ylabel:
        ax.set_ylabel(ylabel)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["train", "val", "test"])
    ax.set_xlabel(" ")

    # n labels centered above each category
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, n in enumerate(ns):
        ax.text(
            i, TEXTOFF, f"n={n}",
            transform=trans, ha="center", va="bottom",
            fontsize=9, clip_on=False
        )

    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
            ha="center", va="center", alpha=0.6)


# ----------------------------
# Load labels (PS optional)
# ----------------------------
labels_s2 = safe_read_labels(LABELS_S2, "S2")
labels_ae = safe_read_labels(LABELS_AE, "AE")
labels_ps = safe_read_labels(LABELS_PS, "PS")  # None if empty

loaded = [g for g in (labels_s2, labels_ae, labels_ps) if g is not None]
if not loaded:
    raise RuntimeError("No label files could be loaded (S2/AE/PS).")

# Harmonize CRS across loaded layers (you said already UTM 8N)
base_crs = loaded[0].crs
for i in range(len(loaded)):
    if loaded[i].crs != base_crs:
        loaded[i] = loaded[i].to_crs(base_crs)

all_labels = gpd.GeoDataFrame(pd.concat(loaded, ignore_index=True), crs=base_crs)

# ----------------------------
# Load training areas and align CRS
# ----------------------------
training_areas = gpd.read_file(TRAINING_AREAS)
print(f"found {len(training_areas)} training areas")

if training_areas.crs != all_labels.crs:
    training_areas = training_areas.to_crs(all_labels.crs)

training_areas = training_areas.copy()
training_areas["training_area_id"] = training_areas.index.astype(int)

# Best-effort geometry fix
all_labels = make_valid(all_labels)
training_areas = make_valid(training_areas)

# Clip labels to training areas
all_labels_clipped = gpd.clip(all_labels, training_areas)

# Spatial join to attach training_area_id
all_labels_joined = gpd.sjoin(
    all_labels_clipped,
    training_areas[["training_area_id", "geometry"]],
    how="left",
    predicate="intersects",
)

# If overlapping training areas create duplicates, keep first match
if all_labels_joined.index.duplicated().any():
    all_labels_joined = all_labels_joined[~all_labels_joined.index.duplicated(keep="first")]

# Remove sjoin artifact columns
all_labels = all_labels_joined.drop(
    columns=[c for c in all_labels_joined.columns if c.startswith("index_")],
    errors="ignore",
)
all_labels["training_area_id"] = all_labels["training_area_id"].astype("Int64")

# ----------------------------
# Load split lists and assign split
# ----------------------------
with open(DATA_SPLIT_AA, "r") as f:
    aa_frames = json.load(f)

training_frames_list = aa_frames["training_frames"]
validation_frames_list = aa_frames["validation_frames"]
testing_frames_list = aa_frames["testing_frames"]

all_labels["data_split"] = "unknown"
all_labels.loc[all_labels["training_area_id"].isin(training_frames_list), "data_split"] = "training"
all_labels.loc[all_labels["training_area_id"].isin(validation_frames_list), "data_split"] = "validation"
all_labels.loc[all_labels["training_area_id"].isin(testing_frames_list), "data_split"] = "testing"

print(all_labels["data_split"].value_counts(dropna=False))

# ----------------------------
# Compute area (UTM => m²) and log10 transform for plotting
# ----------------------------
all_labels["area_m2"] = all_labels.geometry.area.astype(float)
all_labels.loc[all_labels["area_m2"] <= 0, "area_m2"] = np.nan

# Violin is drawn in log10 space (better behaved for heavy tails)
all_labels["log10_area_m2"] = np.log10(all_labels["area_m2"])

split_order = SPLIT_ORDER
sources_fixed = PANEL_ORDER
panel_labels = ["(a)", "(b)", "(c)"]

df_plot = all_labels.loc[
    all_labels["data_split"].isin(split_order) & np.isfinite(all_labels["log10_area_m2"]),
    ["source", "data_split", "log10_area_m2"],
].copy()

vals = df_plot["log10_area_m2"].to_numpy()
if len(vals) == 0:
    raise RuntimeError("No valid areas to plot (check geometry / split assignment).")

# Global y-lims in log10 units (full range; nothing cut off)
ymin = float(np.nanmin(vals))
ymax = float(np.nanmax(vals))
pad = 0.06 * max(1e-6, (ymax - ymin))
ymin_plot, ymax_plot = ymin - pad, ymax + pad

# Decade ticks (labels will be shown in m²)
kmin = int(np.floor(ymin_plot))
kmax = int(np.ceil(ymax_plot))
decade_ticks = np.arange(kmin, kmax + 1, 1)

# ----------------------------
# Plot styling (paper-ish)
# ----------------------------
mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

sns.set_style("white")

# ----------------------------
# MODALITY-SPECIFIC PALETTES
# ----------------------------
palettes_by_source = {}
for src in sources_fixed:
    cmap_name = MODALITY_CMAPS.get(src, "Greys")
    cmap = get_cmap(cmap_name)
    shades = cmap(np.linspace(PALETTE_SHADE_MIN, PALETTE_SHADE_MAX, 3))
    palettes_by_source[src] = {
        "training": shades[0],
        "validation": shades[1],
        "testing": shades[2],
    }

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(10.0, 3.9), constrained_layout=True)

for ax, source, plab in zip(axes, sources_fixed, panel_labels):
    sub = df_plot[df_plot["source"] == source]
    title = f"{plab} {source}"

    # n centered above each boxplot (train/val/test)
    ns = [int((sub["data_split"] == sp).sum()) for sp in split_order]

    if sub.empty:
        empty_panel(ax, title, ns, ylabel="Area (m²)" if ax is axes[0] else None)
        ax.set_ylim(ymin_plot, ymax_plot)
        ax.set_yticks(decade_ticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: sci_compact(10.0 ** y)))
        continue

    ax.set_title(title, loc="left", pad=14)

    split_palette = palettes_by_source[source]

    sns.boxplot(
        data=sub,
        x="data_split",
        y="log10_area_m2",
        order=split_order,
        palette=split_palette,
        linewidth=0.9,
        width=0.6,
        ax=ax,
    )

    ax.set_xticklabels(["train", "val", "test"])
    ax.set_xlabel(" ")

    # n labels centered above each boxplot
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, n in enumerate(ns):
        ax.text(
            i, TEXTOFF, f"n={n}",
            transform=trans, ha="center", va="bottom",
            fontsize=9, clip_on=False
        )

    # Y axis: plotted in log10-space, but labeled in m²
    ax.set_ylim(ymin_plot, ymax_plot)
    ax.set_yticks(decade_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: sci_compact(10.0 ** y)))

    ax.grid(True, axis="y", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[0].set_ylabel("Area (m²)")

fig.savefig(OUT_PDF, bbox_inches="tight")
fig.savefig(OUT_PNG, dpi=450, bbox_inches="tight")
plt.show()

print(f"Saved plots: {OUT_PDF} and {OUT_PNG}")

# ----------------------------
# Save training areas with split attribute
# ----------------------------
training_areas_out = training_areas.copy()
training_areas_out["data_split"] = "unknown"
training_areas_out.loc[training_areas_out["training_area_id"].isin(training_frames_list), "data_split"] = "training"
training_areas_out.loc[training_areas_out["training_area_id"].isin(validation_frames_list), "data_split"] = "validation"
training_areas_out.loc[training_areas_out["training_area_id"].isin(testing_frames_list), "data_split"] = "testing"

training_areas_out.to_file(OUT_TRAINING_AREAS_SPLIT)
print(f"Saved: {OUT_TRAINING_AREAS_SPLIT}")
