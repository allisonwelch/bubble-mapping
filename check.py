import os
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")
GT_PATH = f"{REPO_PATH}/data/results/SWIN/AE/20260428-1537_SWINxAE.weights/gt_seeps.gpkg"

gt = gpd.read_file(GT_PATH)

# Columns from seep_feature_table.aggregate_clusters
area_col = "total_area_m2" if "total_area_m2" in gt.columns else "area_m2"
nb_col = "n_bubbles" if "n_bubbles" in gt.columns else None

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Area: log-spaced bins (long right tail expected)
a = gt[area_col].to_numpy()
a = a[a > 0]
bins_a = np.logspace(np.log10(a.min()), np.log10(a.max()), 30)
axes[0].hist(a, bins=bins_a, edgecolor="black")
axes[0].set_xscale("log")
axes[0].set_xlabel(f"{area_col} (m^2, log)")
axes[0].set_ylabel("count")
axes[0].set_title(f"GT seep area  (n={len(a)})")
for q in (0.25, 0.5, 0.75, 0.95):
    axes[0].axvline(np.quantile(a, q), color="red", ls="--", lw=0.8,
                    label=f"q{int(q * 100)}={np.quantile(a, q):.3f}")
axes[0].legend(fontsize=8)


solidity_col = "solidity" if "solidity" in gt.columns else None
if solidity_col:
  s = gt[solidity_col].dropna().to_numpy()
  axes[1].hist(s, bins=30, edgecolor="black")
  axes[1].set_xlabel("solidity (area / convex-hull area)")
  axes[1].set_ylabel("count")
  axes[1].set_title(f"GT solidity  (n={len(s)})")
  for q in (0.25, 0.5, 0.75):
      axes[1].axvline(np.quantile(s, q), color="red", ls="--", lw=0.8,
                      label=f"q{int(q*100)}={np.quantile(s, q):.2f}")
  axes[1].legend(fontsize=8)


plt.tight_layout()
plt.show()

# Print stratum suggestion: quartiles on log(area)
print("\nSuggested area strata (quartile-based):")
qs = np.quantile(a, [0, 0.25, 0.5, 0.75, 1.0])
print("\nSuggested area strata (quartile-based):")
qs = np.quantile(a, [0, 0.25, 0.5, 0.75, 1.0])
for i in range(4):
    in_bin = ((a >= qs[i]) & (a <= qs[i + 1])).sum()
    print(f"  [{qs[i]:.4f}, {qs[i + 1]:.4f}] m^2  ->  {in_bin} seeps")
