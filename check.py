import geopandas as gpd

REPO_PATH = os.path.expanduser("~/git_repos/bubble-mapping")


pred = gpd.read_file(f"{REPO_PATH}/data/results/SWIN/AE/20260428-1537_SWINxAE.weights/pred_seeps.gpkg")
gt = gpd.read_file(f"{REPO_PATH}/data/results/SWIN/AE/20260428-1537_SWINxAE.weights/gt_seeps.gpkg")

n_multi = 0
n_total = 0
for img, p_chip in pred.groupby("image"):
  g_chip = gt[gt["image"] == img]
  if g_chip.empty:
      n_total += len(p_chip)
      continue
  joined = gpd.sjoin(p_chip, g_chip[["seep_id", "geometry"]],
                     how="left", predicate="intersects")
  counts = joined.groupby("cluster_id")["seep_id"].nunique()
  n_total += len(counts)
  n_multi += int((counts >= 2).sum())

print(f"clusters spanning >=2 GT polys: {n_multi} / {n_total} "
    f"({100*n_multi/max(n_total,1):.1f}%)")
