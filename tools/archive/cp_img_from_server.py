import geopandas as gpd
import os
from tqdm import tqdm
import shutil

AOI = gpd.read_file("/isipd/projects/p_planetdw/data/methods_test/auxilliary_data/aoi_ext.gpkg")
NAME_COL = 'target_name'

FROM_DIR = "/isipd/projects-noreplica/p_macsprocessing/PermaX_MACS/PermaX_2025/data_products/"
TO_DIR = "/isipd/projects/p_planetdw/data/methods_test/training_images/AE"

aoi_names = AOI[NAME_COL].tolist()


# check if TO_DIR and FROM_DIR exist
if not os.path.exists(TO_DIR):
    raise Exception(f"TO_DIR {TO_DIR} does not exist")

if not os.path.exists(FROM_DIR):
    raise Exception(f"FROM_DIR {FROM_DIR} does not exist")

# from FROM_DIR and its subfolders copy all files that contain any aoi_name and end with _Ortho.tif to TO_DIR


for root, dirs, files in os.walk(FROM_DIR):
    for file in tqdm(files):
        if file.endswith("_Ortho.tif") and any(aoi_name in file for aoi_name in aoi_names):
            shutil.copy(os.path.join(root, file), TO_DIR)
            print(f"Copied {file} to {TO_DIR}")