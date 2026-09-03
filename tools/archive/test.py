import rasterio
import numpy as np
import os
import tqdm

img_dir = r'/isipd/projects/p_planetdw/data/methods_test/training_data/PS/20250604-0816_Unet_Planet_utm8'

img_paths = os.listdir(img_dir)
img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
appendix = '_8'

#delete all images with 4 bands
def delete_images_with_4_bands(img_dir):
    img_paths = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
    for img_path in tqdm.tqdm(img_paths):
        with rasterio.open(img_path) as src:
            img = src.read()
            num_bands = img.shape[0]
            if num_bands == 4:
                os.remove(img_path)
                print(f"Deleted {img_path}")
            else:
                print(f"Kept {img_path} with {num_bands} bands")

#delete images where the first 4 bands include exaytly the value 0
def delete_images_with_0_in_first_4_bands(img_dir):
    img_paths = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
    for img_path in tqdm.tqdm(img_paths):
        with rasterio.open(img_path) as src:
            bands = src.read(range(1, 5))
            if np.any(bands == 0):
                os.remove(img_path)
                print(f"Deleted {img_path}")

def append_test_to_img_filename(img_dir, appendix):
    img_paths = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
    for img_path in tqdm.tqdm(img_paths):
        new_img_path = img_path.replace('.tif', f'{appendix}.tif')
        os.rename(img_path, new_img_path)
        print(f"Renamed {img_path} to {new_img_path}")

def delete_images_with_only_0_in_last_band(img_dir):
    img_paths = os.listdir(img_dir)
    img_paths = [os.path.join(img_dir, img_path) for img_path in img_paths if img_path.endswith('.tif')]
    for img_path in tqdm.tqdm(img_paths):
        with rasterio.open(img_path) as src:
            last_band = src.read(src.count)
            if np.all(last_band == 0):
                os.remove(img_path)
                print(f"Deleted {img_path}")

import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
import warnings

# Suppress warnings for non-georeferenced images
warnings.simplefilter("ignore", NotGeoreferencedWarning)

def process_tif_files(root_dir, target_filename, threshold):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.tif') and target_filename in filename:
                full_path = os.path.join(dirpath, filename)
                print(f"Processing: {full_path}")
                process_single_tif(full_path, threshold)

def process_single_tif(file_path, threshold):
    with rasterio.open(file_path, 'r+') as src:
        data = src.read(1).astype('float32')  # Read first band
        nodata = src.nodata if src.nodata is not None else np.nan

        # Set values above threshold to NaN
        data[data < threshold] = np.nan

        # Update the nodata value if it's not already set to NaN
        if src.nodata != np.nan:
            src.nodata = np.nan

        # Write modified data back
        src.write(data, 1)

def update_nodata_value_for_all_tifs(root_dir, new_nodata_value):
    """
    Updates the NoData value for all .tif files in a directory and its subdirectories.
    
    Parameters:
        root_dir (str): The root directory to search.
        new_nodata_value (float or int): The NoData value to set.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.tif'):
                filepath = os.path.join(dirpath, filename)
                print(f"Updating NoData value for: {filepath}")
                set_nodata(filepath, new_nodata_value)

def set_nodata(filepath, nodata_value):
    """
    Sets the NoData value for a single GeoTIFF file.
    """
    with rasterio.open(filepath, 'r+') as src:
        src.nodata = nodata_value




if __name__ == "__main__":
    append_test_to_img_filename(img_dir, appendix)
    delete_images_with_0_in_first_4_bands(img_dir)
    delete_images_with_only_0_in_last_band(img_dir)
