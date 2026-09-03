import os
import glob
import torch
import time
import numpy as np
import rasterio
import tempfile
import pandas as pd
import shutil
import geosam
from geosam import ImageEncoder
from tqdm import tqdm

# Config
checkpoint_path = '/isipd/projects/p_planetdw/data/methods_test/training_images/SAM_checkpoints/sam_vit_l_0b3195.pth'
image_folder = '/isipd/projects/p_planetdw/data/methods_test/training_images/PS/to_encode/'
feature_dir = '/isipd/projects/p_planetdw/data/methods_test/training_images/PS/encodings'

GPU_ID = 0

def float32_to_uint16(image):
    """Convert float32 image to uint16 by shifting by 10000 """
    image = image* 10000
    image = np.clip(image, 0, 65535)  # Clip to valid range for uint16
    image = image.astype(np.uint16)  # Convert to uint16
    return image

if __name__ == "__main__":
    # Diagnostic info
    print(f"[INFO] PyTorch version: {torch.__version__}")
    print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA version: {torch.version.cuda}")
        print(f"[INFO] Number of CUDA devices: {torch.cuda.device_count()}")
        print(f"[INFO] Using GPU: {GPU_ID}")
    
    # Get all .tif image paths
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*.tif")))
    print(f"[INFO] Found {len(image_paths)} images in {image_folder}")

    # check if first image in image path is float32
    with rasterio.open(image_paths[0]) as src:
        dtype = src.dtypes[0]

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPUs available.")

    # Initialize encoder on specified GPU
    device = torch.device(f'cuda:{GPU_ID}')
    torch.cuda.set_device(device)
    print(f"[INFO] Using GPU {GPU_ID}: {torch.cuda.get_device_name(GPU_ID)}")
    
    img_encoder = ImageEncoder(checkpoint_path, gpu_id=GPU_ID)

    # Encode all images sequentially
    for image_path in tqdm(image_paths, desc="Encoding images"):
        try:
            start = time.time()
            img_encoder.encode_image(str(image_path), feature_dir)
            print(f"[GPU {GPU_ID}] Encoded {os.path.basename(image_path)} in {time.time() - start:.2f}s")
        except Exception as e:
            print(f"[GPU {GPU_ID} ERROR] Failed on {image_path}: {e}")
    
    # Fix CSV files
    csv_files = glob.glob(os.path.join(feature_dir, "**", "*.csv"), recursive=True)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df['res'].dtype == 'object':
            df['res'] = df['res'].apply(lambda x: float(x.split(',')[0].replace('(', '')))
            df.to_csv(csv_file, index=False)

    print("[INFO] All encoding complete.")