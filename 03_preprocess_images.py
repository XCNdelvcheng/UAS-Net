"""
Preprocess CT images and masks into standardized .npy arrays using MONAI.
"""
import os
# Force CPU if you encountered driver issues during preprocessing. 
# Change to "" if you want to enable GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from monai.data.image_reader import NibabelReader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, 
    Spacingd, ScaleIntensityRanged, EnsureTyped
)

def main():
    csv_path = "./data/dataset_index.csv"
    output_dir = "./data/npy_data"

    os.makedirs(output_dir, exist_ok=True)

    custom_reader = NibabelReader(squeeze_end_dims=False)
    preprocess_transforms = Compose([
        LoadImaged(keys=["image", "label"], reader=custom_reader),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True),
    ])

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Index file not found at {csv_path}")
        return

    good_files, bad_files = 0, 0

    for i in tqdm(range(len(df)), desc="Preprocessing and saving .npy files"):
        row = df.iloc[i]
        file_dict = {
            "image": row['ct_path'],
            "label": row['seg_path']
        }

        try:
            if not os.path.exists(file_dict['image']) or not os.path.exists(file_dict['label']):
                bad_files += 1
                continue

            transformed_data = preprocess_transforms(file_dict)
            image_tensor = transformed_data['image']
            label_tensor = transformed_data['label']

            if image_tensor.shape != label_tensor.shape:
                bad_files += 1
                continue

            base_filename = f"{i:03d}"
            img_save_path = os.path.join(output_dir, f"{base_filename}_img.npy")
            seg_save_path = os.path.join(output_dir, f"{base_filename}_seg.npy")

            np.save(img_save_path, image_tensor.numpy())
            np.save(seg_save_path, label_tensor.numpy())

            good_files += 1

        except Exception as e:
            bad_files += 1

    print(f"\nPreprocessing Complete. Saved: {good_files}, Skipped: {bad_files}")

if __name__ == "__main__":
    main()
