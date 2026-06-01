# -*- coding: utf-8 -*-
"""
[V-Final-Fix, 脚本 1/2]：预处理脚本

[V-Final-Fix 更新]:
1. [修复] 针对 "OSError: [WinError 1114] c10.dll 失败" 的终极修复。
2. [解决方案] 在 import torch 之前，
   设置 CUDA_VISIBLE_DEVICES="-1"，
   强制此脚本只使用 CPU 运行，从而绕过 NVIDIA 驱动错误。
"""

# [FIX V-Final-Fix] 强制使用 CPU
# 必须在 import torch 之前
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch  # 现在 import torch 时，它会忽略 GPU

from monai.data.image_reader import NibabelReader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped
)


def main():
    # --- 1. 配置 ---

    csv_path = "./dataset_index.csv"

    # !!! 关键：请修改为你希望保存 .npy 文件的文件夹
    output_dir = r"D:\z_up\code\z\npy_data"  # <-- !!! 确保这里是你创建的文件夹 !!!

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出文件夹: {output_dir}")

    # --- 2. 定义我们的确定性变换 ---

    custom_reader = NibabelReader(squeeze_end_dims=False)

    preprocess_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"], reader=custom_reader),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            ScaleIntensityRanged(keys=["image"], a_min=-200.0, a_max=300.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    # --- 3. 加载 CSV ---
    print(f"正在从 {csv_path} 加载数据索引...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"错误: 索引文件未找到: {csv_path}")
        return

    print(f"共找到 {len(df)} 个样本路径。")
    print("预处理将在 CPU 上运行... (这可能需要一些时间)")

    # --- 4. 循环、处理、保存 ---

    good_files_count = 0
    bad_files_count = 0

    for i in tqdm(range(len(df)), desc="预处理并保存 .npy 文件 (CPU)"):
        row = df.iloc[i]
        file_dict = {
            "image": row['ct_path'],
            "label": row['seg_path']
        }

        try:
            if not os.path.exists(file_dict['image']) or not os.path.exists(file_dict['label']):
                print(f"!! 跳过 (索引 {i}): 文件未找到。")
                bad_files_count += 1
                continue

            # 运行所有确定性变换 (在 CPU 上)
            transformed_data = preprocess_transforms(file_dict)

            image_tensor = transformed_data['image']
            label_tensor = transformed_data['label']

            if image_tensor.shape != label_tensor.shape:
                print(f"\n!! 跳过 (索引 {i}): 形状不匹配")
                print(f"!! 文件: {file_dict['label']}")
                print(f"!! Image Shape: {image_tensor.shape}")
                print(f"!! Label Shape: {label_tensor.shape}")
                bad_files_count += 1
                continue

            base_filename = f"{i:03d}"
            img_save_path = os.path.join(output_dir, f"{base_filename}_img.npy")
            seg_save_path = os.path.join(output_dir, f"{base_filename}_seg.npy")

            # .numpy() 即可，因为数据已经在 CPU 上了
            np.save(img_save_path, image_tensor.numpy())
            np.save(seg_save_path, label_tensor.numpy())

            good_files_count += 1

        except Exception as e:
            print(f"\n!! 跳过 (索引 {i}): 预处理时发生未知错误")
            print(f"!! 文件: {file_dict['label']}")
            print(f"!! 错误: {e}")
            bad_files_count += 1

    print("\n--- 预处理完成 ---")
    print(f"成功保存 {good_files_count} 个文件对。")
    print(f"跳过了 {bad_files_count} 个“坏”文件。")
    print(f"NPY 文件保存在: {output_dir}")


if __name__ == "__main__":
    main()