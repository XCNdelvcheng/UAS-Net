"""
Train 3D Masked Autoencoder (MAE) for self-supervised anatomical prior extraction.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from tqdm import tqdm
from torch.cuda import amp as cuda_amp
import matplotlib
matplotlib.use('Agg')
import matplotlib.subplots as plt
import torch.multiprocessing as mp

from monai.transforms import (
    Compose, RandFlipd, RandRotate90d, RandGaussianNoised,
    Resized, ToTensord
)
from monai.networks.nets import ViTAutoEnc

torch.backends.cudnn.benchmark = True

class CFG:
    NPY_ROOT = "./data/npy_data"
    SAVE_DIR = "./checkpoints"
    SAVE_PATH = os.path.join(SAVE_DIR, "best_tumor_aware_encoder.pth")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    IMG_SIZE = (128, 128, 128)
    PATCH_SIZE = (16, 16, 16)
    HIDDEN_SIZE = 768
    EPOCHS = 300
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    TUMOR_LOSS_WEIGHT = 5.0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class EsophagusCTDataset(Dataset):
    def __init__(self, npy_root, transform=None):
        self.npy_root = npy_root
        self.transform = transform
        self.file_pairs = []

        img_files = sorted([f for f in os.listdir(self.npy_root) if f.endswith("_img.npy")])
        for img_file in img_files:
            base_name = img_file.replace("_img.npy", "")
            seg_file = f"{base_name}_seg.npy"
            seg_path = os.path.join(self.npy_root, seg_file)
            
            if os.path.exists(seg_path):
                self.file_pairs.append((os.path.join(self.npy_root, img_file), seg_path))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        try:
            img_path, seg_path = self.file_pairs[idx]
            
            volume = np.load(img_path)
            if volume.ndim == 3: volume = volume[None, :, :, :]
            elif volume.shape[0] > 1: volume = volume[0:1, :, :, :]

            seg_volume = np.load(seg_path)
            if seg_volume.ndim == 3: seg_volume = seg_volume[None, :, :, :]
            elif seg_volume.shape[0] > 1: seg_volume = seg_volume[0:1, :, :, :]

            data_dict = {
                "image": torch.tensor(volume, dtype=torch.float32),
                "segmentation": torch.tensor(seg_volume, dtype=torch.float32)
            }
            if self.transform:
                data_dict = self.transform(data_dict)
            return data_dict
        except Exception:
            return None

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)

def main():
    set_seed(CFG.SEED)
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    num_gpus = torch.cuda.device_count()

    train_transforms = Compose([
        Resized(keys=["image", "segmentation"], spatial_size=CFG.IMG_SIZE, mode=('trilinear', 'nearest'), align_corners=(False, None)),
        RandFlipd(keys=["image", "segmentation"], spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90d(keys=["image", "segmentation"], prob=0.5, max_k=3),
        RandGaussianNoised(keys=["image"], prob=0.2),
        ToTensord(keys=["image", "segmentation"])
    ])

    dataset = EsophagusCTDataset(CFG.NPY_ROOT, transform=train_transforms)
    dataloader = DataLoader(dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                            num_workers=CFG.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)

    model = ViTAutoEnc(
        in_channels=1, img_size=CFG.IMG_SIZE, patch_size=CFG.PATCH_SIZE,
        hidden_size=CFG.HIDDEN_SIZE, mlp_dim=3072, num_layers=12, num_heads=12
    )

    if num_gpus > 1: model = nn.DataParallel(model)
    model = model.to(CFG.DEVICE)

    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scaler = cuda_amp.GradScaler(enabled=(CFG.DEVICE == "cuda"))

    for epoch in range(CFG.EPOCHS):
        model.train()
        epoch_loss = 0.0
        batches_processed = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{CFG.EPOCHS}")

        for batch in pbar:
            if batch is None: continue

            imgs = batch["image"].to(CFG.DEVICE, non_blocking=True)
            segs = batch["segmentation"].to(CFG.DEVICE, non_blocking=True)
            
            batches_processed += 1
            optimizer.zero_grad(set_to_none=True)

            with cuda_amp.autocast(enabled=(CFG.DEVICE == "cuda")):
                recon, _ = model(imgs)
                loss_pixels = criterion(recon, imgs)
                weights = torch.ones_like(imgs).to(CFG.DEVICE)
                weights[segs > 0] = CFG.TUMOR_LOSS_WEIGHT
                loss = (loss_pixels * weights).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        if batches_processed > 0:
            avg_loss = epoch_loss / batches_processed
            print(f"Epoch [{epoch + 1}/{CFG.EPOCHS}] Avg Loss: {avg_loss:.6f}")

    final_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(final_model.state_dict(), CFG.SAVE_PATH)
    print(f"Pretraining finished! Encoder saved to {CFG.SAVE_PATH}")

if __name__ == "__main__":
    if os.name == 'nt':
        try: mp.set_start_method("spawn", force=True)
        except RuntimeError: pass
    main()
