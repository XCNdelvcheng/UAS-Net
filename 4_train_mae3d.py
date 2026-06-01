import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pydicom
import numpy as np
from monai.transforms import (
    Compose, RandFlip, RandRotate90, RandGaussianNoise,
    ToTensor, Resize
)
from monai.networks.nets import ViTAutoEnc
from tqdm import tqdm
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

torch.backends.cudnn.benchmark = True

# ===================== 参数 =====================
DATA_ROOT = r"D:\z_up\code\z\npy_data"
BATCH_SIZE = 8
EPOCHS = 300
LR = 1e-4
SAVE_DIR = "./checkpoints"
SAVE_PATH = os.path.join(SAVE_DIR, "pretrained_encoder_dualGPU.pth")
RESUME_PATH = None

os.makedirs(SAVE_DIR, exist_ok=True)


# ===================== 数据集 =====================
class EsophagusCTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir

        self.image_files = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.endswith("_img.npy")
        ])

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        npy_path = self.image_files[idx]

        try:
            # 加载 numpy 数组 (尺寸不一, 通道可能也不一)
            volume = np.load(npy_path)

            # [FIX 4] 强制单通道
            # 检查通道维度 (shape[0])
            if volume.shape[0] > 1:
                # print(f"Warning: {npy_path} has {volume.shape[0]} channels. Selecting first channel.")
                # 如果通道数大于1 (例如 [2, 264, 264, 166]),
                # 我们只选择第一个通道，并保持维度 (切片 [0:1])
                volume = volume[0:1, :, :, :]

            # 经过此处理, volume 保证是 (1, H, W, D) 形状

        except Exception as e:
            print(f"Error loading {npy_path}: {e}. Skipping.")
            return None

        if self.transform:
            # 应用数据增强 (包括 Resize)
            # transform 接收 (1, H, W, D)
            # transform 输出 (1, 128, 128, 128)
            volume = self.transform(volume)

        return volume


# ===================== 可视化函数 =====================
def visualize_reconstruction(img, recon, epoch, save_dir=SAVE_DIR):
    if img.nelement() == 0 or recon.nelement() == 0:
        print(f"Skipping visualization for epoch {epoch} due to empty batch.")
        return

    img_np = img[0, 0, :, :, 64].detach().cpu().numpy()
    recon_np = recon[0, 0, :, :, 64].detach().cpu().numpy()
    diff = np.abs(img_np - recon_np)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_np, cmap="gray");
    axs[0].set_title("Original")
    axs[1].imshow(recon_np, cmap="gray");
    axs[1].set_title("Reconstructed")
    axs[2].imshow(diff, cmap="hot");
    axs[2].set_title("Difference")
    for ax in axs: ax.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_dir}/recon_epoch{epoch}.png")
    plt.close()


# ===================== Collate Fn =====================
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return torch.empty(0)
    try:
        # 现在所有 batch item 都保证是 [1, 128, 128, 128]
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        print(f"Error in collate_fn: {e}")
        for i, item in enumerate(batch):
            print(f"  Item {i} shape: {item.shape if hasattr(item, 'shape') else 'No shape'}")
        return torch.empty(0)


# ===================== 主程序 =====================
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    print(f"🔧 Using {num_gpus} GPU(s) for training")

    # 变换
    train_transforms = Compose([
        # 必需：将所有不同大小的 .npy 统一调整为模型需要的 (128, 128, 128)
        Resize(spatial_size=(128, 128, 128), mode='trilinear'),

        # 数据增强
        RandFlip(spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90(prob=0.5, max_k=3),
        RandGaussianNoise(prob=0.2),
        ToTensor()  # 确保最终是 Tensor
    ])

    # 数据集 & DataLoader
    dataset = EsophagusCTDataset(DATA_ROOT, transform=train_transforms)

    if len(dataset) == 0:
        print(f"错误: 在 {DATA_ROOT} 中没有找到任何 '*_img.npy' 文件。")
        print("请检查 DATA_ROOT 路径是否正确，以及预处理是否已运行。")
        return

    print(f"成功找到 {len(dataset)} 个 _img.npy 文件。")

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True,
                            collate_fn=collate_fn)

    # 模型 & 优化器
    model = ViTAutoEnc(
        in_channels=1,
        img_size=(128, 128, 128),
        patch_size=(16, 16, 16),
        hidden_size=768,
        mlp_dim=3072,
        num_layers=12,
        num_heads=12
    )

    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs via nn.DataParallel.")
        model = nn.DataParallel(model)

    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 混合精度训练
    scaler = GradScaler(device='cuda')

    # 断点续训
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        print(f"🔄 Resuming from {RESUME_PATH}")
        model_dict = torch.load(RESUME_PATH, map_location=DEVICE)

        if num_gpus > 1:
            model.load_state_dict(model_dict)
        else:
            if "module." in list(model_dict.keys())[0]:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_dict.items():
                    name = k.replace("module.", "")
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(model_dict)

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}", ncols=100)

        for imgs in pbar:
            if imgs.nelement() == 0:
                continue

            imgs = imgs.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                recon, _ = model(imgs)
                loss = criterion(recon, imgs)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        if len(dataloader) == 0 or epoch_loss == 0.0:
            print("DataLoader is empty or no data was processed, skipping epoch.")
            continue

        avg_loss = epoch_loss / len(dataloader)
        print(f"✅ Epoch [{epoch + 1}/{EPOCHS}] Avg Loss: {avg_loss:.6f}")

        torch.cuda.empty_cache()

        # 保存模型与重建可视化
        if (epoch + 1) % 50 == 0:

            save_model = model.module if isinstance(model, nn.DataParallel) else model
            torch.save(save_model.state_dict(), f"{SAVE_DIR}/mae_epoch{epoch + 1}.pth")

            with torch.no_grad():
                try:
                    sample_batch = next(iter(dataloader))
                    if sample_batch.nelement() > 0:
                        sample = sample_batch.to(DEVICE)

                        model.eval()
                        with autocast(device_type='cuda'):
                            recon = model(sample)

                        visualize_reconstruction(sample, recon, epoch + 1)
                        model.train()

                except StopIteration:
                    print("DataLoader empty, skipping visualization.")
                except Exception as e:
                    print(f"Error during visualization: {e}")
                    model.train()

    # 最终保存
    final_model = model.module if isinstance(model, nn.DataParallel) else model
    torch.save(final_model.state_dict(), SAVE_PATH)
    print(f"🎯 Pretraining finished! Encoder saved to {SAVE_PATH}")


# ===================== Windows 入口 =====================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()