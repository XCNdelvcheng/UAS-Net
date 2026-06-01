import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import ViTAutoEnc
from monai.transforms import (
    Compose, Resize, RandFlip, RandRotate90, RandGaussianNoise, RandAffine
)
from tqdm import tqdm
import torch.nn.functional as F
import copy
# lifelines 用于 C-Index
from lifelines.utils import concordance_index as lifelines_cindex
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR

# AMP / autocast & GradScaler
from torch.cuda import amp as cuda_amp

warnings.filterwarnings("ignore", category=UserWarning)


class CFG:
    TRAIN_FILE = "./train_processed.csv"
    VAL_FILE = "./val_processed.csv"
    TEST_FILE = "./test_processed.csv"
    NPY_ROOT = "D:/z_up/code/z/npy_data"
    PRETRAINED_ENCODER_PATH = "D:/z_up/code/z/checkpoints/pretrained_encoder_dualGPU.pth"
    SAVE_DIR = "./checkpoints_finetune"  # <--- MODIFIED: 保存到新目录

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    IMG_SIZE = (128, 128, 128)

    EPOCHS = 150
    BATCH_SIZE = 32
    NUM_WORKERS = 8

    # <--- MODIFIED: 定义主学习率和微调学习率
    # LR = 3e-5  # 预测头 (Projector + Head) 的学习率
    # LR_FINETUNE = 3e-6  # Encoder 微调学习率 (LR * 0.1)
    #
    # WEIGHT_DECAY = 1e-3  # 保持对预测头的强正则化
    LR = 1e-3  # Correct LR for head/projector
    LR_FINETUNE = 1e-5  # LR for finetuning encoder layers
    WEIGHT_DECAY = 1e-5  # Consistent with successful clinical run

    MAE_EMBED_DIM = 768
    IMG_FEAT_DIM = 128
    CLIN_FEAT_DIM = 19
    DROPOUT_RATE = 0.5

    EARLY_STOPPING_PATIENCE = 20  # <--- 也许可以适当延长到 30


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


set_seed(CFG.SEED)
os.makedirs(CFG.SAVE_DIR, exist_ok=True)


def get_train_transforms(img_size):
    return Compose([
        Resize(spatial_size=img_size, mode='trilinear', align_corners=False),
        RandFlip(spatial_axis=[0, 1, 2], prob=0.5),
        RandRotate90(prob=0.5, max_k=3),
        # <--- MODIFIED: 按照 "资料一" 建议，加入更强的数据增强
        RandAffine(
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),  # 3D 旋转
            scale_range=(0.9, 1.1),
            translate_range=(0.05, 0.05, 0.05),
            padding_mode="zeros"
        ),
        RandGaussianNoise(prob=0.1),
    ])


def get_val_transforms(img_size):
    return Compose([
        Resize(spatial_size=img_size, mode='trilinear', align_corners=False),
    ])


class SurvivalCTDataset(Dataset):
    def __init__(self, df, npy_root, img_size, transform=None):
        self.df = df.reset_index(drop=True)
        self.npy_root = npy_root
        self.transform = transform
        self.img_size = img_size
        self.img_files = [os.path.join(self.npy_root, f"{int(row['Num']):03d}_img.npy") for _, row in df.iterrows()]
        cols_to_drop = [col for col in ['Num', 'Time', 'State'] if col in df.columns]
        self.clin_data = torch.tensor(df.drop(columns=cols_to_drop).values, dtype=torch.float32)
        self.times = torch.tensor(df['Time'].values, dtype=torch.float32) if 'Time' in df else None
        self.events = torch.tensor(df['State'].values, dtype=torch.int64) if 'State' in df else None
        self.clin_feat_names = df.drop(columns=cols_to_drop).columns.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        time = self.times[idx] if self.times is not None else torch.tensor(-1.0, dtype=torch.float32)
        event = self.events[idx] if self.events is not None else torch.tensor(-1, dtype=torch.int64)
        clin_vec = self.clin_data[idx]
        try:
            npy_path = self.img_files[idx]
            if not os.path.exists(npy_path):
                raise FileNotFoundError(f"File not found: {npy_path}")
            volume = np.load(npy_path)
            if volume.ndim == 3:
                volume = volume[None, :, :, :]
            elif volume.ndim == 4 and volume.shape[0] > 1:
                volume = volume[0:1, :, :, :]
            volume = torch.as_tensor(volume, dtype=torch.float32)
            if self.transform:
                volume = self.transform(volume)
        except Exception as e:
            print(f"!! 错误: 加载或转换失败 {self.img_files[idx]}: {e}")
            return None
        return volume, clin_vec, time, event


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        print(f"!! Collate Error: {e}")
        for i, item_tuple in enumerate(batch):
            print(f"  Item {i} shapes: {[it.shape if hasattr(it, 'shape') else type(it) for it in item_tuple]}")
        return None


# (保持简化的预测头不变)
class SurvivalHead(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5):
        super(SurvivalHead, self).__init__()
        # 简化为：[Linear -> BN -> ReLU -> Dropout] -> [Linear_Out]
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),  # 单个隐藏层
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 高 Dropout 率
            nn.Linear(128, 1)  # 输出
        )
        print(f"--- 初始化的预测头 (简化版) ---")
        print(self.head)
        print(f"---------------------------------")

    def forward(self, x):
        return self.head(x)


class SemiSurvivalNet(nn.Module):
    def __init__(self, mae_model, mae_embed_dim, img_feat_dim, clin_feat_dim, dropout_rate):
        super(SemiSurvivalNet, self).__init__()
        self.mae_model = mae_model
        self.projector = nn.Linear(mae_embed_dim, img_feat_dim)
        self.clin_feat_dim = clin_feat_dim
        fused_dim = img_feat_dim + clin_feat_dim
        self.surv_head = SurvivalHead(fused_dim, dropout_rate)

    def get_img_features(self, x):
        """
        Robust: 支持多种 ViT/MAE 实现。
        优先尝试：使用 cls_token + pos_embed（若存在）
        否则：使用 patch embedding 后的 token 做池化(mean 或首 token)
        返回: z_img (projected, B x IMG_FEAT_DIM), z_img_raw (原始 embedding, B x MAE_EMBED_DIM)
        """
        # <--- MODIFIED: 确保 mae_model 在 .train() 模式（如果需要微调）
        # <--- 或者在 eval() 模式（如果 BN 层需要冻结）
        # <--- ViTAutoEnc 没有 BN, 所以 .train() 是安全的
        # self.mae_model.train() # 确保解冻的层（和 dropout, if any）是激活的

        # ensure input is float and on same device as model
        x = x.to(next(self.mae_model.parameters()).device, dtype=torch.float32)

        # Case A: model exposes patch embedding + cls_token + pos_embed + blocks + norm (like many ViT variants)
        try:
            if hasattr(self.mae_model, "patch_embedding") and hasattr(self.mae_model, "cls_token") and hasattr(
                    self.mae_model, "pos_embed"):
                x_pe = self.mae_model.patch_embedding(x)
                x_pe = x_pe.flatten(2).transpose(1, 2)
                cls_token = self.mae_model.cls_token.expand(x_pe.shape[0], -1, -1)
                x_pe = torch.cat((cls_token, x_pe), dim=1)
                pos = self.mae_model.pos_embed
                if pos.shape[1] != x_pe.shape[1]:
                    if pos.shape[1] < x_pe.shape[1]:
                        diff = x_pe.shape[1] - pos.shape[1]
                        pos = torch.cat([pos, pos[:, :diff, :].clone()], dim=1)
                    else:
                        pos = pos[:, :x_pe.shape[1], :].clone()
                x_pe = x_pe + pos
                if hasattr(self.mae_model, "blocks"):
                    for blk in self.mae_model.blocks:
                        x_pe = blk(x_pe)
                if hasattr(self.mae_model, "norm"):
                    x_pe = self.mae_model.norm(x_pe)
                z_img_raw = x_pe[:, 0]
                z_img = self.projector(z_img_raw)
                return z_img, z_img_raw
        except Exception as e:
            pass

        # Case B: model has a forward_features / encode / encoder method we can leverage
        try:
            if hasattr(self.mae_model, "forward_features"):
                z = self.mae_model.forward_features(x)
                if z.ndim == 3:
                    z_img_raw = z[:, 0]
                else:
                    z_img_raw = z
                z_img = self.projector(z_img_raw)
                return z_img, z_img_raw
            if hasattr(self.mae_model, "encode") and callable(self.mae_model.encode):
                z = self.mae_model.encode(x)
                if z.ndim == 3:
                    z_img_raw = z[:, 0]
                else:
                    z_img_raw = z
                z_img = self.projector(z_img_raw)
                return z_img, z_img_raw
        except Exception:
            pass

        # Case C (fallback):
        try:
            if hasattr(self.mae_model, "patch_embedding"):
                x_pe = self.mae_model.patch_embedding(x)
                x_pe = x_pe.flatten(2).transpose(1, 2)  # (B, N, D)
                z_img_raw = x_pe.mean(dim=1)  # (B, D)
                if z_img_raw.shape[1] != self.projector.in_features:
                    tmp = nn.Linear(z_img_raw.shape[1], self.projector.in_features).to(z_img_raw.device)
                    z_img_raw = tmp(z_img_raw)
                z_img = self.projector(z_img_raw)
                return z_img, z_img_raw
        except Exception:
            pass

        # 极端情况
        with torch.no_grad():
            pooled = x.view(x.size(0), x.size(1), -1).mean(-1)  # (B, C)
        if pooled.shape[1] != self.projector.in_features:
            tmp = nn.Linear(pooled.shape[1], self.projector.in_features).to(pooled.device)
            z_img_raw = tmp(pooled)
        else:
            z_img_raw = pooled
        z_img = self.projector(z_img_raw)
        return z_img, z_img_raw

    def forward(self, img, clin):
        # <--- MODIFIED: Encoder 现在参与梯度计算
        # <--- get_img_features 内部不再需要 no_grad()
        z_img, z_img_raw = self.get_img_features(img)

        # (保持不变)
        z_fused = torch.cat([z_img, clin], dim=1)
        risk = self.surv_head(z_fused)
        return risk, z_img_raw


def cox_ph_loss(log_risks, times, events, eps=1e-7):
    times, sort_idx = torch.sort(times, descending=True)
    events = events[sort_idx]
    log_risks = log_risks[sort_idx]
    log_risk_exp = torch.exp(log_risks)
    cumsum_exp_risks = torch.cumsum(log_risk_exp.flip(0), dim=0).flip(0)
    log_cumsum_exp_risks = torch.log(cumsum_exp_risks + eps)
    observed_log_risks = log_risks[events == 1]
    observed_log_cumsum = log_cumsum_exp_risks[events == 1]
    loss = - (observed_log_risks - observed_log_cumsum).sum()
    num_events = (events == 1).sum()
    if num_events > 0:
        loss = loss / num_events
    else:
        loss = torch.tensor(0.0, device=log_risks.device, dtype=log_risks.dtype)
    return loss


@torch.no_grad()
def evaluate_cindex(model, loader, device):
    model.eval()
    all_risks = []
    all_times = []
    all_events = []
    for batch in tqdm(loader, desc="Validating", leave=False, ncols=80):
        if batch is None: continue
        try:
            img, clin, times, events = [item.to(device, non_blocking=True) for item in batch]
            with cuda_amp.autocast(enabled=(device == "cuda")):
                risks, _ = model(img, clin)
            all_risks.append(risks.squeeze(1).cpu())
            all_times.append(times.cpu())
            all_events.append(events.cpu())
        except Exception as e:
            print(f"!! 评估时出错 (Batch): {e}")
            continue

    if not all_risks:
        return 0.0
    all_risks = torch.cat(all_risks).numpy()
    all_times = torch.cat(all_times).numpy()
    all_events = torch.cat(all_events).numpy()
    if len(all_risks) < 2:
        return 0.0
    c_index = lifelines_cindex(all_times, -all_risks, all_events)
    return c_index


def main():
    print("--- 实验配置 (纯监督 + 微调) ---")  # <--- MODIFIED
    print(f"设备: {CFG.DEVICE}, Batch Size: {CFG.BATCH_SIZE}, Workers: {CFG.NUM_WORKERS}")
    print(f"LR (Head): {CFG.LR}, LR (Finetune): {CFG.LR_FINETUNE}, Weight Decay: {CFG.WEIGHT_DECAY}")  # <--- MODIFIED
    print(f"Early Stopping Patience: {CFG.EARLY_STOPPING_PATIENCE}\n")

    try:
        df_train_full = pd.read_csv(CFG.TRAIN_FILE)
        df_val = pd.read_csv(CFG.VAL_FILE)
        df_test = pd.read_csv(CFG.TEST_FILE)
        CFG.CLIN_FEAT_DIM = len(df_train_full.drop(
            columns=[col for col in ['Num', 'Time', 'State'] if col in df_train_full.columns]).columns)
        print(f"检测到 {CFG.CLIN_FEAT_DIM} 个临床特征。")
    except FileNotFoundError:
        print(f"!! 错误: 找不到 CSV 文件 ({CFG.TRAIN_FILE} 或其他)。请确保它们在脚本同目录下。")
        return
    except Exception as e:
        print(f"!! 加载 CSV 时出错: {e}")
        return

    print(f"总训练数据: {len(df_train_full)} (全部用于监督训练)")
    print(f"验证: {len(df_val)}, 测试: {len(df_test)}\n")

    train_transforms = get_train_transforms(CFG.IMG_SIZE)
    val_transforms = get_val_transforms(CFG.IMG_SIZE)
    try:
        train_dataset = SurvivalCTDataset(df_train_full, CFG.NPY_ROOT, CFG.IMG_SIZE, transform=train_transforms)
        val_dataset = SurvivalCTDataset(df_val, CFG.NPY_ROOT, CFG.IMG_SIZE, transform=val_transforms)
        test_dataset = SurvivalCTDataset(df_test, CFG.NPY_ROOT, CFG.IMG_SIZE, transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                                  num_workers=CFG.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn,
                                  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                                num_workers=CFG.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                                 num_workers=CFG.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
        print("DataLoaders 创建完毕。\n")
    except Exception as e:
        print(f"!! 创建 Dataset/DataLoader 时出错: {e}")
        print(f"  NPY Root: {CFG.NPY_ROOT}")
        print(f"  Train CSV: {CFG.TRAIN_FILE}")
        return

    print("--- 正在构建模型... ---")
    try:
        mae_vit = ViTAutoEnc(in_channels=1, img_size=CFG.IMG_SIZE, patch_size=(16, 16, 16),
                             hidden_size=CFG.MAE_EMBED_DIM, mlp_dim=3072, num_layers=12, num_heads=12)
        print(f"正在加载预训练权重: {CFG.PRETRAINED_ENCODER_PATH}")
        if os.path.exists(CFG.PRETRAINED_ENCODER_PATH):
            mae_weights = torch.load(CFG.PRETRAINED_ENCODER_PATH, map_location=CFG.DEVICE)
            new_state = {}
            for k, v in mae_weights.items():
                new_key = k
                if k.startswith("module."):
                    new_key = k.replace("module.", "")
                new_state[new_key] = v
            try:
                mae_vit.load_state_dict(new_state, strict=False)
                print("Encoder 权重加载成功 (strict=False).")
            except Exception as e:
                print(f"!! 加载 encoder 权重时出现问题: {e}. 将继续使用随机初始化的 encoder。")
        else:
            print(f"!! 警告: 未找到预训练权重 {CFG.PRETRAINED_ENCODER_PATH}。模型将随机初始化。")
    except Exception as e:
        print(f"!! 构建 ViTAutoEnc 失败: {e}. 尝试使用默认初始化模型。")
        mae_vit = ViTAutoEnc(in_channels=1, img_size=CFG.IMG_SIZE, patch_size=(16, 16, 16),
                             hidden_size=CFG.MAE_EMBED_DIM, mlp_dim=3072, num_layers=12, num_heads=12)

    student = SemiSurvivalNet(mae_vit, CFG.MAE_EMBED_DIM, CFG.IMG_FEAT_DIM, CFG.CLIN_FEAT_DIM, CFG.DROPOUT_RATE).to(
        CFG.DEVICE)
    print("Student 模型构建成功.\n")

    # <--- MODIFIED: 按照方案 2.1, 冻结所有层，然后解冻最后两层
    print("--- 正在设置 Encoder 参数 (Finetune)... ---")
    unfrozen_params = 0
    for name, param in student.mae_model.named_parameters():
        param.requires_grad = False  # 默认全部冻结

        # 假设您的 ViTAutoEnc 内部 ViT 叫做 "vit" 并且有 "blocks"
        # "资料一" 中提到 'blocks.11' 和 'blocks.10'
        # MONAI ViTAutoEnc 的 ViT 模块通常就叫 'vit'
        if 'vit.blocks.11' in name or 'vit.blocks.10' in name:
            param.requires_grad = True
            unfrozen_params += 1
            print(f"  解冻层: {name}")

    print(f"--- {unfrozen_params} 个 Encoder 参数已被解冻 (用于微调) ---")
    if unfrozen_params == 0:
        print("!! 警告: 没有任何层被解冻。请检查 'vit.blocks.11' 命名是否正确。")
        print("!! 尝试打印模型结构以确认:")
        # print(student.mae_model) # (如果需要，取消此行注释来查看结构)

    # <--- MODIFIED: 按照方案 2.2, 设置差分学习率
    param_groups = [
        # 1. 预测头 (Projector + Head)，使用标准 LR
        {'params': student.projector.parameters(), 'lr': CFG.LR},
        {'params': student.surv_head.parameters(), 'lr': CFG.LR, 'weight_decay': CFG.WEIGHT_DECAY},  # 对 Head 施加 L2 正则

        # 2. 已解冻的 Encoder 层，使用更小的 Finetune LR
        {'params': [p for n, p in student.mae_model.named_parameters() if p.requires_grad], 'lr': CFG.LR_FINETUNE}
    ]

    # 打印出有多少参数组
    print(f"优化器分组: {len(param_groups)} 组")
    print(f"  - 预测头 LR: {CFG.LR}")
    print(f"  - 微调层 LR: {CFG.LR_FINETUNE}")

    optimizer = torch.optim.AdamW(param_groups, lr=CFG.LR)  # 默认 LR 在这里不起作用，以 group 内为准
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.LR * 0.01)
    scaler = cuda_amp.GradScaler(enabled=(CFG.DEVICE == "cuda"))

    print("--- 开始纯监督 (微调) 训练 ---")
    best_val_cindex = -1.0
    best_model_path = os.path.join(CFG.SAVE_DIR, "best_finetune_model.pth")
    epochs_no_improve = 0
    train_losses, val_cindices = [], []

    for epoch in range(CFG.EPOCHS):
        # <--- MODIFIED: 必须设置为 .train() 模式，以激活解冻的层
        student.train()
        epoch_loss_total = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CFG.EPOCHS}", ncols=100)
        batches_processed = 0

        for batch in pbar:
            if batch is None: continue

            try:
                img_l, clin_l, times_l, events_l = [item.to(CFG.DEVICE, non_blocking=True) for item in batch]
            except Exception as e:
                print(f"!! 移动数据到 GPU 失败: {e}")
                continue

            optimizer.zero_grad(set_to_none=True)

            with cuda_amp.autocast(enabled=(CFG.DEVICE == "cuda")):
                risk_l, _ = student(img_l, clin_l)
                loss_sup = cox_ph_loss(risk_l.squeeze(1), times_l, events_l)
                loss_total = loss_sup

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                print(f"!! 检测到 NaN/Inf 损失，跳过更新。")
                continue

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss_total += loss_total.item() if not torch.isnan(loss_total) else 0.0
            batches_processed += 1

            pbar.set_postfix({
                "L_sup": f"{loss_total.item():.4f}",
                "LR_Head": f"{optimizer.param_groups[0]['lr']:.1e}",
                "LR_Tune": f"{optimizer.param_groups[2]['lr']:.1e}"  # <--- 监控微调学习率
            })

        if batches_processed == 0:
            print(f"!! 警告: Epoch {epoch + 1} 未处理任何 batch。")
            continue

        avg_loss = epoch_loss_total / batches_processed
        train_losses.append(avg_loss)
        print(
            f"\nEpoch {epoch + 1}/{CFG.EPOCHS} | Avg Supervised Loss: {avg_loss:.4f}")

        # <--- MODIFIED: 评估时必须切换回 .eval()
        val_cindex = evaluate_cindex(student, val_loader, CFG.DEVICE)
        val_cindices.append(val_cindex)
        print(f"Validation C-Index: {val_cindex:.4f}")

        if val_cindex > best_val_cindex:
            best_val_cindex = val_cindex
            print(f"🎉 新的最佳 C-Index: {best_val_cindex:.4f}。保存模型到 {best_model_path}")
            torch.save(student.state_dict(), best_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CFG.EARLY_STOPPING_PATIENCE:
                print(f"\n!! Early stopping 触发: 验证 C-Index 已连续 {CFG.EARLY_STOPPING_PATIENCE} 轮未改进。")
                break

        scheduler.step()  # (在 epoch 结束时 step)
        if CFG.DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\n--- 训练完成 (最佳验证 C-Index: {best_val_cindex:.4f}) ---")

    if os.path.exists(best_model_path):
        print(f"加载最佳模型进行最终测试...")
        state = torch.load(best_model_path, map_location=CFG.DEVICE)
        student.load_state_dict(state)
        test_cindex = evaluate_cindex(student, test_loader, CFG.DEVICE)
        print(f"\n--- 最终测试结果 ---")
        print(f"Test C-Index: {test_cindex:.4f}")
    else:
        print("!! 未找到最佳模型，无法进行最终测试。")

    print("\n--- 脚本执行完毕 ---")


if __name__ == "__main__":
    main()
