import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import ViTAutoEnc
from monai.transforms import Compose, Resize
from tqdm import tqdm
from lifelines.utils import concordance_index as lifelines_cindex
import warnings
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda import amp as cuda_amp

warnings.filterwarnings("ignore", category=UserWarning)


class CFG:
    TRAIN_FILE = "./train_processed.csv"
    VAL_FILE = "./val_processed.csv"
    TEST_FILE = "./test_processed.csv"
    NPY_ROOT = "D:/z_up/code/z/npy_data"
    SAVE_DIR = "./checkpoints_clinical_tuned_hp"  # <--- MODIFIED: Save to new dir

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42

    IMG_SIZE = (128, 128, 128)

    EPOCHS = 150
    BATCH_SIZE = 32  # <--- 32 很好，因为事件很充足
    NUM_WORKERS = 8

    # <--- MODIFIED: 关键修复！
    # 使用适合从零训练 MLP 的标准超参数
    LR = 1e-3  # (0.001) 标准 MLP 学习率 (原为 3e-5)
    WEIGHT_DECAY = 1e-5  # (0.00001) 标准 L2 正则 (原为 1e-3)

    CLIN_FEAT_DIM = 19  # (这个值应该会在 main() 中被覆盖)
    DROPOUT_RATE = 0.5
    EARLY_STOPPING_PATIENCE = 30


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


def get_train_transforms(img_size): return None


def get_val_transforms(img_size): return None


class SurvivalCTDataset(Dataset):
    def __init__(self, df, npy_root, img_size, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size

        # (自动检测临床特征维度)
        cols_to_drop = [col for col in ['Num', 'Time', 'State'] if col in df.columns]
        self.clin_data = torch.tensor(df.drop(columns=cols_to_drop).values, dtype=torch.float32)
        CFG.CLIN_FEAT_DIM = self.clin_data.shape[1]  # <--- 动态设置

        self.times = torch.tensor(df['Time'].values, dtype=torch.float32) if 'Time' in df else None
        self.events = torch.tensor(df['State'].values, dtype=torch.int64) if 'State' in df else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        time = self.times[idx] if self.times is not None else torch.tensor(-1.0, dtype=torch.float32)
        event = self.events[idx] if self.events is not None else torch.tensor(-1, dtype=torch.int64)
        clin_vec = self.clin_data[idx]
        volume = torch.empty(1, *self.img_size)  # Dummy image tensor
        return volume, clin_vec, time, event


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    try:
        return torch.utils.data.dataloader.default_collate(batch)
    except RuntimeError as e:
        print(f"!! Collate Error: {e}")
        return None


class SurvivalHead(nn.Module):
    def __init__(self, in_features, dropout_rate=0.5):  # in_features will be 19+
        super(SurvivalHead, self).__init__()
        hidden_dim = 64
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),  # (e.g., 19 -> 64)
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)  # (64 -> 1)
        )
        print(f"--- 初始化的预测头 (Clinical-Only / 简化版) ---")
        print(f"--- Input Features: {in_features} ---")
        print(self.head)
        print(f"---------------------------------")

    def forward(self, x):
        return self.head(x)


class SemiSurvivalNet(nn.Module):
    def __init__(self, clin_feat_dim, dropout_rate):  # (Removed unused args)
        super(SemiSurvivalNet, self).__init__()
        self.clin_feat_dim = clin_feat_dim
        fused_dim = clin_feat_dim
        self.surv_head = SurvivalHead(fused_dim, dropout_rate)

    def forward(self, img, clin):
        risk = self.surv_head(clin)  # Ignore img
        return risk, None


def cox_ph_loss(log_risks, times, events, eps=1e-7):
    # (Unchanged)
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
        # (This should almost never happen now, but good to keep)
        print("!! 警告: 0 个事件在 batch 中, 损失为 0")
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

    if not all_risks: return 0.0
    all_risks = torch.cat(all_risks).numpy()
    all_times = torch.cat(all_times).numpy()
    all_events = torch.cat(all_events).numpy()
    if len(all_risks) < 2: return 0.0
    c_index = lifelines_cindex(all_times, -all_risks, all_events)
    return c_index


def main():
    print("--- 实验配置 (Clinical-Only, Tuned HP) ---")  # <--- MODIFIED
    print(f"设备: {CFG.DEVICE}, Batch Size: {CFG.BATCH_SIZE}, Workers: {CFG.NUM_WORKERS}")
    print(f"LR: {CFG.LR}, Weight Decay: {CFG.WEIGHT_DECAY}")  # <--- MODIFIED
    print(f"Early Stopping Patience: {CFG.EARLY_STOPPING_PATIENCE}\n")

    try:
        df_train_full = pd.read_csv(CFG.TRAIN_FILE)
        df_val = pd.read_csv(CFG.VAL_FILE)
        df_test = pd.read_csv(CFG.TEST_FILE)
    except FileNotFoundError:
        print(f"!! 错误: 找不到 CSV 文件。请先运行 '5_preClinical_v2.py'")
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

        # (动态设置的 CFG.CLIN_FEAT_DIM 会在这里被 train_dataset 第一次初始化时设置好)
        print(f"检测到 {CFG.CLIN_FEAT_DIM} 个临床特征。")

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
        return

    print("--- 正在构建模型... ---")
    student = SemiSurvivalNet(
        clin_feat_dim=CFG.CLIN_FEAT_DIM,
        dropout_rate=CFG.DROPOUT_RATE
    ).to(CFG.DEVICE)
    print("Student 模型 (Clinical-Only) 构建成功.\n")

    param_groups = [
        {'params': student.surv_head.parameters(), 'lr': CFG.LR, 'weight_decay': CFG.WEIGHT_DECAY}
    ]

    print(f"优化器分组: 1 组 (仅 Survival Head)")
    print(f"  - 预测头 LR: {CFG.LR}")

    optimizer = torch.optim.AdamW(param_groups, lr=CFG.LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=CFG.LR * 0.01)
    scaler = cuda_amp.GradScaler(enabled=(CFG.DEVICE == "cuda"))

    print("--- 开始纯监督 (Clinical-Only, Tuned HP) 训练 ---")
    best_val_cindex = -1.0
    best_model_path = os.path.join(CFG.SAVE_DIR, "best_clinical_tuned_hp_model.pth")
    epochs_no_improve = 0

    for epoch in range(CFG.EPOCHS):
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
                risk_l, _ = student(img_l, clin_l)  # img_l is dummy
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
                "LR": f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        if batches_processed == 0:
            print(f"!! 警告: Epoch {epoch + 1} 未处理任何 batch。")
            continue

        avg_loss = epoch_loss_total / batches_processed
        print(
            f"\nEpoch {epoch + 1}/{CFG.EPOCHS} | Avg Supervised Loss: {avg_loss:.4f}")

        val_cindex = evaluate_cindex(student, val_loader, CFG.DEVICE)
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

        scheduler.step()
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
