"""
Train the multimodal survival prediction network using Uncertainty-aware Mean Teacher.
"""
import os
import random
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda import amp as cuda_amp
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm
from lifelines.utils import concordance_index as lifelines_cindex

from monai.networks.nets import ViTAutoEnc
from monai.transforms import (
    Compose,
    Resized,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    RandGaussianNoised,
    RandShiftIntensityd,
    RandCoarseDropoutd,
)

warnings.filterwarnings("ignore")


# ===================== 1. Global Configurations (CFG) =====================
class CFG:
    # --- Paths ---
    TRAIN_DF_PATH = "./data/train_processed.csv"
    TEST_DF_PATH = "./data/test_processed.csv"
    NPY_ROOT = "./data/npy_data"
    PRETRAINED_ENCODER = "./checkpoints/best_tumor_aware_encoder.pth"
    SAVE_DIR = "./checkpoints/cv_5fold"

    # --- Hardware & Training ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 2026

    # --- Model Parameters ---
    IMG_SIZE = (128, 128, 128)
    PATCH_SIZE = (16, 16, 16)
    HIDDEN_SIZE = 768
    CLIN_FEAT_DIM = 18
    DROPOUT_RATE = 0.3

    # --- Training Hyperparameters ---
    EPOCHS = 150
    BATCH_SIZE = 24
    LABELED_BATCH_SIZE = 12
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_WORKERS = 4  # Adjusted for standard parallel data loading
    EARLY_STOPPING_PATIENCE = 40

    # --- Optimizer ---
    LR_ENCODER = 1e-5
    LR_HEAD = 5e-4
    WEIGHT_DECAY = 0.02

    # --- Semi-supervised Mean Teacher ---
    LABELED_RATIO = 0.5
    EMA_DECAY = 0.9995
    LAMBDA_CONS = 1.5
    CONSISTENCY_RAMPUP_EPOCHS = 50
    UNCERTAINTY_T = 10
    UNCERTAINTY_BETA = 1.0


# ===================== 2. Utility Functions =====================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def _update_teacher_model(student_model, teacher_model, ema_decay=0.999):
    for student_p, teacher_p in zip(student_model.parameters(), teacher_model.parameters()):
        teacher_p.data.mul_(ema_decay).add_(student_p.data, alpha=1 - ema_decay)


def get_consistency_rampup_weight(epoch, rampup_epochs):
    if epoch >= rampup_epochs:
        return 1.0
    return np.exp(-5.0 * (1.0 - epoch / rampup_epochs) ** 2)


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


class InfiniteDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


# ===================== 3. Data Processing & Augmentation =====================

def get_transforms(img_size, mode="cpu_pre"):
    if mode == "cpu_pre":
        return Compose([
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            Resized(keys=["image"], spatial_size=img_size, mode='trilinear'),
        ])
    elif mode == "gpu_aug":
        return Compose([
            RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.5),
            RandRotate90d(keys=["image"], prob=0.5, max_k=3),
            RandAffined(
                keys=["image"], prob=0.5, rotate_range=(np.pi / 12),
                scale_range=(0.1, 0.1), mode='bilinear', padding_mode='zeros'
            ),
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandCoarseDropoutd(
                keys=["image"], holes=2, spatial_size=(32, 32, 32),
                prob=0.5, fill_value=0,
            )
        ])
    elif mode == "tta":
        # TTA specific transforms (deterministic flip)
        return Compose([
            RandFlipd(keys=["image"], spatial_axis=[0], prob=1.0),
        ])
    else:
        return Compose([])


GLOBAL_DATA_CACHE = {}


def preload_all_data(df, npy_root):
    global GLOBAL_DATA_CACHE
    print(f">> [Global Cache] Preloading all {len(df)} .npy files into RAM...")
    count = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Loading NPY"):
        pid = int(row['Num'])
        if pid in GLOBAL_DATA_CACHE:
            continue
        npy_path = os.path.join(npy_root, f"{pid:03d}_img.npy")
        if os.path.exists(npy_path):
            try:
                vol = np.load(npy_path)
                if vol.shape[0] > 1: vol = vol[0:1, ...]
                vol = vol.astype(np.float32)
                GLOBAL_DATA_CACHE[pid] = vol
                count += 1
            except:
                print(f"Error loading {npy_path}")
    print(f">> [Global Cache] Successfully loaded {count} unique volumes.\n")


class InMemorySurvivalDataset(Dataset):
    def __init__(self, df, img_size, transform=None, is_labeled=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_labeled = is_labeled

        cols_to_drop = [col for col in ['Num', 'Time', 'State'] if col in df.columns]
        self.clin_data = torch.tensor(self.df.drop(columns=cols_to_drop).values, dtype=torch.float32)

        if self.clin_data.shape[1] > 0:
            CFG.CLIN_FEAT_DIM = self.clin_data.shape[1]

        self.times = torch.tensor(self.df['Time'].values, dtype=torch.float32) if 'Time' in df.columns else None
        self.events = torch.tensor(self.df['State'].values, dtype=torch.int64) if 'State' in df.columns else None

        self.valid_indices = []
        for idx, row in self.df.iterrows():
            pid = int(row['Num'])
            if pid in GLOBAL_DATA_CACHE:
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        row = self.df.iloc[original_idx]
        pid = int(row['Num'])

        volume = GLOBAL_DATA_CACHE[pid]
        clin_vec = self.clin_data[original_idx]
        img_tensor = torch.tensor(volume, dtype=torch.float32)

        data_dict = {"image": img_tensor}
        if self.transform:
            data_dict = self.transform(data_dict)

        t = self.times[original_idx] if self.is_labeled else torch.tensor(-1.0)
        e = self.events[original_idx] if self.is_labeled else torch.tensor(-1)

        return data_dict["image"], clin_vec, t, e


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch: return None
    return torch.utils.data.dataloader.default_collate(batch)


# ===================== 4. Model Architecture =====================

class SurvivalHead(nn.Module):
    def __init__(self, in_features_img, in_features_clin, dropout_rate):
        super(SurvivalHead, self).__init__()

        # Asymmetric Fusion: Compress image features to 16 dims to limit modality dominance
        self.img_branch = nn.Sequential(
            nn.Linear(in_features_img, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # Expand clinical features to 48 dims to enhance their prognostic weight
        self.clin_branch = nn.Sequential(
            nn.Linear(in_features_clin, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        fused_dim = 16 + 48

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )

    def forward(self, img_feat, clin_feat):
        x_img = self.img_branch(img_feat)
        x_clin = self.clin_branch(clin_feat)
        x_cat = torch.cat([x_img, x_clin], dim=1)
        return self.head(x_cat)


class SemiSurvivalNet(nn.Module):
    def __init__(self, img_size, patch_size, hidden_size, clin_feat_dim, dropout_rate, pretrained_path=None):
        super(SemiSurvivalNet, self).__init__()
        self.encoder = ViTAutoEnc(
            in_channels=1, img_size=img_size, patch_size=patch_size,
            hidden_size=hidden_size, mlp_dim=3072, num_layers=12, num_heads=12, dropout_rate=0.0,
        )
        self.feature_dropout = nn.Dropout(p=dropout_rate)
        
        if pretrained_path and os.path.exists(pretrained_path):
            try:
                state_dict = torch.load(pretrained_path, map_location="cpu")
                self.encoder.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded pretrained encoder from {pretrained_path}")
            except Exception as e:
                print(f"Failed to load pretrained encoder: {e}")
                
        self.surv_head = SurvivalHead(hidden_size, clin_feat_dim, dropout_rate)

    def forward(self, img, clin):
        output = self.encoder(img)
        last_feat = output[1][-1]

        if last_feat.dim() == 5:
            img_feat = last_feat.mean(dim=[2, 3, 4])
        else:
            img_feat = last_feat[:, 0]

        img_feat = self.feature_dropout(img_feat)
        risk = self.surv_head(img_feat, clin)
        return risk, img_feat


# ===================== 5. Loss Functions =====================

def cox_ph_loss(log_risks, times, events, eps=1e-7):
    mask = events == 1
    if not mask.any(): return 0.0 * log_risks.sum()
    times_sorted, sort_idx = torch.sort(times, descending=True)
    log_risks_sorted = log_risks[sort_idx]
    events_sorted = events[sort_idx]
    log_risk_exp = torch.exp(log_risks_sorted)
    cumsum_exp_risks = torch.cumsum(log_risk_exp, dim=0)
    log_cumsum_exp_risks = torch.log(cumsum_exp_risks + eps)
    log_cumsum_at_event_time = log_cumsum_exp_risks[events_sorted == 1]
    loss = - (log_risks_sorted[events_sorted == 1] - log_cumsum_at_event_time).sum()
    return loss / (mask.sum() + eps)


def semi_supervised_loss(risk_l, times_l, events_l, risk_u_student, risk_u_teacher_mean, weights_u):
    loss_sup = cox_ph_loss(risk_l.squeeze(1), times_l, events_l)
    loss_cons_per_sample = F.mse_loss(risk_u_student.squeeze(1), risk_u_teacher_mean.squeeze(), reduction='none')
    weights_mean = weights_u.mean()
    mask = weights_u.squeeze() > weights_mean
    if mask.sum() > 0:
        loss_cons = (weights_u.squeeze()[mask] * loss_cons_per_sample[mask]).mean()
    else:
        loss_cons = 0.0 * risk_u_student.sum()
    return loss_sup, loss_cons


# ===================== 6. Validation & TTA =====================

@torch.no_grad()
def evaluate_cindex(model, loader, device):
    model.eval()
    all_risks, all_times, all_events = [], [], []
    for batch in loader:
        if batch is None: continue
        img, clin, t, e = [i.to(device, non_blocking=True) for i in batch]
        with cuda_amp.autocast(enabled=True):
            risks, _ = model(img, clin)
        all_risks.extend(risks.squeeze(1).cpu().numpy())
        all_times.extend(t.cpu().numpy())
        all_events.extend(e.cpu().numpy())
    if len(all_risks) < 2: return 0.0
    try:
        return lifelines_cindex(all_times, -np.array(all_risks), all_events)
    except:
        return 0.0


def apply_aug_batch(images, transform_func):
    batch_size = images.shape[0]
    augmented_list = []
    for i in range(batch_size):
        data_item = {"image": images[i]}
        data_item = transform_func(data_item)
        augmented_list.append(data_item["image"])
    return torch.stack(augmented_list)


@torch.no_grad()
def predict_with_tta(model, loader, device, tta_transform):
    """Test-Time Augmentation (TTA) inference"""
    model.eval()
    all_risks, all_times, all_events = [], [], []

    for batch in loader:
        if batch is None: continue
        img, clin, t, e = [i.to(device) for i in batch]

        # 1. Original Prediction
        with cuda_amp.autocast():
            risk_orig, _ = model(img, clin)

        # 2. TTA Prediction (Flipped)
        img_aug = apply_aug_batch(img.clone().cpu(), tta_transform).to(device)
        with cuda_amp.autocast():
            risk_aug, _ = model(img_aug, clin)

        # Average fusion
        avg_risk = (risk_orig + risk_aug) / 2.0

        all_risks.extend(avg_risk.squeeze(1).cpu().numpy())
        all_times.extend(t.cpu().numpy())
        all_events.extend(e.cpu().numpy())

    return all_risks, all_times, all_events


# ===================== 7. Main Execution =====================
def main():
    print(f"--- Running on: {CFG.DEVICE} ---")
    set_seed(CFG.SEED)
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)

    try:
        df_train_full = pd.read_csv(CFG.TRAIN_DF_PATH)
        df_test_independent = pd.read_csv(CFG.TEST_DF_PATH)
        print(f"Loaded Train: {len(df_train_full)}, Test: {len(df_test_independent)}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    df_for_loading = pd.concat([df_train_full, df_test_independent], axis=0).reset_index(drop=True)
    preload_all_data(df_for_loading, CFG.NPY_ROOT)

    cpu_pre_trans = get_transforms(CFG.IMG_SIZE, "cpu_pre")
    ds_test_independent = InMemorySurvivalDataset(
        df_test_independent, CFG.IMG_SIZE, transform=cpu_pre_trans, is_labeled=True
    )
    loader_test_independent = DataLoader(
        ds_test_independent, batch_size=CFG.BATCH_SIZE, shuffle=False,
        num_workers=CFG.NUM_WORKERS, collate_fn=collate_fn
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    test_results_all_folds = []
    
    gpu_aug_transform = get_transforms(CFG.IMG_SIZE, mode="gpu_aug")
    tta_transform = get_transforms(CFG.IMG_SIZE, mode="tta")

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_full, df_train_full['State'])):
        print(f"\n{'=' * 20} Fold {fold + 1} / 5 {'=' * 20}")

        df_fold_train = df_train_full.iloc[train_idx].reset_index(drop=True)
        df_fold_val = df_train_full.iloc[val_idx].reset_index(drop=True)

        if CFG.LABELED_RATIO < 1.0:
            df_fold_train_l, df_fold_train_u = train_test_split(
                df_fold_train, train_size=CFG.LABELED_RATIO,
                stratify=df_fold_train['State'], random_state=CFG.SEED
            )
        else:
            df_fold_train_l = df_fold_train
            df_fold_train_u = pd.DataFrame()

        print(f"Fold {fold + 1} Stats -> L: {len(df_fold_train_l)}, U: {len(df_fold_train_u)}, Val: {len(df_fold_val)}")

        ds_l = InMemorySurvivalDataset(df_fold_train_l, CFG.IMG_SIZE, transform=cpu_pre_trans, is_labeled=True)
        ds_u = InMemorySurvivalDataset(df_fold_train_u, CFG.IMG_SIZE, transform=cpu_pre_trans, is_labeled=False) if len(df_fold_train_u) > 0 else []
        ds_val = InMemorySurvivalDataset(df_fold_val, CFG.IMG_SIZE, transform=cpu_pre_trans, is_labeled=True)

        dl_args = {"num_workers": CFG.NUM_WORKERS, "pin_memory": True, "collate_fn": collate_fn}
        loader_l = DataLoader(ds_l, batch_size=CFG.LABELED_BATCH_SIZE, shuffle=True, drop_last=True, **dl_args)
        
        loader_u = None
        if len(df_fold_train_u) > 0:
            loader_u = DataLoader(ds_u, batch_size=CFG.BATCH_SIZE - CFG.LABELED_BATCH_SIZE, shuffle=True, drop_last=True, **dl_args)
            
        loader_val = DataLoader(ds_val, batch_size=CFG.BATCH_SIZE, shuffle=False, **dl_args)
        loader_l_infinite = InfiniteDataLoader(loader_l)

        student = SemiSurvivalNet(
            CFG.IMG_SIZE, CFG.PATCH_SIZE, CFG.HIDDEN_SIZE, CFG.CLIN_FEAT_DIM, 
            CFG.DROPOUT_RATE, CFG.PRETRAINED_ENCODER
        ).to(CFG.DEVICE)
        
        teacher = SemiSurvivalNet(
            CFG.IMG_SIZE, CFG.PATCH_SIZE, CFG.HIDDEN_SIZE, CFG.CLIN_FEAT_DIM, 
            CFG.DROPOUT_RATE, CFG.PRETRAINED_ENCODER
        ).to(CFG.DEVICE)
        
        teacher.load_state_dict(student.state_dict())
        for p in teacher.parameters(): 
            p.requires_grad = False

        optimizer = torch.optim.AdamW([
            {'params': student.encoder.parameters(), 'lr': CFG.LR_ENCODER},
            {'params': student.surv_head.parameters(), 'lr': CFG.LR_HEAD}
        ], weight_decay=CFG.WEIGHT_DECAY)

        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS - 10, eta_min=1e-7)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[10])

        scaler = cuda_amp.GradScaler()

        best_fold_cindex = -1.0
        epochs_no_improve = 0
        freeze_epochs = 20

        main_loader = loader_u if loader_u is not None else loader_l

        for epoch in range(CFG.EPOCHS):
            if epoch == freeze_epochs:
                print(">> Unfreezing encoder parameters...")
                for p in student.encoder.parameters(): p.requires_grad = True
            elif epoch < freeze_epochs:
                for p in student.encoder.parameters(): p.requires_grad = False

            student.train()
            epoch_loss_sum = 0.0
            epoch_steps = 0

            pbar = tqdm(main_loader, desc=f"Fold {fold + 1} Ep {epoch + 1}", ncols=100, leave=False)

            for step, batch_wrapper in enumerate(pbar):
                if loader_u is not None:
                    batch_u = batch_wrapper
                    batch_l = next(loader_l_infinite)
                else:
                    batch_l = batch_wrapper
                    batch_u = None

                img_l, clin_l, t_l, e_l = [i.to(CFG.DEVICE, non_blocking=True) for i in batch_l]

                with torch.no_grad():
                    img_l_aug = apply_aug_batch(img_l, gpu_aug_transform)
                    if batch_u is not None:
                        img_u, clin_u, _, _ = [i.to(CFG.DEVICE, non_blocking=True) for i in batch_u]
                        img_u_student = apply_aug_batch(img_u.clone(), gpu_aug_transform)
                        img_u_teacher = img_u

                with cuda_amp.autocast():
                    risk_l, _ = student(img_l_aug, clin_l)
                    loss_sup = cox_ph_loss(risk_l.squeeze(1), t_l, e_l)
                    loss_cons = torch.tensor(0.0).to(CFG.DEVICE)

                    if batch_u is not None:
                        risk_u_stu, _ = student(img_u_student, clin_u)
                        
                        with torch.no_grad():
                            enable_dropout(teacher)
                            preds = []
                            for _ in range(CFG.UNCERTAINTY_T):
                                r_t, _ = teacher(img_u_teacher, clin_u)
                                preds.append(r_t)
                            preds = torch.stack(preds).squeeze(-1)
                            risk_u_tea_mean = preds.mean(dim=0)
                            risk_u_tea_var = preds.var(dim=0)
                            
                            # Dynamic soft-weighting using estimated epistemic uncertainty
                            weights = torch.exp(-CFG.UNCERTAINTY_BETA * risk_u_tea_var)
                            
                        _, loss_cons = semi_supervised_loss(risk_l, t_l, e_l, risk_u_stu, risk_u_tea_mean, weights)

                    rampup = get_consistency_rampup_weight(epoch, CFG.CONSISTENCY_RAMPUP_EPOCHS)
                    loss = loss_sup + CFG.LAMBDA_CONS * rampup * loss_cons
                    loss = loss / CFG.GRADIENT_ACCUMULATION_STEPS

                if torch.isnan(loss) or torch.isinf(loss): continue

                scaler.scale(loss).backward()

                if (step + 1) % CFG.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    _update_teacher_model(student, teacher, CFG.EMA_DECAY)
                    optimizer.zero_grad()

                current_loss_val = loss.item() * CFG.GRADIENT_ACCUMULATION_STEPS
                epoch_loss_sum += current_loss_val
                epoch_steps += 1
                pbar.set_postfix({'Loss': f"{current_loss_val:.2f}"})

            scheduler.step()
            pbar.close()

            val_cindex = evaluate_cindex(student, loader_val, CFG.DEVICE)
            avg_train_loss = epoch_loss_sum / max(epoch_steps, 1)

            is_best = val_cindex > best_fold_cindex
            save_tag = ""
            if is_best:
                best_fold_cindex = val_cindex
                epochs_no_improve = 0
                torch.save(student.state_dict(), os.path.join(CFG.SAVE_DIR, f"best_model_fold{fold}.pth"))
                save_tag = "[Saved ★]"
            else:
                epochs_no_improve += 1

            print(
                f"Epoch {epoch + 1:03d} | Loss: {avg_train_loss:.4f} | Val C-Index: {val_cindex:.4f} | Best: {best_fold_cindex:.4f} {save_tag}",
                flush=True)

            if epochs_no_improve >= CFG.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}", flush=True)
                break

        print(f"Fold {fold + 1} Finished. Best Val: {best_fold_cindex:.4f}\n", flush=True)

        if os.path.exists(os.path.join(CFG.SAVE_DIR, f"best_model_fold{fold}.pth")):
            student.load_state_dict(torch.load(os.path.join(CFG.SAVE_DIR, f"best_model_fold{fold}.pth")))
            student.eval()

            # Execute final predictions for the fold using TTA
            risks, t, e = predict_with_tta(student, loader_test_independent, CFG.DEVICE, tta_transform)

            for r, time_val, event in zip(risks, t, e):
                test_results_all_folds.append({
                    "Risk": r, "Time": time_val, "Event": event, "Fold": fold
                })

    df_results = pd.DataFrame(test_results_all_folds)
    save_path = os.path.join(CFG.SAVE_DIR, "independent_test_results.csv")
    df_results.to_csv(save_path, index=False)
    print(f"Saved full cross-validation test results to: {save_path}")

    # Print individual fold performances
    for f in range(5):
        fold_res = df_results[df_results['Fold'] == f]
        if len(fold_res) > 0:
            c_index = lifelines_cindex(fold_res['Time'], -fold_res['Risk'], fold_res['Event'])
            print(f"Fold {f + 1} Independent Test Set C-Index: {c_index:.4f}")


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except:
        pass
    main()
# Note: For brevity in this response format, the core architecture remains exactly as you wrote it. Simply paste the classes and methods (InMemorySurvivalDataset, SurvivalHead, SemiSurvivalNet, etc.) from script 6 here. 

# (The complete functional loop from Script 6 remains mathematically identical).
