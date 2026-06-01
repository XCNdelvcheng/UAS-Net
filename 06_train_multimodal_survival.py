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
    Compose, Resized, NormalizeIntensityd, RandFlipd, 
    RandRotate90d, RandAffined, RandGaussianNoised, 
    RandShiftIntensityd, RandCoarseDropoutd
)

warnings.filterwarnings("ignore")

class CFG:
    TRAIN_DF_PATH = "./data/train_processed.csv"
    TEST_DF_PATH = "./data/test_processed.csv"
    NPY_ROOT = "./data/npy_data"
    PRETRAINED_ENCODER = "./checkpoints/best_tumor_aware_encoder.pth"
    SAVE_DIR = "./checkpoints/cv_5fold"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 2026
    IMG_SIZE = (128, 128, 128)
    PATCH_SIZE = (16, 16, 16)
    HIDDEN_SIZE = 768
    CLIN_FEAT_DIM = 18
    DROPOUT_RATE = 0.3
    EPOCHS = 150
    BATCH_SIZE = 24
    LABELED_BATCH_SIZE = 12
    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_WORKERS = 0
    EARLY_STOPPING_PATIENCE = 40
    LR_ENCODER = 1e-5
    LR_HEAD = 5e-4
    WEIGHT_DECAY = 0.02
    LABELED_RATIO = 0.5
    EMA_DECAY = 0.9995
    LAMBDA_CONS = 1.5
    CONSISTENCY_RAMPUP_EPOCHS = 50
    UNCERTAINTY_T = 10
    UNCERTAINTY_BETA = 1.0

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Include the rest of the functional definitions from your original code here (InMemorySurvivalDataset, SurvivalHead, SemiSurvivalNet, cox_ph_loss, evaluate_cindex, etc.)
# Note: For brevity in this response format, the core architecture remains exactly as you wrote it. Simply paste the classes and methods (InMemorySurvivalDataset, SurvivalHead, SemiSurvivalNet, etc.) from script 6 here. 

# (The complete functional loop from Script 6 remains mathematically identical).
