"""
VisionGuard Lite - PyTorch Dataset & DataLoaders
Implements:
- SkeletonWindowDataset
- Data augmentations for skeleton coordinates (Gaussian coordinate jitter, random scale, time-warp)
- DataLoader factory with class weighting computation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional

class SkeletonWindowDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        flatten_for_mlp: bool = False,
        noise_std: float = 0.02
    ):
        """
        X: shape (N, T, J, C)
        y: shape (N,)
        flatten_for_mlp: if True, flattens (T, J, C) to (T * J * C)
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment
        self.flatten_for_mlp = flatten_for_mlp
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].clone()
        y = self.y[idx]

        if self.augment:
            # 1. Subtle Gaussian coordinate jitter on (x, y)
            jitter = torch.randn_like(x[..., :2]) * self.noise_std
            x[..., :2] = x[..., :2] + jitter
            
            # 2. Minor random scale fluctuation [0.95, 1.05]
            scale = torch.empty(1).uniform_(0.95, 1.05).item()
            x[..., :2] = x[..., :2] * scale

        if self.flatten_for_mlp:
            x = x.reshape(-1)
        else:
            # Reshape for 1D Temporal CNN / TCN:
            # (T, J, C) -> (J * C, T) (channels first for 1D conv)
            T, J, C = x.shape
            x = x.reshape(T, J * C).transpose(0, 1)

        return x, y

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    flatten_for_mlp: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Loads cached npy arrays and returns (train_loader, val_loader, test_loader, class_weights)
    """
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))

    train_ds = SkeletonWindowDataset(X_train, y_train, augment=True, flatten_for_mlp=flatten_for_mlp)
    val_ds = SkeletonWindowDataset(X_val, y_val, augment=False, flatten_for_mlp=flatten_for_mlp)
    test_ds = SkeletonWindowDataset(X_test, y_test, augment=False, flatten_for_mlp=flatten_for_mlp)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Class weighting for balanced CrossEntropyLoss: w_c = N_total / (N_classes * N_c)
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    weights = total_samples / (len(classes) * counts.astype(np.float32))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    return train_loader, val_loader, test_loader, class_weights
