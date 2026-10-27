"""
VisionGuard Lite - Model A: BaselineMLP
Fast non-temporal flattening baseline with linear layers, batchnorm, dropout.
"""

import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    def __init__(self, in_features: int = 1530, hidden_dim: int = 64, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input has (B, C, T) or (B, T, J, C), flatten
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.net(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
