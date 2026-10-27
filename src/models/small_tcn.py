"""
VisionGuard Lite - Model C: SmallTCN
Temporal Convolutional Network with dilated causal/symmetric convolutions,
residual connections, and receptive field covering the entire window.
"""

import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        out = self.drop2(self.relu2(self.bn2(self.conv2(out))))
        return self.relu(out + res)

class SmallTCN(nn.Module):
    def __init__(self, in_channels: int = 51, num_classes: int = 2, num_channels: list = [48, 64, 64], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_c = in_channels if i == 0 else num_channels[i - 1]
            out_c = num_channels[i]
            padding = (kernel_size - 1) * dilation_size // 2
            layers.append(
                TemporalBlock(in_c, out_c, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout)
            )

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C).transpose(1, 2)
            
        feat = self.network(x)
        pooled = self.global_pool(feat).squeeze(-1)
        return self.fc(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
