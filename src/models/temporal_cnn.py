"""
VisionGuard Lite - Model B: TemporalCNN
1D Temporal Convolutional Neural Network for skeleton activity classification.
Takes (Batch, In_Channels=J*C, Seq_Len=T), extracts temporal feature maps,
uses global average pooling, and outputs class logits.
Ultra-lightweight (~25k params), ideal for ONNX edge deployment.
"""

import torch
import torch.nn as nn

class TemporalCNN(nn.Module):
    def __init__(self, in_channels: int = 51, num_classes: int = 2, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 96, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(96)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(96, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input shape: (B, Channels, T)
        if x.dim() == 4:
            # (B, T, J, C) -> (B, J*C, T)
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C).transpose(1, 2)
            
        h = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        h = self.pool2(self.relu2(self.bn2(self.conv2(h))))
        h = self.relu3(self.bn3(self.conv3(h)))
        
        pooled = self.global_pool(h).squeeze(-1) # (B, 64)
        out = self.classifier(self.dropout(pooled))
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
