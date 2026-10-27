"""
VisionGuard Lite - Model Trainer
Implements:
- Training loop with AdamW and learning rate scheduling
- Validation early stopping
- Class-balanced loss computation
- Automatic checkpointing to Drive
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional

from src.evaluation.metrics import compute_classification_metrics

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: Optional[torch.Tensor] = None,
        max_epochs: int = 40,
        patience: int = 8,
        checkpoint_path: str = 'best_model.pt'
    ):
        self.model = model.to(device)
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_path = checkpoint_path

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs, eta_min=1e-6)

        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': []
        }

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(y)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_probs = []
        all_targets = []
        all_logits = []

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * len(y)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_logits.append(logits.cpu().numpy())
                all_probs.append(probs[:, 1].cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        all_logits = np.concatenate(all_logits, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        metrics = compute_classification_metrics(all_targets, all_preds, all_probs)
        metrics['loss'] = total_loss / len(dataloader.dataset)
        metrics['logits'] = all_logits
        metrics['probs'] = all_probs
        metrics['preds'] = all_preds
        metrics['targets'] = all_targets
        return metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        best_val_loss = float('inf')
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            t_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            v_loss = val_metrics['loss']
            v_f1 = val_metrics['f1_score']

            self.scheduler.step()
            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['val_f1'].append(v_f1)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
                os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)), exist_ok=True)
                torch.save(best_model_state, self.checkpoint_path)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.patience:
                break

        # Load best weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return self.history
