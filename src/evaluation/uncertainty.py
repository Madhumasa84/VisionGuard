"""
VisionGuard Lite - Uncertainty Estimation, Calibration & Stress Testing
Implements:
- Temperature Scaling for post-hoc probability calibration (ECE optimization)
- Expected Calibration Error (ECE) calculation & reliability diagrams
- Selective prediction policy (abstention under ambiguity or low visibility)
- Edge distribution shift stress-testing (occlusions, coordinate noise, distance shifts)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple, List

class TemperatureScaler(nn.Module):
    """
    Post-hoc calibration: outputs logits / T where T is optimized on validation data.
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray, max_iter: int = 50) -> float:
        logits = torch.tensor(val_logits, dtype=torch.float32)
        labels = torch.tensor(val_labels, dtype=torch.long)
        
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)
        with torch.no_grad():
            self.temperature.clamp_(min=0.05, max=10.0)
            
        return float(self.temperature.item())

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """
    Computes Expected Calibration Error (ECE) and bin statistics.
    probs: (N, C) softmax probabilities or (N,) maximum confidence
    labels: (N,) true labels
    """
    if probs.ndim == 2:
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
    else:
        confidences = probs
        predictions = (probs >= 0.5).astype(int)

    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    bin_stats = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_stats.append({
                'bin': i,
                'lower': bin_lower,
                'upper': bin_upper,
                'count': int(np.sum(in_bin)),
                'accuracy': float(accuracy_in_bin),
                'confidence': float(avg_confidence_in_bin)
            })

    return {
        'ece': float(ece),
        'bin_stats': bin_stats
    }

class UncertaintyPolicy:
    """
    Safety-oriented selective classification policy:
    If confidence < threshold (or entropy > max_entropy), mark UNCERTAIN.
    """
    def __init__(self, confidence_threshold: float = 0.75, max_entropy: float = 0.55):
        self.confidence_threshold = confidence_threshold
        self.max_entropy = max_entropy

    def predict_with_rejection(self, probs: np.ndarray) -> List[Dict[str, Any]]:
        results = []
        for p in probs:
            if p.ndim == 0:
                p_fall = float(p)
                p_arr = np.array([1.0 - p_fall, p_fall])
            else:
                p_arr = p
                p_fall = float(p_arr[1])
                
            conf = float(np.max(p_arr))
            pred_class = int(np.argmax(p_arr))
            
            eps = 1e-8
            entropy = -float(np.sum(p_arr * np.log(p_arr + eps)))
            
            is_uncertain = (conf < self.confidence_threshold) or (entropy > self.max_entropy)
            decision = 'UNCERTAIN' if is_uncertain else ('FALL' if pred_class == 1 else 'NORMAL')
            
            results.append({
                'decision': decision,
                'predicted_class': pred_class,
                'confidence': conf,
                'entropy': entropy,
                'fall_probability': p_fall,
                'is_uncertain': is_uncertain
            })
        return results

def evaluate_distribution_shifts(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates model resilience under synthetic domain shifts:
    1. Clean Test
    2. 40% Joint Occlusion (zeroing out random joints)
    3. Gaussian Coordinate Noise (sigma = 0.15)
    4. Post-norm Scale Compression (0.6x feature-level height collapse)
    """
    from src.evaluation.metrics import compute_classification_metrics
    
    model.eval()
    shifts = {
        'clean': X_test.copy(),
        'occlusion_40pct': X_test.copy(),
        'gaussian_noise_0.15': X_test.copy(),
        'post_norm_compression_0.6x': X_test.copy()
    }
    
    # 1. 40% Occlusion
    N, T, J, C = shifts['occlusion_40pct'].shape
    mask = np.random.rand(N, 1, J, 1) < 0.4
    shifts['occlusion_40pct'][mask.repeat(T, axis=1).repeat(C, axis=3)] = 0.0

    # 2. Gaussian Noise
    shifts['gaussian_noise_0.15'] += np.random.normal(0, 0.15, size=shifts['gaussian_noise_0.15'].shape).astype(np.float32)

    # 3. Post-norm Scale Compression (0.6x)
    shifts['post_norm_compression_0.6x'] *= 0.6

    policy = UncertaintyPolicy(confidence_threshold=0.75, max_entropy=0.55)

    results = {}
    for shift_name, X_data in shifts.items():
        N, T, J, C = X_data.shape
        x_tensor = torch.tensor(X_data, dtype=torch.float32).reshape(N, T, J * C).transpose(1, 2).to(device)
        
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs_2d = torch.softmax(logits, dim=1).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
        m = compute_classification_metrics(y_test, preds, probs)
        
        # Uncertainty policy rejection rate
        policy_decisions = policy.predict_with_rejection(probs_2d)
        abstentions = sum(1 for d in policy_decisions if d['is_uncertain'])
        abstention_rate = abstentions / float(len(policy_decisions))

        results[shift_name] = {
            'f1_score': m['f1_score'],
            'recall': m['recall'],
            'precision': m['precision'],
            'auroc': m['auroc'],
            'brier_score': m['brier_score'],
            'abstention_rate': round(abstention_rate, 4),
            'accepted_samples': len(policy_decisions) - abstentions
        }
    return results
