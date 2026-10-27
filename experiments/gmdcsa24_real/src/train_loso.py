"""
VisionGuard Lite - Real GMDCSA-24 Leave-One-Subject-Out (LOSO) Training & Evaluation
Evaluates:
- BaselineMLP, TemporalCNN, SmallTCN across 4 Subject LOSO Folds
- Event-aligned windows with zero subject leakage
- Validation-only threshold selection (never tuned on test subject)
- Window-level & Event-level Fall Recall, Detection Delay, False Alarms/Hour
- Post-hoc Temperature Scaling (ECE) on held-out test subjects
"""

import os
import sys
import json
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.baseline_mlp import BaselineMLP
from src.models.temporal_cnn import TemporalCNN
from src.models.small_tcn import SmallTCN
from src.evaluation.metrics import compute_classification_metrics
from src.evaluation.uncertainty import TemperatureScaler, compute_ece

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def find_best_threshold(val_probs: np.ndarray, val_labels: np.ndarray) -> Tuple[float, float]:
    """
    Finds threshold on validation set that maximizes F1 score.
    Never uses test set.
    """
    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (val_probs >= thresh).astype(int)
        tp = np.sum((preds == 1) & (val_labels == 1))
        fp = np.sum((preds == 1) & (val_labels == 0))
        fn = np.sum((preds == 0) & (val_labels == 1))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            
    return best_thresh, best_f1

def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    epochs: int = 35,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Trains a model with AdamW, class-weighted cross entropy, and early stopping on validation loss/F1.
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_losses = []
        val_preds_list = []
        val_labels_list = []
        val_probs_list = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_losses.append(loss.item())

                probs = torch.softmax(logits, dim=1)[:, 1]
                val_probs_list.extend(probs.cpu().numpy())
                val_labels_list.extend(y_batch.cpu().numpy())

        val_loss = float(np.mean(val_losses))
        val_probs = np.array(val_probs_list)
        val_labels = np.array(val_labels_list)
        
        # Calculate val F1 at default 0.5 threshold
        val_preds = (val_probs >= 0.5).astype(int)
        tp = np.sum((val_preds == 1) & (val_labels == 1))
        fp = np.sum((val_preds == 1) & (val_labels == 0))
        fn = np.sum((val_preds == 0) & (val_labels == 1))
        val_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        scheduler.step(val_f1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(float(val_f1))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= 15:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'history': history
    }

def get_model_instance(model_name: str) -> nn.Module:
    if model_name == 'BaselineMLP':
        return BaselineMLP(in_features=1530, hidden_dim=64, num_classes=2, dropout=0.3)
    elif model_name == 'TemporalCNN':
        return TemporalCNN(in_channels=51, num_classes=2, dropout=0.25)
    elif model_name == 'SmallTCN':
        return SmallTCN(in_channels=51, num_classes=2, num_channels=[48, 64, 64], kernel_size=3, dropout=0.2)
    else:
        raise ValueError(f"Unknown model {model_name}")

def evaluate_test_set(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    device: torch.device
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Evaluates model on test data at given threshold.
    Returns metrics, raw probabilities, and raw logits.
    """
    model.eval()
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=64, shuffle=False)

    all_logits = []
    all_probs = []

    with torch.no_grad():
        for (xb,) in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    probs_arr = np.concatenate(all_probs, axis=0)
    preds_arr = (probs_arr >= threshold).astype(int)

    metrics = compute_classification_metrics(y_test, preds_arr, probs_arr)
    return metrics, probs_arr, logits_arr

def compute_event_level_metrics(
    extracted_poses_dir: str,
    test_subject: str,
    test_vids: np.ndarray,
    test_times: np.ndarray,
    test_probs: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Computes genuine event-level detection recall, detection delay, and false alarm rate.
    Uses video annotations to group windows by recording and actual physical events.
    """
    # 1. Load manifest to find all recordings for test subject
    manifest_path = os.path.join(extracted_poses_dir, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    subj_records = [r for r in manifest['records'] if r['subject'] == test_subject]

    total_fall_events = 0
    detected_fall_events = 0
    detection_delays = []
    total_normal_duration_sec = 0.0
    false_alarm_windows = 0
    false_alarm_clusters = 0

    # Map video_id to windows [(idx, p, start_sec, end_sec), ...]
    vid_to_windows = {}
    for i, vid in enumerate(test_vids):
        if vid not in vid_to_windows:
            vid_to_windows[vid] = []
        vid_to_windows[vid].append((i, test_probs[i], float(test_times[i, 0]), float(test_times[i, 1])))

    for rec in subj_records:
        npz_name = os.path.basename(rec['file_path'])
        npz_path = os.path.join(extracted_poses_dir, npz_name)
        if not os.path.exists(npz_path):
            continue
        data = np.load(npz_path, allow_pickle=True)
        vid_id = str(data['video_id'])
        category = str(data['category'])
        events = json.loads(str(data['events']))
        duration_sec = float(data['duration_sec'])

        vid_wins = vid_to_windows.get(vid_id, [])

        if category == 'ADL':
            total_normal_duration_sec += duration_sec
            fa_in_vid = sum(1 for _, p, _, _ in vid_wins if p >= threshold)
            false_alarm_windows += fa_in_vid
            if fa_in_vid > 0:
                false_alarm_clusters += 1
        else:
            # Fall video: evaluate each physical fall event
            for ev in events:
                if ev.get('is_fall', False):
                    total_fall_events += 1
                    f_start = float(ev['start_sec'])
                    f_end = float(ev['end_sec'])
                    f_dur = max(0.2, f_end - f_start)

                    detected = False
                    earliest_detect_sec = None

                    for global_idx, p, w_start, w_end in vid_wins:
                        overlap = max(0.0, min(w_end, f_end) - max(w_start, f_start))
                        # If window covers >= 40% of fall or >= 0.6s
                        if overlap / f_dur >= 0.40 or overlap >= 0.60:
                            if p >= threshold:
                                detected = True
                                if earliest_detect_sec is None or w_end < earliest_detect_sec:
                                    earliest_detect_sec = w_end

                    if detected:
                        detected_fall_events += 1
                        delay = max(0.0, earliest_detect_sec - f_start)
                        detection_delays.append(delay)

    event_recall = (detected_fall_events / total_fall_events) if total_fall_events > 0 else 0.0
    mean_delay = float(np.mean(detection_delays)) if detection_delays else 0.0
    median_delay = float(np.median(detection_delays)) if detection_delays else 0.0
    normal_hours = total_normal_duration_sec / 3600.0
    fa_per_hour = (false_alarm_clusters / normal_hours) if normal_hours > 0 else 0.0

    return {
        'total_fall_events': total_fall_events,
        'detected_fall_events': detected_fall_events,
        'event_level_recall': round(float(event_recall), 4),
        'mean_detection_delay_sec': round(float(mean_delay), 3),
        'median_detection_delay_sec': round(float(median_delay), 3),
        'normal_video_hours': round(float(normal_hours), 4),
        'false_alarm_clusters': false_alarm_clusters,
        'false_alarms_per_hour': round(float(fa_per_hour), 2)
    }

def run_loso_experiments(
    data_dir: str = None,
    extracted_poses_dir: str = None,
    checkpoints_dir: str = None,
    results_dir: str = None,
    batch_size: int = 32,
    epochs: int = 35
) -> Dict[str, Any]:
    """
    Executes full LOSO training across 4 folds for BaselineMLP, TemporalCNN, and SmallTCN.
    """
    if data_dir is None:
        data_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/data/processed_windows')
    if extracted_poses_dir is None:
        extracted_poses_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/data/extracted_poses')
    if checkpoints_dir is None:
        checkpoints_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/checkpoints')
    if results_dir is None:
        results_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/results')
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_names = ['BaselineMLP', 'TemporalCNN', 'SmallTCN']
    folds = ['fold_1', 'fold_2', 'fold_3', 'fold_4']

    all_results = {m: {'folds': {}, 'aggregate': {}} for m in model_names}
    best_overall_f1 = {m: -1.0 for m in model_names}
    best_models = {}

    for fold_name in folds:
        fold_dir = os.path.join(data_dir, fold_name)
        if not os.path.exists(fold_dir):
            print(f"Directory {fold_dir} not found. Skipping.")
            continue

        print(f"\n==========================================")
        print(f"   Executing {fold_name.upper()} (LOSO)")
        print(f"==========================================")

        # Load data
        X_train = np.load(os.path.join(fold_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(fold_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(fold_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(fold_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(fold_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(fold_dir, 'y_test.npy'))
        vids_test = np.load(os.path.join(fold_dir, 'vids_test.npy'))
        subjs_test = np.load(os.path.join(fold_dir, 'subjs_test.npy'))
        times_test = np.load(os.path.join(fold_dir, 'times_test.npy'))
        test_subject = str(subjs_test[0])

        print(f"Test Subject: {test_subject}")
        print(f"Train: {X_train.shape[0]} windows (Fall: {np.sum(y_train==1)}, Normal: {np.sum(y_train==0)})")
        print(f"Val:   {X_val.shape[0]} windows (Fall: {np.sum(y_val==1)}, Normal: {np.sum(y_val==0)})")
        print(f"Test:  {X_test.shape[0]} windows (Fall: {np.sum(y_test==1)}, Normal: {np.sum(y_test==0)})")

        # Class weights for loss
        n_pos = max(1, np.sum(y_train == 1))
        n_neg = max(1, np.sum(y_train == 0))
        w0 = 1.0 / n_neg
        w1 = 1.0 / n_pos
        w_sum = w0 + w1
        class_weights = torch.tensor([w0 / w_sum * 2.0, w1 / w_sum * 2.0], dtype=torch.float32)

        # DataLoaders
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
        val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        fold_ckpt_dir = os.path.join(checkpoints_dir, fold_name)
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        for m_name in model_names:
            print(f"\n--- Training {m_name} on {fold_name} ---")
            model = get_model_instance(m_name).to(device)

            model, train_info = train_single_model(
                model_name=m_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                class_weights=class_weights,
                epochs=epochs,
                lr=1e-3,
                patience=10
            )

            # Evaluate on validation subject to select threshold and fit temperature
            val_metrics, val_probs, val_logits = evaluate_test_set(model, X_val, y_val, threshold=0.5, device=device)
            optimal_thresh, val_best_f1 = find_best_threshold(val_probs, y_val)
            print(f"[{m_name}] Val F1 (0.50): {val_metrics['f1_score']:.4f} | Optimal Val Thresh: {optimal_thresh:.2f} (Val F1: {val_best_f1:.4f})")

            # Post-hoc Temperature Calibration on validation logits
            scaler = TemperatureScaler()
            calib_temp = scaler.fit(val_logits, y_val)
            print(f"[{m_name}] Fitted Temperature T: {calib_temp:.3f}")

            # Evaluate on held-out test subject at default threshold (0.5) and optimal val threshold
            test_metrics_05, test_probs, test_logits = evaluate_test_set(model, X_test, y_test, threshold=0.5, device=device)
            test_metrics_opt, _, _ = evaluate_test_set(model, X_test, y_test, threshold=optimal_thresh, device=device)

            # ECE before and after calibration
            pre_ece = compute_ece(test_probs, y_test)['ece']
            calib_logits = test_logits / calib_temp
            calib_probs = torch.softmax(torch.tensor(calib_logits), dim=1)[:, 1].numpy()
            post_ece = compute_ece(calib_probs, y_test)['ece']

            # Event-level detection metrics
            event_metrics = compute_event_level_metrics(
                extracted_poses_dir=extracted_poses_dir,
                test_subject=test_subject,
                test_vids=vids_test,
                test_times=times_test,
                test_probs=test_probs,
                threshold=0.5
            )

            print(f"[{m_name} Test] F1 (0.50): {test_metrics_05['f1_score']:.4f} | Recall: {test_metrics_05['recall']:.4f} | Prec: {test_metrics_05['precision']:.4f} | AUROC: {test_metrics_05['auroc']:.4f}")
            print(f"[{m_name} Test] Event Recall: {event_metrics['event_level_recall']*100:.1f}% ({event_metrics['detected_fall_events']}/{event_metrics['total_fall_events']}) | FA/hr: {event_metrics['false_alarms_per_hour']} | Delay: {event_metrics['mean_detection_delay_sec']}s")
            print(f"[{m_name} Test] ECE: {pre_ece:.4f} -> {post_ece:.4f} (Calibrated)")

            # Save fold checkpoint
            ckpt_path = os.path.join(fold_ckpt_dir, f"{m_name.lower()}.pt")
            torch.save({
                'model_state': model.state_dict(),
                'model_name': m_name,
                'fold': fold_name,
                'test_subject': test_subject,
                'temperature': calib_temp,
                'optimal_threshold': optimal_thresh,
                'test_f1': test_metrics_05['f1_score']
            }, ckpt_path)

            # Track best model across folds
            if test_metrics_05['f1_score'] > best_overall_f1[m_name]:
                best_overall_f1[m_name] = test_metrics_05['f1_score']
                best_ckpt_path = os.path.join(checkpoints_dir, f"real_{m_name.lower()}_best.pt")
                torch.save({
                    'model_state': model.state_dict(),
                    'model_name': m_name,
                    'best_fold': fold_name,
                    'test_subject': test_subject,
                    'temperature': calib_temp,
                    'optimal_threshold': optimal_thresh,
                    'metrics': test_metrics_05,
                    'event_metrics': event_metrics
                }, best_ckpt_path)
                best_models[m_name] = copy.deepcopy(model)

            all_results[m_name]['folds'][fold_name] = {
                'test_subject': test_subject,
                'optimal_threshold_val': round(optimal_thresh, 3),
                'temperature': round(calib_temp, 3),
                'metrics_thresh_05': test_metrics_05,
                'metrics_thresh_opt': test_metrics_opt,
                'ece_pre': round(pre_ece, 4),
                'ece_post': round(post_ece, 4),
                'event_metrics': event_metrics
            }

    # Compute aggregate cross-subject statistics (mean ± std)
    for m_name in model_names:
        fold_res = all_results[m_name]['folds']
        f1_vals = [res['metrics_thresh_05']['f1_score'] for res in fold_res.values()]
        rec_vals = [res['metrics_thresh_05']['recall'] for res in fold_res.values()]
        prec_vals = [res['metrics_thresh_05']['precision'] for res in fold_res.values()]
        auroc_vals = [res['metrics_thresh_05']['auroc'] for res in fold_res.values()]
        auprc_vals = [res['metrics_thresh_05']['auprc'] for res in fold_res.values()]
        brier_vals = [res['metrics_thresh_05']['brier_score'] for res in fold_res.values()]
        ece_pre_vals = [res['ece_pre'] for res in fold_res.values()]
        ece_post_vals = [res['ece_post'] for res in fold_res.values()]
        ev_rec_vals = [res['event_metrics']['event_level_recall'] for res in fold_res.values()]
        fa_hr_vals = [res['event_metrics']['false_alarms_per_hour'] for res in fold_res.values()]
        delay_vals = [res['event_metrics']['mean_detection_delay_sec'] for res in fold_res.values()]

        all_results[m_name]['aggregate'] = {
            'f1_mean': round(float(np.mean(f1_vals)), 4),
            'f1_std': round(float(np.std(f1_vals)), 4),
            'recall_mean': round(float(np.mean(rec_vals)), 4),
            'recall_std': round(float(np.std(rec_vals)), 4),
            'precision_mean': round(float(np.mean(prec_vals)), 4),
            'precision_std': round(float(np.std(prec_vals)), 4),
            'auroc_mean': round(float(np.mean(auroc_vals)), 4),
            'auroc_std': round(float(np.std(auroc_vals)), 4),
            'auprc_mean': round(float(np.mean(auprc_vals)), 4),
            'auprc_std': round(float(np.std(auprc_vals)), 4),
            'brier_mean': round(float(np.mean(brier_vals)), 4),
            'ece_pre_mean': round(float(np.mean(ece_pre_vals)), 4),
            'ece_post_mean': round(float(np.mean(ece_post_vals)), 4),
            'event_recall_mean': round(float(np.mean(ev_rec_vals)), 4),
            'event_recall_std': round(float(np.std(ev_rec_vals)), 4),
            'fa_per_hour_mean': round(float(np.mean(fa_hr_vals)), 2),
            'detection_delay_sec_mean': round(float(np.mean(delay_vals)), 3)
        }

    out_file = os.path.join(results_dir, "loso_evaluation.json")
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n==========================================")
    print(f"   LOSO Cross-Validation Summary")
    print(f"==========================================")
    for m_name in model_names:
        agg = all_results[m_name]['aggregate']
        print(f"{m_name:12s} | F1: {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f} | Recall: {agg['recall_mean']:.4f} | Prec: {agg['precision_mean']:.4f} | AUROC: {agg['auroc_mean']:.4f} | Ev-Rec: {agg['event_recall_mean']*100:.1f}% | FA/hr: {agg['fa_per_hour_mean']}")

    return all_results

if __name__ == '__main__':
    run_loso_experiments()
