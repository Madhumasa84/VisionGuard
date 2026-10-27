"""
VisionGuard Lite - Quantitative Evaluation of Temporal Confirmation Filtering
Compares raw window detection (K=1) against temporal confirmation filters (K=2, 3, 4)
across all 4 GMDCSA-24 LOSO held-out subject folds.
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.temporal_cnn import TemporalCNN

def evaluate_confirmation_policy(
    data_dir: Path = REPO_ROOT / "experiments/gmdcsa24_real/data/processed_windows",
    extracted_poses_dir: Path = REPO_ROOT / "experiments/gmdcsa24_real/data/extracted_poses",
    checkpoints_dir: Path = REPO_ROOT / "experiments/gmdcsa24_real/checkpoints",
    results_dir: Path = REPO_ROOT / "experiments/gmdcsa24_real/results",
    threshold: float = 0.50
):
    manifest_path = extracted_poses_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    folds = ["fold_1", "fold_2", "fold_3", "fold_4"]
    k_values = [1, 2, 3, 4]
    
    # Results container
    policy_results = {k: {"folds": {}, "aggregate": {}} for k in k_values}

    for fold_name in folds:
        fold_dir = data_dir / fold_name
        ckpt_file = checkpoints_dir / fold_name / "temporalcnn.pt"
        if not fold_dir.exists() or not ckpt_file.exists():
            continue

        # Load test set for this fold
        X_test = np.load(fold_dir / "X_test.npy")
        y_test = np.load(fold_dir / "y_test.npy")
        vids_test = np.load(fold_dir / "vids_test.npy")
        times_test = np.load(fold_dir / "times_test.npy")
        subjs_test = np.load(fold_dir / "subjs_test.npy")
        test_subj = str(subjs_test[0])

        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        temp = float(ckpt.get("temperature", 1.0))
        model = TemporalCNN(in_channels=51, num_classes=2)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        with torch.no_grad():
            x_tensor = torch.from_numpy(X_test)
            logits = model(x_tensor)
            probs = torch.softmax(logits / temp, dim=1)[:, 1].numpy()

        # Map video_id to list of windows: (window_idx, prob, start_sec, end_sec)
        vid_to_windows = {}
        for i, vid in enumerate(vids_test):
            if vid not in vid_to_windows:
                vid_to_windows[vid] = []
            vid_to_windows[vid].append((i, float(probs[i]), float(times_test[i, 0]), float(times_test[i, 1])))

        subj_records = [r for r in manifest["records"] if r["subject"] == test_subj]

        for k in k_values:
            total_fall_events = 0
            detected_fall_events = 0
            delays = []
            total_adl_hours = 0.0
            false_alarm_events = 0

            for rec in subj_records:
                npz_name = os.path.basename(rec["file_path"])
                npz_p = extracted_poses_dir / npz_name
                if not npz_p.exists():
                    continue

                data = np.load(npz_p, allow_pickle=True)
                vid_id = str(data["video_id"])
                category = str(data["category"])
                events = json.loads(str(data["events"]))
                duration_sec = float(data["duration_sec"])

                wins = vid_to_windows.get(vid_id, [])
                # Apply K-window temporal confirmation
                consec = 0
                confirmed_flags = []
                for _, p, w_s, w_e in wins:
                    if p >= threshold:
                        consec += 1
                    else:
                        consec = 0
                    confirmed_flags.append(consec >= k)

                if category == "ADL":
                    total_adl_hours += duration_sec / 3600.0
                    # Check if any confirmed fall event fired in normal video
                    if any(confirmed_flags):
                        false_alarm_events += 1
                else:
                    # Fall video: evaluate detection per annotated fall event
                    for ev in events:
                        if ev.get("is_fall", False):
                            total_fall_events += 1
                            f_start = float(ev["start_sec"])
                            f_end = float(ev["end_sec"])
                            f_dur = max(0.2, f_end - f_start)

                            detected = False
                            detect_time = None

                            for w_idx, (_, p, w_s, w_e) in enumerate(wins):
                                if confirmed_flags[w_idx]:
                                    overlap = max(0.0, min(w_e, f_end) - max(w_s, f_start))
                                    if (overlap / f_dur >= 0.40) or (overlap >= 0.60):
                                        detected = True
                                        if detect_time is None or w_e < detect_time:
                                            detect_time = w_e

                            if detected:
                                detected_fall_events += 1
                                delays.append(max(0.0, detect_time - f_start))

            event_rec = (detected_fall_events / total_fall_events) if total_fall_events > 0 else 0.0
            mean_delay = float(np.mean(delays)) if delays else 0.0
            fa_per_hr = (false_alarm_events / total_adl_hours) if total_adl_hours > 0 else 0.0

            policy_results[k]["folds"][fold_name] = {
                "test_subject": test_subj,
                "event_recall": round(float(event_rec), 4),
                "detected_events": detected_fall_events,
                "total_events": total_fall_events,
                "mean_delay_sec": round(float(mean_delay), 3),
                "false_alarms_per_hour": round(float(fa_per_hr), 2)
            }

    # Aggregate across folds
    for k in k_values:
        f_res = policy_results[k]["folds"]
        recs = [r["event_recall"] for r in f_res.values()]
        delays = [r["mean_delay_sec"] for r in f_res.values()]
        fas = [r["false_alarms_per_hour"] for r in f_res.values()]
        total_det = sum(r["detected_events"] for r in f_res.values())
        total_ev = sum(r["total_events"] for r in f_res.values())

        policy_results[k]["aggregate"] = {
            "mean_event_recall": round(float(np.mean(recs)), 4),
            "total_detected_events": total_det,
            "total_events": total_ev,
            "overall_event_recall": round(float(total_det / max(1, total_ev)), 4),
            "mean_detection_delay_sec": round(float(np.mean(delays)), 3),
            "mean_false_alarms_per_hour": round(float(np.mean(fas)), 2)
        }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "confirmation_evaluation.json"
    with open(out_file, "w") as f:
        json.dump(policy_results, f, indent=2)

    print("\n" + "=" * 65)
    print("   Temporal Confirmation Policy Evaluation Summary (GMDCSA-24)")
    print("=" * 65)
    for k in k_values:
        agg = policy_results[k]["aggregate"]
        print(f"K={k} Policy | Event Recall: {agg['overall_event_recall']*100:.1f}% ({agg['total_detected_events']}/{agg['total_events']}) | Mean Delay: {agg['mean_detection_delay_sec']:.2f}s | FA/hr: {agg['mean_false_alarms_per_hour']:.2f}")

    return policy_results

if __name__ == "__main__":
    evaluate_confirmation_policy()
