"""
VisionGuard Lite - GMDCSA-24 Real Data Processor
Implements:
- Event-aligned training label policy (removing flawed 50% window duration threshold)
- Sliding 30-frame window extraction (3.0s @ 10 FPS)
- Verified normal intervals for negative examples
- Exclusion of ambiguous transition segments
- Leave-One-Subject-Out (LOSO) cross-validation folds with zero participant leakage
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.schema import WINDOW_SIZE, WINDOW_STRIDE, EVAL_STRIDE
from src.data.preprocessing import PosePreprocessor

class GMDCSARealProcessor:
    def __init__(self, target_window_size: int = WINDOW_SIZE):
        self.target_window_size = target_window_size
        self.preprocessor = PosePreprocessor(target_window_size=target_window_size)

    def label_window_event_aligned(
        self,
        w_start_sec: float,
        w_end_sec: float,
        events: List[Dict[str, Any]],
        is_fall_video: bool
    ) -> Tuple[int, str]:
        """
        Event-Aligned Training Label Policy:
        - If ADL video: Label = 0 (NORMAL)
        - If Fall video:
          * If window covers >= 40% of the physical fall event OR overlap >= 0.6 seconds: Label = 1 (FALL)
          * If window ends >= 1.0 second before fall onset: Label = 0 (NORMAL)
          * Otherwise (boundary transition or post-fall recovery): Label = -1 (AMBIGUOUS / EXCLUDE)
        """
        if not is_fall_video:
            return 0, "normal_adl"

        fall_events = [e for e in events if e.get('is_fall', False)]
        if not fall_events:
            return 0, "normal_no_fall"

        max_overlap_sec = 0.0
        max_fall_coverage_ratio = 0.0
        earliest_fall_start = min(e['start_sec'] for e in fall_events)

        for e in fall_events:
            f_start = e['start_sec']
            f_end = e['end_sec']
            f_dur = max(0.2, f_end - f_start)

            o_start = max(w_start_sec, f_start)
            o_end = min(w_end_sec, f_end)
            overlap = max(0.0, o_end - o_start)

            if overlap > max_overlap_sec:
                max_overlap_sec = overlap
                max_fall_coverage_ratio = overlap / float(f_dur)

        # 1. Clear Positive: contains significant portion of fall event (>= 40% of fall or >= 0.6s)
        if max_fall_coverage_ratio >= 0.40 or max_overlap_sec >= 0.60:
            return 1, "fall_event"

        # 2. Clear Negative: strictly pre-fall with 1.0s safety margin
        if w_end_sec <= (earliest_fall_start - 1.0) and max_overlap_sec == 0.0:
            return 0, "clean_pre_fall"

        # 3. Ambiguous Boundary Transition: exclude from clean training
        return -1, "ambiguous_transition"

    def process_sequence(
        self,
        npz_path: str,
        stride: int = WINDOW_STRIDE
    ) -> List[Dict[str, Any]]:
        """
        Loads an extracted video npz, normalizes landmarks, and segments into windows.
        """
        data = np.load(npz_path, allow_pickle=True)
        coco_kps = data['coco_landmarks']  # (T, 17, 3)
        timestamps = data['timestamps']    # (T,)
        subject = str(data['subject'])
        category = str(data['category'])
        video_id = str(data['video_id'])
        events = json.loads(str(data['events']))
        is_fall = (category == 'Fall')

        T = len(timestamps)
        if T < self.target_window_size:
            return []

        # 1. Normalize and impute
        norm_kps = self.preprocessor.normalize_skeleton(coco_kps)
        imputed_kps = self.preprocessor.impute_missing_landmarks(norm_kps, min_conf=0.3)

        windows_data = []
        for start_idx in range(0, T - self.target_window_size + 1, stride):
            end_idx = start_idx + self.target_window_size
            w_start_sec = float(timestamps[start_idx])
            w_end_sec = float(timestamps[end_idx - 1])

            label, label_desc = self.label_window_event_aligned(
                w_start_sec, w_end_sec, events, is_fall
            )

            # Extract window feature: (30, 17, 3)
            window_arr = imputed_kps[start_idx:end_idx]

            windows_data.append({
                'window': window_arr,
                'label': label,
                'label_desc': label_desc,
                'subject': subject,
                'video_id': video_id,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'start_sec': w_start_sec,
                'end_sec': w_end_sec,
                'is_fall_video': is_fall
            })

        return windows_data

def build_loso_folds(
    extracted_dir: str = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """
    Creates 4 Leave-One-Subject-Out (LOSO) cross-validation folds.
    """
    if extracted_dir is None:
        extracted_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/data/extracted_poses')
    if output_dir is None:
        output_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/data/processed_windows')
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(extracted_dir, 'manifest.json')
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    processor = GMDCSARealProcessor(target_window_size=30)
    all_extracted_windows = []
    
    total_excluded = 0
    total_fall_windows = 0
    total_normal_windows = 0

    for rec in manifest['records']:
        npz_name = os.path.basename(rec['file_path'])
        npz_p = os.path.join(extracted_dir, npz_name)
        if not os.path.exists(npz_p):
            continue
        wins = processor.process_sequence(npz_p, stride=5) # 5-frame stride (0.5s) for thorough coverage
        for w in wins:
            if w['label'] == -1:
                total_excluded += 1
                continue
            elif w['label'] == 1:
                total_fall_windows += 1
            else:
                total_normal_windows += 1
            all_extracted_windows.append(w)

    print(f"Total valid windows: {len(all_extracted_windows)} (Fall: {total_fall_windows}, Normal: {total_normal_windows})")
    print(f"Total ambiguous transition windows excluded: {total_excluded}")

    # Generate 4 LOSO folds
    subjects = ['Subject 1', 'Subject 2', 'Subject 3', 'Subject 4']
    fold_info = {}

    for fold_idx, test_subj in enumerate(subjects, start=1):
        train_val_subjs = [s for s in subjects if s != test_subj]
        # Designate one train subject as validation
        val_subj = train_val_subjs[-1]
        train_subjs = train_val_subjs[:-1]

        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        splits = {
            'train': [w for w in all_extracted_windows if w['subject'] in train_subjs],
            'val': [w for w in all_extracted_windows if w['subject'] == val_subj],
            'test': [w for w in all_extracted_windows if w['subject'] == test_subj]
        }

        fold_stats = {}
        for s_name, w_list in splits.items():
            X = np.array([w['window'] for w in w_list], dtype=np.float32)
            y = np.array([w['label'] for w in w_list], dtype=np.int64)
            subjs = np.array([w['subject'] for w in w_list])
            vids = np.array([w['video_id'] for w in w_list])
            times = np.array([[w['start_sec'], w['end_sec']] for w in w_list], dtype=np.float32)

            np.save(os.path.join(fold_dir, f"X_{s_name}.npy"), X)
            np.save(os.path.join(fold_dir, f"y_{s_name}.npy"), y)
            np.save(os.path.join(fold_dir, f"subjs_{s_name}.npy"), subjs)
            np.save(os.path.join(fold_dir, f"vids_{s_name}.npy"), vids)
            np.save(os.path.join(fold_dir, f"times_{s_name}.npy"), times)

            fold_stats[s_name] = {
                'total_windows': len(y),
                'fall_windows': int(np.sum(y == 1)),
                'normal_windows': int(np.sum(y == 0)),
                'fall_ratio': float(np.mean(y == 1)) if len(y) > 0 else 0.0,
                'subjects': list(sorted(set(subjs)))
            }

        fold_info[f"fold_{fold_idx}"] = {
            'test_subject': test_subj,
            'val_subject': val_subj,
            'train_subjects': train_subjs,
            'stats': fold_stats
        }

    meta_file = os.path.join(output_dir, "loso_metadata.json")
    with open(meta_file, 'w') as f:
        json.dump(fold_info, f, indent=2)

    return fold_info

if __name__ == '__main__':
    build_loso_folds()

