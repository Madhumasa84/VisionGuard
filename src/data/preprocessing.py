"""
VisionGuard Lite - Preprocessing Pipeline
Implements:
- Person centering & scale normalization
- Missing joint imputation & confidence masking
- 30-frame temporal sliding window segmentation
- Strict interval-based fall labeling (preventing video-level false positives)
- Exclusion of ambiguous transition boundaries from clean supervised training
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.data.schema import WINDOW_SIZE, WINDOW_STRIDE, EVAL_STRIDE

class PosePreprocessor:
    def __init__(self, target_window_size: int = WINDOW_SIZE):
        self.target_window_size = target_window_size

    def normalize_skeleton(self, keypoints: np.ndarray) -> np.ndarray:
        """
        keypoints: shape (T, J, C) where C >= 2 (x, y, [conf])
        Applies:
        1. Person-centering: translate hip midpoint to (0, 0)
        2. Scale normalization: scale by torso height (hip midpoint to shoulder midpoint)
        """
        normalized = keypoints.copy()
        T, J, C = normalized.shape
        
        # Determine hip and shoulder indices depending on joint count
        if J == 17:
            l_hip, r_hip = 11, 12
            l_sh, r_sh = 5, 6
        elif J == 33:
            l_hip, r_hip = 23, 24
            l_sh, r_sh = 11, 12
        else:
            l_hip, r_hip = 0, 1
            l_sh, r_sh = 0, 1

        for t in range(T):
            # Compute hip midpoint
            hip_mid = (normalized[t, l_hip, :2] + normalized[t, r_hip, :2]) / 2.0
            sh_mid = (normalized[t, l_sh, :2] + normalized[t, r_sh, :2]) / 2.0
            torso_len = np.linalg.norm(sh_mid - hip_mid)
            if torso_len < 1e-4:
                torso_len = 0.35  # default fall-back scale
            
            # Center and scale (x, y)
            normalized[t, :, :2] = (normalized[t, :, :2] - hip_mid) / torso_len
            
        return normalized

    def impute_missing_landmarks(self, keypoints: np.ndarray, min_conf: float = 0.3) -> np.ndarray:
        """
        Linear interpolation for joints with low confidence (< min_conf).
        """
        imputed = keypoints.copy()
        T, J, C = imputed.shape
        if C < 3:
            return imputed
            
        for j in range(J):
            low_conf_mask = imputed[:, j, 2] < min_conf
            if np.any(low_conf_mask) and not np.all(low_conf_mask):
                valid_idx = np.where(~low_conf_mask)[0]
                invalid_idx = np.where(low_conf_mask)[0]
                imputed[invalid_idx, j, 0] = np.interp(invalid_idx, valid_idx, imputed[valid_idx, j, 0])
                imputed[invalid_idx, j, 1] = np.interp(invalid_idx, valid_idx, imputed[valid_idx, j, 1])
                imputed[invalid_idx, j, 2] = min_conf # flagged
        return imputed

    def create_sliding_windows(
        self,
        keypoints: np.ndarray,
        stride: int,
        fall_intervals: List[Tuple[int, int]],
        is_fall_video: bool
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Extracts sliding windows of size target_window_size.
        Labeling policy:
        - If window overlaps fall interval >= 50%: label = 1 (FALL)
        - If window overlap == 0%: label = 0 (NON-FALL)
        - If window has partial overlap (0% < overlap < 50%): label = -1 (AMBIGUOUS / EXCLUDE)
        """
        T, J, C = keypoints.shape
        windows = []
        labels = []
        metadata = []
        
        if T < self.target_window_size:
            return np.empty((0, self.target_window_size, J, C)), np.empty(0, dtype=np.int64), []

        for start in range(0, T - self.target_window_size + 1, stride):
            end = start + self.target_window_size
            window = keypoints[start:end]
            
            label = 0
            is_ambiguous = False
            
            if is_fall_video and len(fall_intervals) > 0:
                max_overlap_ratio = 0.0
                for f_start, f_end in fall_intervals:
                    overlap_start = max(start, f_start)
                    overlap_end = min(end, f_end)
                    overlap_len = max(0, overlap_end - overlap_start)
                    ratio = overlap_len / float(self.target_window_size)
                    if ratio > max_overlap_ratio:
                        max_overlap_ratio = ratio
                
                if max_overlap_ratio >= 0.5:
                    label = 1
                elif max_overlap_ratio > 0.0:
                    label = -1 # Ambiguous transition boundary
                    is_ambiguous = True
                else:
                    label = 0 # Pre-fall or post-fall clean sequence
            
            windows.append(window)
            labels.append(label)
            metadata.append({
                'start_frame': start,
                'end_frame': end,
                'is_ambiguous': is_ambiguous
            })
            
        return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64), metadata

def process_and_cache_dataset(
    manifest_path: str,
    output_dir: str,
    stride_train: int = WINDOW_STRIDE,
    stride_eval: int = EVAL_STRIDE
) -> Dict[str, Any]:
    """
    Processes all raw sequences into normalized subject-partitioned numpy arrays.
    Saves:
    - X_train.npy, y_train.npy, subjs_train.npy
    - X_val.npy, y_val.npy, subjs_val.npy
    - X_test.npy, y_test.npy, subjs_test.npy
    - preprocessing_meta.json
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Subject split
    train_subjs = {'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08'}
    val_subjs = {'P09', 'P10'}
    test_subjs = {'P11', 'P12'}

    preprocessor = PosePreprocessor(target_window_size=WINDOW_SIZE)
    
    splits = {
        'train': {'X': [], 'y': [], 'subj': []},
        'val': {'X': [], 'y': [], 'subj': []},
        'test': {'X': [], 'y': [], 'subj': []}
    }
    
    ambiguous_excluded_counts = {'train': 0, 'val': 0, 'test': 0}

    for record in manifest['records']:
        data = np.load(record['file_path'])
        keypoints = data['keypoints']
        fall_intervals = record['fall_intervals']
        is_fall = record['is_fall']
        subj = record['subject']
        
        if subj in train_subjs:
            split_name = 'train'
            stride = stride_train
        elif subj in val_subjs:
            split_name = 'val'
            stride = stride_eval
        else:
            split_name = 'test'
            stride = stride_eval
            
        norm_kp = preprocessor.normalize_skeleton(keypoints)
        imputed_kp = preprocessor.impute_missing_landmarks(norm_kp)
        windows, labels, meta = preprocessor.create_sliding_windows(
            imputed_kp, stride, fall_intervals, is_fall
        )
        
        for w, l in zip(windows, labels):
            if l == -1:
                ambiguous_excluded_counts[split_name] += 1
                continue # Exclude ambiguous boundary from clean supervised training/testing
            splits[split_name]['X'].append(w)
            splits[split_name]['y'].append(l)
            splits[split_name]['subj'].append(subj)

    saved_info = {}
    for s_name, d in splits.items():
        X_arr = np.array(d['X'], dtype=np.float32)
        y_arr = np.array(d['y'], dtype=np.int64)
        subj_arr = np.array(d['subj'])
        
        np.save(os.path.join(output_dir, f'X_{s_name}.npy'), X_arr)
        np.save(os.path.join(output_dir, f'y_{s_name}.npy'), y_arr)
        np.save(os.path.join(output_dir, f'subjs_{s_name}.npy'), subj_arr)
        
        fall_cnt = int(np.sum(y_arr == 1))
        non_fall_cnt = int(np.sum(y_arr == 0))
        saved_info[s_name] = {
            'total_windows': len(y_arr),
            'fall_windows': fall_cnt,
            'non_fall_windows': non_fall_cnt,
            'fall_ratio': float(fall_cnt / max(1, len(y_arr))),
            'ambiguous_excluded': ambiguous_excluded_counts[s_name],
            'unique_subjects': list(sorted(set(d['subj'])))
        }

    meta_file = os.path.join(output_dir, 'preprocessing_meta.json')
    with open(meta_file, 'w') as f:
        json.dump(saved_info, f, indent=2)

    return saved_info
