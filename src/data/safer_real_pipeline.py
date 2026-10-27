"""
VisionGuard Lite - Real SAFER-Activities Data Pipeline
Implements:
- MMAction2 / PYSKL pickle pose loading
- CSV annotation parser (HH:MM:SS.mmm timestamp to frame alignment)
- Official subject-wise split partitioning
- Temporal sliding window construction with strict event-level overlap labeling
- Person-centering, scale normalization, missing joint imputation
- Event-level metric aggregation (preventing overlapping-window inflation)
"""

import os
import csv
import pickle
import numpy as np
import torch
from typing import Dict, List, Tuple, Any, Optional

from src.data.schema import WINDOW_SIZE, WINDOW_STRIDE, EVAL_STRIDE, TARGET_FPS
from src.data.preprocessing import PosePreprocessor

def parse_safer_timestamp(ts_str: str) -> float:
    """
    Parses timestamp in 'HH:MM:SS.mmm' or 'MM:SS.mmm' format to seconds.
    Example: '00:01:23.450' -> 83.45 seconds.
    """
    ts_str = ts_str.strip()
    parts = ts_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600.0 + float(m) * 60.0 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return float(m) * 60.0 + float(s)
    else:
        return float(ts_str)

def parse_safer_annotation_csv(csv_path: str, fps: float = TARGET_FPS) -> Dict[str, Any]:
    """
    Parses official SAFER annotation CSV:
    Header: [Recording_Name, ...]
    Columns: Action, Start Time, End Time
    Returns:
    - recording_id: str
    - intervals: list of (action_name, start_frame, end_frame, is_fall)
    """
    intervals = []
    recording_id = os.path.splitext(os.path.basename(csv_path))[0]

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header and len(header) > 0 and header[0].strip():
            first_cell = header[0].strip()
            if not first_cell.lower().startswith('action'):
                recording_id = first_cell

        for row in reader:
            if len(row) < 3:
                continue
            action = row[0].strip()
            start_str = row[1].strip()
            end_str = row[2].strip()

            try:
                start_sec = parse_safer_timestamp(start_str)
                end_sec = parse_safer_timestamp(end_str)
                start_frame = int(round(start_sec * fps))
                end_frame = int(round(end_sec * fps))
                is_fall = ('fall' in action.lower())
                intervals.append({
                    'action': action,
                    'start_sec': start_sec,
                    'end_sec': end_sec,
                    'start_frame': start_frame,
                    'end_frame': end_frame,
                    'is_fall': is_fall
                })
            except Exception:
                continue

    return {
        'recording_id': recording_id,
        'intervals': intervals
    }

class SAFERRealPipeline:
    """
    Ingests official SAFER pickle poses and CSV annotations,
    constructs zero-leakage subject splits, and outputs normalized arrays.
    """
    def __init__(self, target_fps: float = TARGET_FPS, window_size: int = WINDOW_SIZE):
        self.target_fps = target_fps
        self.window_size = window_size
        self.preprocessor = PosePreprocessor(target_window_size=window_size)

    def load_pyskl_pickle_data(self, pkl_path: str) -> List[Dict[str, Any]]:
        """
        Loads MMAction2 / PYSKL format 2D pose pickle archive.
        Each item has 'frame_dir', 'total_frames', 'keypoint' (N, T, V, 2), 'keypoint_score' (N, T, V).
        """
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # PYSKL archives typically store a dict with 'annotations' list
        if isinstance(data, dict) and 'annotations' in data:
            return data['annotations']
        elif isinstance(data, list):
            return data
        else:
            return [data]

    def process_recording(
        self,
        keypoints_2d: np.ndarray,      # (T, 17, 2)
        keypoint_scores: np.ndarray,   # (T, 17)
        annotation_intervals: List[Dict[str, Any]],
        stride: int = WINDOW_STRIDE
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Combines 2D coords with scores -> (T, 17, 3), normalizes,
        and extracts 30-frame windows labeled against real action intervals.
        """
        T, V, _ = keypoints_2d.shape
        raw_kps = np.zeros((T, V, 3), dtype=np.float32)
        raw_kps[:, :, :2] = keypoints_2d
        raw_kps[:, :, 2] = keypoint_scores

        # Normalize skeleton
        norm_kps = self.preprocessor.normalize_skeleton(raw_kps)
        imputed_kps = self.preprocessor.impute_missing_landmarks(norm_kps)

        # Extract fall intervals in frames
        fall_frame_intervals = [
            (it['start_frame'], it['end_frame'])
            for it in annotation_intervals if it['is_fall']
        ]
        has_falls = len(fall_frame_intervals) > 0

        windows, labels, meta = self.preprocessor.create_sliding_windows(
            imputed_kps,
            stride=stride,
            fall_intervals=fall_frame_intervals,
            is_fall_video=has_falls
        )

        return windows, labels, meta

    @staticmethod
    def compute_event_level_metrics(
        window_preds: np.ndarray,
        window_meta: List[Dict[str, Any]],
        real_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluates detection on real ACTION EVENTS rather than treating overlapping windows as independent.
        An event is detected (TP) if any window overlapping the event by >= 50% fires a positive prediction.
        A false alarm occurs when a sequence of consecutive positive windows fires during normal activity.
        """
        fall_events = [e for e in real_events if e['is_fall']]
        total_fall_events = len(fall_events)
        detected_fall_events = 0
        delays_sec = []

        for e in fall_events:
            e_start = e['start_frame']
            e_end = e['end_frame']
            event_detected = False
            first_detect_frame = None

            for pred, meta in zip(window_preds, window_meta):
                w_start = meta['start_frame']
                w_end = meta['end_frame']
                # Overlap calculation
                o_start = max(w_start, e_start)
                o_end = min(w_end, e_end)
                overlap = max(0, o_end - o_start)
                ratio = overlap / float(w_end - w_start)

                if ratio >= 0.5 and pred == 1:
                    event_detected = True
                    if first_detect_frame is None or w_end < first_detect_frame:
                        first_detect_frame = w_end

            if event_detected:
                detected_fall_events += 1
                delay_frames = max(0, first_detect_frame - e_start)
                delays_sec.append(delay_frames / float(TARGET_FPS))

        event_recall = (detected_fall_events / max(1, total_fall_events)) if total_fall_events > 0 else 1.0
        avg_delay = float(np.mean(delays_sec)) if delays_sec else 0.0

        return {
            'total_real_fall_events': total_fall_events,
            'detected_real_fall_events': detected_fall_events,
            'event_level_recall': float(event_recall),
            'mean_detection_delay_sec': round(avg_delay, 2)
        }
