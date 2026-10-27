"""
Unit tests for SAFER real data pipeline and pose schema mapping
"""

import pytest
import numpy as np
import tempfile
import os

from src.data.pose_schema_audit import PoseSchemaAuditor, MEDIAPIPE_TO_COCO_INDICES, COCO_17_JOINTS
from src.data.safer_real_pipeline import parse_safer_timestamp, parse_safer_annotation_csv, SAFERRealPipeline

def test_safer_timestamp_parsing():
    assert parse_safer_timestamp("00:01:23.450") == 83.45
    assert parse_safer_timestamp("01:00:00.000") == 3600.0
    assert parse_safer_timestamp("02:15.500") == 135.5
    assert parse_safer_timestamp("12.34") == 12.34

def test_mediapipe_to_coco_mapping():
    assert len(COCO_17_JOINTS) == 17
    assert len(MEDIAPIPE_TO_COCO_INDICES) == 17

    # Synthetic MediaPipe sequence (10, 33, 4)
    mp_kps = np.zeros((10, 33, 4), dtype=np.float32)
    # Put identifiable coordinate at MediaPipe left_shoulder (11)
    mp_kps[:, 11, :2] = [0.42, 0.58]
    mp_kps[:, 11, 3] = 0.95

    coco_kps = PoseSchemaAuditor.mediapipe_to_coco17(mp_kps)
    assert coco_kps.shape == (10, 17, 3)
    # In COCO-17, left_shoulder is index 5
    assert abs(coco_kps[0, 5, 0] - 0.42) < 1e-5
    assert abs(coco_kps[0, 5, 1] - 0.58) < 1e-5
    assert abs(coco_kps[0, 5, 2] - 0.95) < 1e-5

def test_event_level_metrics_calculation():
    pipeline = SAFERRealPipeline()
    # 2 real fall events: Event 1 (frames 20-50), Event 2 (frames 80-110)
    real_events = [
        {'action': 'Forward Fall', 'start_frame': 20, 'end_frame': 50, 'is_fall': True},
        {'action': 'Backward Fall', 'start_frame': 80, 'end_frame': 110, 'is_fall': True}
    ]

    # Windows:
    # Window 0: 0-30 (overlap 10 frames with Event 1 -> 33% -> pred=0)
    # Window 1: 20-50 (overlap 30 frames with Event 1 -> 100% -> pred=1)
    # Window 2: 30-60 (overlap 20 frames with Event 1 -> 66% -> pred=1)
    # Window 3: 80-110 (overlap 30 frames with Event 2 -> 100% -> pred=1)
    window_meta = [
        {'start_frame': 0, 'end_frame': 30},
        {'start_frame': 20, 'end_frame': 50},
        {'start_frame': 30, 'end_frame': 60},
        {'start_frame': 80, 'end_frame': 110}
    ]
    window_preds = np.array([0, 1, 1, 1])

    metrics = pipeline.compute_event_level_metrics(window_preds, window_meta, real_events)
    assert metrics['total_real_fall_events'] == 2
    assert metrics['detected_real_fall_events'] == 2
    assert metrics['event_level_recall'] == 1.0
