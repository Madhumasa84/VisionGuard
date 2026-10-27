"""
VisionGuard Lite - Schema & Landmark Definitions
Defines activity categories, skeleton landmark schemas (COCO-17 and MediaPipe-33),
windowing configurations, and data structures.
"""

from enum import Enum
from typing import List, Dict, Any
import json

class ActivityClass(Enum):
    NORMAL_WALKING = 0
    NORMAL_SITTING = 1
    NORMAL_STANDING = 2
    NORMAL_LYING = 3
    NORMAL_BENDING = 4
    TRANSITION_STUMBLE = 5
    FALL_EVENT = 6

ACTIVITY_NAMES = {
    0: 'walking',
    1: 'sitting',
    2: 'standing',
    3: 'lying',
    4: 'bending',
    5: 'stumble',
    6: 'fall'
}

# Binary classification mapping
# Fall (1) vs Non-Fall (0)
def to_binary_label(multi_class_label: int) -> int:
    return 1 if multi_class_label == ActivityClass.FALL_EVENT.value else 0

# 17 COCO Keypoints as used in open-source skeleton benchmarks (SAFER/OpenPose/YOLO-Pose)
COCO_17_LANDMARKS: List[str] = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]

# MediaPipe 33 Landmark names
MEDIAPIPE_33_LANDMARKS: List[str] = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Standard windowing configuration
WINDOW_SIZE: int = 30        # 30 frames
TARGET_FPS: float = 10.0     # 10 Hz -> 3.0 seconds per temporal window
WINDOW_STRIDE: int = 10      # 10 frames -> 1.0 second stride for training
EVAL_STRIDE: int = 5         # 5 frames -> 0.5 second stride for continuous evaluation

# Joint connectivity pairs for skeleton rendering
COCO_SKELETON_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms / Shoulders
    (5, 11), (6, 12), (11, 12),               # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)    # Legs
]

MEDIAPIPE_SKELETON_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

def export_schema_json(filepath: str):
    data = {
        'activity_classes': {k.name: k.value for k in ActivityClass},
        'activity_names': ACTIVITY_NAMES,
        'coco_17_landmarks': COCO_17_LANDMARKS,
        'mediapipe_33_landmarks': MEDIAPIPE_33_LANDMARKS,
        'window_size': WINDOW_SIZE,
        'target_fps': TARGET_FPS,
        'window_stride': WINDOW_STRIDE
    }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
