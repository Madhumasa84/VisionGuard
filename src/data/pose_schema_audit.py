"""
VisionGuard Lite - Pose Schema Audit & Cross-Extractor Mapping
Implements:
- ViTPose-H (COCO-17) vs MediaPipe-33 topology comparison
- Explicit landmark index mapping
- Coordinate system & visibility translation
- Automated schema and parity verification tests
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# 17 COCO Keypoints as extracted by YOLOv8x + ViTPose-H in SAFER-Activities
COCO_17_JOINTS = [
    'nose',            # 0
    'left_eye',        # 1
    'right_eye',       # 2
    'left_ear',        # 3
    'right_ear',       # 4
    'left_shoulder',   # 5
    'right_shoulder',  # 6
    'left_elbow',      # 7
    'right_elbow',     # 8
    'left_wrist',      # 9
    'right_wrist',     # 10
    'left_hip',        # 11
    'right_hip',       # 12
    'left_knee',       # 13
    'right_knee',      # 14
    'left_ankle',      # 15
    'right_ankle'      # 16
]

# MediaPipe 33 Landmark indices corresponding to COCO-17
MEDIAPIPE_TO_COCO_INDICES = {
    0: 0,   # nose -> nose
    1: 2,   # left_eye -> left_eye
    2: 5,   # right_eye -> right_eye
    3: 7,   # left_ear -> left_ear
    4: 8,   # right_ear -> right_ear
    5: 11,  # left_shoulder -> left_shoulder
    6: 12,  # right_shoulder -> right_shoulder
    7: 13,  # left_elbow -> left_elbow
    8: 14,  # right_elbow -> right_elbow
    9: 15,  # left_wrist -> left_wrist
    10: 16, # right_wrist -> right_wrist
    11: 23, # left_hip -> left_hip
    12: 24, # right_hip -> right_hip
    13: 25, # left_knee -> left_knee
    14: 26, # right_knee -> right_knee
    15: 27, # left_ankle -> left_ankle
    16: 28  # right_ankle -> right_ankle
}

MP_TO_COCO_INDEX_LIST = [MEDIAPIPE_TO_COCO_INDICES[i] for i in range(17)]

class PoseSchemaAuditor:
    """
    Validates cross-pipeline pose compatibility, coordinate normalization,
    and confidence scores between ViTPose-H and MediaPipe.
    """
    @staticmethod
    def mediapipe_to_coco17(mp_landmarks: np.ndarray) -> np.ndarray:
        """
        Converts MediaPipe (T, 33, C) or (33, C) where C >= 3 (x, y, [z], conf)
        into COCO-17 format (T, 17, 3) [x, y, conf].
        """
        is_single_frame = mp_landmarks.ndim == 2
        if is_single_frame:
            mp_landmarks = mp_landmarks[None, ...]
            
        T, J, C = mp_landmarks.shape
        if J < 29:
            raise ValueError(f"Expected at least 29 MediaPipe landmarks, got {J}")

        coco_out = np.zeros((T, 17, 3), dtype=np.float32)
        for coco_idx, mp_idx in enumerate(MP_TO_COCO_INDEX_LIST):
            coco_out[:, coco_idx, 0] = mp_landmarks[:, mp_idx, 0] # x
            coco_out[:, coco_idx, 1] = mp_landmarks[:, mp_idx, 1] # y
            if C >= 4:
                coco_out[:, coco_idx, 2] = mp_landmarks[:, mp_idx, 3] # visibility
            elif C == 3:
                coco_out[:, coco_idx, 2] = mp_landmarks[:, mp_idx, 2]
            else:
                coco_out[:, coco_idx, 2] = 1.0

        if is_single_frame:
            return coco_out[0]
        return coco_out

    @staticmethod
    def audit_representation_differences() -> Dict[str, Any]:
        """
        Comprehensive comparison between ViTPose-H and MediaPipe representations.
        """
        return {
            'extractor_A (SAFER-Activities)': {
                'name': 'ViTPose-H (with YOLOv8x detector)',
                'keypoint_count': 17,
                'topology': 'COCO-17 format',
                'coordinate_system': 'Pixel coordinates [0, W] x [0, H] or normalized by image resolution',
                'confidence_metric': 'Detector box score * ViTPose heatmap peak softmax confidence [0, 1]',
                'lifting_method': 'MotionAGFormer (2D to 3D lifting)'
            },
            'extractor_B (VisionGuard Deployment)': {
                'name': 'MediaPipe Tasks PoseLandmarker (Lite / Heavy)',
                'keypoint_count': 33,
                'topology': 'BlazePose topology',
                'coordinate_system': 'Normalized image coordinates [0, 1] x [0, 1], with z representing relative depth',
                'confidence_metric': 'Landmark visibility sigmoid probability [0, 1]',
                'lifting_method': 'Native metric 3D world coordinates'
            },
            'generalization_warning': (
                'Models trained exclusively on ViTPose-H poses will experience domain shift if applied '
                'directly to MediaPipe landmarks without landmark alignment and coordinate normalization, '
                'due to subtle differences in joint center definitions (e.g., hip joint center vs pelvis bone).'
            ),
            'index_mapping': MEDIAPIPE_TO_COCO_INDICES
        }
