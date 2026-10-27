"""
VisionGuard Lite - MediaPipe Pose Extractor & Visualizer
Extracts 33 landmarks from video frames using MediaPipe Tasks API (PoseLandmarker),
generates normalized temporal windows, and draws skeleton overlays.
"""

import os
import sys
import ctypes
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Ensure bundled GLESv2 library is preloaded on Linux headless environments if needed
REPO_ROOT = Path(__file__).resolve().parents[2]
if sys.platform.startswith('linux'):
    lib_dir = REPO_ROOT / 'lib' / 'x86_64-linux-gnu'
    gles_path = lib_dir / 'libGLESv2.so.2'
    if gles_path.exists():
        try:
            ctypes.CDLL(str(gles_path))
        except Exception:
            pass

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.data.schema import MEDIAPIPE_SKELETON_EDGES

class MediaPipePoseExtractor:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = str(REPO_ROOT / 'checkpoints' / 'pose_landmarker_lite.task')
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def extract_from_video(
        self,
        video_path: str,
        max_frames: int = 300,
        target_fps: float = 10.0
    ) -> Dict[str, Any]:
        """
        Reads video, downsamples to target_fps if needed, extracts landmarks.
        Returns:
        - landmarks: (T, 33, 4) -> [x, y, z, visibility]
        - annotated_frames: list of annotated BGR frames
        - fps: actual video fps
        - duration_sec: video duration in seconds
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0 or np.isnan(orig_fps):
            orig_fps = 30.0

        step = max(1, int(round(orig_fps / target_fps)))
        
        all_landmarks = []
        annotated_frames = []
        frame_idx = 0

        while cap.isOpened() and len(annotated_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                h, w, _ = frame.shape
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self.detector.detect(mp_image)

                frame_kps = np.zeros((33, 4), dtype=np.float32)
                annotated = frame.copy()

                if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                    lms = detection_result.pose_landmarks[0]
                    for j_idx, lm in enumerate(lms):
                        frame_kps[j_idx] = [lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.9)]
                        px, py = int(lm.x * w), int(lm.y * h)
                        if 0 <= px < w and 0 <= py < h:
                            cv2.circle(annotated, (px, py), 3, (0, 255, 0), -1)

                    for j1, j2 in MEDIAPIPE_SKELETON_EDGES:
                        if j1 < len(lms) and j2 < len(lms):
                            p1 = (int(lms[j1].x * w), int(lms[j1].y * h))
                            p2 = (int(lms[j2].x * w), int(lms[j2].y * h))
                            if 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h:
                                cv2.line(annotated, p1, p2, (255, 120, 0), 2)
                else:
                    # Heuristic fallback if frame has motion/figure
                    pass

                all_landmarks.append(frame_kps)
                annotated_frames.append(annotated)

            frame_idx += 1

        cap.release()

        landmarks_arr = np.array(all_landmarks, dtype=np.float32) if len(all_landmarks) > 0 else np.zeros((0, 33, 4))
        duration = len(annotated_frames) / float(target_fps) if len(annotated_frames) > 0 else 0.0

        return {
            'landmarks': landmarks_arr,
            'annotated_frames': annotated_frames,
            'frame_count': len(annotated_frames),
            'fps': target_fps,
            'duration_sec': duration
        }

    def close(self):
        try:
            self.detector.close()
        except:
            pass
