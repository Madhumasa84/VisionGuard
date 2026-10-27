"""
VisionGuard Lite - Dataset Downloader & Real Data Ingestion Manager
Supports official SAFER-Activities gated dataset fetching via Hugging Face Hub,
local manual archive ingestion, and representative kinematics benchmarking.
"""

import os
import sys
import json
import zipfile
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from huggingface_hub import HfApi, hf_hub_download, login

from src.data.schema import (
    ActivityClass,
    ACTIVITY_NAMES,
    COCO_17_LANDMARKS,
    MEDIAPIPE_33_LANDMARKS,
    TARGET_FPS
)

class SAFERDatasetManager:
    REPO_ID = 'g-b-r-l/SAFER-Activities'

    @staticmethod
    def check_access(token: Optional[str] = None) -> Tuple[bool, str]:
        """Check if HF token has access to the gated SAFER-Activities repo."""
        try:
            api = HfApi(token=token)
            info = api.dataset_info(SAFERDatasetManager.REPO_ID)
            return True, f"Authenticated! Access granted to repository: {info.id}"
        except Exception as e:
            return False, (
                f"Gated access restriction or unauthenticated: {str(e)}.\n"
                f"To access SAFER-Activities, please accept the CC BY-NC-SA 4.0 license at "
                f"https://huggingface.co/datasets/{SAFERDatasetManager.REPO_ID} "
                f"and provide your Hugging Face read token."
            )

    @staticmethod
    def download_real_dataset(
        output_dir: str = '/content/VisionGuardLite/datasets/raw/safer_real',
        token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Downloads official SAFER-Activities archive from gated Hugging Face repository.
        Requires user authentication and repository agreement approval.
        """
        os.makedirs(output_dir, exist_ok=True)
        if token:
            login(token=token, add_to_git_credential=False)

        access_ok, msg = SAFERDatasetManager.check_access(token)
        if not access_ok:
            raise PermissionError(f"Cannot download official SAFER-Activities dataset: {msg}")

        print(f"Downloading official dataset files from {SAFERDatasetManager.REPO_ID}...")
        api = HfApi(token=token)
        repo_files = api.list_repo_files(SAFERDatasetManager.REPO_ID, repo_type='dataset')
        
        downloaded = []
        for rf in repo_files:
            if rf.endswith('.zip') or rf.endswith('.json') or rf.endswith('.csv') or rf.endswith('.parquet'):
                local_f = hf_hub_download(
                    repo_id=SAFERDatasetManager.REPO_ID,
                    filename=rf,
                    repo_type='dataset',
                    local_dir=output_dir,
                    token=token
                )
                downloaded.append(local_f)
                print(f"Downloaded: {rf}")

        # Extract any zip archives
        for f in os.listdir(output_dir):
            if f.endswith('.zip'):
                zip_path = os.path.join(output_dir, f)
                print(f"Extracting {f}...")
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(output_dir)

        return {
            'status': 'success',
            'output_dir': output_dir,
            'files': downloaded
        }

    @staticmethod
    def generate_representative_dataset(
        output_dir: str,
        num_subjects: int = 12,
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Generates realistic representative skeleton sequences matching SAFER schema:
        17 COCO landmarks (x, y, confidence) at 10 FPS with real kinematic dynamics
        for demonstration and pipeline validation.
        PROVENANCE: DEMONSTRATION / SYNTHETIC ONLY.
        """
        os.makedirs(output_dir, exist_ok=True)
        np.random.seed(seed)
        
        subjects = [f'P{i:02d}' for i in range(1, num_subjects + 1)]
        dataset_records = []
        
        base_offsets = {
            0: (0.0, -0.45),   # nose
            1: (-0.03, -0.47), # l_eye
            2: (0.03, -0.47),  # r_eye
            3: (-0.06, -0.45), # l_ear
            4: (0.06, -0.45),  # r_ear
            5: (-0.15, -0.30), # l_shoulder
            6: (0.15, -0.30),  # r_shoulder
            7: (-0.22, -0.15), # l_elbow
            8: (0.22, -0.15),  # r_elbow
            9: (-0.25, 0.0),   # l_wrist
            10: (0.25, 0.0),   # r_wrist
            11: (-0.10, 0.05), # l_hip
            12: (0.10, 0.05),  # r_hip
            13: (-0.12, 0.28), # l_knee
            14: (0.12, 0.28),  # r_knee
            15: (-0.12, 0.50), # l_ankle
            16: (0.12, 0.50),  # r_ankle
        }

        activity_types = [
            (ActivityClass.NORMAL_WALKING, 4, 100),
            (ActivityClass.NORMAL_SITTING, 3, 80),
            (ActivityClass.NORMAL_STANDING, 3, 70),
            (ActivityClass.NORMAL_BENDING, 3, 80),
            (ActivityClass.TRANSITION_STUMBLE, 2, 80),
            (ActivityClass.FALL_EVENT, 4, 100),
        ]

        total_sequences = 0
        for subj in subjects:
            subj_height_scale = np.random.uniform(0.9, 1.1)
            subj_speed_scale = np.random.uniform(0.85, 1.15)
            
            for act_class, num_vids, num_frames in activity_types:
                for v_idx in range(num_vids):
                    video_id = f'{subj}_{act_class.name.lower()}_v{v_idx:02d}'
                    frames = np.zeros((num_frames, 17, 3), dtype=np.float32)
                    fall_intervals = []
                    
                    root_x = 0.5 + np.random.uniform(-0.1, 0.1)
                    root_y = 0.5 + np.random.uniform(-0.05, 0.05)
                    
                    if act_class == ActivityClass.NORMAL_WALKING:
                        walk_dir = 1.0 if np.random.rand() > 0.5 else -1.0
                        for t in range(num_frames):
                            phase = 2 * np.pi * (t / 10.0) * subj_speed_scale
                            cx = root_x + walk_dir * (t / num_frames) * 0.2
                            cy = root_y + np.sin(phase * 2) * 0.015
                            
                            for j in range(17):
                                ox, oy = base_offsets[j]
                                if j in [13, 15]:
                                    oy += np.sin(phase) * 0.05
                                    ox += np.cos(phase) * 0.04
                                elif j in [14, 16]:
                                    oy += np.sin(phase + np.pi) * 0.05
                                    ox += np.cos(phase + np.pi) * 0.04
                                elif j in [7, 9]:
                                    ox += np.cos(phase + np.pi) * 0.04
                                elif j in [8, 10]:
                                    ox += np.cos(phase) * 0.04
                                
                                conf = np.random.uniform(0.85, 0.99)
                                frames[t, j] = [cx + ox * subj_height_scale, cy + oy * subj_height_scale, conf]

                    elif act_class == ActivityClass.NORMAL_SITTING:
                        sit_frame = num_frames // 3
                        for t in range(num_frames):
                            progress = min(1.0, max(0.0, (t - sit_frame) / 20.0))
                            cy = root_y + progress * 0.18
                            for j in range(17):
                                ox, oy = base_offsets[j]
                                if j in [11, 12, 13, 14]:
                                    oy *= (1.0 - progress * 0.4)
                                conf = np.random.uniform(0.88, 0.99)
                                frames[t, j] = [root_x + ox * subj_height_scale, cy + oy * subj_height_scale, conf]

                    elif act_class == ActivityClass.NORMAL_STANDING:
                        for t in range(num_frames):
                            sway_x = np.sin(t * 0.1) * 0.005
                            sway_y = np.cos(t * 0.08) * 0.003
                            for j in range(17):
                                ox, oy = base_offsets[j]
                                conf = np.random.uniform(0.90, 0.99)
                                frames[t, j] = [root_x + ox + sway_x, root_y + oy + sway_y, conf]

                    elif act_class == ActivityClass.NORMAL_BENDING:
                        bend_start = num_frames // 4
                        bend_dur = num_frames // 2
                        for t in range(num_frames):
                            rel_t = (t - bend_start) / bend_dur
                            bend_amount = np.sin(rel_t * np.pi) if 0 <= rel_t <= 1 else 0.0
                            for j in range(17):
                                ox, oy = base_offsets[j]
                                if j in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
                                    oy += bend_amount * 0.28
                                    ox += bend_amount * 0.12
                                conf = np.random.uniform(0.80, 0.98)
                                frames[t, j] = [root_x + ox * subj_height_scale, root_y + oy * subj_height_scale, conf]

                    elif act_class == ActivityClass.TRANSITION_STUMBLE:
                        stumble_start = num_frames // 3
                        for t in range(num_frames):
                            if stumble_start <= t <= stumble_start + 15:
                                jerk_x = np.sin((t - stumble_start) / 15.0 * np.pi) * 0.15
                                jerk_y = np.sin((t - stumble_start) / 15.0 * np.pi) * 0.08
                            else:
                                jerk_x, jerk_y = 0.0, 0.0
                            for j in range(17):
                                ox, oy = base_offsets[j]
                                conf = np.random.uniform(0.75, 0.95)
                                frames[t, j] = [root_x + ox + jerk_x, root_y + oy + jerk_y, conf]

                    elif act_class == ActivityClass.FALL_EVENT:
                        fall_onset = int(num_frames * 0.35)
                        fall_duration = 12
                        impact_frame = fall_onset + fall_duration
                        fall_intervals.append((fall_onset, impact_frame + 20))
                        
                        fall_angle = np.random.choice([np.pi/2, -np.pi/2, np.pi/2.5])
                        
                        for t in range(num_frames):
                            if t < fall_onset:
                                cy = root_y
                                cx = root_x
                                rot = 0.0
                            elif t <= impact_frame:
                                progress = (t - fall_onset) / float(fall_duration)
                                cy = root_y + (progress ** 2) * 0.32
                                cx = root_x + progress * 0.15 * np.sign(fall_angle)
                                rot = progress * fall_angle
                            else:
                                cy = root_y + 0.32
                                cx = root_x + 0.15 * np.sign(fall_angle)
                                rot = fall_angle
                                
                            cos_r, sin_r = np.cos(rot), np.sin(rot)
                            for j in range(17):
                                ox, oy = base_offsets[j]
                                rx = ox * cos_r - oy * sin_r
                                ry = ox * sin_r + oy * cos_r
                                conf = np.random.uniform(0.70, 0.95)
                                frames[t, j] = [cx + rx * subj_height_scale, cy + ry * subj_height_scale, conf]

                    seq_path = os.path.join(output_dir, f'{video_id}.npz')
                    np.savez_compressed(
                        seq_path,
                        keypoints=frames,
                        subject=subj,
                        activity_class=act_class.value,
                        activity_name=ACTIVITY_NAMES[act_class.value],
                        fall_intervals=np.array(fall_intervals, dtype=np.int32),
                        fps=TARGET_FPS
                    )
                    
                    dataset_records.append({
                        'video_id': video_id,
                        'subject': subj,
                        'file_path': seq_path,
                        'activity_id': act_class.value,
                        'activity_name': ACTIVITY_NAMES[act_class.value],
                        'is_fall': act_class == ActivityClass.FALL_EVENT,
                        'num_frames': num_frames,
                        'fall_intervals': fall_intervals,
                        'provenance': 'synthetic_kinematic_demonstration'
                    })
                    total_sequences += 1

        index_file = os.path.join(output_dir, 'manifest.json')
        with open(index_file, 'w') as f:
            json.dump({
                'provenance': 'synthetic_kinematic_demonstration',
                'description': 'Demonstration sequences generated for end-to-end pipeline verification under gated dataset access',
                'num_subjects': num_subjects,
                'total_sequences': total_sequences,
                'subjects': subjects,
                'records': dataset_records
            }, f, indent=2)

        return {
            'output_dir': output_dir,
            'total_sequences': total_sequences,
            'subjects': subjects,
            'manifest': index_file,
            'provenance': 'synthetic_kinematic_demonstration'
        }
