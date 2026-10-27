"""
VisionGuard Lite - GMDCSA-24 Real Feature Extractor
Extracts 33 MediaPipe landmarks and 17 COCO landmarks from all 160 real human videos,
aligns them with exact action timestamps from official CSV annotations,
and saves persistent per-video npz archives.
"""

import os
import sys
import csv
import re
import cv2
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from src.data.pose_schema_audit import PoseSchemaAuditor

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def parse_classes_field(text: str) -> List[Tuple[str, float, float]]:
    """
    Parses 'Action[start to end]; Action[start to end]'
    Example: 'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]' -> [('Falling (SW)', 3.4, 6.0), ('Sitting', 0.0, 3.4)]
    """
    items = text.split(';')
    parsed = []
    for it in items:
        it = it.strip()
        m = re.match(r'^(.*?)\s*\[\s*([0-9.]+)\s*to\s*([0-9.]+)\s*\]$', it)
        if m:
            act_name = m.group(1).strip()
            s = float(m.group(2))
            e = float(m.group(3))
            parsed.append((act_name, s, e))
    return parsed

def load_all_annotations(raw_base_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Reads all Subject {1..4}/{Fall,ADL}.csv files and maps by (subject, category, filename).
    """
    annotations_map = {}
    for s_idx in [1, 2, 3, 4]:
        subj_name = f"Subject {s_idx}"
        for cat in ['Fall', 'ADL']:
            csv_path = os.path.join(raw_base_dir, subj_name, f"{cat}.csv")
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if not row or len(row) < 6:
                        continue
                    fn = row[0].strip()
                    if not fn.endswith('.mp4'):
                        fn = f"{fn}.mp4"
                    length_sec = float(row[1].strip()) if row[1].strip() else 0.0
                    classes_str = row[5].strip()
                    events = parse_classes_field(classes_str)

                    key = f"{subj_name}_{cat}_{fn}"
                    annotations_map[key] = {
                        'subject': subj_name,
                        'category': cat,
                        'file_name': fn,
                        'nominal_length_sec': length_sec,
                        'description': row[4].strip() if len(row) > 4 else '',
                        'events': [
                            {
                                'action': e[0],
                                'start_sec': e[1],
                                'end_sec': e[2],
                                'is_fall': ('fall' in e[0].lower())
                            } for e in events
                        ]
                    }
    return annotations_map

def extract_all_poses(
    raw_base_dir: str = None,
    output_dir: str = None,
    target_fps: float = 10.0,
    model_path: str = None
):
    if raw_base_dir is None:
        raw_base_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/data/raw/gmdcsa24')
    if output_dir is None:
        output_dir = str(REPO_ROOT / 'experiments/gmdcsa24_real/data/extracted_poses')
    if model_path is None:
        model_path = str(REPO_ROOT / 'checkpoints/pose_landmarker_lite.task')
    os.makedirs(output_dir, exist_ok=True)
    annotations_map = load_all_annotations(raw_base_dir)
    print(f"Loaded annotations for {len(annotations_map)} recordings.")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    manifest_records = []
    video_count = 0
    t_start = time.perf_counter()

    for s_idx in [1, 2, 3, 4]:
        subj_name = f"Subject {s_idx}"
        for cat in ['Fall', 'ADL']:
            video_folder = os.path.join(raw_base_dir, subj_name, cat)
            if not os.path.exists(video_folder):
                continue
            vids = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4')])

            for vid_fn in vids:
                video_count += 1
                vid_path = os.path.join(video_folder, vid_fn)
                key = f"{subj_name}_{cat}_{vid_fn}"
                meta = annotations_map.get(key, {
                    'subject': subj_name,
                    'category': cat,
                    'file_name': vid_fn,
                    'events': []
                })

                cap = cv2.VideoCapture(vid_path)
                if not cap.isOpened():
                    print(f"Warning: could not open {vid_path}")
                    continue

                orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                step = max(1, int(round(orig_fps / target_fps)))

                all_mp_lms = []
                all_timestamps = []
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % step == 0:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        res = detector.detect(mp_img)

                        lms = np.zeros((33, 4), dtype=np.float32)
                        if res.pose_landmarks and len(res.pose_landmarks) > 0:
                            for j, lm in enumerate(res.pose_landmarks[0]):
                                lms[j] = [lm.x, lm.y, lm.z, getattr(lm, 'visibility', 0.9)]

                        all_mp_lms.append(lms)
                        all_timestamps.append(frame_idx / float(orig_fps))

                    frame_idx += 1

                cap.release()

                all_mp_lms = np.array(all_mp_lms, dtype=np.float32)
                all_timestamps = np.array(all_timestamps, dtype=np.float32)
                missing_mask = (all_mp_lms[..., 3] < 0.3)

                # Convert to COCO-17 for classifier input
                coco_lms = PoseSchemaAuditor.mediapipe_to_coco17(all_mp_lms)

                # Frame index alignment for action events
                events_with_frames = []
                for ev in meta['events']:
                    s_sec = ev['start_sec']
                    e_sec = ev['end_sec']
                    # Find closest sampled frame index
                    if len(all_timestamps) > 0:
                        s_idx_fr = int(np.argmin(np.abs(all_timestamps - s_sec)))
                        e_idx_fr = int(np.argmin(np.abs(all_timestamps - e_sec)))
                    else:
                        s_idx_fr = int(round(s_sec * target_fps))
                        e_idx_fr = int(round(e_sec * target_fps))

                    events_with_frames.append({
                        'action': ev['action'],
                        'start_sec': s_sec,
                        'end_sec': e_sec,
                        'start_frame': s_idx_fr,
                        'end_frame': e_idx_fr,
                        'is_fall': ev['is_fall']
                    })

                safe_id = f"s{s_idx}_{cat.lower()}_{vid_fn.replace('.mp4', '')}"
                out_npz_path = os.path.join(output_dir, f"{safe_id}.npz")

                np.savez_compressed(
                    out_npz_path,
                    mp_landmarks=all_mp_lms,
                    coco_landmarks=coco_lms,
                    timestamps=all_timestamps,
                    missing_mask=missing_mask,
                    subject=subj_name,
                    category=cat,
                    video_name=vid_fn,
                    video_id=safe_id,
                    fps=target_fps,
                    duration_sec=float(all_timestamps[-1]) if len(all_timestamps) > 0 else 0.0,
                    events=json.dumps(events_with_frames)
                )

                manifest_records.append({
                    'video_id': safe_id,
                    'subject': subj_name,
                    'category': cat,
                    'is_fall_video': (cat == 'Fall'),
                    'file_path': out_npz_path,
                    'frame_count': len(all_timestamps),
                    'duration_sec': float(all_timestamps[-1]) if len(all_timestamps) > 0 else 0.0,
                    'num_events': len(events_with_frames)
                })

                if video_count % 20 == 0 or video_count == 160:
                    elapsed = time.perf_counter() - t_start
                    print(f"[{video_count}/160] Processed {safe_id} ({len(all_timestamps)} frames, {elapsed:.1f}s elapsed)")

    try:
        detector.close()
    except Exception:
        pass

    manifest_file = os.path.join(output_dir, "manifest.json")
    with open(manifest_file, "w") as f:
        json.dump({
            'dataset': 'GMDCSA-24 v2.1 (Real Human Activity and Fall Videos)',
            'provenance': 'real_human_video_recordings',
            'publication': 'PMC11416611 (Data in Brief, 2024)',
            'zenodo_doi': '10.5281/zenodo.13354453',
            'license': 'MIT',
            'total_videos': len(manifest_records),
            'target_fps': target_fps,
            'records': manifest_records
        }, f, indent=2)

    print(f"\nAll {len(manifest_records)} real videos extracted successfully!")
    print(f"Manifest written to: {manifest_file}")
    return manifest_file

if __name__ == '__main__':
    extract_all_poses()
