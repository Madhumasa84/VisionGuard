"""Regression checks for the user-facing analysis workflow."""
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from demo import app
from demo.ui import build_interface, result_summary


@pytest.mark.parametrize("probabilities,k,expected,delay", [
    ([.8, .2, .8, .9], 2, [0, 0, 0, 1], .5),
    ([.8, .8, .8, .8], 4, [0, 0, 0, 1], 1.5),
    ([.2, .8, .2, .8], 2, [0, 0, 0, 0], None),
    ([.5, .2, .7, .8], 1, [1, 0, 1, 1], 0),
])
def test_confirmation_policy(probabilities, k, expected, delay):
    states, actual_delay = app.confirm_windows(probabilities, [3, 3.5, 4, 4.5], .5, k)
    assert states == expected
    assert actual_delay == delay


@pytest.mark.parametrize("settings", [
    {"confirmation_k": 0}, {"confirmation_k": 2.5}, {"confirmation_k": True},
    {"confidence_thresh": float('nan')}, {"confidence_thresh": .95},
    {"backend_choice": "unknown"},
])
def test_invalid_settings(settings):
    result = app.process_video_pipeline(str(app.REPO_ROOT / 'demo/sample_activity.mp4'), **settings)
    assert result[0] is None
    assert result[3]['Status'] == 'Invalid Settings'


def test_extractor_closes_on_failure(monkeypatch):
    extractor = Mock()
    extractor.extract_from_video.side_effect = ValueError('Unreadable video')
    monkeypatch.setattr(app, 'MediaPipePoseExtractor', lambda: extractor)
    result = app.process_video_pipeline(str(app.REPO_ROOT / 'demo/sample_activity.mp4'))
    assert result[3]['Status'] == 'Extraction Failure'
    extractor.close.assert_called_once()


def fake_extractor(monkeypatch, frame_count=40, fps=8):
    # A deterministic pose allows pipeline branches to be tested without MediaPipe.
    landmarks = np.zeros((frame_count, 33, 4), dtype=np.float32)
    landmarks[:, :, 3] = .9
    landmarks[:, 11:13, 1] = .2
    landmarks[:, 23:25, 1] = .6
    extractor = Mock()
    extractor.extract_from_video.return_value = dict(
        landmarks=landmarks, frame_count=frame_count, fps=fps,
        duration_sec=frame_count / fps,
        annotated_frames=[np.zeros((120, 320, 3), dtype=np.uint8) for _ in range(frame_count)])
    monkeypatch.setattr(app, 'MediaPipePoseExtractor', lambda: extractor)
    monkeypatch.setattr(app.manager, 'predict_window', lambda *args, **kwargs: .8)
    return extractor


def test_outputs_are_isolated_and_timing_uses_sample_rate(monkeypatch):
    fake_extractor(monkeypatch)
    path = str(app.REPO_ROOT / 'demo/sample_activity.mp4')
    first = app.process_video_pipeline(path)
    second = app.process_video_pipeline(path)
    assert first[0] != second[0] and first[1] != second[1]
    assert Path(first[0]).stat().st_size > 0
    assert Path(first[1]).stat().st_size > 0
    assert first[3]['Confirmation Policy Delay'] == '0.62 s'
    cap = cv2.VideoCapture(first[0])
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(8)
        assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 40
    finally:
        cap.release()


def test_short_clip_returns_clear_error(monkeypatch):
    extractor = fake_extractor(monkeypatch, frame_count=5)
    result = app.process_video_pipeline(str(app.REPO_ROOT / 'demo/sample_activity.mp4'))
    assert result[0] is None
    assert result[3]['Status'] == 'Video Too Short'
    extractor.close.assert_called_once()


def test_encoder_failure_is_reported(monkeypatch):
    fake_extractor(monkeypatch)
    writer = Mock()
    writer.isOpened.return_value = False
    monkeypatch.setattr(app.cv2, 'VideoWriter', lambda *args: writer)
    result = app.process_video_pipeline(str(app.REPO_ROOT / 'demo/sample_activity.mp4'))
    assert result[0] is None
    assert result[3]['Status'] == 'Output Failure'
    writer.release.assert_called_once()


def test_result_summary_escapes_values():
    assert '<script>' not in result_summary({'Status': '<script>alert(1)</script>'})
    metrics = {'Decision Status': 'NORMAL ACTIVITY', 'Peak Fall Probability': '<img src=x>',
               'Mean Landmark Visibility': '90%', 'Effective Processing FPS': '20 FPS'}
    summary = result_summary(metrics)
    assert '<img' not in summary
    assert 'NORMAL ACTIVITY' in summary
    assert 'role="status"' in summary


def test_interface_exposes_analysis_endpoint():
    interface = build_interface(Mock(), app.REPO_ROOT)
    config = interface.get_config_file()
    assert config['analytics_enabled'] is False
    analysis = [dep for dep in config['dependencies'] if dep.get('api_name') == 'analyze_video']
    assert len(analysis) == 1
    assert len(analysis[0]['inputs']) == 4
    assert len(analysis[0]['outputs']) == 5


@pytest.mark.parametrize('source_fps,expected_fps', [(5, 5), (24, 12), (30, 10)])
def test_extractor_preserves_sampled_timebase(monkeypatch, source_fps, expected_fps):
    from src.inference import mediapipe_extractor as module
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = source_fps
    capture.read.side_effect = [(True, np.zeros((64, 64, 3), dtype=np.uint8))] * 12 + [(False, None)]
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda *args: capture)
    extractor = module.MediaPipePoseExtractor.__new__(module.MediaPipePoseExtractor)
    extractor.detector = Mock()
    extractor.detector.detect.return_value.pose_landmarks = []
    result = extractor.extract_from_video('fake.mp4', target_fps=10)
    assert result['fps'] == expected_fps
    assert result['duration_sec'] == pytest.approx(result['frame_count'] / expected_fps)
    capture.release.assert_called_once()


def test_extractor_releases_capture_on_detection_error(monkeypatch):
    from src.inference import mediapipe_extractor as module
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 30
    capture.read.return_value = (True, np.zeros((64, 64, 3), dtype=np.uint8))
    monkeypatch.setattr(module.cv2, 'VideoCapture', lambda *args: capture)
    extractor = module.MediaPipePoseExtractor.__new__(module.MediaPipePoseExtractor)
    extractor.detector = Mock()
    extractor.detector.detect.side_effect = RuntimeError('Detection failed')
    with pytest.raises(RuntimeError, match='Detection failed'):
        extractor.extract_from_video('fake.mp4')
    capture.release.assert_called_once()


@pytest.mark.parametrize('probability,visibility,k,expected', [
    (.9, .0, 2, 'UNCERTAIN / LOW TRACKING CONFIDENCE'),
    (.1, .0, 2, 'UNCERTAIN / LOW TRACKING CONFIDENCE'),
    (.9, .9, 2, 'CONFIRMED FALL EVENT'),
    (.9, .9, 4, 'UNCONFIRMED SPIKE (Filtered by K-Window Policy)'),
    (.1, .9, 2, 'NORMAL ACTIVITY'),
    (.4, .9, 2, 'UNCERTAIN / LOW TRACKING CONFIDENCE'),
])
def test_tracking_quality_and_decision_states(monkeypatch, probability, visibility, k, expected):
    extractor = fake_extractor(monkeypatch)
    extractor.extract_from_video.return_value['landmarks'][:, :, 3] = visibility
    monkeypatch.setattr(app.manager, 'predict_window', lambda *args, **kwargs: probability)
    result = app.process_video_pipeline(str(app.REPO_ROOT / 'demo/sample_activity.mp4'), confirmation_k=k)
    assert result[3]['Decision Status'] == expected
    assert result[3]['Safety Verifier Passed'] is True
    if visibility == 0:
        assert result[3]['Windows With Usable Tracking'] == 0
        assert result[3]['Confirmation Policy Delay'].startswith('N/A')
        assert 'Uncertain' in result[2]


def test_real_video_pytorch_backend():
    result = app.process_video_pipeline(str(app.REPO_ROOT / 'demo/sample_activity.mp4'), backend_choice='PyTorch (CPU)')
    assert Path(result[0]).is_file(), result[2]
    assert result[3]['Inference Engine'] == 'PYTORCH (Consumer CPU)'
    assert result[3]['Safety Verifier Passed'] is True


def test_padded_clip_remains_analyzable(monkeypatch):
    fake_extractor(monkeypatch, frame_count=20)
    result = app.process_video_pipeline(str(app.REPO_ROOT / 'demo/sample_activity.mp4'))
    assert Path(result[0]).is_file()
    assert result[3]['Temporal Windows Analyzed'] == 1
    assert result[3]['Decision Status'].startswith('UNCONFIRMED SPIKE')


def test_interface_theme_and_styles():
    from demo.ui import interface_style
    style = interface_style()
    assert 'max-width: 700px' in style['css']
    assert style['theme'].body_text_color_subdued == '#475569'
