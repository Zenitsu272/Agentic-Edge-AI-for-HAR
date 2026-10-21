from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from ultralytics.engine.results import Results

from inference import InferencePipeline
from pose_activity import PersonState


def upright(category='Standing'):
    return dict(category=category, reliable=True, provisional=False, torso_angle=0,
                shoulder_y=100, hip_y=200, scale=100, down=False, shoulders_visible=True)


@pytest.mark.parametrize('initial', ['Standing', 'Sitting'])
def test_sudden_partial_collapse_emits_before_confirmation(initial):
    state = PersonState(1, [0, 0, 100, 300], 0)
    state.update(upright(initial), 0)
    state.update(upright(initial), .15)
    partial = dict(upright(), category='Unknown', torso_angle=55,
                   shoulder_y=180, hip_y=230, down=False)
    assert state.update(partial, .4)[2] == 'PARTIAL_FALL'
    assert not state.confirmed and state.candidate_since is None


def test_abrupt_shoulder_descent_with_lost_hips_is_a_partial_signal():
    state = PersonState(1, [0, 0, 100, 300], 0)
    state.update(upright(), 0)
    state.update(upright(), .15)
    partial = dict(upright(), category='Partial', reliable=False,
                   shoulder_y=210, hip_y=None, torso_angle=None)
    assert state.update(partial, .4)[2] == 'PARTIAL_FALL'


def test_occlusion_without_descent_does_not_alert():
    state = PersonState(1, [0, 0, 100, 300], 0)
    state.update(upright(), 0)
    state.update(upright(), .15)
    partial = dict(upright(), category='Partial', reliable=False,
                   hip_y=None, torso_angle=None)
    assert state.update(partial, .4)[2] == 'STABLE'


def test_seated_collapse_can_alert_while_hips_remain_at_chair_height():
    state = PersonState(1, [0, 0, 100, 300], 0)
    state.update(upright('Sitting'), 0)
    state.update(upright('Sitting'), .15)
    partial = dict(upright(), category='Unknown', torso_angle=60,
                   shoulder_y=180, hip_y=200)
    assert state.update(partial, .4)[2] == 'PARTIAL_FALL'
    assert not state.confirmed


def test_forward_collapse_can_alert_without_horizontal_torso_rotation():
    state = PersonState(1, [0, 0, 100, 300], 0)
    state.update(upright(), 0)
    state.update(upright(), .15)
    partial = dict(upright(), category='Unknown', torso_angle=20,
                   shoulder_y=180, hip_y=215, scale=40)
    assert state.update(partial, .4)[2] == 'PARTIAL_FALL'
    assert not state.confirmed


def test_ordinary_fast_sitting_keeps_torso_upright_and_does_not_alert():
    state = PersonState(1, [0, 0, 100, 300], 0)
    state.update(upright(), 0)
    state.update(upright(), .15)
    seated = dict(upright('Sitting'), shoulder_y=180, hip_y=280)
    assert state.update(seated, .4)[2] == 'STABLE'


def test_pipeline_holds_a_single_partial_signal_after_recovery_and_disappearance():
    # Fake only model landmarks; run actual geometry, tracking, incident and latch.
    frame = np.zeros((600, 800, 3), dtype=np.uint8)
    points = np.zeros((17, 3), dtype=np.float32)
    points[:, 2] = .95
    coords = [(200,65),(190,60),(210,60),(180,65),(220,65),
              (180,100),(220,100),(165,160),(235,160),(160,210),(240,210),
              (185,200),(215,200),(185,300),(215,300),(185,400),(215,400)]
    points[:, :2] = coords
    partial = points.copy()
    partial[5:7, :2] = [(120,180),(160,180)]
    partial[11:13, :2] = [(180,220),(220,220)]

    class Model:
        current = points

        def predict(self, *args, **kwargs):
            if self.current is None:
                return []
            return [Results(frame, path='', names={0:'person'},
                            boxes=torch.tensor([[100,40,300,420,.95,0]], dtype=torch.float32),
                            keypoints=torch.from_numpy(self.current[None]))]

    pipeline = object.__new__(InferencePipeline)
    pipeline.pose_model = Model()
    pipeline.device = 'cpu'
    pipeline.model_path = Path('test-pose.pt')
    pipeline.imgsz = 640
    pipeline.reset()
    with patch('inference.database.log_event'), patch('pose_activity.time.monotonic') as clock:
        clock.return_value = 100
        pipeline.process_unified(frame, timestamp=0)
        pipeline.process_unified(frame, timestamp=.15)
        pipeline.pose_model.current = partial
        result = pipeline.process_unified(frame, timestamp=.4)
        assert result['stage3_event']['fall_alert_active']
        assert result['stage3_event']['fall_detection_level'] == 'possible_partial'
        assert not result['stage3_event']['fall_confirmed']
        assert result['stage3_event']['active_alert']['detector_status'] == 'PARTIAL_FALL'
        pipeline.pose_model.current = points
        clock.return_value = 105
        assert pipeline.process_unified(frame, timestamp=.5)['stage3_event']['fall_alert_active']
        pipeline.pose_model.current = None
        clock.return_value = 109.9
        result = pipeline.process_unified(frame, timestamp=.6)
        assert result['stage3_event']['fall_alert_active'] and not result['poses']
        clock.return_value = 110
        assert not pipeline.process_unified(frame, timestamp=.7)['stage3_event']['fall_alert_active']
