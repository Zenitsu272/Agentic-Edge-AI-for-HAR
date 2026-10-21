import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main


def candidate_result():
    return {
        'frame_width': 640, 'frame_height': 480, 'poses': [],
        'stage1_presence': {'person_present': True},
        'stage2_posture': {'posture': 'Lying / Resting'},
        'stage3_event': {'fall_confirmed': True, 'state': 'FALL_CONFIRMED'},
    }


def test_live_pose_mode_never_dispatches_vlm_or_memory():
    with patch.object(main, 'VLM_ENABLED', False), \
         patch.object(main.pipeline, 'new_session', return_value=main.pipeline), \
         patch.object(main.pipeline, 'decode_frame', return_value=np.zeros((480, 640, 3), np.uint8)), \
         patch.object(main.pipeline, 'process_unified', side_effect=lambda *a, **k: candidate_result()), \
         patch.object(main.agent.coordinator, 'should_trigger', side_effect=AssertionError('VLM must be bypassed')), \
         patch.object(main.chroma_memory, 'log_new_event', side_effect=AssertionError('Memory must be bypassed')):
        with TestClient(main.app).websocket_connect('/ws/pipeline') as ws:
            ws.send_text('test frame')
            result = ws.receive_json()
            assert result['central_server']['verification_mode'] == 'POSE_ONLY'
            assert result['stage3_event']['fall_confirmed']


def test_replay_pose_mode_bypasses_vlm(tmp_path):
    import cv2
    path = tmp_path / 'test.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (640, 480))
    writer.write(np.zeros((480, 640, 3), np.uint8))
    writer.release()
    with patch.object(main, 'REPLAYS_DIR', tmp_path), \
         patch.object(main, 'VLM_ENABLED', False), \
         patch.object(main.pipeline, 'new_session', return_value=main.pipeline), \
         patch.object(main.pipeline, 'process_unified', side_effect=lambda *a, **k: candidate_result()), \
         patch.object(main.agent.coordinator, 'should_trigger', side_effect=AssertionError('VLM must be bypassed')):
        with TestClient(main.app).websocket_connect('/ws/replay_stream?video=test.avi') as ws:
            result = ws.receive_json()
            assert result['central_server']['verification_mode'] == 'POSE_ONLY'
            assert result['rendered_frame'].startswith('data:image/jpeg;base64,')
            ws.send_json({'command': 'stop'})
