import time
from unittest.mock import AsyncMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import database
import main
from incident_verification import IncidentVerificationService


def video_bytes(tmp_path, frames=4, fps=10):
    path=tmp_path/'source.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),fps,(120,100))
    assert writer.isOpened()
    for _ in range(frames):writer.write(np.zeros((100,120,3),np.uint8))
    writer.release()
    return path.read_bytes()


def test_upload_validates_video_and_preserves_files_with_the_same_name(tmp_path):
    data=video_bytes(tmp_path)
    existing=tmp_path/'same.avi';existing.write_bytes(b'existing recording')
    with patch.object(main,'REPLAYS_DIR',tmp_path):
        client=TestClient(main.app)
        first=client.post('/api/upload_video',files={'file':('same.avi',data,'video/x-msvideo')})
        second=client.post('/api/upload_video',files={'file':('same.avi',data,'video/x-msvideo')})
        assert first.status_code==second.status_code==200
        assert first.json()['filename']!=second.json()['filename']
        assert first.json()['display_name']=='same.avi'
        assert first.json()['total_frames']==4
        assert first.json()['duration_sec']==.4
        assert existing.read_bytes()==b'existing recording'
        client.close()


@pytest.mark.parametrize('name',['not-video.txt','corrupt.mp4'])
def test_bad_upload_is_rejected_without_leaving_a_file(tmp_path,name):
    with patch.object(main,'REPLAYS_DIR',tmp_path):
        client=TestClient(main.app)
        response=client.post('/api/upload_video',files={'file':(name,b'not a video','application/octet-stream')})
        assert response.status_code==400
        assert list(tmp_path.iterdir())==[]
        client.close()


def packet(timestamp, active=False):
    return {'timestamp':timestamp,'poses':[],'stage1_presence':{},'stage2_posture':{},
            'stage3_event':{'fall_alert_active':active,'fall_confirmed':False,
                            'active_alert':{'alert_id':'upload-incident','track_id':'Person 1'} if active else None}}


@pytest.mark.parametrize('active',[False,True])
def test_uploaded_video_checks_each_frame_once_and_completes_normally(tmp_path,active):
    (tmp_path/'check.avi').write_bytes(video_bytes(tmp_path))
    def process(frame,timestamp):return packet(timestamp,active)
    with patch.object(main,'REPLAYS_DIR',tmp_path),patch.object(main,'VLM_ENABLED',False), \
         patch.object(main.pipeline,'new_session',return_value=main.pipeline), \
         patch.object(main.pipeline,'process_unified',side_effect=process) as inference:
        with TestClient(main.app).websocket_connect('/ws/replay_stream?video=check.avi&loop=false') as ws:
            results=[ws.receive_json() for _ in range(4)]
            complete=ws.receive_json()
            assert [r['video_progress']['processed_frames'] for r in results]==[1,2,3,4]
            assert results[-1]['video_progress']['percent']==100
            assert complete['type']=='video_complete' and complete['processed_frames']==4
            assert inference.call_count==4
            if active:assert complete['incidents'][0]['status']=='DISABLED'
            else:assert complete['incidents']==[]
            with pytest.raises(WebSocketDisconnect):ws.receive_json()


def test_vlm_finishes_and_can_be_polled_after_uploaded_video_ends(tmp_path):
    class Adapter:
        status='ONLINE'
        status_detail='Test adapter'
        model_name='Test VLM'
        calls=0
        def verify_video_event(self,candidate,frames):
            self.calls+=1
            time.sleep(.05)
            return {'status':'NOT_FALL','event_id':candidate['id'],'is_true_fall':False,
                    'verification_method':'VLM_MODEL','scene_observation':'Controlled sitting.'}
    adapter=Adapter();service=IncidentVerificationService(adapter);service.ready.set()
    (tmp_path/'check.avi').write_bytes(video_bytes(tmp_path,frames=13))
    path=tmp_path/'audit.db';database.init_db(str(path))
    def process(frame,timestamp):return packet(timestamp,timestamp>=.6)
    with patch.object(database,'DB_PATH',path),patch.object(main,'REPLAYS_DIR',tmp_path), \
         patch.object(main,'VLM_ENABLED',True),patch.object(main,'verification_service',service), \
         patch.object(service,'start',new=AsyncMock()), \
         patch.object(main.pipeline,'new_session',return_value=main.pipeline), \
         patch.object(main.pipeline,'process_unified',side_effect=process):
        with TestClient(main.app) as client:
            with client.websocket_connect('/ws/replay_stream?video=check.avi&loop=false') as ws:
                results=[ws.receive_json() for _ in range(13)]
                complete=ws.receive_json()
                assert complete['type']=='video_complete'
                assert complete['incidents'][0]['event_id']=='upload-incident'
            deadline=time.monotonic()+3
            while time.monotonic()<deadline:
                saved=client.get('/api/verification/upload-incident').json()['verification']
                if saved['status']=='NOT_FALL':break
                time.sleep(.01)
            assert saved['status']=='NOT_FALL' and adapter.calls==1
