import copy
from unittest.mock import AsyncMock, patch
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import database
import main
from incident_verification import IncidentVerificationService


@pytest.mark.parametrize('mode',['live','replay'])
def test_camera_and_replay_publish_a_real_verification_job(mode,tmp_path):
    class Adapter:
        status='ONLINE'
        status_detail='Test model'
        model_name='Test VLM'
        calls=0

        def verify_video_event(self,candidate,frames):
            self.calls+=1
            return {'event_id':candidate['id'],'status':'NOT_FALL','is_true_fall':False,
                    'verification_method':'VLM_MODEL','scene_observation':'Controlled movement.'}

    adapter=Adapter()
    service=IncidentVerificationService(adapter)
    service.ready.set()
    frame=np.zeros((100,120,3),np.uint8)
    position=[0]

    def process(*args,**kwargs):
        t=position[0]*.3
        position[0]+=1
        active=t>=.9
        return {'timestamp':t,'poses':[], 'stage1_presence':{},'stage2_posture':{},
                'stage3_event':{'fall_alert_active':active,'fall_confirmed':False,
                'active_alert':{'alert_id':'test-job','track_id':'Person 1'} if active else None}}

    db_path=tmp_path/'audit.db'
    database.init_db(str(db_path))
    video=tmp_path/'test.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),5,(120,100))
    for _ in range(20):writer.write(frame)
    writer.release()
    with patch.object(database,'DB_PATH',db_path), patch.object(main,'VLM_ENABLED',True), \
         patch.object(main,'verification_service',service),patch.object(service,'start',new=AsyncMock()), \
         patch.object(main,'REPLAYS_DIR',tmp_path), \
         patch.object(main.pipeline,'new_session',return_value=main.pipeline), \
         patch.object(main.pipeline,'decode_frame',return_value=frame), \
         patch.object(main.pipeline,'process_unified',side_effect=process):
        with TestClient(main.app) as client:
            endpoint='/ws/pipeline' if mode=='live' else '/ws/replay_stream?video=test.avi'
            with client.websocket_connect(endpoint) as ws:
                results=[]
                for _ in range(12):
                    if mode=='live':ws.send_text('test image')
                    results.append(ws.receive_json())
                assert any(r['verification']['status']=='COLLECTING' for r in results)
                deadline=time.monotonic()+3
                while time.monotonic()<deadline:
                    saved=client.get('/api/verification/test-job')
                    if saved.status_code==200 and saved.json()['verification']['status']=='NOT_FALL':break
                    time.sleep(.01)
                assert saved.json()['verification']['status']=='NOT_FALL'
                assert saved.json()['verification']['evidence_frames_count']>=3
                assert adapter.calls==1
                if mode=='replay':ws.send_json({'command':'stop'})
