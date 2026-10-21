import asyncio
import copy
import json
import time
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import database
from incident_verification import IncidentVerificationService, prepare_evidence_images
from vlm_adapter import VLMAdapter


class Adapter:
    status = 'ONLINE'
    status_detail = 'Test adapter'
    model_name = 'Test VLM'

    def __init__(self, status='NOT_FALL', delay=0):
        self.calls = []
        self.verdict = status
        self.delay = delay

    def verify_video_event(self, candidate, frames):
        self.calls.append((candidate, frames))
        if self.delay: time.sleep(self.delay)
        return {'status':self.verdict,'event_id':candidate['id'],
                'verification_method':'VLM_MODEL','is_true_fall':self.verdict=='CONFIRMED_FALL',
                'scene_observation':'Visible sequence assessed.'}


def test_evidence_keeps_full_scene_when_a_pose_is_lost():
    import cv2
    frame=np.zeros((100,120,3),np.uint8)
    _,encoded=cv2.imencode('.jpg',frame)
    first={'timestamp_sec':0,'jpeg':encoded.tobytes(),'boxes':{'Person 1':[100,100,800,900]}}
    missing={'timestamp_sec':1,'jpeg':encoded.tobytes(),'boxes':{}}
    images=prepare_evidence_images([first,missing],.5)
    assert all(i['frame'].shape==frame.shape for i in images)
    assert [i['relative_sec'] for i in images]==[-.5,.5]


def packet(t, event=False, event_id='fall-1', remaining=10):
    return {'timestamp':t, 'poses':[{'track_id':1,'box':[10,10,90,90]}],
            'stage3_event':{'fall_alert_active':event,'fall_confirmed':False,
                            'fall_hold_remaining_sec':remaining,
                            'active_alert':{'alert_id':event_id,'track_id':'Person 1'} if event else None}}


async def feed_incident(service, event_id='fall-1', pre=True):
    frame = np.zeros((100,120,3),np.uint8)
    if pre:
        for t in (0,.3,.6): service.observe('stream',packet(t),frame)
    result = packet(.9,True,event_id)
    service.observe('stream',result,frame)
    for t in (1.2,1.5,1.8,2.1,2.4): service.observe('stream',packet(t,True,event_id),frame)
    await asyncio.gather(*list(service.tasks))
    return frame


def test_ordered_before_trigger_after_frames_and_exactly_once_dispatch():
    async def run():
        adapter = Adapter()
        saved = []
        service = IncidentVerificationService(adapter,persist=lambda r:saved.append(copy.deepcopy(r)))
        service.ready.set()
        frame = await feed_incident(service)
        assert len(adapter.calls)==1
        candidate,images=adapter.calls[0]
        offsets=[f['relative_sec'] for f in images]
        assert 3<=len(images)<=6 and offsets==sorted(offsets)
        assert offsets[0]<0 and offsets[-1]>0 and 0 in offsets
        assert all(i['frame'].shape == (100,120,3) for i in images)
        assert candidate['track_id']=='Person 1'
        for t in (2.7,3,3.3): service.observe('stream',packet(t,True),frame)
        assert len(adapter.calls)==1 and not service.tasks
        assert saved[-1]['status']=='NOT_FALL'
        assert saved[-1]['evidence_frames_count']==len(images)
        await service.close()
    asyncio.run(run())


def test_negative_verdict_does_not_silence_an_active_pose_alert():
    async def run():
        service=IncidentVerificationService(Adapter(),persist=lambda r:None);service.ready.set()
        frame=await feed_incident(service)
        result=packet(2.7,True,remaining=5)
        service.observe('stream',result,frame)
        assert result['stage3_event']['fall_alert_active']
        result=packet(3,True,remaining=0)
        service.observe('stream',result,frame)
        assert result['stage3_event']['fall_alert_active']
        assert result['verification']['status']=='NOT_FALL'
        await service.close()
    asyncio.run(run())


def test_no_fall_frames_do_not_invoke_the_vlm():
    async def run():
        adapter=Adapter();service=IncidentVerificationService(adapter,persist=lambda r:None)
        for t in (0,.3,.6): service.observe('stream',packet(t),np.zeros((100,120,3),np.uint8))
        assert not service.tasks and not adapter.calls
        await service.close()
    asyncio.run(run())


def test_missing_pre_event_context_is_inconclusive():
    async def run():
        adapter=Adapter();service=IncidentVerificationService(adapter,persist=lambda r:None);service.ready.set()
        await feed_incident(service,pre=False)
        assert service.records['fall-1']['status']=='INCONCLUSIVE' and not adapter.calls
        await service.close()
    asyncio.run(run())


def test_unavailable_model_does_not_fabricate_a_verdict():
    async def run():
        adapter=Adapter();adapter.status='WEIGHTS_MISSING'
        service=IncidentVerificationService(adapter,persist=lambda r:None);service.ready.set()
        await feed_incident(service)
        result=service.records['fall-1']
        assert result['status']=='UNAVAILABLE' and result['is_true_fall'] is None
        assert not adapter.calls
        await service.close()
    asyncio.run(run())


def test_timeout_cannot_later_be_overwritten_by_a_late_positive():
    async def run():
        adapter=Adapter('CONFIRMED_FALL',delay=.06)
        service=IncidentVerificationService(adapter,persist=lambda r:None,model_timeout=.01);service.ready.set()
        await feed_incident(service)
        assert service.records['fall-1']['status']=='TIMEOUT'
        assert service.records['fall-1']['is_true_fall'] is None
        await service.close()
    asyncio.run(run())


def test_queue_is_bounded_and_reports_an_unverified_busy_incident():
    async def run():
        service=IncidentVerificationService(Adapter(),persist=lambda r:None,max_pending=1)
        frame=np.zeros((100,120,3),np.uint8)
        service.observe('stream',packet(0,True,'first'),frame)
        service.observe('stream',packet(.3,True,'second'),frame)
        assert service.records['second']['status']=='BUSY'
        assert len(service.tasks)==1
        await service.close()
    asyncio.run(run())


def test_two_people_get_independent_jobs_and_one_rejection_cannot_clear_the_other():
    async def run():
        adapter=Adapter(); service=IncidentVerificationService(adapter,persist=lambda r:None);service.ready.set()
        frame=np.zeros((100,120,3),np.uint8)
        for t in (0,.3,.6): service.observe('stream',packet(t),frame)
        for t in (.9,1.2,1.5,1.8,2.1,2.4,2.7):
            data=packet(t,True,'fall-2',remaining=0)
            data['stage3_event']['active_incidents']=[{'alert_id':'fall-1','track_id':'Person 1'},{'alert_id':'fall-2','track_id':'Person 2'}]
            service.observe('stream',data,frame)
        await asyncio.gather(*list(service.tasks))
        assert len(adapter.calls)==2
        assert {candidate['track_id'] for candidate,_ in adapter.calls}=={'Person 1','Person 2'}
        service.records['fall-1']['status']='CONFIRMED_FALL'
        data['timestamp']=3
        service.observe('stream',data,frame)
        assert data['stage3_event']['fall_alert_active']
        await service.close()
    asyncio.run(run())


def test_verdict_storage_retains_incident_and_acknowledgement(tmp_path):
    path=str(tmp_path/'audit.db');database.init_db(path)
    database.log_event({'id':'fall-1','track_id':'Person 1','details':'Original pose alert'},path)
    database.acknowledge_event_by_id('fall-1',path)
    record={'event_id':'fall-1','status':'NOT_FALL','scene_observation':'Controlled sitting.'}
    database.save_verification(record,path)
    assert database.get_verification('fall-1',path)==record
    event=database.get_events(db_path=path)[0]
    assert event['details']=='Original pose alert' and event['acknowledged']
    assert event['vlm_verification']['status']=='NOT_FALL'


@pytest.mark.parametrize('output,expected',[
    ('[{"classification":"FALL","scene_observation":"One person collapses."},{"classification":"NO_FALL","scene_observation":"Another person remains standing."}]','CONFIRMED_FALL'),
    ('[{"classification":"NO_FALL","scene_observation":"Controlled squat."}]','NOT_FALL'),
    ('[{"classification":"UNCERTAIN","scene_observation":"Movement is obscured."}]','INCONCLUSIVE'),
    ('[{"classification":"FALL"}]','INCONCLUSIVE'),
    ('{"classification":"FALL","scene_observation":"The person loses balance and falls to the floor."}','CONFIRMED_FALL'),
    ('{"classification":"NO_FALL","scene_observation":"The person squats deliberately."}','NOT_FALL'),
    ('{"classification":"UNCERTAIN","scene_observation":"The onset is obscured."}','INCONCLUSIVE'),
    ('{"classification":"FALL"}','INCONCLUSIVE'),
    ('{"is_true_fall":true,"uncertainty":false,"scene_observation":"Visible loss of balance."}','CONFIRMED_FALL'),
    ('{"is_true_fall":false,"uncertainty":false,"scene_observation":"Controlled sitting."}','NOT_FALL'),
    ('{"is_true_fall":true,"uncertainty":true,"scene_observation":"Onset is obscured."}','INCONCLUSIVE'),
    ('{"is_true_fall":"false","uncertainty":false,"scene_observation":"Controlled sitting."}','INCONCLUSIVE'),
    ('This is not a fall.','INCONCLUSIVE'),
    ('{"is_true_fall":false}','INCONCLUSIVE')])
def test_only_well_typed_model_json_can_verify_or_reject(output,expected):
    adapter=VLMAdapter(auto_load=False);adapter.status='ONLINE'
    adapter.model=MagicMock();adapter.model.device=torch.device('cpu')
    adapter.model.generate.return_value=torch.tensor([[1,2,3]])
    adapter.processor=MagicMock()
    adapter.processor.return_value={'input_ids':torch.tensor([[1,2]])}
    adapter.processor.decode.return_value=output
    result=adapter.verify_video_event({'id':'fall-1'},[np.zeros((30,30,3),np.uint8)]*3)
    assert result['status']==expected and adapter.model.generate.called
    assert (result['is_true_fall'] is None)==(expected=='INCONCLUSIVE')


def test_strict_production_adapter_never_uses_pose_fallback():
    adapter=VLMAdapter(auto_load=False);adapter.status='WEIGHTS_MISSING'
    result=adapter.verify_video_event({'id':'fall-1','kinematics':{'posture':'Lying / Resting','dwell_sec':20}},[np.zeros((30,30,3))]*3)
    assert result['status']=='UNAVAILABLE' and result['is_true_fall'] is None
