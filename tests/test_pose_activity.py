import numpy as np
import pytest
from pose_activity import analyze_pose, PoseActivityTracker


def pose(kind='Standing', scale=1., shift=(0,0)):
    p=np.zeros((17,3)); p[:,2]=.95
    coords={0:(200,65),1:(190,60),2:(210,60),3:(180,65),4:(220,65),
            5:(180,100),6:(220,100),7:(165,160),8:(235,160),9:(160,210),10:(240,210),
            11:(185,200),12:(215,200),13:(185,300),14:(215,300),15:(185,400),16:(215,400)}
    if kind=='Sitting':
        coords.update({13:(230,230),14:(260,230),15:(230,330),16:(260,330)})
    if kind=='Lying':
        coords.update({5:(180,250),6:(180,275),11:(280,260),12:(280,285),13:(370,270),14:(370,295),15:(450,275),16:(450,300)})
    for i,xy in coords.items(): p[i,:2]=np.array(xy)*scale+shift
    return p


def features(p, shape=(600,800)):
    return analyze_pose(p,[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())],shape)


def detection(p, shape=(600,800)):
    box=[float(p[:,0].min()-10),float(p[:,1].min()-10),float(p[:,0].max()+10),float(p[:,1].max()+10)]
    return dict(keypoints=p.tolist(),box=box,features=analyze_pose(p,box,shape),detection_confidence=.9)


@pytest.mark.parametrize('kind',['Standing','Sitting'])
@pytest.mark.parametrize('scale',[.5,1,1.8])
@pytest.mark.parametrize('mirror',[False,True])
def test_scale_and_mirror_invariance(kind,scale,mirror):
    p=pose(kind,scale, (30,20))
    if mirror: p[:,0]=1000-p[:,0]
    assert features(p,(1200,1200))['category']==kind


def test_missing_ankles_are_not_sitting_evidence():
    p=pose(); p[15:,2]=.05
    assert features(p)['category']=='Standing'


def test_cropped_standing_is_provisional_not_sitting():
    p=pose(); p[13:,2]=.05
    f=features(p,(260,800))
    assert f['category']=='Standing' and f['provisional']


def test_missing_lower_body_without_boundary_evidence_is_uncertain():
    p=pose(); p[13:,2]=0
    assert features(p)['category']=='Unknown'


def test_sitting_with_occluded_ankles():
    p=pose('Sitting'); p[15:,2]=0
    assert features(p)['category']=='Sitting'


def test_raised_arms_do_not_change_standing():
    p=pose(); p[7:11,1]=20
    assert features(p)['category']=='Standing'


def test_bending_with_extended_legs_is_not_down():
    p=pose(); p[5:7,0]-=90; p[5:7,1]+=65
    f=features(p)
    assert not f['down'] and f['category']=='Standing'


def test_low_confidence_zero_and_nan_points_are_not_used():
    p=pose(); p[13:,2]=.1; p[13,:2]=np.nan; p[14,:2]=0
    assert features(p)['category']=='Unknown'


def test_lone_arm_is_rejected():
    p=pose(); p[:,2]=0; p[5:10:2,2]=.9
    assert not features(p)['usable']


def run_sequence(sequence):
    tracker=PoseActivityTracker(); out=[]
    for t,p in sequence: out.append(tracker.update([detection(p)],t,(600,800))[0])
    return out


def test_rapid_collapse_requires_persistent_down_pose():
    seq=[(i*.1,pose()) for i in range(10)]+[(1+i*.1,pose('Lying')) for i in range(20)]
    out=run_sequence(seq)
    assert out[10]['state']=='VERIFYING'
    assert out[12]['category']!='Fall'
    assert out[-1]['category']=='Fall'


def test_lying_at_start_is_not_a_fall():
    out=run_sequence([(i*.1,pose('Lying')) for i in range(30)])
    assert all(p['category']!='Fall' for p in out)


def test_sitting_transition_is_not_a_fall():
    out=run_sequence([(i*.1,pose()) for i in range(10)]+[(1+i*.1,pose('Sitting')) for i in range(20)])
    assert out[-1]['category']=='Sitting'
    assert all(p['state']=='STABLE' for p in out)


def test_deliberate_slow_lowering_is_not_a_fall():
    upright,down=pose(),pose('Lying')
    seq=[(i*.1,upright) for i in range(10)]
    seq += [(1+i*.1, upright+(down-upright)*min(i/50,1)) for i in range(65)]
    assert all(p['category']!='Fall' for p in run_sequence(seq))


def test_brief_horizontal_pose_with_recovery_is_not_a_fall():
    seq=[(i*.1,pose()) for i in range(10)]+[(1+i*.1,pose('Lying')) for i in range(3)]+[(1.3+i*.1,pose()) for i in range(10)]
    assert all(p['category']!='Fall' for p in run_sequence(seq))


def test_no_confirmation_during_observation_gap():
    out=run_sequence([(0,pose()),(.2,pose()),(.4,pose('Lying')),(2,pose('Lying')),(2.2,pose('Lying'))])
    assert all(p['category']!='Fall' for p in out)


def test_identity_is_independent_of_detection_order():
    tr=PoseActivityTracker()
    standing=detection(pose()); seated=detection(pose('Sitting',shift=(400,0)),(600,1200))
    a=tr.update([standing,seated],0,(600,1200))
    b=tr.update([seated,standing],.1,(600,1200))
    assert a[0]['track_id']==b[1]['track_id'] and a[1]['track_id']==b[0]['track_id']
    assert b[0]['category']=='Sitting' and b[1]['category']=='Standing'


def test_replay_loop_clears_temporal_history():
    tr=PoseActivityTracker()
    tr.update([detection(pose())],5,(600,800))
    out=tr.update([detection(pose('Lying'))],0,(600,800))
    assert out[0]['state']=='STABLE'


def test_user_screenshot_regressions():
    import json
    from pathlib import Path
    rows=json.loads((Path(__file__).resolve().parents[1]/'data/pose_regressions.json').read_text())
    for row in rows:
        f=analyze_pose(row['keypoints'],row['box'],row['frame_shape'])
        assert f['category']==row['expected']
        assert f['provisional']==row['provisional']


@pytest.mark.parametrize('kind',['Standing','Sitting'])
def test_small_landmark_jitter_does_not_change_class(kind):
    rng=np.random.default_rng(123)
    for _ in range(60):
        p=pose(kind);p[:,:2]+=rng.normal(0,3,(17,2))
        assert features(p)['category']==kind


def test_two_people_do_not_share_a_fall_history():
    tr=PoseActivityTracker()
    for i in range(10):
        out=tr.update([detection(pose()),detection(pose('Sitting',shift=(500,0)),(600,1300))],i*.1,(600,1300))
    for i in range(20):
        out=tr.update([detection(pose('Lying')),detection(pose('Sitting',shift=(500,0)),(600,1300))],1+i*.1,(600,1300))
    assert out[0]['category']=='Fall' and out[1]['category']=='Sitting'


def test_recovery_clears_fall_after_stable_upright_pose():
    seq=[(i*.1,pose()) for i in range(10)]+[(1+i*.1,pose('Lying')) for i in range(20)]+[(3+i*.1,pose()) for i in range(10)]
    out=run_sequence(seq)
    assert out[24]['category']=='Fall' and out[-1]['category']=='Standing'
