from pose_activity import FallAlertLatch, PersonState


def test_alert_survives_recovery_and_disappearance_for_ten_seconds():
    latch = FallAlertLatch()
    event = {'alert_id': 'fall-1', 'details': 'original incident'}
    assert latch.update(True, event, now=100)['active']
    assert latch.update(False, now=100.1)['held']
    last = latch.update(False, now=109.99)
    assert last['active'] and last['event'] == event
    assert not latch.update(False, now=110)['active']


def test_same_incident_does_not_keep_restarting_hold():
    latch = FallAlertLatch()
    event = {'alert_id': 'fall-1'}
    latch.update(True, event, now=10)
    latch.update(True, event, now=19)
    assert not latch.update(False, now=20)['active']


def test_ongoing_fall_remains_visible_after_hold_period():
    latch = FallAlertLatch()
    event = {'alert_id': 'fall-1'}
    latch.update(True, event, now=10)
    assert latch.update(True, event, now=25)['active']


def test_new_incident_restarts_hold():
    latch = FallAlertLatch()
    latch.update(True, {'alert_id': 'fall-1'}, now=0)
    latch.update(True, {'alert_id': 'fall-2'}, now=8)
    assert latch.update(False, now=17.9)['active']
    assert not latch.update(False, now=18)['active']


def test_confirmation_updates_incident_without_restarting_ten_second_hold():
    latch = FallAlertLatch()
    latch.update(True, {'alert_id':'fall-1','detector_status':'PARTIAL_FALL'}, now=0)
    result = latch.update(True, {'alert_id':'fall-1','detector_status':'FALL_CONFIRMED'}, now=2)
    assert result['remaining_sec'] == 8
    assert result['event']['detector_status'] == 'FALL_CONFIRMED'
    assert not latch.update(False, now=10)['active']


def test_small_or_moderate_drop_cannot_start_fall_candidate():
    # Isolate the stricter descent gate: all other fall evidence is strong.
    state = PersonState(1, [0, 0, 100, 300], 0)
    upright = dict(category='Standing', reliable=True, provisional=False,
                   torso_angle=0, shoulder_y=100, hip_y=200, scale=100, down=False)
    state.update(upright, 0)
    state.update(upright, .15)
    down = dict(upright, category='Lying', torso_angle=80, down=True,
                shoulder_y=165, hip_y=240)
    assert state.update(down, .4)[2] == 'PARTIAL_FALL'
    assert state.candidate_since is None
    assert not state.confirmed


def test_moderate_rotation_cannot_start_fall_candidate():
    state = PersonState(1, [0, 0, 100, 300], 0)
    upright = dict(category='Standing', reliable=True, provisional=False,
                   torso_angle=25, shoulder_y=100, hip_y=200, scale=100, down=False)
    state.update(upright, 0)
    state.update(upright, .15)
    down = dict(upright, category='Lying', torso_angle=72, down=True,
                shoulder_y=200, hip_y=250)
    assert state.update(down, .4)[2] == 'PARTIAL_FALL'
    assert state.candidate_since is None and not state.confirmed
