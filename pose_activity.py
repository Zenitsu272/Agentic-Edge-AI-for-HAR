"""Confidence-aware pose geometry and per-person temporal activity classification.

The scores are rule support, not calibrated model probabilities. Fall is a
transition observed over time; a horizontal first frame is not a fall event.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import time
import numpy as np

VERSION = 'pose-temporal-v2.2'
KEYPOINT_MIN = 0.35
FALL_MIN_TILT_DEG = 70
FALL_MIN_ROTATION_DEG = 50
FALL_MIN_DROP_TORSOS = 0.70
FALL_MIN_SPEED_TORSOS_PER_SEC = 1.40
FALL_MIN_HIP_DROP_TORSOS = 0.10
FALL_CONFIRM_SECONDS = 1.20
FALL_CONFIRM_OBSERVATIONS = 6
FALL_ALERT_HOLD_SECONDS = 10.0
# Early warning gates; the stricter confirmation gates above remain unchanged.
PARTIAL_FALL_MIN_TILT_DEG = 45
PARTIAL_FALL_MIN_ROTATION_DEG = 25
PARTIAL_FALL_MIN_DROP_TORSOS = 0.55
PARTIAL_FALL_MIN_SPEED = 1.40
OCCLUDED_FALL_MIN_DROP_TORSOS = 0.90
OCCLUDED_FALL_MIN_SPEED = 1.80


class FallAlertLatch:
    """Keep a detected incident visible using elapsed real time, including replay.

    Holding an alert does not change the person's current pose or create another
    incident. A new incident restarts the hold; ordinary frames do not.
    """
    def __init__(self):
        self.until = 0.0
        self.detected_before = False
        self.event = None

    def update(self, detected, event=None, now=None):
        now = time.monotonic() if now is None else now
        new_id = event.get('alert_id') if event else None
        old_id = self.event.get('alert_id') if self.event else None
        if detected and (not self.detected_before or new_id != old_id):
            self.until = now + FALL_ALERT_HOLD_SECONDS
        if detected:
            # Update a partial incident to confirmed without restarting its timer.
            self.event = dict(event) if event else None
        self.detected_before = detected
        remaining = max(0.0, self.until - now)
        return {'active': bool(detected or remaining > 0),
                'held': bool(not detected and remaining > 0),
                'remaining_sec': round(remaining, 1),
                'event': self.event}


def angle(a, b, c):
    u, v = a - b, c - b
    den = float(np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / den, -1, 1)))) if den > 1e-6 else None


def analyze_pose(keypoints, box, frame_shape):
    """Scale-independent geometry. Missing joints are never invented or zero-filled."""
    pts = np.asarray(keypoints, dtype=float)
    h, w = frame_shape[:2]
    valid = (pts[:, 2] >= KEYPOINT_MIN) & np.isfinite(pts).all(axis=1)
    valid &= (pts[:, 0] > 1) & (pts[:, 0] < w - 1) & (pts[:, 1] > 1) & (pts[:, 1] < h - 1)
    def center(indices):
        ids = [i for i in indices if valid[i]]
        return np.average(pts[ids, :2], axis=0, weights=pts[ids, 2]) if ids else None
    shoulders, hips = center([5, 6]), center([11, 12])
    visible = int(valid.sum())
    body_visible = int(valid[5:].sum())
    result = dict(category='Unknown', posture='Insufficient view', score=0., provisional=True,
                  reason='Show shoulders, hips and knees to classify posture.', visible_keypoints=visible,
                  lower_body_visible=int(valid[11:].sum()), torso_angle=None, knee_angle=None,
                  thigh_vertical=None, shoulder_y=None, hip_y=None, scale=None,
                  down=False, reliable=False, valid=valid.tolist(),
                  shoulders_visible=bool(valid[5] and valid[6]))
    if shoulders is not None:
        result['shoulder_y'] = float(shoulders[1])
    # Reject clothing/limbs hallucinated as people; still keep genuine close-up faces.
    result['usable'] = bool((shoulders is not None and hips is not None and body_visible >= 3)
                            or (shoulders is not None and visible >= 5))
    if shoulders is None or hips is None:
        result.update(category='Partial', posture='Partial view', reason='Lower body is not visible; activity is uncertain.')
        return result
    torso = hips - shoulders
    scale = float(np.linalg.norm(torso))
    if scale < max(8, min(h, w) * .018):
        result['usable'] = False
        return result
    tilt = math.degrees(math.atan2(abs(torso[0]), max(abs(torso[1]), 1e-6)))
    thighs, bends, thigh_tilts = [], [], []
    for hip, knee, ankle in [(11, 13, 15), (12, 14, 16)]:
        if valid[hip] and valid[knee]:
            vec = pts[knee, :2] - pts[hip, :2]
            thighs.append(float(vec[1] / scale))
            thigh_tilts.append(math.degrees(math.atan2(abs(vec[0]), max(abs(vec[1]), 1e-6))))
            if valid[ankle]:
                bend = angle(pts[hip, :2], pts[knee, :2], pts[ankle, :2])
                if bend is not None: bends.append(bend)
    thigh = float(np.median(thighs)) if thighs else None
    bend = float(np.median(bends)) if bends else None
    result.update(torso_angle=round(tilt, 1), knee_angle=round(bend, 1) if bend is not None else None,
                  thigh_vertical=round(thigh, 3) if thigh is not None else None,
                  shoulder_y=float(shoulders[1]), hip_y=float(hips[1]), scale=scale, reliable=True)
    # Extended vertical thighs contradict a horizontal-body inference (e.g. bending).
    legs_vertical = thigh is not None and thigh > .55 and min(thigh_tilts) < 35
    horizontal = tilt >= 62 and not legs_vertical
    if horizontal:
        result.update(category='Lying', posture='Lying / resting', score=.8, provisional=False,
                      down=True, reason='Horizontal torso. No fall is inferred without a rapid transition.')
    elif legs_vertical and (tilt < 48 or (bend is not None and bend > 145)):
        result.update(category='Standing', posture='Standing', score=.9 if bends else .78,
                      provisional=False, reason='Hips above knees; thighs extended vertically.')
    elif thigh is not None and -.3 <= thigh < .58 and tilt < 62:
        # A shortened projected thigh is useful for front-facing seated people.
        # Unlike the old algorithm, absence of ankles contributes no seated evidence.
        supported_bend = bend is not None and bend < 150
        shortened_thigh = thigh < .42
        if supported_bend or shortened_thigh or np.median(thigh_tilts) > 48:
            result.update(category='Sitting', posture='Sitting', score=.88 if supported_bend else .75,
                          provisional=False, reason='Bent or shortened thighs with knees near hip level.')
        else:
            result.update(posture='Changing posture', reason='Leg geometry is between seated and upright; waiting for a stable pose.')
    elif not thighs and tilt < 28 and hips[1] > shoulders[1] and (h - hips[1]) / scale < .75:
        # A torso extending through the lower image boundary is consistent with standing,
        # but seated people can produce the same 2D crop. Explicitly mark limited evidence.
        result.update(category='Standing', posture='Standing', score=.55, provisional=True,
                      reason='Upright torso; legs leave the frame. Standing is provisional until knees are visible.')
    elif legs_vertical:
        result.update(category='Standing', posture='Standing', score=.68, provisional=True,
                      reason='Legs remain extended while the torso bends or reaches.')
    return result


@dataclass
class PersonState:
    track_id: int
    box: list
    last_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=90))
    votes: deque = field(default_factory=lambda: deque(maxlen=12))
    category: str = 'Unknown'
    candidate_since: float | None = None
    down_frames: int = 0
    confirmed: bool = False
    recovery_since: float | None = None
    event_id: str | None = None
    last_logged_category: str | None = None
    drop_speed: float = 0.
    fall_signal_until: float = -1.
    fall_episode: int = 0
    last_logged_fall: tuple | None = None

    def update(self, features, now):
        gap = now - self.last_seen
        if gap > .65 or gap < 0:
            self.history.clear(); self.votes.clear()
            self.candidate_since = None; self.down_frames = 0; self.confirmed = False
            self.recovery_since = None; self.event_id = None; self.category = 'Unknown'
            self.fall_signal_until = -1.
        self.last_seen = now
        raw = features['category']
        self.votes.append((now, raw))
        while self.votes and now - self.votes[0][0] > .6: self.votes.popleft()
        # Do not hide missing evidence behind a stale confident label.
        if raw in ('Unknown', 'Partial'):
            stable = raw
        else:
            support = sum(c == raw for _, c in self.votes) / len(self.votes)
            stable = raw if support >= .6 or self.category in ('Unknown', 'Partial') else self.category
        self.category = stable
        self.drop_speed = 0.
        baselines = [(t, f) for t, f in self.history if .12 <= now - t <= 1.1
                     and f['category'] in ('Standing', 'Sitting')
                     and f['torso_angle'] < 45 and not f['provisional']]
        if not baselines:
            baselines = [(t, f) for t, f in self.history if .12 <= now - t <= 1.1
                         and f['category'] == 'Standing' and f['torso_angle'] < 28]
        rapid = False
        partial_rapid = False
        if len(baselines) < 2:
            baselines = []
        for t, old in baselines:
            if features.get('shoulder_y') is None:
                continue
            dt = now - t
            scale = old['scale']
            shoulder_drop = (features['shoulder_y'] - old['shoulder_y']) / scale
            speed = shoulder_drop / dt
            self.drop_speed = max(self.drop_speed, speed)
            if features['reliable']:
                hip_drop = (features['hip_y'] - old['hip_y']) / scale
                rotation = features['torso_angle'] - old['torso_angle']
                seated_collapse = bool(
                    old['category'] == 'Sitting' and hip_drop > -.15
                    and rotation > 40 and shoulder_drop > .70 and speed > 1.80)
                partial_rapid |= bool(
                    features['torso_angle'] >= PARTIAL_FALL_MIN_TILT_DEG
                    and rotation > PARTIAL_FALL_MIN_ROTATION_DEG
                    and shoulder_drop > PARTIAL_FALL_MIN_DROP_TORSOS
                    and speed > PARTIAL_FALL_MIN_SPEED
                    and (hip_drop > FALL_MIN_HIP_DROP_TORSOS or seated_collapse))
                # Forward/backward collapse may foreshorten an upright-looking
                # torso instead of rotating it horizontally in the image.
                partial_rapid |= bool(
                    features['scale'] / scale < .55
                    and shoulder_drop > .70 and speed > 1.80
                    and hip_drop > -.15)
                rapid |= bool(
                    features['down'] and features['torso_angle'] >= FALL_MIN_TILT_DEG
                    and rotation > FALL_MIN_ROTATION_DEG
                    and shoulder_drop > FALL_MIN_DROP_TORSOS
                    and speed > FALL_MIN_SPEED_TORSOS_PER_SEC
                    and hip_drop > FALL_MIN_HIP_DROP_TORSOS)
            else:
                # Abrupt torso loss can be an incomplete fall. Occlusion alone
                # cannot trigger: both shoulders must visibly descend rapidly.
                partial_rapid |= bool(
                    features.get('shoulders_visible')
                    and shoulder_drop > OCCLUDED_FALL_MIN_DROP_TORSOS
                    and speed > OCCLUDED_FALL_MIN_SPEED)
        if rapid or partial_rapid:
            if now >= self.fall_signal_until and self.candidate_since is None and not self.confirmed:
                self.fall_episode += 1
            self.fall_signal_until = now + .45
        if features['reliable']:
            if rapid and self.candidate_since is None and not self.confirmed:
                self.candidate_since = now
                self.down_frames = 0
            if self.candidate_since is not None and not self.confirmed:
                if features['down'] and features['torso_angle'] >= FALL_MIN_TILT_DEG:
                    self.down_frames += 1
                    if (now - self.candidate_since >= FALL_CONFIRM_SECONDS
                            and self.down_frames >= FALL_CONFIRM_OBSERVATIONS):
                        self.confirmed = True
                else:
                    self.candidate_since = None; self.down_frames = 0
            if self.confirmed:
                if raw in ('Standing', 'Sitting') and not features['down']:
                    if self.recovery_since is None: self.recovery_since = now
                    if now - self.recovery_since >= .5:
                        self.confirmed = False; self.candidate_since = None; self.down_frames = 0; self.event_id = None
                else: self.recovery_since = None
            self.history.append((now, features.copy()))
        else:
            # An occluded body cannot confirm a fall by merely waiting out a timer.
            self.candidate_since = None; self.down_frames = 0
        if self.confirmed:
            return 'Fall', 'Fall detected', 'FALL_CONFIRMED'
        if self.candidate_since is not None:
            return 'Unknown', 'Possible fall — checking', 'VERIFYING'
        if now < self.fall_signal_until:
            return 'Unknown', 'Partial fall detected', 'PARTIAL_FALL'
        return stable, {'Partial':'Partial view', 'Unknown':'Uncertain', 'Lying':'Lying / resting'}.get(stable, stable), 'STABLE'


class PoseActivityTracker:
    """One-to-one spatial association and independent temporal state per person."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.last_time = None
        self.frame_shape = None

    def reset(self):
        self.__init__()

    def update(self, detections, now, shape):
        if self.last_time is not None and (now < self.last_time or now - self.last_time > 2 or self.frame_shape != tuple(shape[:2])):
            self.reset()
        self.last_time = now; self.frame_shape = tuple(shape[:2])
        self.tracks = {i:t for i,t in self.tracks.items() if now - t.last_seen < 1.2}
        pairs = []
        for tid, track in self.tracks.items():
            a = np.array(track.box)
            for i, d in enumerate(detections):
                b = np.array(d['box'])
                extent = max(np.linalg.norm(a[2:] - a[:2]), np.linalg.norm(b[2:] - b[:2]), 1)
                distance = np.linalg.norm((a[:2]+a[2:] - b[:2]-b[2:])/2) / extent
                inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
                union = np.maximum(0, a[2:]-a[:2]).prod() + np.maximum(0,b[2:]-b[:2]).prod() - inter
                iou = inter / max(union, 1)
                if distance < .65: pairs.append((distance + .35*(1-iou), tid, i))
        used_tracks, assignments = set(), {}
        for _, tid, i in sorted(pairs):
            if tid not in used_tracks and i not in assignments:
                assignments[i] = tid; used_tracks.add(tid)
        output = []
        for i, detection in enumerate(detections):
            tid = assignments.get(i)
            if tid is None:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = PersonState(tid, detection['box'], now)
            track = self.tracks[tid]
            previous = track.box
            dt = now - track.last_seen
            displacement = np.linalg.norm((np.array(previous[:2])+previous[2:] - np.array(detection['box'][:2])-detection['box'][2:])/2)
            movement = 'Moving' if dt > .01 and displacement / max(detection['features']['scale'] or 100, 1) / dt > .18 else 'Stationary'
            track.box = detection['box']
            category, label, state = track.update(detection['features'], now)
            output.append(dict(detection, track_id=tid, category=category, posture=label, state=state,
                               movement=movement, evidence_score=round(detection['features']['score']*100),
                               provisional=detection['features']['provisional'],
                               reason=detection['features']['reason'],
                               fall_episode=track.fall_episode,
                               drop_speed=round(track.drop_speed, 2),
                               dwell_sec=round(now-track.candidate_since,2) if track.candidate_since is not None else 0.))
        return output
