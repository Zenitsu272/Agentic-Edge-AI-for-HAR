"""Local pose inference with isolated, per-stream activity tracking."""
import os
import time
import uuid
import base64
from collections import deque
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import database
from pose_activity import PoseActivityTracker, FallAlertLatch, analyze_pose, VERSION


class InferencePipeline:
    def __init__(self):
        self.device = os.getenv('CAREWATCH_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        models = Path(__file__).parent / 'models'
        preferred = models / 'yolo11s-pose.pt'
        self.model_path = Path(os.getenv('CAREWATCH_POSE_MODEL', str(preferred if preferred.exists() else models / 'yolo11n-pose.pt')))
        self.imgsz = int(os.getenv('CAREWATCH_IMAGE_SIZE', '960' if self.device.startswith('cuda') else '640'))
        self.pose_model = YOLO(str(self.model_path))
        self.pose_model.to(self.device)
        print(f'[Pose] {self.model_path.name} on {self.device}, input={self.imgsz}, classifier={VERSION}')
        self.reset()

    def reset(self):
        self.activity_tracker = PoseActivityTracker()
        self.fall_alert_latch = FallAlertLatch()
        self.prev_gray_frame = None
        self.frame_buffer = deque(maxlen=12)
        self.pose_history = deque(maxlen=30)  # Compatibility for replay consumers.
        self.state = 'STABLE'
        self.active_alert = None
        self.session_id = uuid.uuid4().hex[:10]

    def new_session(self):
        # Share read-only model weights; never share motion history between viewers.
        session = object.__new__(InferencePipeline)
        session.device, session.model_path, session.imgsz = self.device, self.model_path, self.imgsz
        session.pose_model = self.pose_model
        session.reset()
        return session

    def decode_frame(self, data):
        try:
            raw = data.split(',', 1)[-1]
            return cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)
        except (ValueError, cv2.error):
            return None

    def compute_motion_flux(self, frame):
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (11,11), 0)
        old = self.prev_gray_frame
        self.prev_gray_frame = gray
        if old is None or old.shape != gray.shape: return 0., False
        flux = round(float(np.mean(cv2.absdiff(old, gray) > 25) * 100), 2)
        return flux, flux > 1.2

    def _persist_changes(self, pose, now):
        track = self.activity_tracker.tracks[pose['track_id']]
        label = pose['category']
        fall_signal = pose['state'] in ('PARTIAL_FALL', 'VERIFYING', 'FALL_CONFIRMED')
        confirmed = pose['state'] == 'FALL_CONFIRMED'
        if fall_signal:
            phase = 'FALL_CONFIRMED' if confirmed else 'PARTIAL_FALL'
            incident_key = (pose['fall_episode'], phase)
            if incident_key == track.last_logged_fall: return
            if not confirmed and track.last_logged_fall == (pose['fall_episode'], 'FALL_CONFIRMED'): return
            track.last_logged_fall = incident_key
        else:
            if label not in ('Standing', 'Sitting') or pose['provisional']: return
            if label == track.last_logged_category: return
        # Avoid logging a single first-frame classification as an activity transition.
        support = [t for t,c in track.votes if c == label]
        if not fall_signal and (len(support) < 3 or now - min(support) < .3): return
        if not fall_signal: track.last_logged_category = label
        event_id = (f'POSE-{self.session_id}-{track.track_id}-FALL-{pose["fall_episode"]}'
                    if fall_signal else f'POSE-{self.session_id}-{track.track_id}-{uuid.uuid4().hex[:8]}')
        event = {
            'id': event_id, 'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'timestamp_sec': time.time(), 'track_id': f'Person {track.track_id}',
            'event_type': ('Fall detected (pose)' if confirmed else 'Possible / partial fall detected') if fall_signal else f'Activity: {label}',
            'severity': ('ALERT' if confirmed else 'WARNING') if fall_signal else 'NORMAL',
            'detector_status': ('FALL_CONFIRMED' if confirmed else 'PARTIAL_FALL') if fall_signal else pose['state'], 'vlm_status': 'NOT_REQUESTED',
            'policy_decision': 'HUMAN REVIEW' if fall_signal else 'LOG',
            'details': ('Rapid torso descent and rotation followed by sustained horizontal pose.' if confirmed else 'Sudden collapse from standing or sitting; partial fall signal awaiting confirmation.') if fall_signal else pose['reason'],
            'posture_trajectory': label, 'acknowledged': False,
            'metadata': {'classifier_version': VERSION, 'provenance': 'REAL_PIPELINE', 'is_synthetic': False,
                         'video_time': now, 'evidence_score': pose['evidence_score']},
        }
        database.log_event(event)
        if fall_signal:
            track.event_id = event_id
            self.active_alert = dict(event, alert_id=event_id, type=event['event_type'])

    def process_unified(self, frame, timestamp=None):
        start = time.perf_counter()
        now = time.monotonic() if timestamp is None else float(timestamp)
        self.frame_buffer.append(frame)
        flux, motion = self.compute_motion_flux(frame)
        results = self.pose_model.predict(frame, conf=.4, iou=.55, imgsz=self.imgsz, device=self.device, verbose=False)
        detections = []
        for result in results:
            if result.keypoints is None or result.boxes is None: continue
            for index, points in enumerate(result.keypoints.data.cpu().numpy()):
                box = result.boxes.xyxy[index].cpu().tolist()
                features = analyze_pose(points, box, frame.shape)
                if not features['usable']: continue
                detections.append(dict(keypoints=points.tolist(), box=box, features=features,
                                       detection_confidence=round(float(result.boxes.conf[index]),3)))
        poses = self.activity_tracker.update(detections, now, frame.shape)
        for pose in poses: self._persist_changes(pose, now)
        active_incidents = []
        for pose in poses:
            if pose['state'] in ('PARTIAL_FALL','VERIFYING','FALL_CONFIRMED'):
                track = self.activity_tracker.tracks[pose['track_id']]
                pose['incident_id'] = track.event_id
                if track.event_id:
                    active_incidents.append({'alert_id':track.event_id, 'track_id':f"Person {pose['track_id']}",
                                             'detector_status':pose['state']})
        # Prefer a fall, then a person with usable body geometry, never arbitrary model order.
        primary = max(poses, key=lambda p: (p['category']=='Fall', p['state'] in ('VERIFYING', 'PARTIAL_FALL'),
                      p['category'] in ('Standing','Sitting'), p['features']['visible_keypoints'],
                      p['detection_confidence'])) if poses else None
        f = primary['features'] if primary else {}
        state = primary['state'] if primary else 'STABLE'
        self.state = state
        boxes = [dict(x1=p['box'][0],y1=p['box'][1],x2=p['box'][2],y2=p['box'][3],conf=p['detection_confidence']) for p in poses]
        fall = next((p for p in poses if p['category']=='Fall'), None)
        fall_signal = any(p['state'] in ('PARTIAL_FALL', 'VERIFYING', 'FALL_CONFIRMED') for p in poses)
        alert = self.fall_alert_latch.update(fall_signal, self.active_alert if fall_signal else None)
        confirmed_alert = bool(alert['active'] and alert['event'] and alert['event'].get('detector_status') == 'FALL_CONFIRMED')
        if alert['active']:
            state = 'FALL_CONFIRMED' if confirmed_alert else 'PARTIAL_FALL'
            self.state = state
        return {
            'timestamp': now, 'frame_width': int(frame.shape[1]), 'frame_height': int(frame.shape[0]),
            'classifier_version': VERSION, 'model': self.model_path.name, 'device': self.device,
            'poses': poses, 'inference_ms': round((time.perf_counter()-start)*1000,1),
            'stage1_presence': {
                'person_present': bool(poses), 'persons_count': len(poses), 'active_tracks_count': len(poses),
                'track_id': f"Person {primary['track_id']}" if primary else None,
                'motion_flux_pct': flux, 'motion_active': motion,
                'movement_status': primary['movement'] if primary else 'Idle',
                'status_text': f'{len(poses)} people tracked' if poses else 'Scene idle', 'boxes': boxes,
            },
            'stage2_posture': {
                'category': primary['category'] if primary else 'Unknown',
                'posture': primary['posture'] if primary else 'Awaiting subject',
                'confidence': primary['evidence_score'] if primary else 0,
                'provisional': primary['provisional'] if primary else False,
                'torso_angle_deg': f.get('torso_angle'), 'knee_angle_deg': f.get('knee_angle'),
                'visible_keypoints': f.get('visible_keypoints',0),
                'aspect_ratio': ((primary['box'][2]-primary['box'][0])/max(primary['box'][3]-primary['box'][1],1)) if primary else 0,
                'details': primary['reason'] if primary else 'Waiting for a person with visible keypoints.',
                'keypoints': primary['keypoints'] if primary else [], 'box': primary['box'] if primary else None,
            },
            'stage3_event': {
                'state': state, 'state_label': 'Fall detected' if alert['active'] else (primary['posture'] if primary else 'Monitoring'),
                'fall_confirmed': confirmed_alert,
                'fall_alert_active': alert['active'],
                'fall_detection_level': 'confirmed' if confirmed_alert else ('possible_partial' if alert['active'] else None),
                'fall_detected_now': fall is not None,
                'partial_fall_detected_now': fall_signal and fall is None,
                'fall_alert_held': alert['held'],
                'fall_hold_remaining_sec': alert['remaining_sec'],
                'drop_velocity': round((primary['drop_speed']*(f.get('scale') or 1)),1) if primary else 0,
                'normalized_drop_speed': primary['drop_speed'] if primary else 0,
                'ground_dwell_sec': primary['dwell_sec'] if primary else 0,
                'active_alert': alert['event'] if alert['active'] else None,
                'active_incidents': active_incidents,
            }
        }

    def _classify_biomechanics(self, kpts, confs, box):
        # Compatibility adapter for scripts using the old single-frame API.
        pts = np.column_stack([kpts, confs if confs is not None else np.ones(len(kpts))])
        h, w = max(pts[:,1].max()+2,box[3]+2), max(pts[:,0].max()+2,box[2]+2)
        f = analyze_pose(pts,box,(h,w))
        return f['category'],f['posture'],f['score'],f['torso_angle'] or 0,(box[2]-box[0])/max(box[3]-box[1],1),f['hip_y'],f['reason']

    def process_motion(self, frame):
        r = self.process_unified(frame); s = r['stage1_presence']
        return dict(status=s['status_text'], motion=s['motion_active'],persons_count=s['persons_count'],inference_ms=r['inference_ms'],boxes=s['boxes'])

    def process_har(self, frame):
        r = self.process_unified(frame)
        return dict(activities=[dict(activity=p['posture'],confidence=p['evidence_score'],details=p['reason'],box=p['box'],keypoints=p['keypoints']) for p in r['poses']],count=len(r['poses']),fall_alert=r['stage3_event']['fall_alert_active'],inference_ms=r['inference_ms'])
