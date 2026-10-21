"""Bounded incident evidence capture and asynchronous, local VLM verification."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import copy
import time
import cv2
import numpy as np
import database


@dataclass
class EvidenceSession:
    frames: deque = field(default_factory=lambda: deque(maxlen=40))
    last_sample: float = -float('inf')
    last_time: float = -float('inf')
    pending: list = field(default_factory=list)
    latest_id: str | None = None


def select_keyframes(frames, trigger_time, max_frames=12):
    """Keep chronological context, onset, and the final observed posture."""
    if not frames:
        return []
    indices = set(np.linspace(0, len(frames)-1, min(max_frames, len(frames)), dtype=int).tolist())
    onset = min(range(len(frames)), key=lambda i: abs(frames[i]['timestamp_sec']-trigger_time))
    if onset not in indices:
        interior = [i for i in indices if i not in (0, len(frames)-1)]
        if interior: indices.remove(min(interior, key=lambda i: abs(i-onset)))
        indices.add(onset)
    return [frames[i] for i in sorted(indices)]


def prepare_evidence_images(selected, trigger_time):
    """Keep the whole scene, including the floor and support surfaces."""
    images = []
    for sample in selected:
        decoded = cv2.imdecode(np.frombuffer(sample['jpeg'],np.uint8),cv2.IMREAD_COLOR)
        if decoded is None: continue
        images.append({'frame':decoded,'relative_sec':sample['timestamp_sec']-trigger_time})
    return images


class IncidentVerificationService:
    def __init__(self, adapter, persist=database.save_verification, post_seconds=1.5,
                 max_pending=4, model_timeout=60, evidence_idle_timeout=10):
        self.adapter = adapter
        self.persist = persist
        self.post_seconds = post_seconds
        self.max_pending = max_pending
        self.model_timeout = model_timeout
        self.evidence_idle_timeout = evidence_idle_timeout
        self.records = {}
        self.tasks = set()
        self.sessions = {}
        self.ready = asyncio.Event()
        self.model_lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='carewatch-vlm')
        self.loader = None

    async def start(self):
        if self.loader is not None: return
        async def load():
            try:
                await asyncio.get_running_loop().run_in_executor(self.executor, self.adapter._init_adapter)
            finally:
                self.ready.set()
        self.loader = asyncio.create_task(load())

    async def close(self):
        for task in list(self.tasks): task.cancel()
        if self.tasks: await asyncio.gather(*list(self.tasks), return_exceptions=True)
        if self.loader and not self.loader.done(): self.loader.cancel()
        self.executor.shutdown(wait=False, cancel_futures=True)

    def new_session(self, session_id):
        self.sessions[session_id] = EvidenceSession()

    def close_session(self, session_id):
        session = self.sessions.pop(session_id, None)
        if session:
            for job in session.pending: job['ready'].set()

    def _save(self, record):
        public = {k:v for k,v in record.items() if not k.startswith('_')}
        self.persist(public)

    def _finish(self, record, status, reason):
        record.update(status=status, is_true_fall=None, scene_observation=reason,
                      verification_method='UNAVAILABLE', completed_at=time.time())
        self._save(record)

    def observe(self, session_id, result, frame):
        session = self.sessions.setdefault(session_id, EvidenceSession())
        source_time = float(result.get('timestamp', time.monotonic()))
        if source_time < session.last_time:
            # Never combine the end of one replay loop with the beginning of another.
            for job in session.pending: job['ready'].set()
            session.frames.clear(); session.pending.clear(); session.last_sample = -float('inf')
        session.last_time = source_time
        event = result.get('stage3_event', {}).get('active_alert')
        triggered = bool(result.get('stage3_event', {}).get('fall_alert_active') or result.get('stage3_event', {}).get('fall_confirmed'))
        display_event_id = (event.get('alert_id') or event.get('id')) if event else None
        events = result.get('stage3_event', {}).get('active_incidents') or ([event] if event else [])
        fresh_events = [e for e in events if triggered and (e.get('alert_id') or e.get('id')) not in self.records]
        fresh = bool(fresh_events)
        if source_time-session.last_sample >= .10-1e-6 or fresh:
            h,w = frame.shape[:2]
            scale = min(1., 640/max(h,w))
            resized = cv2.resize(frame, (max(1,round(w*scale)),max(1,round(h*scale))))
            ok, encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY,80])
            if ok:
                boxes = {f"Person {p['track_id']}":[round(p['box'][0]/w*1000),round(p['box'][1]/h*1000),
                          round(p['box'][2]/w*1000),round(p['box'][3]/h*1000)] for p in result.get('poses',[])}
                sample = {'timestamp_sec':source_time, 'jpeg':encoded.tobytes(), 'boxes':boxes}
                session.frames.append(sample); session.last_sample = source_time
                for job in list(session.pending):
                    if not job['ready'].is_set() and (not job['frames'] or source_time > job['frames'][-1]['timestamp_sec']):
                        job['frames'].append(sample)
                        job['last_progress'] = time.monotonic()
                        if source_time >= job['trigger']+self.post_seconds: job['ready'].set()
        for event in fresh_events:
            event_id = event.get('alert_id') or event.get('id')
            session.latest_id = event_id
            record = {'event_id':event_id, 'status':'COLLECTING', 'model':self.adapter.model_name,
                      'trigger_time_sec':source_time,
                      'verification_scope':'SCENE',
                      'final_classification':'FALL', 'decision_source':'POSE_MODEL',
                      'track_id':event.get('track_id'), 'is_true_fall':None,
                      'scene_observation':'Collecting frames before and after the motion.',
                      'evidence_frames_count':0, 'verification_method':'PENDING', 'created_at':time.time()}
            self.records[event_id] = record
            if len(self.tasks) >= self.max_pending:
                self._finish(record, 'BUSY', 'The local verifier is busy; this incident remains unverified. Review the person.')
            else:
                before = [s for s in session.frames if source_time-2.0 <= s['timestamp_sec'] <= source_time]
                job = {'frames':before, 'trigger':source_time, 'ready':asyncio.Event(),
                       'last_progress':time.monotonic(),
                       'candidate':{'id':event_id, 'camera_id':'Camera 01', 'track_id':event.get('track_id')}}
                session.pending.append(job)
                self._save(record)
                task = asyncio.create_task(self._verify(job, record, session))
                self.tasks.add(task); task.add_done_callback(self.tasks.discard)
            # Keep only recent completed records in RAM. Audit results remain in SQLite.
            for key in list(self.records)[:-128]:
                if self.records[key]['status'] not in ('COLLECTING','QUEUED','VERIFYING'): self.records.pop(key)
        current = self.records.get(display_event_id or session.latest_id)
        result['verification_incidents'] = {e.get('alert_id') or e.get('id'):copy.deepcopy(self.records.get(e.get('alert_id') or e.get('id')))
                                            for e in events}
        result['verification'] = copy.deepcopy(current) if current else {
            'status':'READY' if self.adapter.status == 'ONLINE' else self.adapter.status,
            'model':self.adapter.model_name, 'scene_observation':(
                'A fall alert will send a short frame sequence for local scene verification.'
                if self.adapter.status == 'ONLINE' else self.adapter.status_detail),
            'is_true_fall':None, 'evidence_frames_count':0}
        return result

    async def _verify(self, job, record, session):
        try:
            # A replay can take longer than real time under GPU load. Wait
            # until its source window is complete, EOF, or no frames arrive.
            while not job['ready'].is_set():
                idle_remaining = self.evidence_idle_timeout - (time.monotonic()-job['last_progress'])
                if idle_remaining <= 0: break
                try:
                    await asyncio.wait_for(job['ready'].wait(), timeout=idle_remaining)
                except asyncio.TimeoutError:
                    continue
            frames = job['frames']
            if (len(frames)<3 or frames[-1]['timestamp_sec']-job['trigger'] < .35
                    or job['trigger']-frames[0]['timestamp_sec'] < .25):
                self._finish(record, 'INCONCLUSIVE', 'Insufficient before/after frames to verify the motion. Review the person.')
                return
            selected = select_keyframes(frames, job['trigger'])
            record['evidence_frames_count'] = len(selected)
            record['evidence_offsets_sec'] = [round(s['timestamp_sec']-job['trigger'],2) for s in selected]
            record.update(status='QUEUED', scene_observation='Waiting for the local VLM.')
            self._save(record)
            await asyncio.wait_for(self.ready.wait(), timeout=120)
            async with self.model_lock:
                if self.adapter.status != 'ONLINE':
                    self._finish(record, 'UNAVAILABLE', self.adapter.status_detail or 'The local VLM is unavailable.')
                    return
                images = prepare_evidence_images(selected,job['trigger'])
                record.update(status='VERIFYING', scene_observation='VLM is examining the frame sequence.')
                self._save(record)
                future = asyncio.get_running_loop().run_in_executor(
                    self.executor, self.adapter.verify_video_event, job['candidate'], images)
                try:
                    verdict = await asyncio.wait_for(asyncio.shield(future), timeout=self.model_timeout)
                except asyncio.TimeoutError:
                    self._finish(record,'TIMEOUT','VLM verification timed out. The fall remains unverified.')
                    # Retain exclusive access until native generation actually ends.
                    await future
                    return
                if (verdict.get('status') in ('CONFIRMED_FALL','NOT_FALL')
                        and verdict.get('verification_method') != 'VLM_MODEL'):
                    self._finish(record,'UNAVAILABLE','No actual VLM inference was available; the incident remains unverified.')
                    return
                record.update(verdict, completed_at=time.time())
                # The VLM opinion never changes the pose model's incident decision.
                record.update(final_classification='FALL', decision_source='POSE_MODEL')
                record['evidence_frames_count'] = len(selected)
                self._save(record)
        except asyncio.CancelledError:
            self._finish(record,'INTERRUPTED','Verification was interrupted; no VLM verdict is available.')
            raise
        except asyncio.TimeoutError:
            self._finish(record,'TIMEOUT','The VLM did not become ready in time.')
        except Exception as error:
            self._finish(record,'ERROR',f'Verification error: {type(error).__name__}: {str(error)[:160]}')
        finally:
            if job in session.pending: session.pending.remove(job)
            job['frames'].clear()

    def telemetry(self):
        return {'pending_incidents':len(self.tasks), 'capacity':self.max_pending,
                'model_status':self.adapter.status, 'frames_per_incident':12}
