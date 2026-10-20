import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

class TrackedPerson:
    def __init__(self, track_id: int, bbox: List[float], conf: float, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.conf = conf
        self.last_seen = timestamp
        self.first_seen = timestamp
        self.missed_frames = 0
        
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.centroid = (cx, cy)
        
        # History: list of (timestamp, centroid, bbox, displacement_px)
        self.history = [(timestamp, (cx, cy), bbox, 0.0)]
        self.max_history = 60
        
        # Movement metrics
        self.displacement_px_sec = 0.0
        self.normalized_displacement = 0.0
        self.is_moving = False
        self.movement_status = "Stationary"

    def update(self, bbox: List[float], conf: float, timestamp: float):
        self.bbox = bbox
        self.conf = conf
        self.last_seen = timestamp
        self.missed_frames = 0
        
        new_cx = (bbox[0] + bbox[2]) / 2.0
        new_cy = (bbox[1] + bbox[3]) / 2.0
        
        dt = max(timestamp - self.history[-1][0], 1e-4) if self.history else 0.033
        prev_cx, prev_cy = self.centroid
        dx = new_cx - prev_cx
        dy = new_cy - prev_cy
        disp = np.sqrt(dx*dx + dy*dy)
        
        self.centroid = (new_cx, new_cy)
        self.displacement_px_sec = disp / dt
        
        # Normalize by bounding box height
        box_h = max(bbox[3] - bbox[1], 10.0)
        self.normalized_displacement = round(float(disp / box_h), 3)
        
        # Movement hysteresis: > 15 px/sec or > 0.05 normalized displacement
        self.is_moving = self.displacement_px_sec > 25.0 or self.normalized_displacement > 0.08
        self.movement_status = "Active Movement" if self.is_moving else "Stationary"
        
        self.history.append((timestamp, (new_cx, new_cy), bbox, float(disp)))
        if len(self.history) > self.max_history:
            self.history.pop(0)

class CentroidTracker:
    """
    Lightweight, deterministic multi-person tracker with hysteresis,
    centroid displacement tracking, and track lifecycle management.
    """
    def __init__(self, max_missed_frames: int = 15, max_distance: float = 120.0):
        self.next_track_id = 1
        self.tracks: Dict[int, TrackedPerson] = {}
        self.max_missed_frames = max_missed_frames
        self.max_distance = max_distance

    def update(self, detections: List[Dict[str, Any]], timestamp: Optional[float] = None) -> List[TrackedPerson]:
        """
        detections: list of {"box": [x1, y1, x2, y2], "conf": float}
        """
        if timestamp is None:
            timestamp = time.time()

        if len(detections) == 0:
            # Increment missed frames for all existing tracks
            dead_tracks = []
            for tid, track in self.tracks.items():
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    dead_tracks.append(tid)
            for tid in dead_tracks:
                del self.tracks[tid]
            return list(self.tracks.values())

        det_centroids = []
        for d in detections:
            b = d["box"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            det_centroids.append((cx, cy))

        if len(self.tracks) == 0:
            for i, d in enumerate(detections):
                track = TrackedPerson(self.next_track_id, d["box"], d["conf"], timestamp)
                self.tracks[self.next_track_id] = track
                self.next_track_id += 1
            return list(self.tracks.values())

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        # Compute Euclidean distance matrix
        cost_matrix = np.zeros((len(track_centroids), len(det_centroids)))
        for i, tc in enumerate(track_centroids):
            for j, dc in enumerate(det_centroids):
                dist = np.hypot(tc[0] - dc[0], tc[1] - dc[1])
                cost_matrix[i, j] = dist

        assigned_tracks = set()
        assigned_dets = set()

        # Greedy match closest centroids within max_distance
        while cost_matrix.size > 0:
            min_val = cost_matrix.min()
            if min_val > self.max_distance or min_val == np.inf:
                break
            min_idx = np.unravel_index(cost_matrix.argmin(), cost_matrix.shape)
            r, c = min_idx[0], min_idx[1]
            
            tid = track_ids[r]
            self.tracks[tid].update(detections[c]["box"], detections[c]["conf"], timestamp)
            assigned_tracks.add(tid)
            assigned_dets.add(c)
            
            cost_matrix[r, :] = np.inf
            cost_matrix[:, c] = np.inf

        # Unassigned existing tracks
        for r, tid in enumerate(track_ids):
            if tid not in assigned_tracks:
                self.tracks[tid].missed_frames += 1

        # Remove dead tracks
        dead_tracks = [tid for tid, tr in self.tracks.items() if tr.missed_frames > self.max_missed_frames]
        for tid in dead_tracks:
            del self.tracks[tid]

        # Create new tracks for unassigned detections
        for c, d in enumerate(detections):
            if c not in assigned_dets:
                new_track = TrackedPerson(self.next_track_id, d["box"], d["conf"], timestamp)
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        return list(self.tracks.values())
