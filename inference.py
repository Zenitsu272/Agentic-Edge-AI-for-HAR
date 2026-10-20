import io
import time
import base64
from collections import deque
from datetime import datetime
import numpy as np
import cv2
import torch
from ultralytics import YOLO

import database
from tracker import CentroidTracker

class InferencePipeline:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        print(f"[InferencePipeline] Target Device: {self.device} (CUDA available: {torch.cuda.is_available()})")
        print("[InferencePipeline] Loading YOLOv8n (Person Presence) & YOLO11n-Pose (Biomechanics)...")
        
        # Stage 1: Person presence gate
        self.motion_model = YOLO("yolov8n.pt")
        # Stage 2: Biomechanical 17-keypoint Human Activity & Posture model
        self.pose_model = YOLO("yolo11n-pose.pt")
        
        if self.device == 0:
            self.motion_model.to("cuda")
            self.pose_model.to("cuda")
        print("[InferencePipeline] Models initialized successfully on GPU.")

        # Real motion flux tracking (grayscale frame differencing)
        self.prev_gray_frame = None

        # Multi-person centroid tracker with track history and displacement metrics
        self.tracker = CentroidTracker(max_missed_frames=15, max_distance=120.0)

        # Rolling frame buffer (Stage 2 -> Stage 3 packaging)
        self.frame_buffer = deque(maxlen=45)

        # Stage 3: Temporal Fall Event State Machine
        # States: 'STABLE', 'RAPID_DROP', 'VERIFYING', 'FALL_CONFIRMED', 'RECOVERED'
        self.state = "STABLE"
        self.state_start_time = time.time()
        self.drop_detected_time = None
        self.confirmed_fall_time = None
        
        # Rolling temporal history: records (timestamp, posture, hip_y, box_y_center, box_h)
        self.pose_history = deque(maxlen=30)
        
        # Active alert record
        self.active_alert = None
        self.alert_counter = 1001

    def decode_frame(self, data_url_or_b64: str):
        if "," in data_url_or_b64:
            data_url_or_b64 = data_url_or_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(data_url_or_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    def compute_motion_flux(self, frame):
        """
        Computes genuine pixel-level motion flux between consecutive frames.
        Returns flux percentage (0.0% to 100.0%) and motion_detected boolean.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray_frame is None or self.prev_gray_frame.shape != gray.shape:
            self.prev_gray_frame = gray
            return 0.0, False

        diff = cv2.absdiff(self.prev_gray_frame, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_pixels = int(np.count_nonzero(thresh))
        total_pixels = int(thresh.size)
        flux_pct = round((motion_pixels / max(total_pixels, 1)) * 100.0, 2)

        self.prev_gray_frame = gray
        # Motion considered active if pixel flux > 1.2%
        motion_active = flux_pct >= 1.2
        return flux_pct, motion_active

    def process_unified(self, frame):
        """
        Single unified edge pipeline:
        Stage 1: Person Presence & Genuine Motion Flux
        Stage 2: Biomechanical Posture Estimation (Standing, Sitting, Lying/Resting)
        Stage 3: Temporal Fall Event State Machine (Drop Velocity -> Verification -> Fall Confirmed / Recovered)
        """
        t0 = time.time()
        now = t0
        self.frame_buffer.append(frame)

        # --- STAGE 1: PERSON PRESENCE, TRACKING & MOTION ---
        flux_pct, motion_active = self.compute_motion_flux(frame)
        motion_results = self.motion_model.predict(frame, classes=[0], conf=0.35, device=self.device, verbose=False)
        
        persons_count = 0
        boxes = []
        for r in motion_results:
            for box in r.boxes:
                persons_count += 1
                b = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append({
                    "x1": round(b[0]),
                    "y1": round(b[1]),
                    "x2": round(b[2]),
                    "y2": round(b[3]),
                    "conf": round(conf, 2)
                })

        person_present = persons_count > 0
        
        # Track persons across frames using centroid tracker
        tracker_inputs = [{"box": [b["x1"], b["y1"], b["x2"], b["y2"]], "conf": b["conf"]} for b in boxes]
        active_tracks = self.tracker.update(tracker_inputs, now)
        
        primary_track = active_tracks[0] if active_tracks else None
        track_id_str = f"Track #{primary_track.track_id}" if primary_track else "None"
        movement_status = primary_track.movement_status if primary_track else ("Stationary" if person_present else "Idle")
        displacement_px_s = round(primary_track.displacement_px_sec, 1) if primary_track else 0.0
        norm_disp = round(primary_track.normalized_displacement, 3) if primary_track else 0.0
        self.current_track_id = track_id_str

        # --- STAGE 2: BIOMECHANICAL POSTURE ESTIMATION ---
        posture = "Awaiting Subject"
        posture_category = "Unknown"
        posture_conf = 0.0
        posture_details = "No resident in field of view"
        torso_angle = 0.0
        aspect_ratio = 0.0
        skeleton_pts = []
        primary_box = None
        hip_y = None
        box_y_center = None

        if person_present:
            pose_results = self.pose_model.predict(frame, conf=0.35, device=self.device, verbose=False)
            
            for r in pose_results:
                if r.keypoints is None or len(r.keypoints) == 0:
                    continue

                kpts_all = r.keypoints.xy.cpu().numpy()
                confs_all = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else None
                boxes_all = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None

                if len(kpts_all) > 0:
                    kpts = kpts_all[0]
                    c = confs_all[0] if confs_all is not None else None
                    box = boxes_all[0].tolist() if boxes_all is not None and len(boxes_all) > 0 else []
                    primary_box = [round(float(v)) for v in box] if len(box) >= 4 else None

                    if primary_box:
                        box_y_center = (primary_box[1] + primary_box[3]) / 2.0

                    # Format keypoints for frontend skeleton rendering
                    for pt_idx, (px, py) in enumerate(kpts):
                        pconf = float(c[pt_idx]) if c is not None else 1.0
                        skeleton_pts.append([round(float(px), 1), round(float(py), 1), round(float(pconf), 2)])

                    # Biomechanical classification
                    posture_category, posture, posture_conf, torso_angle, aspect_ratio, hip_y, posture_details = self._classify_biomechanics(kpts, c, primary_box)
                    break

        # Record history for temporal analysis
        if person_present and hip_y is not None:
            self.pose_history.append({
                "time": now,
                "category": posture_category,
                "posture": posture,
                "hip_y": hip_y,
                "box_y_center": box_y_center,
                "torso_angle": torso_angle
            })

        # --- STAGE 3: TEMPORAL FALL EVENT STATE MACHINE ---
        fall_event_data = self._update_temporal_state_machine(now, person_present, posture_category, posture, torso_angle)

        latency_ms = (time.time() - t0) * 1000.0

        return {
            "timestamp": round(now, 3),
            "inference_ms": round(float(latency_ms), 1),
            "stage1_presence": {
                "person_present": person_present,
                "persons_count": persons_count,
                "track_id": track_id_str,
                "active_tracks_count": len(active_tracks),
                "motion_flux_pct": flux_pct,
                "motion_active": motion_active,
                "movement_status": movement_status,
                "displacement_px_sec": displacement_px_s,
                "normalized_displacement": norm_disp,
                "status_text": f"PERSON DETECTED ({track_id_str})" if person_present else "SCENE IDLE",
                "boxes": boxes
            },
            "stage2_posture": {
                "category": posture_category,
                "posture": posture,
                "confidence": round(float(posture_conf) * 100.0, 1),
                "torso_angle_deg": round(float(torso_angle), 1),
                "aspect_ratio": round(float(aspect_ratio), 2),
                "details": posture_details,
                "keypoints": skeleton_pts,
                "box": primary_box
            },
            "stage3_event": fall_event_data
        }

    def _classify_biomechanics(self, kpts, confs, box):
        """
        Biomechanical posture estimation based on 17 COCO keypoints:
        0: nose, 5: l_shoulder, 6: r_shoulder, 11: l_hip, 12: r_hip,
        13: l_knee, 14: r_knee, 15: l_ankle, 16: r_ankle.
        
        Returns: (category, display_label, confidence, torso_angle, aspect_ratio, hip_y, details)
        Categories: 'Standing', 'Sitting', 'Lying', 'Transition', 'Partial'
        """
        if kpts is None or len(kpts) < 17:
            return "Unknown", "Unknown", 0.0, 0.0, 0.0, None, "Insufficient keypoints"

        # Visibility checks
        has_shoulders = (confs[5] > 0.3 or confs[6] > 0.3) if confs is not None else True
        has_hips = (confs[11] > 0.25 or confs[12] > 0.25) if confs is not None else True
        has_knees = (confs[13] > 0.25 or confs[14] > 0.25) if confs is not None else True
        has_ankles = (confs[15] > 0.25 or confs[16] > 0.25) if confs is not None else True

        # Extract midpoints
        s_pts = [kpts[i] for i in [5, 6] if confs is None or confs[i] > 0.2]
        shoulder_y = float(np.mean([pt[1] for pt in s_pts])) if s_pts else float(kpts[5][1])
        shoulder_x = float(np.mean([pt[0] for pt in s_pts])) if s_pts else float(kpts[5][0])

        h_pts = [kpts[i] for i in [11, 12] if confs is None or confs[i] > 0.2]
        hip_y = float(np.mean([pt[1] for pt in h_pts])) if h_pts else (shoulder_y + 110.0)
        hip_x = float(np.mean([pt[0] for pt in h_pts])) if h_pts else shoulder_x

        knee_y = float((kpts[13][1] + kpts[14][1]) / 2.0)
        ankle_y = float((kpts[15][1] + kpts[16][1]) / 2.0)

        # Torso vector and angle relative to vertical axis
        dx = float(hip_x - shoulder_x)
        dy = float(hip_y - shoulder_y)
        angle_rad = float(np.arctan2(abs(dx), max(abs(dy), 1e-4)))
        torso_angle = float(np.degrees(angle_rad))

        # Bounding box aspect ratio (width / height)
        if box and len(box) >= 4:
            w = max(box[2] - box[0], 1)
            h = max(box[3] - box[1], 1)
            aspect_ratio = float(w / h)
        else:
            aspect_ratio = 0.5

        torso_height = float(abs(hip_y - shoulder_y))
        leg_height = float(abs(ankle_y - hip_y))
        hip_knee_delta = float(abs(knee_y - hip_y))

        # Close-up / Partial view check
        if not has_hips and not has_knees:
            return "Partial", "Partial View (Upper Body)", 0.85, torso_angle, aspect_ratio, hip_y, "Resident close to camera"

        # --- 1. LYING / RESTING POSTURE ---
        # Characterized by:
        # - Horizontal torso angle (> 55 degrees)
        # - Full-body horizontal aspect ratio (aspect_ratio > 1.15)
        # - Shoulders and hips at nearly the same vertical level (|shoulder_y - hip_y| small)
        is_lying = False
        if has_shoulders and has_hips:
            if torso_angle > 55.0 and (aspect_ratio > 1.1 or abs(shoulder_y - hip_y) < 60.0):
                is_lying = True
            elif aspect_ratio > 1.35:
                is_lying = True

        if is_lying:
            # Note: This is classified as Lying / Resting (POSTURE, NOT FALL EVENT)
            return "Lying", "Lying / Resting", 0.92, torso_angle, aspect_ratio, hip_y, f"Horizontal posture (tilt {torso_angle:.1f}°, aspect {aspect_ratio:.2f})"

        # --- 2. SITTING POSTURE ---
        # When sitting (in chair or on floor):
        # - Knees are bent and at or near hip level (small vertical hip-knee delta)
        # - Thighs are horizontal
        # - Even if resident leans forward (faint/reading), hips remain anchored and knees remain bent!
        is_sitting = False
        if has_knees and has_hips:
            if hip_knee_delta < max(torso_height * 0.85, 55.0):
                is_sitting = True
            elif leg_height < torso_height * 1.35:
                is_sitting = True
        elif not has_ankles and has_hips:
            # Sitting at desk/table where lower legs are occluded
            is_sitting = True

        if is_sitting:
            if torso_angle > 38.0:
                return "Sitting", "Sitting (Leaning Forward)", 0.88, torso_angle, aspect_ratio, hip_y, f"Seated with forward tilt ({torso_angle:.1f}°), hips anchored"
            return "Sitting", "Sitting", 0.94, torso_angle, aspect_ratio, hip_y, f"Stable seated posture (hip-knee delta {hip_knee_delta:.1f}px)"

        # --- 3. BENDING / TRANSITION ---
        if torso_angle > 40.0 and has_knees and hip_knee_delta > torso_height:
            return "Transition", "Bending / Reaching", 0.86, torso_angle, aspect_ratio, hip_y, f"Forward torso bend ({torso_angle:.1f}°), legs extended"

        # --- 4. STANDING POSTURE ---
        return "Standing", "Standing", 0.95, torso_angle, aspect_ratio, hip_y, f"Upright posture (tilt {torso_angle:.1f}°, aspect {aspect_ratio:.2f})"

    def _update_temporal_state_machine(self, now, person_present, posture_category, posture_label, torso_angle):
        """
        Temporal Fall Event State Machine:
        STABLE -> RAPID_DROP -> VERIFYING (1.5s dwell) -> FALL_CONFIRMED (Alert logged) -> RECOVERED
        
        Key Distinctions:
        1. A sleeping or lying person who was already horizontal stays STABLE (Lying/Resting). No false alert!
        2. A sitting person leaning forward stays STABLE (Sitting). No false alert!
        3. A real fall requires:
           - Prior upright posture (Standing or Sitting)
           - Rapid downward displacement of center-of-mass (drop velocity > threshold within 0.3s - 0.8s)
           - Resulting in Lying posture on the ground
           - Persistent floor dwell without immediate recovery for > 1.5s
        """
        if not person_present:
            return {
                "state": "IDLE",
                "state_label": "No Resident Present",
                "fall_confirmed": False,
                "drop_velocity": 0.0,
                "ground_dwell_sec": 0.0,
                "active_alert": self.active_alert
            }

        drop_velocity = 0.0
        dwell_time = 0.0

        # Calculate downward velocity across recent 0.3s - 0.8s window
        if len(self.pose_history) >= 4:
            curr = self.pose_history[-1]
            # Look back ~0.5s ago
            past_records = [p for p in self.pose_history if 0.25 <= (curr["time"] - p["time"]) <= 0.85]
            if past_records:
                past = past_records[0]
                dt = max(curr["time"] - past["time"], 0.05)
                dy = (curr["hip_y"] - past["hip_y"]) if (curr["hip_y"] and past["hip_y"]) else 0.0
                drop_velocity = dy / dt  # pixels per second (positive = downward drop)
                
                # Check for rapid drop transition:
                # Started upright (Standing or Sitting) -> now Lying, with significant downward velocity
                is_rapid_drop = (
                    past["category"] in ["Standing", "Sitting", "Transition"] and
                    posture_category == "Lying" and
                    dy > 55.0 and
                    drop_velocity > 90.0
                )

                if self.state == "STABLE" and is_rapid_drop:
                    self.state = "RAPID_DROP"
                    self.drop_detected_time = now
                    self.state_start_time = now
                    print(f"[TemporalStateMachine] Rapid drop detected! Velocity: {drop_velocity:.1f}px/s, transition: {past['category']} -> {posture_category}")

        # STATE MACHINE TRANSITIONS
        if self.state == "RAPID_DROP":
            # Transition to VERIFYING
            self.state = "VERIFYING"
            self.state_start_time = now

        elif self.state == "VERIFYING":
            dwell_time = now - self.drop_detected_time if self.drop_detected_time else 0.0
            
            # Check for recovery (person stood up or sat back up during verification window)
            if posture_category in ["Standing", "Sitting"]:
                self.state = "RECOVERED"
                self.state_start_time = now
                print(f"[TemporalStateMachine] Candidate fall recovered to {posture_category} within {dwell_time:.1f}s.")
            elif dwell_time >= 1.5:  # Persisted down for at least 1.5 seconds
                self.state = "FALL_CONFIRMED"
                self.confirmed_fall_time = now
                self.state_start_time = now
                
                # Trigger active alert
                self.alert_counter += 1
                self.active_alert = {
                    "alert_id": f"EVT-{self.alert_counter}",
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "severity": "CRITICAL EMERGENCY",
                    "type": "Confirmed Fall Event",
                    "details": f"Rapid downward collapse detected ({drop_velocity:.1f} px/s). Persistent ground dwell.",
                    "posture_trajectory": "Upright → Rapid Drop → Floor Collapse",
                    "acknowledged": False
                }
                print(f"[TemporalStateMachine] *** EMERGENCY: Fall Confirmed! Alert {self.active_alert['alert_id']} logged. ***")
                
                # Persist to SQLite relational database
                try:
                    database.log_event({
                        "id": self.active_alert["alert_id"],
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "timestamp_sec": now,
                        "camera_id": "Camera 01",
                        "track_id": getattr(self, "current_track_id", "Track-1"),
                        "event_type": "Confirmed Fall Incident",
                        "severity": "CRITICAL EMERGENCY",
                        "detector_status": "FALL_CONFIRMED",
                        "vlm_status": "PENDING_VERIFICATION",
                        "policy_decision": "EMERGENCY / ALERT",
                        "details": self.active_alert["details"],
                        "posture_trajectory": self.active_alert["posture_trajectory"],
                        "acknowledged": False,
                        "metadata": {
                            "drop_velocity": round(float(drop_velocity), 1),
                            "dwell_sec": round(float(dwell_time), 1),
                            "posture": posture_label,
                            "torso_angle": round(float(torso_angle), 1)
                        }
                    })
                except Exception as ex:
                    print("[TemporalStateMachine] Database event log error:", ex)

        elif self.state == "FALL_CONFIRMED":
            dwell_time = now - self.confirmed_fall_time if self.confirmed_fall_time else 0.0
            # If the resident recovers after a confirmed fall
            if posture_category in ["Standing", "Sitting"]:
                self.state = "RECOVERED"
                self.state_start_time = now
                print(f"[TemporalStateMachine] Resident recovered from fall after {dwell_time:.1f}s down.")

        elif self.state == "RECOVERED":
            # After 3 seconds of stable recovery, return to STABLE
            if (now - self.state_start_time) > 3.0:
                self.state = "STABLE"
                self.drop_detected_time = None
                self.confirmed_fall_time = None

        elif self.state == "STABLE":
            # If already in Lying posture without a drop (e.g. resting / sleeping in bed):
            # Remain completely STABLE with no alert!
            pass

        # Formatting state label
        state_labels = {
            "STABLE": "Normal Monitoring (Stable)",
            "RAPID_DROP": "Drop Detected (Analyzing Velocity)",
            "VERIFYING": f"Verifying Fall Dwell ({dwell_time:.1f}s)",
            "FALL_CONFIRMED": f"CRITICAL: Confirmed Fall ({dwell_time:.1f}s on ground)",
            "RECOVERED": "Resident Recovered"
        }

        return {
            "state": self.state,
            "state_label": state_labels.get(self.state, self.state),
            "fall_confirmed": (self.state == "FALL_CONFIRMED"),
            "drop_velocity": round(float(drop_velocity), 1),
            "ground_dwell_sec": round(float(dwell_time), 1),
            "active_alert": self.active_alert
        }

    # Backward compatibility helpers for older individual endpoints if still requested
    def process_motion(self, frame):
        res = self.process_unified(frame)
        return {
            "status": res["stage1_presence"]["status_text"],
            "motion": res["stage1_presence"]["motion_active"],
            "persons_count": res["stage1_presence"]["persons_count"],
            "inference_ms": res["inference_ms"],
            "boxes": res["stage1_presence"]["boxes"]
        }

    def process_har(self, frame):
        res = self.process_unified(frame)
        activities = []
        if res["stage1_presence"]["person_present"]:
            activities.append({
                "activity": res["stage2_posture"]["posture"],
                "confidence": res["stage2_posture"]["confidence"],
                "details": res["stage2_posture"]["details"],
                "box": res["stage2_posture"]["box"] or [],
                "keypoints": res["stage2_posture"]["keypoints"]
            })
        return {
            "activities": activities,
            "count": len(activities),
            "fall_alert": res["stage3_event"]["fall_confirmed"],
            "inference_ms": res["inference_ms"]
        }
