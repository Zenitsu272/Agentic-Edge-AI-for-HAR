import os
import sys
import time
import json
import numpy as np
import pytest

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import database
from tracker import CentroidTracker
from vlm_adapter import VLMAdapter
from policy import EscalationPolicy
from central_server import ChromaAgentMemory
from agent import CareWatchAgent

class TestCareWatchPlatform:
    @classmethod
    def setup_class(cls):
        database.init_db()
        cls.tracker = CentroidTracker(max_missed_frames=10, max_distance=100.0)
        cls.vlm = VLMAdapter()
        cls.policy = EscalationPolicy()
        cls.chroma = ChromaAgentMemory()
        cls.agent = CareWatchAgent(cls.chroma, cls.vlm, cls.policy)

    # 1. Person present and stationary vs moving
    def test_person_presence_and_movement(self):
        tracker = CentroidTracker(max_missed_frames=10, max_distance=150.0)
        # Frame 1: person at (100, 120)
        tracks1 = tracker.update([{"box": [80, 80, 120, 160], "conf": 0.9}], timestamp=1.0)
        assert len(tracks1) == 1
        assert tracks1[0].track_id == 1
        assert tracks1[0].movement_status == "Stationary"

        # Frame 2: person moving smoothly to (130, 150) (displacement ~42px in 0.05s -> 840 px/s)
        tracks2 = tracker.update([{"box": [110, 110, 150, 190], "conf": 0.9}], timestamp=1.05)
        assert len(tracks2) == 1
        assert tracks2[0].track_id == 1
        assert tracks2[0].is_moving or tracks2[0].displacement_px_sec > 50.0

    # 2. Tracking lifecycle & missed detection tolerance
    def test_track_missed_frames_and_termination(self):
        tracker = CentroidTracker(max_missed_frames=5, max_distance=100.0)
        tracker.update([{"box": [50, 50, 100, 150], "conf": 0.9}], timestamp=1.0)
        
        # 1 missed frame does not kill track
        tracks = tracker.update([], timestamp=1.1)
        assert len(tracks) == 1
        assert tracks[0].missed_frames == 1

        # 6 missed frames (exceeding max_missed_frames=5) terminates track
        for i in range(6):
            tracks = tracker.update([], timestamp=1.2 + i*0.1)
        assert len(tracks) == 0

    # 3. Posture classification: Sitting vs Lying vs Standing
    def test_posture_classification_rules(self):
        from inference import InferencePipeline
        # Verify rule logic directly
        # Synthetic standing: small torso angle, small aspect ratio
        cand_standing = {"kinematics": {"posture": "Standing", "torso_angle": 8.0, "dwell_sec": 0.0, "drop_velocity": 0.0}}
        res_standing = self.vlm.verify_candidate_event(cand_standing)
        assert res_standing["is_true_fall"] is False
        assert res_standing["confidence"] >= 0.90

        # Synthetic sitting with forward reach: hips anchored, non-fall
        cand_sitting = {"kinematics": {"posture": "Sitting (Leaning Forward)", "torso_angle": 38.0, "dwell_sec": 0.0, "drop_velocity": 0.0}}
        res_sitting = self.vlm.verify_candidate_event(cand_sitting)
        assert res_sitting["is_true_fall"] is False

        # Synthetic intentional lying / resting without acute drop velocity
        cand_resting = {"kinematics": {"posture": "Lying / Resting", "torso_angle": 75.0, "dwell_sec": 0.5, "drop_velocity": 5.0}}
        pol_resting = self.policy.evaluate(cand_resting, {"is_true_fall": False, "recovery_observed": False}, [])
        assert pol_resting["decision"] == "NORMAL / LOG"

    # 4. Staged Fall Incident Confirmation
    def test_staged_fall_confirmation(self):
        cand_fall = {
            "id": "TEST-EVT-FALL-01",
            "camera_id": "Camera 01",
            "track_id": "Track #1",
            "state": "FALL_CONFIRMED",
            "kinematics": {
                "posture": "Lying / Resting",
                "torso_angle": 82.0,
                "drop_velocity": 94.5,
                "dwell_sec": 2.1
            }
        }
        vlm_res = self.vlm.verify_candidate_event(cand_fall)
        assert vlm_res["is_true_fall"] is True

        pol_res = self.policy.evaluate(cand_fall, vlm_res, [])
        assert pol_res["decision"] == "EMERGENCY / ALERT"
        assert pol_res["severity"] == "CRITICAL EMERGENCY"

    # 5. Fall Followed by Recovery
    def test_fall_followed_by_recovery(self):
        cand_recovered = {
            "state": "RECOVERED",
            "kinematics": {
                "posture": "Standing",
                "torso_angle": 12.0,
                "drop_velocity": -25.0,
                "dwell_sec": 0.0
            }
        }
        vlm_res = self.vlm.verify_candidate_event(cand_recovered)
        assert vlm_res["recovery_observed"] is True
        pol_res = self.policy.evaluate(cand_recovered, vlm_res, [])
        assert pol_res["decision"] == "NORMAL / LOG"

    # 6. VLM Offline / Fallback Transparency
    def test_vlm_fallback_transparency(self):
        # Ensure status is reported transparently
        assert self.vlm.status in ["ONLINE", "WEIGHTS_INCOMPLETE", "OFFLINE_FALLBACK", "WEIGHTS_MISSING"]
        cand = {"kinematics": {"posture": "Standing", "torso_angle": 5.0, "dwell_sec": 0.0, "drop_velocity": 0.0}}
        res = self.vlm.verify_candidate_event(cand)
        assert "verification_method" in res
        assert "adapter_status" in res

    # 7. Agent Bounded Tool Loop & Operational Audit Traces
    def test_agent_bounded_investigation(self):
        candidate = {
            "id": "EVT-TEST-AGENT-101",
            "camera_id": "Camera 01",
            "track_id": "Track #1",
            "state": "STABLE",
            "kinematics": {
                "posture": "Standing",
                "torso_angle": 10.0,
                "drop_velocity": 0.0,
                "dwell_sec": 0.0
            }
        }
        investigation = self.agent.run_investigation(candidate)
        assert investigation["run_id"].startswith("AGENT-RUN-")
        assert len(investigation["tools_executed"]) <= 4
        assert investigation["escalation_decision"] == "NORMAL / LOG"

        # Check SQLite audit trace persistence
        traces = database.get_recent_agent_traces(limit=5)
        assert len(traces) > 0
        latest = traces[0]
        assert latest["event_id"] == "EVT-TEST-AGENT-101"
        assert len(latest["tools_called"]) > 0

    # 8. ChromaDB Persistent Retrieval & Storage
    def test_chroma_db_persistence_and_retrieval(self):
        # Insert test memory
        test_id = f"EVT-TEST-CHROMA-{int(time.time())}"
        doc = "Camera 01 Test Incident: verified collapse near bed with 95 px/s downward velocity."
        meta = {"category": "Fall", "camera": "Camera 01", "drop_velocity_px_s": 95.0}
        self.chroma.log_new_event(test_id, doc, meta)

        # Retrieve
        matches = self.chroma.retrieve_context("collapse near bed with rapid drop", n_results=2)
        assert len(matches) > 0
        assert any(test_id in str(m) or "collapse" in m["document"] for m in matches)

    # 9. SQLite Event Storage & Human Acknowledgement Persistence
    def test_sqlite_event_persistence_and_acknowledgement(self):
        evt_id = f"EVT-TEST-ACK-{int(time.time()*1000)%100000}"
        database.log_event({
            "id": evt_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_sec": time.time(),
            "camera_id": "Camera 01",
            "track_id": "Track-99",
            "event_type": "Test Verification Fall",
            "severity": "CRITICAL EMERGENCY",
            "detector_status": "FALL_CONFIRMED",
            "vlm_status": "VERIFIED",
            "policy_decision": "EMERGENCY / ALERT",
            "details": "Simulated unit test fall event for persistence test.",
            "posture_trajectory": "Upright -> Fall",
            "acknowledged": False
        })

        # Verify present in SQLite
        evts = database.get_events(limit=10)
        found = [e for e in evts if e["id"] == evt_id]
        assert len(found) == 1
        assert found[0]["acknowledged"] is False

        # Acknowledge
        acked = database.acknowledge_event_by_id(evt_id)
        assert acked is True

        # Verify acknowledgement survived in SQLite
        evts_after = database.get_events(limit=10)
        found_after = [e for e in evts_after if e["id"] == evt_id]
        assert found_after[0]["acknowledged"] is True

    # 10. Measured Benchmark Results File Validity
    def test_benchmark_results_structure(self):
        bench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json"))
        assert os.path.exists(bench_path)
        with open(bench_path, "r") as f:
            b = json.load(f)
        assert "total_frames_evaluated" in b
        assert "event_level_precision_pct" in b
        assert "event_level_recall_pct" in b
        assert "p95_tail_latency_ms" in b
        assert b["ground_truth_events"] == 2
        assert b["true_positives"] == 2
        assert b["false_positives"] == 0
        assert b["event_level_f1_pct"] == 100.0
