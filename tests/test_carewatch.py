import os
import sys
import time
import json
import asyncio
from unittest.mock import MagicMock
import numpy as np
import pytest
from PIL import Image

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import database
from tracker import CentroidTracker
from vlm_adapter import VLMAdapter
from policy import EscalationPolicy
from central_server import ChromaAgentMemory
from agent import CareWatchAgent
from agent_graph import IncidentInvestigationGraph, IncidentCoordinator
from benchmark_evaluation import evaluate_predictions_against_ground_truth

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
        tracks1 = tracker.update([{"box": [80, 80, 120, 160], "conf": 0.9}], timestamp=1.0)
        assert len(tracks1) == 1
        assert tracks1[0].track_id == 1
        assert tracks1[0].movement_status == "Stationary"

        tracks2 = tracker.update([{"box": [110, 110, 150, 190], "conf": 0.9}], timestamp=1.05)
        assert len(tracks2) == 1
        assert tracks2[0].track_id == 1
        assert tracks2[0].is_moving or tracks2[0].displacement_px_sec > 50.0

    # 2. Tracking lifecycle & missed detection tolerance
    def test_track_missed_frames_and_termination(self):
        tracker = CentroidTracker(max_missed_frames=5, max_distance=100.0)
        tracker.update([{"box": [50, 50, 100, 150], "conf": 0.9}], timestamp=1.0)
        
        tracks = tracker.update([], timestamp=1.1)
        assert len(tracks) == 1
        assert tracks[0].missed_frames == 1

        for i in range(6):
            tracks = tracker.update([], timestamp=1.2 + i*0.1)
        assert len(tracks) == 0

    # 3. Posture classification: Sitting vs Lying vs Standing
    def test_posture_classification_rules(self):
        cand_standing = {"kinematics": {"posture": "Standing", "torso_angle": 8.0, "dwell_sec": 0.0, "drop_velocity": 0.0}}
        res_standing = self.vlm.verify_candidate_event(cand_standing)
        assert res_standing["is_true_fall"] is False
        assert res_standing["confidence"] >= 0.90

        cand_sitting = {"kinematics": {"posture": "Sitting (Leaning Forward)", "torso_angle": 38.0, "dwell_sec": 0.0, "drop_velocity": 0.0}}
        res_sitting = self.vlm.verify_candidate_event(cand_sitting)
        assert res_sitting["is_true_fall"] is False

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
        assert self.vlm.status in ["ONLINE", "WEIGHTS_INCOMPLETE", "OFFLINE_FALLBACK", "WEIGHTS_MISSING"]
        cand = {"kinematics": {"posture": "Standing", "torso_angle": 5.0, "dwell_sec": 0.0, "drop_velocity": 0.0}}
        res = self.vlm.verify_candidate_event(cand)
        assert "verification_method" in res
        assert "adapter_status" in res

    # 7. Agent Bounded LangGraph Workflow & Operational Audit Traces
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
        assert investigation["event_id"] == "EVT-TEST-AGENT-101"
        assert len(investigation["tools_executed"]) <= 6
        assert investigation["escalation_decision"] == "NORMAL / LOG"
        assert investigation["orchestrator"] == "LangGraph"

        traces = database.get_recent_agent_traces(limit=5)
        assert len(traces) > 0
        latest = traces[0]
        assert latest["event_id"] == "EVT-TEST-AGENT-101"
        assert len(latest["tools_called"]) > 0

    # 8. ChromaDB Persistent Retrieval & Storage
    def test_chroma_db_persistence_and_retrieval(self):
        test_id = f"EVT-TEST-CHROMA-{int(time.time()*1000)%100000}"
        doc = "Camera 01 Test Incident: verified collapse near bed with 95 px/s downward velocity."
        meta = {"category": "Fall", "camera": "Camera 01", "drop_velocity_px_s": 95.0, "provenance": "REAL_PIPELINE"}
        self.chroma.log_new_event(test_id, doc, meta)

        matches = self.chroma.retrieve_context("collapse near bed with rapid drop", n_results=2)
        assert len(matches) > 0
        top = matches[0]
        assert "raw_distance" in top
        assert "cosine_similarity" in top
        assert "provenance" in top

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
            "vlm_status": "RULE_BASED_FALLBACK",
            "policy_decision": "EMERGENCY / ALERT",
            "details": "Simulated unit test fall event for persistence test.",
            "posture_trajectory": "Upright -> Fall",
            "acknowledged": False
        })

        evts = database.get_events(limit=10)
        found = [e for e in evts if e["id"] == evt_id]
        assert len(found) == 1
        assert found[0]["acknowledged"] is False

        acked = database.acknowledge_event_by_id(evt_id)
        assert acked is True

        evts_after = database.get_events(limit=10)
        found_after = [e for e in evts_after if e["id"] == evt_id]
        assert found_after[0]["acknowledged"] is True

    # 10. Measured Benchmark Results File Validity
    def test_benchmark_results_structure(self):
        bench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "benchmark_results.json"))
        if not os.path.exists(bench_path):
            bench_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "benchmark_results.json"))
        assert os.path.exists(bench_path)
        with open(bench_path, "r") as f:
            b = json.load(f)
        assert "total_frames_evaluated" in b
        assert "event_level_precision_pct" in b
        assert "event_level_recall_pct" in b
        assert "p95_tail_latency_ms" in b
        assert b["true_positives"] == 2
        assert b["false_positives"] == 0
        assert b["event_level_f1_pct"] == 100.0

    # 11. Support and Decoding of .avi Video Format
    def test_avi_video_decoding_and_pipeline(self):
        import cv2
        from inference import InferencePipeline
        pipe = InferencePipeline()
        
        avi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "replays", "sample_fall_clip.avi"))
        if not os.path.exists(avi_path):
            avi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample_fall_clip.avi"))
        assert os.path.exists(avi_path), f".avi test video missing at {avi_path}"
        
        cap = cv2.VideoCapture(avi_path)
        assert cap.isOpened(), "OpenCV failed to open .avi video"
        
        ret, frame = cap.read()
        cap.release()
        assert ret is True and frame is not None, "Failed to read frame from .avi video"
        
        res = pipe.process_unified(frame)
        assert "stage1_presence" in res
        assert "stage2_posture" in res
        assert "stage3_event" in res
        assert res["inference_ms"] > 0.0

    # ==================== AUDIT SPECIFIC REGRESSION TESTS (P0 & P1) ====================

    # 12. P0-1 Genuine VLM Online Inference Forward Pass & Output Parsing
    def test_genuine_vlm_forward_pass_and_structured_parsing(self):
        adapter = VLMAdapter(auto_load=False)
        adapter.status = "ONLINE"
        
        # Mock model and processor to verify genuine forward pass without requiring GPU weights download
        mock_processor = MagicMock()
        mock_model = MagicMock()
        
        # Mock tokenizer/chat template output
        mock_processor.apply_chat_template.return_value = "<user> inspect </user>"
        import torch
        mock_inputs = {
            "input_ids": torch.tensor([[101, 102, 103]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_processor.return_value = MagicMock(
            to=lambda dev: mock_inputs,
            items=lambda: mock_inputs.items()
        )
        # Mock generated token IDs
        mock_model.device = torch.device("cpu")
        mock_model.generate.return_value = torch.tensor([[101, 102, 103, 201, 202, 203]])
        
        # Mock JSON response from Qwen2.5-VL
        expected_json = json.dumps({
            "action_observed": "Subject lost balance and collapsed onto the carpeted floor.",
            "posture": "lying_floor",
            "floor_contact": True,
            "voluntary_recovery": False,
            "is_true_fall": True,
            "confidence": 0.94,
            "uncertainty": False,
            "scene_observation": "Clear floor collapse verified with prolonged dwell."
        })
        mock_processor.decode.return_value = f"```json\n{expected_json}\n```"
        
        adapter.model = mock_model
        adapter.processor = mock_processor
        
        # Create synthetic test frame
        dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        cand = {
            "id": "EVT-MOCK-VLM-01",
            "camera_id": "Camera 01",
            "kinematics": {"posture": "Lying / Resting", "drop_velocity": 85.0, "dwell_sec": 2.0}
        }
        
        res = adapter.verify_candidate_event(cand, evidence_frames=[{"frame": dummy_frame, "timestamp_sec": 12.5}])
        
        assert res["verification_method"] == "VLM_MODEL"
        assert res["is_true_fall"] is True
        assert res["confidence"] == 0.94
        assert res["evidence_frames_count"] == 1
        assert "floor collapse" in res["scene_observation"]
        assert mock_model.generate.called, "model.generate() must be called during VLM forward pass"

    # 13. P0-1 VLM Failure / Incomplete Weights Never Claims VLM_MODEL
    def test_vlm_failure_never_claims_vlm_model(self):
        adapter = VLMAdapter(auto_load=False)
        adapter.status = "WEIGHTS_INCOMPLETE"
        adapter.status_detail = "Incomplete weights detected on disk."
        
        cand = {
            "id": "EVT-FALLBACK-01",
            "kinematics": {"posture": "Lying / Resting", "drop_velocity": 90.0, "dwell_sec": 2.0}
        }
        res = adapter.verify_candidate_event(cand)
        assert res["verification_method"] == "RULE_BASED_FALLBACK"
        assert res["verification_method"] != "VLM_MODEL"
        assert res["adapter_status"] == "WEIGHTS_INCOMPLETE"

    # 14. P0-2 Normal Video Frames Do Not Trigger LangGraph Investigation
    def test_ordinary_frames_do_not_trigger_investigation(self):
        coordinator = self.agent.coordinator
        # Normal frame with STABLE state
        stage3_normal = {
            "state": "STABLE",
            "fall_confirmed": False,
            "drop_velocity": 0.0,
            "ground_dwell_sec": 0.0
        }
        assert coordinator.should_trigger(stage3_normal) is False

        # Candidate frame triggers
        stage3_candidate = {
            "state": "VERIFYING",
            "fall_confirmed": False,
            "drop_velocity": 85.0,
            "ground_dwell_sec": 0.8
        }
        assert coordinator.should_trigger(stage3_candidate) is True

    # 15. P0-2 Deduplication of Concurrent Investigations for Same Incident
    def test_investigation_deduplication(self):
        async def _run():
            coordinator = self.agent.coordinator
            cand = {
                "id": "EVT-DEDUP-001",
                "camera_id": "Camera 01",
                "state": "RAPID_DROP",
                "kinematics": {"posture": "Lying / Resting", "drop_velocity": 90.0, "dwell_sec": 1.5}
            }
            
            task1 = asyncio.create_task(coordinator.dispatch_investigation_async(cand))
            task2 = asyncio.create_task(coordinator.dispatch_investigation_async(cand))
            await asyncio.gather(task1, task2)
            assert coordinator.queue_depth <= 1

        asyncio.run(_run())

    # 16. P0-2 LangGraph Bounded Deadline Enforcement
    def test_langgraph_deadline_enforcement(self):
        graph = IncidentInvestigationGraph(self.chroma, self.vlm, self.policy, timeout_sec=0.001)
        # Force expired deadline
        cand = {
            "id": "EVT-EXPIRED-01",
            "deadline": time.time() - 10.0,
            "kinematics": {"posture": "Standing"}
        }
        res = graph.execute(cand)
        assert res["incident_status"] in ["EXPIRED", "NORMAL", "FALLBACK"]

    # 17. P0-3 Incident Lifecycle Transitions
    def test_incident_lifecycle_stages(self):
        # Candidate stage
        cand_warn = {
            "id": "EVT-LIFE-01",
            "state": "RAPID_DROP",
            "kinematics": {"posture": "Standing", "drop_velocity": 70.0, "dwell_sec": 0.5}
        }
        pol_warn = self.policy.evaluate(cand_warn, {"is_true_fall": False, "verification_method": "RULE_BASED_FALLBACK"}, [])
        assert pol_warn["stage"] == "CANDIDATE"
        assert pol_warn["decision"] == "WARNING / HUMAN REVIEW"

        # Confirmed emergency stage
        cand_emerg = {
            "id": "EVT-LIFE-02",
            "state": "FALL_CONFIRMED",
            "kinematics": {"posture": "Lying / Resting", "drop_velocity": 90.0, "dwell_sec": 2.0}
        }
        pol_emerg = self.policy.evaluate(cand_emerg, {"is_true_fall": True, "verification_method": "RULE_BASED_FALLBACK"}, [])
        assert pol_emerg["stage"] == "FALLBACK"
        assert pol_emerg["decision"] == "EMERGENCY / ALERT"

    # 18. P0-4 Production Mode Disables Synthetic Demo Seeding
    def test_production_mode_no_demo_seeding(self, tmp_path):
        test_db = str(tmp_path / "prod_carewatch.db")
        # Ensure CAREWATCH_DEMO_MODE is not set or 0
        old_env = os.environ.get("CAREWATCH_DEMO_MODE")
        try:
            os.environ["CAREWATCH_DEMO_MODE"] = "0"
            database.init_db(test_db)
            evts = database.get_events(limit=50, db_path=test_db)
            assert len(evts) == 0, "A fresh production database must contain zero seeded synthetic events"
        finally:
            if old_env is not None:
                os.environ["CAREWATCH_DEMO_MODE"] = old_env
            else:
                os.environ.pop("CAREWATCH_DEMO_MODE", None)

    # 19. P0-4 Explicit Demo Mode Tags Synthetic Provenance
    def test_demo_mode_tags_synthetic_provenance(self, tmp_path):
        test_db = str(tmp_path / "demo_carewatch.db")
        old_env = os.environ.get("CAREWATCH_DEMO_MODE")
        try:
            os.environ["CAREWATCH_DEMO_MODE"] = "1"
            database.init_db(test_db)
            evts = database.get_events(limit=50, db_path=test_db)
            assert len(evts) > 0, "Demo mode must seed demo records"
            for e in evts:
                assert e["provenance"] == "DEMO_SYNTHETIC"
                assert e["is_synthetic"] is True
        finally:
            if old_env is not None:
                os.environ["CAREWATCH_DEMO_MODE"] = old_env
            else:
                os.environ.pop("CAREWATCH_DEMO_MODE", None)

    # 20. P0-5 Strict One-to-One Matching and Two Nearby Falls Independence
    def test_one_to_one_matching_and_nearby_falls(self):
        gt = {
            "positive_fall_events": [
                {"event_id": "GT-1", "start_sec": 78.0, "end_sec": 82.0, "peak_sec": 79.03},
                {"event_id": "GT-2", "start_sec": 85.0, "end_sec": 89.0, "peak_sec": 86.53}
            ],
            "matching_criteria": {"temporal_tolerance_sec": 3.0}
        }
        
        # Two distinct predictions 7.5s apart matching the two distinct GT falls
        predictions = [
            {"event_id": "P-1", "timestamp_sec": 79.03},
            {"event_id": "P-2", "timestamp_sec": 86.53}
        ]
        
        res = evaluate_predictions_against_ground_truth(predictions, gt)
        assert res["true_positives"] == 2
        assert res["false_positives"] == 0
        assert res["false_negatives"] == 0
        assert res["f1_pct"] == 100.0
        assert len(res["matches"]) == 2
        # Verify 1-to-1: each prediction matched a unique GT
        matched_gt_ids = {m["gt_event_id"] for m in res["matches"]}
        assert len(matched_gt_ids) == 2

    # 21. P0-5 One Prediction Cannot Match Multiple Ground Truths
    def test_single_prediction_cannot_double_count(self):
        gt = {
            "positive_fall_events": [
                {"event_id": "GT-1", "start_sec": 78.0, "end_sec": 82.0, "peak_sec": 79.0},
                {"event_id": "GT-2", "start_sec": 80.0, "end_sec": 84.0, "peak_sec": 81.0}
            ],
            "matching_criteria": {"temporal_tolerance_sec": 3.0}
        }
        # Only 1 prediction
        predictions = [{"event_id": "P-1", "timestamp_sec": 80.5}]
        res = evaluate_predictions_against_ground_truth(predictions, gt)
        assert res["true_positives"] == 1
        assert res["false_negatives"] == 1

    # 22. P1-6 Genuine System Health Reports Unconfigured and Fallback State
    def test_system_health_genuine_reporting(self):
        from main import get_system_health
        health = get_system_health()
        assert health["status"] in ["ONLINE", "DEGRADED", "UNAVAILABLE"]
        comps = health["components"]
        assert comps["remote_notification_provider"]["status"] == "NOT_CONFIGURED"
        assert comps["person_detector"]["status"] == "ONLINE"
        assert comps["pose_estimator"]["status"] == "ONLINE"
        assert comps["langgraph_orchestration"]["status"] == "ONLINE"

    # 23. P1-7 Historical Memory Cannot Independently Turn Non-Fall into Fall
    def test_historical_memory_cannot_fabricate_fall(self):
        # Even if ChromaDB returns past fall matches, current stable standing must remain NORMAL
        cand_standing = {
            "id": "EVT-STABLE-01",
            "camera_id": "Camera 01",
            "state": "STABLE",
            "kinematics": {"posture": "Standing", "drop_velocity": 0.0, "dwell_sec": 0.0}
        }
        vlm_res = {"is_true_fall": False, "recovery_observed": False, "verification_method": "RULE_BASED_FALLBACK"}
        past_fall_matches = [{"document": "Past fall incident on Camera 01", "cosine_similarity": 0.99}]
        
        decision = self.policy.evaluate(cand_standing, vlm_res, past_fall_matches)
        assert decision["decision"] == "NORMAL / LOG"
        assert decision["severity"] == "NORMAL"
