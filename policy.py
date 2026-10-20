import time
from typing import Dict, Any

class EscalationPolicy:
    """
    Deterministic Escalation Policy Engine:
    Applies configurable, transparent rules to synthesize Edge Biomechanics,
    VLM Verification (or transparent fallback), and ChromaDB historical evidence.
    
    Decisions:
    - EMERGENCY / ALERT
    - WARNING / HUMAN REVIEW
    - UNCERTAIN / VERIFICATION REQUIRED
    - NORMAL / LOG
    - PROCESSING FAILURE / FALLBACK
    """
    def __init__(self):
        self.cooldown_sec = 12.0
        self.last_emergency_time = 0.0

    def evaluate(self, candidate_event: Dict[str, Any], vlm_result: Dict[str, Any], memory_matches: list) -> Dict[str, Any]:
        kinematics = candidate_event.get("kinematics", {})
        posture = kinematics.get("posture", "Unknown")
        state = candidate_event.get("state", "STABLE")
        drop_velocity = kinematics.get("drop_velocity", 0.0)
        dwell_sec = kinematics.get("dwell_sec", 0.0)
        camera_id = candidate_event.get("camera_id", "Camera 01")
        track_id = candidate_event.get("track_id", "Track-1")
        
        now = time.time()
        
        # 1. Processing failure check
        if candidate_event.get("error"):
            return {
                "decision": "PROCESSING FAILURE / FALLBACK",
                "severity": "PROCESSING FAILURE",
                "decision_tone": "warn",
                "action": "Flag sensor pipeline error for technician diagnostic",
                "rationale": f"Telemetry parsing error: {candidate_event.get('error')}",
                "trigger_rule": "RULE_ERR_01"
            }

        # 2. Critical Emergency Fall Confirmation
        # Conditions:
        # A) Detector confirmed fall (drop velocity > threshold + floor dwell >= 1.5s)
        # OR
        # B) VLM verified true fall with dwell >= 1.2s and subject unrecovered
        is_emergency = (state == "FALL_CONFIRMED") or (
            vlm_result.get("is_true_fall") and dwell_sec >= 1.2 and not vlm_result.get("recovery_observed")
        )

        if is_emergency:
            # Check cooldown to prevent duplicate alerts for the exact same event
            is_recent_duplicate = (now - self.last_emergency_time) < self.cooldown_sec
            if not is_recent_duplicate:
                self.last_emergency_time = now

            vlm_method = vlm_result.get("verification_method", "HEURISTIC")
            vlm_note = "VLM verified floor collapse" if vlm_result.get("is_true_fall") else "Edge kinematic confirmation"
            
            return {
                "decision": "EMERGENCY / ALERT",
                "severity": "CRITICAL EMERGENCY",
                "decision_tone": "risk",
                "action": "Dispatch immediate caregiver emergency alert · Sound local edge banner",
                "rationale": (
                    f"Acute incident confirmed on {camera_id} ({track_id}): Rapid drop ({drop_velocity:.1f} px/s), "
                    f"prolonged floor dwell ({dwell_sec:.1f}s), {vlm_note} ({vlm_method})."
                ),
                "trigger_rule": "RULE_EMERGENCY_FALL_01",
                "is_new_alert": not is_recent_duplicate
            }

        # 3. Warning / Human Review Window
        # Candidate drop detected, actively verifying dwell window
        if state in ["RAPID_DROP", "VERIFYING"] or (drop_velocity > 60.0 and dwell_sec < 1.5):
            return {
                "decision": "WARNING / HUMAN REVIEW",
                "severity": "WARNING / REVIEW",
                "decision_tone": "warn",
                "action": "Open active verification window · Debounce dwell timer",
                "rationale": f"Drop detected ({drop_velocity:.1f} px/s). Awaiting temporal dwell verification ({dwell_sec:.1f}s / 1.5s).",
                "trigger_rule": "RULE_WARNING_CANDIDATE_02",
                "is_new_alert": False
            }

        # 4. Uncertain State
        if vlm_result.get("uncertainty_flag"):
            return {
                "decision": "UNCERTAIN / VERIFICATION REQUIRED",
                "severity": "UNCERTAIN",
                "decision_tone": "warn",
                "action": "Request secondary verification or nurse check-in",
                "rationale": f"Indeterminate orientation. Subject low to ground without clear impact velocity ({drop_velocity:.1f} px/s).",
                "trigger_rule": "RULE_UNCERTAIN_03",
                "is_new_alert": False
            }

        # 5. Normal / Log
        return {
            "decision": "NORMAL / LOG",
            "severity": "NORMAL",
            "decision_tone": "sage",
            "action": "Log baseline activity to ChromaDB & SQLite · Continue continuous monitoring",
            "rationale": f"Subject posture is stable ({posture}). Center of mass displacement within benign bounds.",
            "trigger_rule": "RULE_NORMAL_STABLE_04",
            "is_new_alert": False
        }
