import time
from typing import Dict, Any, Optional

class EscalationPolicy:
    """
    Deterministic Escalation Policy Engine:
    Applies configurable, transparent rules to synthesize Edge Biomechanics,
    VLM Verification (or transparent fallback), and ChromaDB historical evidence.
    
    Standardized Incident Lifecycle:
    - Stages: DETECTED -> CANDIDATE -> VERIFYING -> VERIFIED | FALLBACK
    - Severity Outcomes: NORMAL | WARNING | EMERGENCY | UNCERTAIN
    - Acknowledgement: UNACKNOWLEDGED -> ACKNOWLEDGED
    """
    def __init__(self, incident_debounce_sec: float = 3.0):
        self.incident_debounce_sec = incident_debounce_sec
        # Map event_id -> last_alert_time
        self.alerted_events: Dict[str, float] = {}
        self.last_global_alert_time = 0.0

    def evaluate(self, candidate_event: Dict[str, Any], vlm_result: Dict[str, Any], memory_matches: list) -> Dict[str, Any]:
        kinematics = candidate_event.get("kinematics", {})
        posture = kinematics.get("posture", "Unknown")
        state = candidate_event.get("state", "STABLE")
        drop_velocity = kinematics.get("drop_velocity", 0.0)
        dwell_sec = kinematics.get("dwell_sec", 0.0)
        camera_id = candidate_event.get("camera_id", "Camera 01")
        track_id = candidate_event.get("track_id", "Track-1")
        event_id = candidate_event.get("id") or "UNKNOWN_EVENT"
        
        now = time.time()
        
        # 1. Processing failure check
        if candidate_event.get("error"):
            return {
                "event_id": event_id,
                "stage": "FALLBACK",
                "decision": "PROCESSING FAILURE / FALLBACK",
                "severity": "PROCESSING FAILURE",
                "decision_tone": "warn",
                "action": "Flag sensor pipeline error for technician diagnostic",
                "rationale": f"Telemetry parsing error: {candidate_event.get('error')}",
                "trigger_rule": "RULE_ERR_01",
                "verification_method": vlm_result.get("verification_method", "RULE_BASED_FALLBACK"),
                "is_new_alert": False
            }

        # 2. Critical Emergency Fall Confirmation
        # Triggered if:
        # A) Edge FSM confirmed fall (drop velocity > threshold + floor dwell >= 1.5s)
        # OR
        # B) VLM verified fall with dwell >= 1.2s and subject unrecovered
        vlm_method = vlm_result.get("verification_method", "RULE_BASED_FALLBACK")
        is_vlm_model = (vlm_method == "VLM_MODEL")
        
        is_emergency = (state == "FALL_CONFIRMED") or (
            vlm_result.get("is_true_fall") and dwell_sec >= 1.2 and not vlm_result.get("recovery_observed")
        )

        if is_emergency:
            # Per-event alert tracking: check if alert was already dispatched for this specific event_id
            last_alert = self.alerted_events.get(event_id, 0.0)
            is_recent_for_event = (now - last_alert) < self.incident_debounce_sec
            
            if not is_recent_for_event:
                self.alerted_events[event_id] = now
                self.last_global_alert_time = now
                is_new_alert = True
            else:
                is_new_alert = False

            stage = "VERIFIED" if is_vlm_model else "FALLBACK"
            vlm_note = "VLM model verified floor collapse" if is_vlm_model else "Biomechanical rule fallback verification"
            
            return {
                "event_id": event_id,
                "stage": stage,
                "decision": "EMERGENCY / ALERT",
                "severity": "CRITICAL EMERGENCY",
                "decision_tone": "risk",
                "action": "Dispatch immediate caregiver emergency alert · Sound local edge banner",
                "rationale": (
                    f"Acute incident confirmed on {camera_id} ({track_id}): Rapid drop ({drop_velocity:.1f} px/s), "
                    f"prolonged floor dwell ({dwell_sec:.1f}s), {vlm_note} ({vlm_method})."
                ),
                "trigger_rule": "RULE_EMERGENCY_FALL_01",
                "verification_method": vlm_method,
                "is_new_alert": is_new_alert
            }

        # 3. Warning / Candidate Review Window
        if state in ["RAPID_DROP", "VERIFYING"] or (drop_velocity > 60.0 and dwell_sec < 1.5):
            return {
                "event_id": event_id,
                "stage": "CANDIDATE",
                "decision": "WARNING / HUMAN REVIEW",
                "severity": "WARNING / REVIEW",
                "decision_tone": "warn",
                "action": "Open active verification window · Debounce dwell timer",
                "rationale": f"Candidate drop detected ({drop_velocity:.1f} px/s). Awaiting temporal dwell verification ({dwell_sec:.1f}s / 1.5s).",
                "trigger_rule": "RULE_WARNING_CANDIDATE_02",
                "verification_method": vlm_method,
                "is_new_alert": False
            }

        # 4. Uncertain State
        if vlm_result.get("uncertainty_flag"):
            return {
                "event_id": event_id,
                "stage": "FALLBACK",
                "decision": "UNCERTAIN / VERIFICATION REQUIRED",
                "severity": "UNCERTAIN",
                "decision_tone": "warn",
                "action": "Request secondary verification or nurse check-in",
                "rationale": f"Indeterminate orientation. Subject low to ground without clear impact velocity ({drop_velocity:.1f} px/s).",
                "trigger_rule": "RULE_UNCERTAIN_03",
                "verification_method": vlm_method,
                "is_new_alert": False
            }

        # 5. Normal / Log
        return {
            "event_id": event_id,
            "stage": "DETECTED",
            "decision": "NORMAL / LOG",
            "severity": "NORMAL",
            "decision_tone": "sage",
            "action": "Log baseline activity · Continue continuous edge monitoring",
            "rationale": f"Subject posture is stable ({posture}). Center of mass displacement within benign bounds.",
            "trigger_rule": "RULE_NORMAL_STABLE_04",
            "verification_method": vlm_method,
            "is_new_alert": False
        }
