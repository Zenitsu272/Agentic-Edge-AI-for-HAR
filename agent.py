import time
import uuid
from typing import Dict, Any, List, Optional
import database
from central_server import ChromaAgentMemory
from vlm_adapter import VLMAdapter
from policy import EscalationPolicy

class CareWatchAgent:
    """
    Bounded, Tool-Using Agent for Candidate Fall Event Investigation.
    Follows a ReAct (Reason -> Act -> Observe) loop bounded to maximum 4 steps.
    
    Tools:
    1. tool_inspect_candidate: inspects kinematic measurements and trajectory.
    2. tool_query_memory: retrieves similar empirical baselines from ChromaDB.
    3. tool_inspect_history: inspects temporal tracking and displacement sequence.
    4. tool_verify_vlm: invokes the VLM adapter on candidate event.
    5. tool_apply_policy: computes deterministic escalation decision.
    """
    def __init__(self, memory: ChromaAgentMemory, vlm: VLMAdapter, policy: EscalationPolicy):
        self.memory = memory
        self.vlm = vlm
        self.policy = policy
        self.max_steps = 4

    def run_investigation(self, candidate_event: Dict[str, Any], tracking_history: Optional[List] = None) -> Dict[str, Any]:
        t_start = time.time()
        run_id = f"AGENT-RUN-{uuid.uuid4().hex[:8].upper()}"
        event_id = candidate_event.get("id") or f"EVT-CAND-{int(t_start*1000)%1000000}"
        
        tools_executed = []
        evidence = {}
        
        # --- STEP 1: Tool Call - Inspect Candidate Event ---
        step1_res = self._tool_inspect_candidate(candidate_event)
        tools_executed.append({
            "step": 1,
            "tool": "inspect_candidate",
            "input": {"candidate_id": event_id},
            "output": step1_res
        })
        evidence["candidate"] = step1_res

        # --- STEP 2: Tool Call - Query ChromaDB Historical Memory ---
        kin = candidate_event.get("kinematics", {})
        query = f"Camera {candidate_event.get('camera_id', '01')} {kin.get('posture', '')} drop {kin.get('drop_velocity', 0):.1f} dwell {kin.get('dwell_sec', 0):.1f}s"
        step2_res = self._tool_query_memory(query, n_results=2)
        tools_executed.append({
            "step": 2,
            "tool": "query_memory",
            "input": {"query": query, "n_results": 2},
            "output": {"matches_count": len(step2_res), "top_match": step2_res[0]["document"] if step2_res else "No match"}
        })
        evidence["historical_memory"] = step2_res

        # --- STEP 3: Tool Call - Verify with VLM Adapter ---
        step3_res = self._tool_verify_vlm(candidate_event)
        tools_executed.append({
            "step": 3,
            "tool": "verify_vlm",
            "input": {"method": step3_res.get("verification_method")},
            "output": {
                "is_true_fall": step3_res.get("is_true_fall"),
                "confidence": step3_res.get("confidence"),
                "scene_observation": step3_res.get("scene_observation", "")[:120]
            }
        })
        evidence["vlm_verification"] = step3_res

        # --- STEP 4: Tool Call - Apply Deterministic Escalation Policy ---
        step4_res = self._tool_apply_policy(candidate_event, step3_res, step2_res)
        tools_executed.append({
            "step": 4,
            "tool": "apply_policy",
            "input": {"rule_set": "EscalationPolicy_v1"},
            "output": {
                "decision": step4_res.get("decision"),
                "severity": step4_res.get("severity"),
                "action": step4_res.get("action")
            }
        })
        evidence["policy"] = step4_res

        duration_ms = round((time.time() - t_start) * 1000.0, 1)

        # Operational summary
        audit_trace = {
            "run_id": run_id,
            "event_id": event_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tools_called": tools_executed,
            "evidence_summary": f"Kinematics: drop {kin.get('drop_velocity', 0):.1f}px/s, dwell {kin.get('dwell_sec', 0):.1f}s; VLM: {step3_res.get('scene_observation')[:100]}",
            "recommendation": step4_res.get("action"),
            "policy_action": step4_res.get("decision"),
            "duration_ms": duration_ms
        }

        # Persist audit trace to SQLite
        try:
            database.log_agent_trace(audit_trace)
        except Exception as e:
            print("[CareWatchAgent] Warning: could not log trace to SQLite:", e)

        return {
            "run_id": run_id,
            "event_id": event_id,
            "duration_ms": duration_ms,
            "tools_executed": tools_executed,
            "evidence": evidence,
            "escalation_decision": step4_res.get("decision"),
            "decision_tone": step4_res.get("decision_tone", "sage"),
            "recommended_action": step4_res.get("action"),
            "agent_rationale": step4_res.get("rationale"),
            "vlm_verification": step3_res,
            "chroma_memory_matches": step2_res
        }

    def _tool_inspect_candidate(self, candidate_event: Dict[str, Any]) -> Dict[str, Any]:
        kin = candidate_event.get("kinematics", {})
        return {
            "event_id": candidate_event.get("id"),
            "camera_id": candidate_event.get("camera_id", "Camera 01"),
            "posture": kin.get("posture", "Unknown"),
            "torso_angle": kin.get("torso_angle", 0.0),
            "drop_velocity": kin.get("drop_velocity", 0.0),
            "dwell_sec": kin.get("dwell_sec", 0.0),
            "state": candidate_event.get("state", "STABLE")
        }

    def _tool_query_memory(self, query: str, n_results: int = 2) -> List[Dict[str, Any]]:
        return self.memory.retrieve_context(query, n_results=n_results)

    def _tool_verify_vlm(self, candidate_event: Dict[str, Any]) -> Dict[str, Any]:
        return self.vlm.verify_candidate_event(candidate_event)

    def _tool_apply_policy(self, candidate_event: Dict[str, Any], vlm_res: Dict[str, Any], memory_matches: list) -> Dict[str, Any]:
        return self.policy.evaluate(candidate_event, vlm_res, memory_matches)
