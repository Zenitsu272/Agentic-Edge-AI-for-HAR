import time
import uuid
from typing import Dict, Any, List, Optional
import database
from central_server import ChromaAgentMemory
from vlm_adapter import VLMAdapter
from policy import EscalationPolicy
from agent_graph import IncidentInvestigationGraph, IncidentCoordinator

class CareWatchAgent:
    """
    Agentic Orchestrator for Incident Verification and Escalation.
    Backed by LangGraph (IncidentInvestigationGraph) with bounded execution,
    deterministic tool calling, transparent provenance, and deduplication.
    """
    def __init__(self, memory: ChromaAgentMemory, vlm: VLMAdapter, policy: EscalationPolicy, max_retries: int = 2):
        self.memory = memory
        self.vlm = vlm
        self.policy = policy
        self.max_steps = 4
        self.graph = IncidentInvestigationGraph(memory, vlm, policy, max_retries=max_retries)
        self.coordinator = IncidentCoordinator(self.graph)

    def run_investigation(self, candidate_event: Dict[str, Any], tracking_history: Optional[List] = None, evidence_frames: Optional[List] = None) -> Dict[str, Any]:
        """
        Executes bounded LangGraph investigation for a candidate incident.
        """
        t_start = time.time()
        event_id = candidate_event.get("id") or f"EVT-CAND-{int(t_start*1000)%1000000}"
        candidate_event["id"] = event_id
        
        # Execute through LangGraph workflow
        lg_state = self.graph.execute(candidate_event, evidence_frames=evidence_frames)
        
        run_id = f"AGENT-LG-{uuid.uuid4().hex[:8].upper()}"
        duration_ms = lg_state.get("duration_ms", round((time.time() - t_start) * 1000.0, 1))
        
        policy_res = lg_state.get("policy_decision", {})
        vlm_res = lg_state.get("vlm_result", {})
        tools_executed = lg_state.get("tools_executed", [])
        
        return {
            "run_id": run_id,
            "event_id": event_id,
            "orchestrator": "LangGraph",
            "duration_ms": duration_ms,
            "tools_executed": tools_executed,
            "evidence": {
                "candidate": self._tool_inspect_candidate(candidate_event),
                "vlm_verification": vlm_res,
                "policy": policy_res,
                "incident_status": lg_state.get("incident_status")
            },
            "escalation_decision": policy_res.get("decision", "NORMAL / LOG"),
            "decision_tone": policy_res.get("decision_tone", "sage"),
            "recommended_action": policy_res.get("action", "MONITOR"),
            "agent_rationale": lg_state.get("agent_rationale", policy_res.get("rationale", "")),
            "vlm_verification": vlm_res,
            "incident_status": lg_state.get("incident_status", "DETECTED"),
            "chroma_memory_matches": []
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
