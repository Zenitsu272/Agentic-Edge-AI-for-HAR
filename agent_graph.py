import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, TypedDict
from langgraph.graph import StateGraph, START, END

import database
from policy import EscalationPolicy
from vlm_adapter import VLMAdapter
from central_server import ChromaAgentMemory

class IncidentState(TypedDict):
    """
    Typed, bounded incident state for LangGraph incident investigation workflow.
    Ensures stable event identity, bounded memory, and explicit lifecycle stages.
    """
    event_id: str
    camera_id: str
    track_id: str
    candidate_start_time: float
    latest_event_time: float
    detection_state: str  # e.g., 'RAPID_DROP', 'VERIFYING', 'FALL_CONFIRMED'
    kinematics: Dict[str, Any]
    evidence_frames: List[Dict[str, Any]]
    evidence_status: str  # 'SUFFICIENT' | 'INSUFFICIENT' | 'REQUEST_MORE'
    vlm_result: Dict[str, Any]
    verification_method: str  # 'VLM_MODEL' | 'RULE_BASED_FALLBACK'
    policy_decision: Dict[str, Any]
    incident_status: str  # 'DETECTED' -> 'CANDIDATE' -> 'VERIFYING' -> 'VERIFIED' | 'FALLBACK'
    severity: str  # 'NORMAL' | 'WARNING' | 'EMERGENCY' | 'UNCERTAIN'
    retry_count: int
    max_retries: int
    deadline: float
    persistence_status: str  # 'PENDING' | 'PERSISTED' | 'FAILED'
    notification_status: str  # 'NOT_REQUIRED' | 'DISPATCHED_LOCAL' | 'REMOTE_NOT_CONFIGURED' | 'FAILED'
    tools_executed: List[Dict[str, Any]]
    agent_rationale: str
    error: Optional[str]
    duration_ms: float

class IncidentInvestigationGraph:
    """
    Bounded LangGraph workflow for incident investigation:
    Validate Event -> Select Evidence Frames -> VLM Verification -> Apply Policy -> Persist Event -> Notify if Required -> Finalize Incident
    """
    def __init__(self, memory: ChromaAgentMemory, vlm: VLMAdapter, policy: EscalationPolicy, max_retries: int = 2, timeout_sec: float = 5.0):
        self.memory = memory
        self.vlm = vlm
        self.policy = policy
        self.max_retries = max_retries
        self.timeout_sec = timeout_sec
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(IncidentState)

        # 1. Define Nodes
        builder.add_node("validate_event", self.node_validate_event)
        builder.add_node("select_evidence_frames", self.node_select_evidence_frames)
        builder.add_node("vlm_verify", self.node_vlm_verify)
        builder.add_node("apply_policy", self.node_apply_policy)
        builder.add_node("persist_event", self.node_persist_event)
        builder.add_node("notify_if_required", self.node_notify_if_required)
        builder.add_node("finalize", self.node_finalize)

        # 2. Define Flow & Conditional Edges
        builder.add_edge(START, "validate_event")

        builder.add_conditional_edges(
            "validate_event",
            self.route_after_validation,
            {
                "valid": "select_evidence_frames",
                "invalid": "finalize"
            }
        )

        builder.add_conditional_edges(
            "select_evidence_frames",
            self.route_after_evidence,
            {
                "ready": "vlm_verify",
                "insufficient_fallback": "apply_policy",
                "finalize": "finalize"
            }
        )

        builder.add_edge("vlm_verify", "apply_policy")
        builder.add_edge("apply_policy", "persist_event")
        builder.add_edge("persist_event", "notify_if_required")
        builder.add_edge("notify_if_required", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile()

    # --- NODE IMPLEMENTATIONS ---

    def node_validate_event(self, state: IncidentState) -> Dict[str, Any]:
        """Validates that candidate event is non-empty, within deadline, and possesses valid tracking identity."""
        tools = list(state.get("tools_executed", []))
        t_now = time.time()
        
        # Check deadline
        if t_now > state.get("deadline", t_now + 10.0):
            tools.append({"step": 1, "tool": "validate_event", "output": "EXPIRED_DEADLINE"})
            return {
                "incident_status": "EXPIRED",
                "error": "Investigation deadline exceeded",
                "tools_executed": tools
            }

        kin = state.get("kinematics", {})
        event_id = state.get("event_id")
        
        if not event_id or not kin:
            tools.append({"step": 1, "tool": "validate_event", "output": "INVALID_SCHEMA"})
            return {
                "incident_status": "INVALID",
                "error": "Missing event_id or kinematics payload",
                "tools_executed": tools
            }

        tools.append({"step": 1, "tool": "validate_event", "output": "VALID_CANDIDATE"})
        return {
            "incident_status": "CANDIDATE",
            "tools_executed": tools
        }

    def route_after_validation(self, state: IncidentState) -> str:
        if state.get("incident_status") in ["INVALID", "EXPIRED"]:
            return "invalid"
        return "valid"

    def node_select_evidence_frames(self, state: IncidentState) -> Dict[str, Any]:
        """Selects and bounds evidence frames for VLM verification."""
        tools = list(state.get("tools_executed", []))
        frames = state.get("evidence_frames", [])
        
        # Bound frames to maximum 4 keyframes
        bounded_frames = frames[-4:] if frames else []
        status = "SUFFICIENT" if bounded_frames else "INSUFFICIENT"
        
        tools.append({
            "step": 2, 
            "tool": "select_evidence_frames", 
            "output": {"frame_count": len(bounded_frames), "evidence_status": status}
        })

        return {
            "incident_status": "VERIFYING",
            "evidence_frames": bounded_frames,
            "evidence_status": status,
            "tools_executed": tools
        }

    def route_after_evidence(self, state: IncidentState) -> str:
        # Even with insufficient visual frames, proceed to fallback policy evaluation
        return "ready"

    def node_vlm_verify(self, state: IncidentState) -> Dict[str, Any]:
        """Invokes VLMAdapter or falls back gracefully to deterministic rule-based verification."""
        tools = list(state.get("tools_executed", []))
        candidate = {
            "id": state.get("event_id"),
            "camera_id": state.get("camera_id"),
            "kinematics": state.get("kinematics", {}),
            "state": state.get("detection_state")
        }
        
        # Pass bounded evidence frames
        vlm_res = self.vlm.verify_candidate_event(candidate, evidence_frames=state.get("evidence_frames", []))
        method = vlm_res.get("verification_method", "RULE_BASED_FALLBACK")
        
        tools.append({
            "step": 3,
            "tool": "vlm_verify",
            "output": {
                "method": method,
                "is_true_fall": vlm_res.get("is_true_fall"),
                "confidence": vlm_res.get("confidence")
            }
        })

        return {
            "vlm_result": vlm_res,
            "verification_method": method,
            "tools_executed": tools
        }

    def node_apply_policy(self, state: IncidentState) -> Dict[str, Any]:
        """Applies deterministic escalation policy to reconcile kinematics, VLM verification, and history."""
        tools = list(state.get("tools_executed", []))
        candidate = {
            "id": state.get("event_id"),
            "camera_id": state.get("camera_id"),
            "kinematics": state.get("kinematics", {}),
            "state": state.get("detection_state")
        }
        vlm_res = state.get("vlm_result", {})
        
        # Bounded historical memory query (n_results=1, strictly for context, cannot override current evidence)
        kin = state.get("kinematics", {})
        q_text = f"Camera {state.get('camera_id')} {kin.get('posture', '')} drop {kin.get('drop_velocity', 0):.1f}"
        mem_matches = self.memory.retrieve_context(q_text, n_results=1)

        policy_res = self.policy.evaluate(candidate, vlm_res, mem_matches)
        
        final_status = "VERIFIED" if state.get("verification_method") == "VLM_MODEL" else "FALLBACK"
        if not vlm_res.get("is_true_fall", False) and policy_res.get("decision") != "EMERGENCY / ALERT":
            final_status = "NORMAL"

        tools.append({
            "step": 4,
            "tool": "apply_policy",
            "output": {
                "decision": policy_res.get("decision"),
                "severity": policy_res.get("severity"),
                "action": policy_res.get("action")
            }
        })

        return {
            "policy_decision": policy_res,
            "severity": policy_res.get("severity", "NORMAL / LOG"),
            "incident_status": final_status,
            "agent_rationale": policy_res.get("rationale", ""),
            "tools_executed": tools
        }

    def node_persist_event(self, state: IncidentState) -> Dict[str, Any]:
        """Persists or updates the event in SQLite idempotently with stable event_id."""
        tools = list(state.get("tools_executed", []))
        event_id = state.get("event_id")
        kin = state.get("kinematics", {})
        vlm_res = state.get("vlm_result", {})
        policy_res = state.get("policy_decision", {})

        evt_record = {
            "id": event_id,
            "timestamp_sec": state.get("latest_event_time", time.time()),
            "camera_id": state.get("camera_id", "Camera 01"),
            "track_id": state.get("track_id", "Track-1"),
            "type": "Fall Incident" if vlm_res.get("is_true_fall") else "Activity Event",
            "severity": state.get("severity", "NORMAL / LOG"),
            "detector_status": state.get("detection_state", "STABLE"),
            "vlm_status": state.get("verification_method", "RULE_BASED_FALLBACK"),
            "policy_decision": policy_res.get("decision", "NORMAL / LOG"),
            "details": vlm_res.get("scene_observation", state.get("agent_rationale", "")),
            "posture_trajectory": f"{kin.get('posture', 'Unknown')} (drop {kin.get('drop_velocity', 0):.1f} px/s, dwell {kin.get('dwell_sec', 0):.1f}s)",
            "acknowledged": 0,
            "metadata": {
                "drop_velocity": kin.get("drop_velocity", 0.0),
                "dwell_sec": kin.get("dwell_sec", 0.0),
                "torso_angle": kin.get("torso_angle", 0.0),
                "provenance": "REAL_PIPELINE",
                "orchestration": "LANGGRAPH"
            }
        }

        persist_status = "PERSISTED"
        try:
            database.log_event(evt_record)
        except Exception as e:
            persist_status = f"FAILED: {str(e)[:80]}"

        tools.append({"step": 5, "tool": "persist_event", "output": persist_status})
        return {
            "persistence_status": persist_status,
            "tools_executed": tools
        }

    def node_notify_if_required(self, state: IncidentState) -> Dict[str, Any]:
        """Evaluates notification dispatch without duplicating alerts."""
        tools = list(state.get("tools_executed", []))
        severity = state.get("severity", "")
        
        notify_status = "NOT_REQUIRED"
        if "EMERGENCY" in severity:
            # Local edge notification is active; remote provider is explicitly not configured
            notify_status = "LOCAL_EDGE_BANNER_ACTIVE (REMOTE_PROVIDER: NOT_CONFIGURED)"

        tools.append({"step": 6, "tool": "notify_if_required", "output": notify_status})
        return {
            "notification_status": notify_status,
            "tools_executed": tools
        }

    def node_finalize(self, state: IncidentState) -> Dict[str, Any]:
        """Computes total execution duration and logs operational audit trace."""
        t_start = state.get("candidate_start_time", time.time())
        duration_ms = round((time.time() - t_start) * 1000.0, 1)

        trace = {
            "run_id": f"LG-{uuid.uuid4().hex[:8].upper()}",
            "event_id": state.get("event_id"),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tools_called": state.get("tools_executed", []),
            "evidence_summary": f"LangGraph execution on {state.get('event_id')}: status={state.get('incident_status')}",
            "recommendation": state.get("policy_decision", {}).get("action", "MONITOR"),
            "policy_action": state.get("policy_decision", {}).get("decision", "NORMAL"),
            "duration_ms": duration_ms
        }

        try:
            database.log_agent_trace(trace)
        except Exception:
            pass

        return {
            "duration_ms": duration_ms
        }

    def execute(self, candidate_event: Dict[str, Any], evidence_frames: Optional[List] = None) -> Dict[str, Any]:
        """Synchronously executes the bounded LangGraph investigation for candidate events."""
        t_now = time.time()
        initial_state: IncidentState = {
            "event_id": candidate_event.get("id") or f"EVT-LG-{int(t_now*1000)%1000000}",
            "camera_id": candidate_event.get("camera_id", "Camera 01"),
            "track_id": candidate_event.get("track_id", "Track-1"),
            "candidate_start_time": t_now,
            "latest_event_time": t_now,
            "detection_state": candidate_event.get("state", "CANDIDATE"),
            "kinematics": candidate_event.get("kinematics", {}),
            "evidence_frames": evidence_frames or candidate_event.get("evidence_frames", []),
            "evidence_status": "PENDING",
            "vlm_result": {},
            "verification_method": "PENDING",
            "policy_decision": {},
            "incident_status": "DETECTED",
            "severity": "NORMAL",
            "retry_count": 0,
            "max_retries": self.max_retries,
            "deadline": t_now + self.timeout_sec,
            "persistence_status": "PENDING",
            "notification_status": "PENDING",
            "tools_executed": [],
            "agent_rationale": "",
            "error": None,
            "duration_ms": 0.0
        }

        final_state = self.graph.invoke(initial_state)
        return final_state


class IncidentCoordinator:
    """
    Event-driven investigation coordinator.
    Ensures:
    1. Normal frames pass through without triggering agent investigations.
    2. Candidate events trigger a bounded LangGraph investigation.
    3. Concurrent investigations for the same event_id are deduplicated.
    4. Blocking inference is executed asynchronously without stalling the WebSocket loop.
    5. Backlog and queue depth are tracked.
    """
    def __init__(self, investigation_graph: IncidentInvestigationGraph, max_concurrency: int = 2):
        self.graph = investigation_graph
        self.max_concurrency = max_concurrency
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.latest_results: Dict[str, Dict[str, Any]] = {}
        self.dropped_frame_count: int = 0
        self.total_investigations: int = 0

    @property
    def queue_depth(self) -> int:
        return len(self.active_tasks)

    def should_trigger(self, stage3_event: Dict[str, Any]) -> bool:
        """Determines if the edge temporal state represents an incident requiring agent verification."""
        state = stage3_event.get("state", "STABLE")
        fall_confirmed = stage3_event.get("fall_confirmed", False)
        # Investigation triggers on rapid drop, verifying dwell, or fall confirmed transitions
        return state in ["RAPID_DROP", "VERIFYING", "FALL_CONFIRMED"] or fall_confirmed

    async def dispatch_investigation_async(self, candidate_event: Dict[str, Any], evidence_frames: Optional[List] = None) -> Optional[Dict[str, Any]]:
        """
        Deduplicated async dispatch.
        If an investigation is already in progress for this event_id, returns the latest known result.
        Otherwise launches a bounded background task.
        """
        event_id = candidate_event.get("id") or "UNKNOWN_INCIDENT"
        
        # Deduplication: already running
        if event_id in self.active_tasks and not self.active_tasks[event_id].done():
            return self.latest_results.get(event_id)

        # Concurrency limit check
        if len(self.active_tasks) >= self.max_concurrency:
            self.dropped_frame_count += 1
            return self.latest_results.get(event_id)

        # Run graph in worker thread to avoid blocking asyncio event loop
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(
            None,
            self.graph.execute,
            candidate_event,
            evidence_frames
        )
        
        self.total_investigations += 1

        async def _wrapper():
            try:
                res = await task
                self.latest_results[event_id] = res
                return res
            finally:
                self.active_tasks.pop(event_id, None)

        coro_task = asyncio.create_task(_wrapper())
        self.active_tasks[event_id] = coro_task
        
        # Return existing result or initial candidate placeholder
        return self.latest_results.get(event_id)

    def get_status_telemetry(self) -> Dict[str, Any]:
        return {
            "orchestrator": "LangGraph (StateGraph)",
            "queue_depth": self.queue_depth,
            "max_concurrency": self.max_concurrency,
            "dropped_investigations": self.dropped_frame_count,
            "total_investigations_triggered": self.total_investigations,
            "active_incident_ids": list(self.active_tasks.keys())
        }
