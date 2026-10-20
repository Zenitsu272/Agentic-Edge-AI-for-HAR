import sqlite3
import json
import time
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

DB_PATH = Path(__file__).parent / "carewatch.db"

def get_db_connection(db_path: Optional[str] = None):
    p = db_path or str(DB_PATH)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[str] = None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Events Table: Structured event store for all detected activities, candidates, and falls
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        timestamp_sec REAL NOT NULL,
        camera_id TEXT NOT NULL,
        track_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        severity TEXT NOT NULL,
        detector_status TEXT NOT NULL,
        vlm_status TEXT NOT NULL,
        policy_decision TEXT NOT NULL,
        details TEXT NOT NULL,
        posture_trajectory TEXT NOT NULL,
        acknowledged INTEGER DEFAULT 0,
        metadata_json TEXT NOT NULL,
        frame_paths_json TEXT DEFAULT '[]'
    )
    """)

    # 2. Agent Operational Audit Traces Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_traces (
        run_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        tools_called_json TEXT NOT NULL,
        evidence_summary TEXT NOT NULL,
        recommendation TEXT NOT NULL,
        policy_action TEXT NOT NULL,
        duration_ms REAL NOT NULL,
        FOREIGN KEY (event_id) REFERENCES events (id)
    )
    """)
    
    conn.commit()
    
    # Production startup: demo-data seeding is DISABLED by default.
    # Seeding is strictly guarded behind the explicit CAREWATCH_DEMO_MODE=1 flag.
    demo_mode_enabled = os.getenv("CAREWATCH_DEMO_MODE", "0").lower() in ("1", "true", "yes")
    
    if demo_mode_enabled:
        cursor.execute("SELECT COUNT(*) FROM events")
        count = cursor.fetchone()[0]
        if count == 0:
            seed_events = [
                (
                    "EVT-DEMO-01",
                    datetime.now().strftime("%Y-%m-%d 16:18:22"),
                    time.time() - 3600,
                    "Camera 01",
                    "Track-1",
                    "[DEMO] Staged Fall Incident",
                    "DEMO / SYNTHETIC",
                    "FALL_CONFIRMED",
                    "RULE_BASED_FALLBACK",
                    "EMERGENCY / ALERT",
                    "SYNTHETIC DEMO: Staged collapse (92.1 px/s drop, 13.1s floor dwell).",
                    "Standing → Rapid Drop (92.1 px/s) → Floor Dwell (13.1s)",
                    0,
                    json.dumps({
                        "drop_velocity": 92.1,
                        "dwell_sec": 13.1,
                        "torso_angle": 78.4,
                        "category": "Fall",
                        "provenance": "DEMO_SYNTHETIC",
                        "is_synthetic": True
                    }),
                    json.dumps([])
                ),
                (
                    "EVT-DEMO-02",
                    datetime.now().strftime("%Y-%m-%d 16:02:14"),
                    time.time() - 4500,
                    "Camera 01",
                    "Track-1",
                    "[DEMO] Baseline Initialization",
                    "DEMO / SYNTHETIC",
                    "STABLE",
                    "NOT_REQUESTED",
                    "NORMAL / LOG",
                    "SYNTHETIC DEMO: Edge AI baseline initialized.",
                    "Camera 01 Init → Continuous Edge Monitoring",
                    1,
                    json.dumps({
                        "category": "Baseline",
                        "state": "STABLE",
                        "provenance": "DEMO_SYNTHETIC",
                        "is_synthetic": True
                    }),
                    json.dumps([])
                )
            ]
            cursor.executemany("""
            INSERT INTO events (
                id, timestamp, timestamp_sec, camera_id, track_id, event_type, severity,
                detector_status, vlm_status, policy_decision, details, posture_trajectory,
                acknowledged, metadata_json, frame_paths_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, seed_events)
            conn.commit()

    conn.close()

def log_event(event: Dict[str, Any], db_path: Optional[str] = None) -> str:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    event_id = event.get("id") or f"EVT-{int(time.time()*1000)%1000000}"
    ts_str = event.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ts_sec = event.get("timestamp_sec", time.time())
    camera_id = event.get("camera_id", "Camera 01")
    track_id = event.get("track_id", "Track-1")
    event_type = event.get("type") or event.get("event_type", "Activity Event")
    severity = event.get("severity", "NORMAL / LOG")
    det_status = event.get("detector_status", "STABLE")
    vlm_status = event.get("vlm_status", "NOT_REQUESTED")
    policy_decision = event.get("policy_decision", severity)
    details = event.get("details", "")
    trajectory = event.get("posture_trajectory", "")
    ack = 1 if event.get("acknowledged") else 0
    meta = json.dumps(event.get("metadata", {}))
    frames = json.dumps(event.get("frame_paths", []))
    
    cursor.execute("""
    INSERT OR REPLACE INTO events (
        id, timestamp, timestamp_sec, camera_id, track_id, event_type, severity,
        detector_status, vlm_status, policy_decision, details, posture_trajectory,
        acknowledged, metadata_json, frame_paths_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (event_id, ts_str, ts_sec, camera_id, track_id, event_type, severity,
          det_status, vlm_status, policy_decision, details, trajectory, ack, meta, frames))
    
    conn.commit()
    conn.close()
    return event_id

def get_events(limit: int = 50, acknowledged: Optional[bool] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    query = "SELECT * FROM events"
    params = []
    if acknowledged is not None:
        query += " WHERE acknowledged = ?"
        params.append(1 if acknowledged else 0)
    query += " ORDER BY timestamp_sec DESC LIMIT ?"
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    
    result = []
    for r in rows:
        meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
        result.append({
            "alert_id": r["id"],
            "id": r["id"],
            "timestamp": r["timestamp"].split(" ")[-1] if " " in r["timestamp"] else r["timestamp"],
            "full_timestamp": r["timestamp"],
            "timestamp_sec": r["timestamp_sec"],
            "camera_id": r["camera_id"],
            "track_id": r["track_id"],
            "type": r["event_type"],
            "severity": r["severity"],
            "detector_status": r["detector_status"],
            "vlm_status": r["vlm_status"],
            "policy_decision": r["policy_decision"],
            "details": r["details"],
            "posture_trajectory": r["posture_trajectory"],
            "acknowledged": bool(r["acknowledged"]),
            "metadata": meta,
            "provenance": meta.get("provenance", "REAL_PIPELINE"),
            "is_synthetic": meta.get("is_synthetic", False),
            "frame_paths": json.loads(r["frame_paths_json"]) if r["frame_paths_json"] else []
        })
    conn.close()
    return result

def acknowledge_event_by_id(event_id: str, db_path: Optional[str] = None) -> bool:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE events SET acknowledged = 1 WHERE id = ?", (event_id,))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated

def log_agent_trace(trace: Dict[str, Any], db_path: Optional[str] = None):
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO agent_traces (
        run_id, event_id, timestamp, tools_called_json, evidence_summary,
        recommendation, policy_action, duration_ms
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trace.get("run_id", f"RUN-{int(time.time()*1000)}"),
        trace.get("event_id", ""),
        trace.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        json.dumps(trace.get("tools_called", [])),
        trace.get("evidence_summary", ""),
        trace.get("recommendation", ""),
        trace.get("policy_action", ""),
        trace.get("duration_ms", 0.0)
    ))
    conn.commit()
    conn.close()

def get_recent_agent_traces(limit: int = 10, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM agent_traces ORDER BY rowid DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    traces = []
    for r in rows:
        traces.append({
            "run_id": r["run_id"],
            "event_id": r["event_id"],
            "timestamp": r["timestamp"],
            "tools_called": json.loads(r["tools_called_json"]),
            "evidence_summary": r["evidence_summary"],
            "recommendation": r["recommendation"],
            "policy_action": r["policy_action"],
            "duration_ms": r["duration_ms"]
        })
    conn.close()
    return traces

# Initialize on module import
init_db()
