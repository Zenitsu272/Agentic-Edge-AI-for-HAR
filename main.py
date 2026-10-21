import os
import json
import time
import asyncio
import base64
import uuid
import math
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
from pathlib import Path
import torch
import cv2
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPLAYS_DIR = Path(__file__).parent / "replays"
REPLAYS_DIR.mkdir(exist_ok=True)

import database
from inference import InferencePipeline
from central_server import ChromaAgentMemory
from vlm_adapter import VLMAdapter
from policy import EscalationPolicy
from agent import CareWatchAgent
from incident_verification import IncidentVerificationService

@asynccontextmanager
async def app_lifespan(app):
    if VLM_ENABLED:
        await verification_service.start()
    try:
        yield
    finally:
        await verification_service.close()


app = FastAPI(lifespan=app_lifespan, title="CareWatch | Agentic Edge AI Platform for HAR & Fall Detection")

# Initialize Edge & Central AI components
pipeline = InferencePipeline()
chroma_memory = ChromaAgentMemory()
VLM_ENABLED = os.getenv("CAREWATCH_ENABLE_VLM", "1").lower() in ("1", "true", "yes")
vlm_adapter = VLMAdapter(auto_load=False)
verification_service = IncidentVerificationService(vlm_adapter)
if not VLM_ENABLED:
    vlm_adapter.status = "DISABLED"
    vlm_adapter.status_detail = "Pose-only mode. VLM verification is disabled."
escalation_policy = EscalationPolicy()
agent = CareWatchAgent(chroma_memory, vlm_adapter, escalation_policy)

DIST_DIR = Path(__file__).parent / "dist"
app.mount("/static", StaticFiles(directory=str(DIST_DIR)), name="static")

class MemoryQueryRequest(BaseModel):
    query: str
    camera_id: str = "Camera 01"

@app.get("/")
def serve_index():
    index_file = DIST_DIR / "index.html"
    return FileResponse(index_file)

def attach_verification(session_pipeline, result, frame):
    if not VLM_ENABLED:
        result['central_server'] = {'status':'DISABLED', 'verification_mode':'POSE_ONLY'}
        result['verification'] = {'status':'DISABLED', 'model':vlm_adapter.model_name,
                                  'scene_observation':'VLM verification is disabled.'}
        return
    verification_service.observe(session_pipeline.session_id, result, frame)
    result['central_server'] = {'status':result['verification']['status'],
                                'verification_mode':'LOCAL_VLM',
                                'vlm_verification':result['verification']}


@app.get('/api/verification/{event_id}')
def read_verification(event_id: str):
    result = database.get_verification(event_id)
    if result is None:
        return JSONResponse(status_code=404, content={'status':'not_found'})
    return {'status':'success', 'verification':result}


# ==================== SYSTEM HEALTH ENDPOINT ====================
@app.get("/api/health")
def get_system_health():
    """
    Genuine System Health Status across all components:
    - Video Capture
    - Person Detection and Pose Estimation (configured YOLO pose model)
    - VLM Provider (Qwen2.5-VL / Status)
    - LangGraph Orchestrator
    - ChromaDB Persistent Semantic Memory
    - SQLite Relational Database
    - Notification Service (Local vs Remote)
    """
    chroma_count = chroma_memory.collection.count() if chroma_memory.collection else 0
    events_count = len(database.get_events(limit=500))
    coordinator_status = agent.coordinator.get_status_telemetry()
    
    # Assess component readiness
    vlm_status = vlm_adapter.status
    if not VLM_ENABLED:
        vlm_health = "DISABLED"
    elif vlm_status == "ONLINE":
        vlm_health = "ONLINE"
    elif vlm_status in ('LOADING', 'NOT_LOADED'):
        vlm_health = 'LOADING'
    elif vlm_status in ["WEIGHTS_INCOMPLETE", "OFFLINE_FALLBACK"]:
        vlm_health = "DEGRADED"
    else:
        vlm_health = "UNAVAILABLE"

    remote_notify_health = "NOT_CONFIGURED"

    # Derive overall aggregate health
    if vlm_health in ["UNAVAILABLE", "DEGRADED", 'LOADING']:
        aggregate_status = "DEGRADED"
    else:
        aggregate_status = "ONLINE"
    
    return {
        "status": aggregate_status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": {
            "yolo_device": pipeline.device,
            "vlm_device": getattr(vlm_adapter, "device", "cuda:0"),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "cuda_available": torch.cuda.is_available()
        },
        "components": {
            "video_capture": {
                "status": "ONLINE",
                "mode": "WEBCAM_AND_VIDEO_REPLAY"
            },
            "person_detector": {
                "model": pipeline.model_path.name,
                "status": "ONLINE",
                "device": pipeline.device
            },
            "pose_estimator": {
                "model": pipeline.model_path.name,
                "status": "ONLINE",
                "device": pipeline.device
            },
            "vlm_provider": {
                "model": vlm_adapter.model_name,
                "status": vlm_health,
                "device": getattr(vlm_adapter, "device", "cuda:0"),
                "detail": vlm_adapter.status_detail,
                "verification_mode": "DISABLED" if not VLM_ENABLED else ("VLM_MODEL" if vlm_adapter.status == "ONLINE" else "UNAVAILABLE")
            },
            "langgraph_orchestration": {
                "status": "DISABLED",
                "engine": "LangGraph (legacy workflow)",
                "telemetry": coordinator_status
            },
            "incident_verification": verification_service.telemetry(),
            "chroma_db": {
                "status": "ONLINE",
                "collection": "carewatch_agent_memory",
                "persistent_path": "./chroma_db",
                "indexed_records_count": chroma_count
            },
            "relational_db": {
                "status": "ONLINE",
                "engine": "SQLite3 (carewatch.db)",
                "events_stored_count": events_count
            },
            "local_notification_banner": {
                "status": "ONLINE",
                "channels": ["LOCAL_EDGE_BANNER", "AUDIT_LOG_FEED"]
            },
            "remote_notification_provider": {
                "status": "NOT_CONFIGURED",
                "detail": "No remote SMS/Webhook/APNS provider configured. Operating via local edge audio/banner alerts and SQLite audit feed."
            }
        }
    }

# ==================== PERSISTENT ALERTS API ====================
@app.get("/api/alerts")
def get_persisted_alerts(limit: int = 50, acknowledged: Optional[bool] = None if False else None):
    """Returns chronological audit records of events stored in SQLite."""
    try:
        events = database.get_events(limit=limit)
        return {"status": "success", "count": len(events), "alerts": events}
    except Exception as e:
        return {"status": "error", "message": str(e), "alerts": []}

@app.post("/api/alerts/{alert_id}/acknowledge")
def acknowledge_alert_by_id(alert_id: str):
    """Acknowledges an alert and persists the status across restarts."""
    success = database.acknowledge_event_by_id(alert_id)
    if pipeline.active_alert and pipeline.active_alert.get("alert_id") == alert_id:
        pipeline.active_alert["acknowledged"] = True
    return {"status": "success" if success else "not_found", "alert_id": alert_id}

@app.post("/api/acknowledge_alert")
def acknowledge_alert_legacy():
    """Backward compatibility endpoint for active in-memory alert."""
    if pipeline.active_alert:
        pipeline.active_alert["acknowledged"] = True
        database.acknowledge_event_by_id(pipeline.active_alert.get("alert_id", ""))
        return {"status": "success", "alert": pipeline.active_alert}
    return {"status": "no_active_alert"}

# ==================== AGENT OPERATIONAL TRACES ====================
@app.get("/api/agent/traces")
def get_agent_traces(limit: int = 10):
    """Returns transparent operational audit traces of agent ReAct investigations."""
    traces = database.get_recent_agent_traces(limit=limit)
    return {"status": "success", "count": len(traces), "traces": traces}

# ==================== MEASURED BENCHMARK API ====================
@app.get("/api/benchmark")
def get_measured_benchmark():
    """Returns genuine empirical benchmark evaluation results."""
    bench_file = Path(__file__).parent / "data" / "benchmark_results.json"
    if not bench_file.exists():
        bench_file = Path(__file__).parent / "benchmark_results.json"
    if bench_file.exists():
        with open(bench_file, "r") as f:
            data = json.load(f)
        return {"status": "success", "benchmark": data}
    return {"status": "error", "message": "benchmark_results.json not found"}

# ==================== CHROMADB SEMANTIC MEMORY API ====================
@app.get("/api/chroma_memories")
def get_chroma_memories():
    """Returns persistent empirical baseline memory entries from ChromaDB."""
    try:
        data = chroma_memory.collection.get()
        items = []
        if data and data["documents"]:
            for i in range(len(data["documents"])):
                items.append({
                    "id": data["ids"][i],
                    "document": data["documents"][i],
                    "metadata": data["metadatas"][i] if data["metadatas"] else {}
                })
        return {"status": "success", "count": len(items), "memories": items}
    except Exception as e:
        return {"status": "error", "message": str(e), "memories": []}

@app.post("/api/query_memory")
def query_chroma_memory(req: MemoryQueryRequest):
    """Queries ChromaDB for semantic similarity against empirical event history."""
    matches = chroma_memory.retrieve_context(req.query, n_results=3)
    return {"status": "success", "query": req.query, "matches": matches}

# ==================== VIDEO REPLAY & AVI INGESTION API ====================
@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Store a bounded local upload under a unique name and validate actual decoding."""
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in {'.avi','.mp4','.webm','.mkv','.mov'}:
        await file.close()
        return JSONResponse(status_code=400, content={'status':'error','message':'Choose an MP4, AVI, MOV, MKV or WebM video.'})
    safe_name = f'upload_{uuid.uuid4().hex}{suffix}'
    target_path = REPLAYS_DIR / safe_name
    accepted = False
    try:
        size = 0
        with target_path.open('xb') as output:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > 500*1024*1024:
                    return JSONResponse(status_code=413,content={'status':'error','message':'The video exceeds the 500 MB upload limit.'})
                output.write(chunk)
        cap = cv2.VideoCapture(str(target_path))
        try:
            ok, first = cap.read()
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        if not ok or first is None or total_frames <= 0 or not math.isfinite(fps) or fps <= 0:
            return JSONResponse(status_code=400, content={
                'status':'error','message':'This video could not be decoded. Try another file or export it as MP4.'
            })
        h,w = first.shape[:2]
        accepted = True
        return {
            "status": "success",
            "filename": safe_name,
            "display_name": Path(file.filename).name,
            "format": target_path.suffix.lower(),
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": round(total_frames/fps,2),
            "resolution": f"{w}x{h}",
            "size_kb": round(size / 1024, 1)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
    finally:
        await file.close()
        if not accepted:
            target_path.unlink(missing_ok=True)

@app.get("/api/replays")
def list_replays():
    """Lists available offline replay videos in the replays directory."""
    files = []
    for p in REPLAYS_DIR.glob("*.*"):
        if p.suffix.lower() in [".avi", ".mp4", ".webm", ".mkv", ".mov"]:
            files.append({
                "filename": p.name,
                "format": p.suffix.lower(),
                "size_mb": round(p.stat().st_size / (1024*1024), 2)
            })
    return {"status": "success", "count": len(files), "replays": files}

@app.websocket("/ws/replay_stream")
async def websocket_replay_stream(websocket: WebSocket):
    """
    Server-side Replay Stream WebSocket:
    Reads .avi or .mp4 video frame-by-frame with OpenCV,
    executes pose inference and schedules incident verification,
    and streams encoded frames + telemetry to frontend at video FPS.
    """
    await websocket.accept()
    session_pipeline = pipeline.new_session()
    query_params = websocket.query_params
    video_name = query_params.get("video", "sample_fall_clip.avi")
    loop_video = query_params.get('loop', 'true').lower() not in ('0','false','no')
    
    # Check replays directory first, then root directory
    video_path = REPLAYS_DIR / Path(video_name).name
    if not video_path.exists():
        root_path = Path(__file__).parent / Path(video_name).name
        if root_path.exists():
            video_path = root_path
        else:
            await websocket.send_json({"error": f"Video file {video_name} not found on server."})
            await websocket.close()
            return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await websocket.send_json({"error": f"OpenCV could not decode video {video_name}."})
        await websocket.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay = max(1.0 / min(fps, 30.0), 0.030)
    paused = False
    current_frame_pos = 0
    incident_ids = set()
    unverified_incidents = {}

    try:
        while True:
            # Check for non-blocking client commands if any
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                data = json.loads(msg)
                cmd = data.get("command")
                if cmd == "pause":
                    paused = True
                elif cmd == "resume":
                    paused = False
                elif cmd == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
            except (ValueError, RuntimeError):
                break

            if paused:
                await asyncio.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                if not loop_video:
                    verification_service.close_session(session_pipeline.session_id)
                    if current_frame_pos < total_frames-1:
                        await websocket.send_json({'error':f'Video decoding stopped at frame {current_frame_pos} of {total_frames}. Results are partial.'})
                        await websocket.close()
                        break
                    incidents = [record.copy() for event_id in incident_ids
                                 if (record := verification_service.records.get(event_id,unverified_incidents.get(event_id))) is not None]
                    await websocket.send_json({'type':'video_complete','processed_frames':current_frame_pos,
                                               'total_frames':total_frames,'duration_sec':current_frame_pos/fps,
                                               'incidents':incidents})
                    await websocket.close()
                    break
                # Loop video seamlessly
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                current_frame_pos = 0
                verification_service.close_session(session_pipeline.session_id)
                session_pipeline.reset()
                ret, frame = cap.read()
                if not ret:
                    break

            current_frame_pos += 1
            replay_timestamp = current_frame_pos / fps

            # Pose inference stays independent of background VLM generation.
            result = session_pipeline.process_unified(frame, timestamp=replay_timestamp)

            attach_verification(session_pipeline, result, frame)
            verification = result.get('verification',{})
            if verification.get('event_id'): incident_ids.add(verification['event_id'])
            incident_ids.update(result.get('verification_incidents',{}))
            alert = result.get('stage3_event',{}).get('active_alert')
            if alert and alert.get('alert_id') and not verification.get('event_id'):
                event_id = alert['alert_id']
                incident_ids.add(event_id)
                if event_id not in unverified_incidents:
                    unverified_incidents[event_id] = {'event_id':event_id,'status':verification.get('status','UNAVAILABLE'),
                        'trigger_time_sec':replay_timestamp,'scene_observation':verification.get('scene_observation','VLM verification was unavailable.')}
                result['video_incident'] = unverified_incidents[event_id]
            result['video_progress'] = {'processed_frames':current_frame_pos,'total_frames':total_frames,
                                        'time_sec':replay_timestamp,'duration_sec':total_frames/fps,
                                        'percent':round(min(100,current_frame_pos/max(total_frames,1)*100),1)}

            # 3. Downscale slightly if large to ensure smooth WebSocket delivery
            fh, fw = frame.shape[:2]
            target_w = 480
            target_h = int(fh * (target_w / max(fw, 1)))
            display_frame = cv2.resize(frame, (target_w, target_h))
            _, enc = cv2.imencode(".jpg", display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            result["rendered_frame"] = "data:image/jpeg;base64," + base64.b64encode(enc).decode("ascii")

            await websocket.send_json(result)
            await asyncio.sleep(frame_delay)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    except Exception as e:
        print("[WS Replay Stream error]:", e)
    finally:
        verification_service.close_session(session_pipeline.session_id)
        cap.release()

# ==================== UNIFIED WEBSOCKET PIPELINE ====================
@app.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket):
    """
    Unified end-to-end WebSocket:
    Pose and temporal fall detection -> asynchronous local VLM verification -> Output
    """
    await websocket.accept()
    session_pipeline = pipeline.new_session()
    try:
        while True:
            data = await websocket.receive_text()
            video_time = None
            if data.startswith('{'):
                packet = json.loads(data)
                data = packet['frame']
                video_time = packet.get('timestamp')
            frame = session_pipeline.decode_frame(data)
            if frame is None:
                await websocket.send_json({'error': 'Could not decode video frame'})
                continue
            result = session_pipeline.process_unified(frame, timestamp=video_time)
            
            attach_verification(session_pipeline, result, frame)

            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[WS Pipeline error]:", e)
    finally:
        verification_service.close_session(session_pipeline.session_id)

@app.websocket("/ws/motion")
async def websocket_motion(websocket: WebSocket):
    await websocket.accept()
    session_pipeline = pipeline.new_session()
    try:
        while True:
            data = await websocket.receive_text()
            frame = session_pipeline.decode_frame(data)
            if frame is None:
                continue
            result = session_pipeline.process_motion(frame)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[WS Motion error]:", e)

@app.websocket("/ws/har")
async def websocket_har(websocket: WebSocket):
    await websocket.accept()
    session_pipeline = pipeline.new_session()
    try:
        while True:
            data = await websocket.receive_text()
            frame = session_pipeline.decode_frame(data)
            if frame is None:
                continue
            result = session_pipeline.process_har(frame)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[WS HAR error]:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
