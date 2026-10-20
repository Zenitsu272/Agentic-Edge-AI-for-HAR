import os
import time
from pathlib import Path
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference import InferencePipeline
from central_server import ChromaAgentMemory, VLMVerifier, AgenticEscalationEngine

app = FastAPI(title="CareWatch | Agentic Edge AI for HAR")

# Initialize Edge & Central AI components
pipeline = InferencePipeline()
chroma_memory = ChromaAgentMemory()
vlm_verifier = VLMVerifier()
agent_engine = AgenticEscalationEngine(chroma_memory, vlm_verifier)

DIST_DIR = Path(__file__).parent / "dist"
app.mount("/static", StaticFiles(directory=str(DIST_DIR)), name="static")

class MemoryQueryRequest(BaseModel):
    query: str
    camera_id: str = "Camera 01"

@app.get("/")
def serve_index():
    index_file = DIST_DIR / "index.html"
    return FileResponse(index_file)

@app.post("/api/acknowledge_alert")
def acknowledge_alert():
    if pipeline.active_alert:
        pipeline.active_alert["acknowledged"] = True
        return {"status": "success", "alert": pipeline.active_alert}
    return {"status": "no_active_alert"}

@app.get("/api/chroma_memories")
def get_chroma_memories():
    """Returns baseline memory entries from ChromaDB."""
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
        return {"status": "error", "message": str(e)}

@app.post("/api/query_memory")
def query_chroma_memory(req: MemoryQueryRequest):
    """Queries ChromaDB for semantic similarity against empirical event history."""
    matches = chroma_memory.retrieve_context(req.query, n_results=3)
    return {"status": "success", "query": req.query, "matches": matches}

# Keep track of logged falls in current streaming session
logged_fall_ids = set()

@app.websocket("/ws/pipeline")
async def websocket_pipeline(websocket: WebSocket):
    """
    Unified end-to-end WebSocket:
    Edge Device (Stages 1-3) -> Central AI Server (VLM + ChromaDB + Escalation) -> Output
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame = pipeline.decode_frame(data)
            if frame is None:
                continue
            
            # Edge Stage 1, 2, 3 processing
            result = pipeline.process_unified(frame)
            
            # Central AI Server (VLM + ChromaDB Agent Memory + Escalation Decision)
            central_analysis = agent_engine.evaluate_incident(result, camera_id="Camera 01")
            result["central_server"] = central_analysis
            
            # Dynamically log verified fall events to ChromaDB persistent memory
            s3 = result["stage3_event"]
            if s3["fall_confirmed"]:
                event_uid = f"EVT-LIVE-{int(time.time() // 10)}"
                if event_uid not in logged_fall_ids:
                    logged_fall_ids.add(event_uid)
                    doc_text = (
                        f"Camera 01 Live Confirmed Fall at {time.strftime('%H:%M:%S')}: "
                        f"Drop velocity {s3['drop_velocity']:.1f} px/s, "
                        f"torso tilt {result['stage2_posture']['torso_angle_deg']:.1f}°, "
                        f"persistent floor dwell {s3['ground_dwell_sec']:.1f}s. "
                        f"Verified by YOLO11n-Pose & Central VLM."
                    )
                    meta = {
                        "camera": "Camera 01",
                        "source": "Live Edge Stream",
                        "timestamp_sec": round(time.time(), 1),
                        "category": "Fall",
                        "state": "FALL_CONFIRMED",
                        "severity": "CRITICAL EMERGENCY",
                        "drop_velocity_px_s": float(s3["drop_velocity"]),
                        "dwell_sec": float(s3["ground_dwell_sec"])
                    }
                    chroma_memory.log_new_event(event_uid, doc_text, meta)

            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[WS Pipeline error]:", e)

@app.websocket("/ws/motion")
async def websocket_motion(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame = pipeline.decode_frame(data)
            if frame is None:
                continue
            result = pipeline.process_motion(frame)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[WS Motion error]:", e)

@app.websocket("/ws/har")
async def websocket_har(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            frame = pipeline.decode_frame(data)
            if frame is None:
                continue
            result = pipeline.process_har(frame)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("[WS HAR error]:", e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
