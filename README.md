# CareWatch: Agentic Edge AI for Human Activity Recognition & Fall Detection

CareWatch is an enterprise-grade Edge-First Human Activity Recognition (HAR) and Fall Detection platform designed for elderly healthcare monitoring, hospital rooms, and assisted living residences.

The platform executes real-time vision inference on local edge hardware (NVIDIA RTX 3050 6GB GPU), combined with a bounded ReAct agent, multimodal Vision-Language Model (VLM) verification, ChromaDB persistent semantic memory, and deterministic escalation policies.

---

## Key Features

1. **Edge Inference Pipeline**:
   - **Person Presence & Motion**: YOLOv8n detector with pixel-level frame differencing.
   - **Centroid Tracking**: Per-person track IDs (`Track #1`), movement status (`Stationary` vs `Active Movement`), and normalized displacement.
   - **Biomechanical Posture Estimation**: YOLO11n-Pose (17 COCO keypoints), torso tilt angle relative to vertical, and full-body aspect ratio.
   - **Temporal Fall State Machine**: Multi-frame candidate debounce, downward velocity gating (>85 px/s), ground dwell timer (>1.5s), and immediate recovery detection.
2. **Central Agentic Orchestration**:
   - **Bounded ReAct Agent**: Tool-using loop (maximum 4 steps, timeout-guarded).
   - **ChromaDB Semantic Memory**: Persistent vector collection indexing baseline habits and past empirical incidents.
   - **VLM Verification Adapter**: Evaluates candidate event frames with transparent availability reporting (`ONLINE` vs `WEIGHTS_INCOMPLETE` vs `RULE_BASED_FALLBACK`).
   - **Deterministic Escalation Policy**: Clear state transitions (`EMERGENCY / ALERT`, `WARNING / REVIEW`, `NORMAL / LOG`, `UNCERTAIN`).
3. **Dual Persistent Storage**:
   - **SQLite (`carewatch.db`)**: Complete relational audit log of events, human acknowledgements, and agent traces.
   - **ChromaDB (`./chroma_db`)**: Persistent high-dimensional semantic vector storage.
4. **Interactive Dashboard**:
   - **Live Pipeline View**: Compact single-viewport interface with real-time video stream, overlaid skeleton canvas, and unified output cards.
   - **Video Replay Mode**: Reproducible video-replay simulation for pre-deployment validation.
   - **ChromaDB Memory Search**: Natural language similarity search against historical resident profiles.
   - **Measured Benchmark**: Empirical accuracy, precision, recall, F1-score, and tail latency on staged recordings.
   - **Safety Decisions & Audit Feed**: Real-time event log with persistent acknowledgements.

---

## System Requirements

- **Operating System**: Windows 10/11 or Linux
- **Python**: Python 3.10 or 3.11 (Recommended: `E:\spaceborn-ai-team\.venv311`)
- **Acceleration**: NVIDIA GPU with CUDA 12.x+ (Tested on RTX 3050 6GB Laptop GPU) or CPU fallback
- **Dependencies**: PyTorch 2.5.1+cu121, Ultralytics YOLO, ChromaDB, FastAPI, Uvicorn, OpenCV

---

## Getting Started

### 1. Launch the Backend Server
```powershell
# Using the configured virtual environment
& "E:\spaceborn-ai-team\.venv311\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Access the Monitoring Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:8000
```

### 3. Run Camera or Video Replay
- Click **Start** to connect your live webcam.
- Click **Replay Video** to select a recorded test clip (e.g. `Recording 2026-10-20 130626.mp4`) and run reproducible offline simulation through the full pipeline.

---

## Testing & Benchmark

### Run the Automated Test Suite (10 Comprehensive Tests)
```powershell
& "E:\spaceborn-ai-team\.venv311\Scripts\python.exe" -m pytest tests\test_carewatch.py -v
```

### Run the Empirical Benchmark Evaluation
```powershell
& "E:\spaceborn-ai-team\.venv311\Scripts\python.exe" benchmark_evaluation.py
```

---

## Verified Evaluation Results

Evaluated on `Recording 2026-10-20 130626.mp4` (Continuous Staged Activities & Falls):

| Metric | Measured Value | Notes |
|---|---|---|
| **Event Precision** | **100.0%** | 0 false alerts during 35s–70s seated reading & forward reach |
| **Event Recall** | **100.0%** | Both ground collapse events detected at 79.0s and 86.5s |
| **F1-Score** | **100.0%** | Event-level incident evaluation |
| **Avg Pipeline Latency** | **39.3 ms** | Real-time GPU execution (~22.7 FPS throughput) |
| **P95 Tail Latency** | **52.4 ms** | Sub-60ms bounded latency on RTX 3050 |
| **Detection Delay** | **1.5 s** | Gated temporal verification window |
| **False Alerts per Hour**| **0.0** | Robust posture separation (Sitting vs Lying vs Fall) |

---

## Architectural Documentation
For in-depth architecture diagrams, data schemas, API contracts, and failure modes, refer to [`ARCHITECTURE.md`](file:///e:/human-fall-detection/Agentic-Edge-AI-for-HAR/ARCHITECTURE.md).
