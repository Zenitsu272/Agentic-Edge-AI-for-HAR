# CareWatch — pose-based activity monitoring

CareWatch runs a local camera/replay dashboard with body landmarks, independent person tracks, posture details, temporal fall observation, SQLite activity/alert history, ChromaDB memory search and saved benchmark results.

## Current focus: stage 2

`pose-temporal-v2.2` replaces the previous pixel-threshold classifier. The three priority outputs are **Standing**, **Sitting**, and **Fall**. Partial, uncertain and already-lying states remain explicit so missing evidence is not forced into one of those labels.

- Uses confidence-filtered shoulders, hips, knees and ankles. Missing ankles are **not** evidence of sitting.
- Uses torso-normalized thigh geometry and knee angles rather than fixed pixel distances or arm-dependent bounding-box proportions.
- Keeps state separately for each person and each camera/replay connection.
- Confirms a fall after observed rapid torso descent/rotation and a sustained down pose (at least 1.2 seconds and six observations). A static lying pose, brief bend, or observation gap cannot confirm a fall.
- A sudden collapse from a previously observed standing or sitting pose immediately displays **Fall detected** with a **possible / partial fall** qualifier. This includes rapid shoulder descent when hips become occluded; losing keypoints alone cannot trigger it. Early warning requires strong normalized downward speed and posture change, while full confirmation keeps the stricter thresholds and dwell requirement above.
- Any possible/partial or confirmed fall alert stays visible for at least 10 seconds of real time, including when the person recovers or leaves the frame. A partial incident retains its ID when confirmed; a new incident restarts the hold. The API separates `fall_alert_active` from the stricter `fall_confirmed` flag.
- Cropped upright torsos may show **Standing · limited view**, explicitly provisional. A seated and standing person can have indistinguishable upper-body crops; show hips and knees for stronger evidence.
- VLM/agent verification remains disabled by default. Existing history and memory remain accessible.

The network uses official pretrained YOLO pose weights; it has **not** been fine-tuned on a new labeled fall dataset. The changes are a larger pose estimator plus a new geometric/temporal activity classifier. Evidence scores represent rule support, not calibrated probabilities.

## Run

From this repository, using the configured local environment:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
~~~

Open <http://127.0.0.1:8000>. Start a camera or replay a video. **Full View** uses native fullscreen; embedded browsers that block fullscreen use the available browser viewport. Escape exits.

Python 3.12 is working locally. Packages include torch, ultralytics, opencv-python, numpy, fastapi, uvicorn, python-multipart, chromadb and langgraph. Use a CUDA-compatible PyTorch build for GPU inference.

The repository includes `models/yolo11s-pose.pt`, the preferred pose model. The pipeline uses CUDA when available and falls back to bundled `yolo11n-pose.pt` if that file is absent.

Optional environment variables: `CAREWATCH_DEVICE=cpu`, `CAREWATCH_IMAGE_SIZE=640`, `CAREWATCH_POSE_MODEL=<path>`. Default GPU input is 960 pixels. `CAREWATCH_ENABLE_VLM=1` restores the optional agent path, subject to model availability.

## Validation and further tuning

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pose_activity.py tests/test_pose_mode.py -q
node --test tests/pose-overlay.test.cjs
~~~

Tests cover scaled/mirrored standing and sitting, missing/low-confidence joints, raised arms, bending, noise, sustained falls, resting, slow lowering, recovery, observation gaps, person ordering, replay resets, and both supplied standing screenshots. `data/pose_regressions.json` stores extracted keypoint coordinates, not images. The source screenshots already contain overlays, so they are regression cases rather than a clean evaluation dataset.

For a new **single-person** raw video, annotate time segments in JSON:

~~~json
[
  {"start": 0, "end": 5, "label": "Standing"},
  {"start": 5, "end": 10, "label": "Sitting"},
  {"start": 11, "end": 14, "label": "Fall"}
]
~~~

~~~powershell
.\.venv\Scripts\python.exe scripts/evaluate_activity.py clip.mp4 labels.json --output data/activity_evaluation.json
~~~

This writes frame-level confusion counts, per-class recall, latency, provisional flags and the prediction timeline. Unknown predictions count as errors. It does not validate fall event recall or train new weights. Use separate people/camera positions for tuning and held-out evaluation.

Saved 100% results in `data/benchmark_results.json` are **historical measurements of the previous pipeline**, not validation of v2. The UI labels them accordingly. Raw labeled sitting and fall clips were not present locally, so no real-video accuracy is claimed for the new classifier. Occlusion, camera movement, identity crossings, deliberate fast lying down and falls directly toward/away from the lens remain challenging.

## References

- [Ultralytics pose output and keypoint confidence](https://docs.ultralytics.com/tasks/pose/)
- [Posture transitions and temporal fall detection](https://www.mdpi.com/2313-433X/7/3/42)

These motivate confidence-aware geometry and temporal checks; they do not validate this implementation's thresholds.
