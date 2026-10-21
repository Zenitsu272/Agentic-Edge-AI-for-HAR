# CareWatch — pose-based activity monitoring

CareWatch runs a local camera/replay dashboard with body landmarks, independent person tracks, posture details, temporal fall observation, SQLite activity/alert history, ChromaDB memory search and saved benchmark results.

## Stage 2: pose and motion

`pose-temporal-v2.2` replaces the previous pixel-threshold classifier. The three priority outputs are **Standing**, **Sitting**, and **Fall**. Partial, uncertain and already-lying states remain explicit so missing evidence is not forced into one of those labels.

- Uses confidence-filtered shoulders, hips, knees and ankles. Missing ankles are **not** evidence of sitting.
- Uses torso-normalized thigh geometry and knee angles rather than fixed pixel distances or arm-dependent bounding-box proportions.
- Keeps state separately for each person and each camera/replay connection.
- Confirms a fall after observed rapid torso descent/rotation and a sustained down pose (at least 1.2 seconds and six observations). A static lying pose, brief bend, or observation gap cannot confirm a fall.
- A sudden collapse from a previously observed standing or sitting pose immediately displays **Fall detected** with a **possible / partial fall** qualifier. This includes rapid shoulder descent when hips become occluded; losing keypoints alone cannot trigger it. Early warning requires strong normalized downward speed and posture change, while full confirmation keeps the stricter thresholds and dwell requirement above.
- Any possible/partial or confirmed fall alert stays visible for at least 10 seconds of real time, including when the person recovers or leaves the frame. A partial incident retains its ID when confirmed; a new incident restarts the hold. The API separates `fall_alert_active` from the stricter `fall_confirmed` flag.
- Cropped upright torsos may show **Standing · limited view**, explicitly provisional. A seated and standing person can have indistinguishable upper-body crops; show hips and knees for stronger evidence.
- Possible/partial and confirmed fall incidents trigger stage 3 verification once per incident. Existing history and memory remain accessible.

The network uses official pretrained YOLO pose weights; it has **not** been fine-tuned on a new labeled fall dataset. The changes are a larger pose estimator plus a new geometric/temporal activity classifier. Evidence scores represent rule support, not calibrated probabilities.

## Run

From this repository, using the configured local environment:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
~~~

Open <http://127.0.0.1:8000>. Start a camera or use **Upload video & check**. **Full View** uses native fullscreen; embedded browsers that block fullscreen use the available browser viewport. Escape exits.

Use **Upload video & check** above the camera to analyze a recording (MP4, AVI, MOV, MKV or WebM, up to 500 MB). The server validates and stores the upload locally, then checks each frame once, including video codecs the browser cannot play. Progress shows the video time and frame count. The results list each fall candidate and its VLM verdict; pending VLM reviews continue after the last video frame. Stop cancels an upload or ends analysis with explicitly partial results. Videos do not loop during a check, and standing/sitting clips without a pose fall candidate do not trigger the VLM.

Python 3.12 is working locally. Packages include torch, ultralytics, opencv-python, numpy, fastapi, uvicorn, python-multipart, chromadb and langgraph. Use a CUDA-compatible PyTorch build for GPU inference.

The repository includes `models/yolo11s-pose.pt`, the preferred pose model. The pipeline uses CUDA when available and falls back to bundled `yolo11n-pose.pt` if that file is absent.

Optional environment variables: `CAREWATCH_DEVICE=cpu`, `CAREWATCH_IMAGE_SIZE=640`, `CAREWATCH_POSE_MODEL=<path>`. Default GPU input is 960 pixels. `CAREWATCH_ENABLE_VLM=0` disables stage 3 for pose-only operation.

## Stage 3: local VLM verification

Fall alerts automatically collect about two seconds before and 1.5 seconds after the trigger. Six chronological frames, including the trigger, go to **Qwen2.5-VL-3B-Instruct**, quantized to 4-bit NF4. Full-scene frames preserve the room, floor and support surfaces. The VLM checks whether a fall occurs anywhere in this scene; its verdict does not independently prove which person fell. Pose tracking supplies the candidate incident/person. Camera images stay local and are held briefly in memory, not saved to the audit database.

The VLM examines visible loss of balance/collapse versus deliberate movement. The dashboard shows **fall confirmed**, **not a fall**, or **inconclusive** with the model's visual explanation. Missing weights, busy queues, timeouts and failed inference remain explicitly unverified; production verification never substitutes pose rules for a model verdict. The original red alert remains for at least 10 seconds. VLM results are separate from the pose alert; a negative result does not automatically dismiss it.

Verification runs on one background worker with a bounded queue, independently for each person's incident. Camera/replay streaming continues during generation, though sharing a GPU can increase pose latency. Incident results are saved in SQLite and appear in activity history; `GET /api/verification/{event_id}` supports polling after a stream stops. Normal standing/sitting frames do not invoke the VLM. The legacy LangGraph investigation path is not used by live/replay verification.

Install the optional model runtime and download the pinned weights once (about 2.47 GB):

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-vlm.txt
.\.venv\Scripts\python.exe scripts/download_vlm_model.py
~~~

Weights default to `.venv/models/Qwen2.5-VL-3B-Instruct-bnb-4bit`; set `CAREWATCH_VLM_MODEL` to use another compatible local directory. Loading uses local files only, without remote model code. The bundled quantized model requires a compatible bitsandbytes runtime; this configuration targets the local CUDA GPU. Health reports loading/unavailability honestly if the model cannot load. The model has not been fine-tuned for this deployment and its verdict can be wrong; inconclusive results require review.

To evaluate an actual video window through the local VLM independently of pose triggering:

~~~powershell
.\.venv\Scripts\python.exe scripts/verify_vlm_clip.py clip.mp4 --start 0 --end 5 --output result.json
~~~

## Validation and further tuning

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pose_activity.py tests/test_pose_mode.py tests/test_incident_verification.py tests/test_verification_websocket.py tests/test_video_upload.py -q
node --test tests/pose-overlay.test.cjs
~~~

Tests cover scaled/mirrored standing and sitting, missing/low-confidence joints, raised arms, bending, noise, sustained falls, resting, slow lowering, recovery, observation gaps, person ordering, replay resets, and both supplied standing screenshots. `data/pose_regressions.json` stores extracted keypoint coordinates, not images. The source screenshots already contain overlays, so they are regression cases rather than a clean evaluation dataset.

Verification unit/integration tests use a fake model to check evidence capture, deduplication, independent people, strict output parsing, unavailable/timeout behavior, SQLite persistence, and live/replay routing. They do not measure model accuracy. Use real clips separately to assess the VLM.

Local VLM smoke checks on the RTX 3050 used six RGB frames from the [UR Fall Detection Dataset](https://fenix.ur.edu.pl/~mkepski/ds/uf.html). The actual camera WebSocket path triggered on `fall-01-cam0` at 3.4 seconds and verified its surrounding window as **FALL** in 6.6 seconds while pose processing continued. `adl-01-cam0` (bending/crouching) returned **NO_FALL** in 3.1 seconds. Sampling six frames across the entire fall clip instead of the trigger window incorrectly returned **NO_FALL**: sampling and prompt wording affect this small model, and these development checks are not a held-out accuracy evaluation. This is why negative VLM results do not dismiss pose alerts. The source preview videos combine depth and RGB; `verify_vlm_clip.py --rgb-right-half` selects the RGB half. Clips remain local and are not redistributed in this repository; follow the dataset's license and attribution terms when obtaining them.

The observed results and tested runtime versions are saved in [data/vlm_smoke_results.json](data/vlm_smoke_results.json).

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

Saved 100% results in `data/benchmark_results.json` are **historical measurements of the previous pipeline**, not validation of v2. The UI labels them accordingly. No broad real-video accuracy is claimed for the new classifier or VLM. Occlusion, camera movement, identity crossings, deliberate fast lying down and falls directly toward/away from the lens remain challenging.

## References

- [Ultralytics pose output and keypoint confidence](https://docs.ultralytics.com/tasks/pose/)
- [Posture transitions and temporal fall detection](https://www.mdpi.com/2313-433X/7/3/42)

These motivate confidence-aware geometry and temporal checks; they do not validate this implementation's thresholds.
