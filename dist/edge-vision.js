/**
 * CareWatch Live Stream & Unified Edge AI Pipeline Client
 * Stages:
 * 1. Person Presence & Genuine Motion Flux (YOLOv8n + Frame Diff)
 * 2. Biomechanical Posture Estimation (YOLO11n-Pose: Standing, Sitting, Lying/Resting)
 * 3. Temporal Fall Event State Machine (Drop Velocity -> Verification -> Fall Confirmed)
 * 4. Real-time Alert Dispatch & Audit Trail
 */

class EdgeVisionClient {
  constructor() {
    this.stream = null;
    this.ws = null;
    this.mode = 'pipeline'; // 'pipeline', 'motion', or 'har'
    this.canvas = document.createElement('canvas');
    this.ctx = this.canvas.getContext('2d');
    this.videoEl = null;
    this.overlayCanvas = null;
    this.overlayCtx = null;
    this.isStreaming = false;
    this.waitingForResponse = false;
    this.lastFrameTime = 0;
    this.smoothedKpts = null;
    this.sentFrameWidth = 480;
    this.sentFrameHeight = 360;
    
    // 17-Keypoint COCO pairs for full-body & facial landmark tracking
    this.SKELETON_PAIRS = [
      // Facial & Head landmarks (Nose, Eyes, Ears)
      [0, 1], [0, 2], [1, 3], [2, 4],
      // Torso & Shoulders
      [5, 6], [5, 11], [6, 12], [11, 12],
      // Arms (Shoulder -> Elbow -> Wrist)
      [5, 7], [7, 9], [6, 8], [8, 10],
      // Legs (Hip -> Knee -> Ankle)
      [11, 13], [13, 15], [12, 14], [14, 16]
    ];
  }

  setMode(mode) {
    if (this.mode !== mode) {
      this.mode = mode;
      console.log(`[EdgeVisionClient] Mode changed to ${mode}`);
      if (this.isStreaming) {
        this.closeWebSocket();
        this.connectWebSocket();
      }
    }
  }

  attachToDOMElements(videoElementId, overlayCanvasId) {
    this.videoEl = document.getElementById(videoElementId);
    this.overlayCanvas = document.getElementById(overlayCanvasId);
    if (this.overlayCanvas) {
      this.overlayCtx = this.overlayCanvas.getContext('2d');
    }
    if (this.videoEl && this.stream) {
      this.videoEl.srcObject = this.stream;
      this.videoEl.play().catch(e => console.warn(e));
    }
  }

  async startCamera(videoElementId, overlayCanvasId) {
    this.videoEl = document.getElementById(videoElementId);
    this.overlayCanvas = document.getElementById(overlayCanvasId);
    if (this.overlayCanvas) {
      this.overlayCtx = this.overlayCanvas.getContext('2d');
      this.overlayCanvas.classList.remove('no-mirror');
    }

    if (!this.stream) {
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
          },
          audio: false
        });
      } catch (err) {
        throw new Error("Camera permission denied or camera not found: " + err.message);
      }
    }

    if (this.videoEl) {
      this.videoEl.srcObject = this.stream;
      await this.videoEl.play();
    }

    this.isStreaming = true;
    this.connectWebSocket();
  }

  async startVideoReplay(file, videoElementId, serverFrameId, overlayCanvasId) {
    this.videoEl = document.getElementById(videoElementId);
    this.serverFrameEl = document.getElementById(serverFrameId);
    this.overlayCanvas = document.getElementById(overlayCanvasId);
    if (this.overlayCanvas) {
      this.overlayCtx = this.overlayCanvas.getContext('2d');
    }

    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }

    if (this.serverFrameEl) {
      this.serverFrameEl.style.display = 'none';
      this.serverFrameEl.src = '';
    }

    if (this.videoEl) {
      this.videoEl.style.display = 'block';
      this.videoEl.srcObject = null;
      this.videoEl.src = URL.createObjectURL(file);
      this.videoEl.loop = true;
      this.videoEl.muted = true;
      
      this.videoEl.onerror = async () => {
        console.warn("[EdgeVisionClient] Browser cannot decode video, falling back to server-side OpenCV replay:", file.name);
        await this.startServerVideoReplay(file, videoElementId, serverFrameId, overlayCanvasId);
      };

      try {
        await this.videoEl.play();
      } catch (e) {
        console.warn("[EdgeVisionClient] Browser playback error, falling back to server-side OpenCV replay:", e);
        await this.startServerVideoReplay(file, videoElementId, serverFrameId, overlayCanvasId);
        return;
      }
    }

    this.isStreaming = true;
    this.connectWebSocket();

    const statusBadge = document.getElementById('camStatusBadge');
    if (statusBadge) {
      statusBadge.innerText = '● REPLAY SIMULATION · RTX 3050 CUDA';
      statusBadge.className = 'chip chip-sage';
    }
  }

  async startServerVideoReplay(file, videoElementId, serverFrameId, overlayCanvasId) {
    this.videoEl = document.getElementById(videoElementId);
    this.serverFrameEl = document.getElementById(serverFrameId);
    this.overlayCanvas = document.getElementById(overlayCanvasId);
    if (this.overlayCanvas) {
      this.overlayCtx = this.overlayCanvas.getContext('2d');
      this.overlayCanvas.classList.add('no-mirror');
    }

    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }

    if (this.videoEl) {
      this.videoEl.pause();
      this.videoEl.style.display = 'none';
    }

    if (this.serverFrameEl) {
      this.serverFrameEl.style.display = 'block';
    }

    const statusBadge = document.getElementById('camStatusBadge');
    if (statusBadge) {
      statusBadge.innerText = `● UPLOADING ${file.name.toUpperCase()}...`;
      statusBadge.className = 'chip chip-amber';
    }

    // 1. Upload to backend
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/api/upload_video', {
      method: 'POST',
      body: formData
    });
    const uploadData = await res.json();
    if (uploadData.status !== 'success') {
      throw new Error(uploadData.message || 'Failed to upload video');
    }

    // 2. Connect to WebSocket replay stream
    this.closeWebSocket();
    this.isStreaming = true;
    this.isServerReplay = true;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}/ws/replay_stream?video=${encodeURIComponent(uploadData.filename)}`;
    console.log(`[EdgeVisionClient] Connecting server video replay -> ${wsUrl}`);
    
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log(`[EdgeVisionClient] Replay stream connected for ${uploadData.filename}`);
      if (statusBadge) {
        statusBadge.innerText = `● REPLAY (${uploadData.format.toUpperCase()} · ${uploadData.resolution}) · RTX 3050 CUDA`;
        statusBadge.className = 'chip chip-sage';
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.error) {
          console.error("Replay error:", data.error);
          return;
        }

        // Render frame directly onto serverFrameEl
        if (data.rendered_frame && this.serverFrameEl) {
          this.serverFrameEl.src = data.rendered_frame;
        }

        // Handle pipeline output
        this.handlePipelineResult(data);

        // Draw skeleton onto overlay canvas
        if (this.overlayCanvas && this.serverFrameEl) {
          const w = this.serverFrameEl.clientWidth || 480;
          const h = this.serverFrameEl.clientHeight || 360;
          if (w > 0 && h > 0) {
            this.overlayCanvas.width = w;
            this.overlayCanvas.height = h;
            this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

            const sx = this.overlayCanvas.width / 480.0;
            const sy = this.overlayCanvas.height / 360.0;

            const s2 = data.stage2_posture;
            if (s2 && s2.keypoints && s2.keypoints.length > 0) {
              const kpts = s2.keypoints;
              const theme = s2.category === 'Lying' ? '#b5179e' : '#00f5d4';
              this.overlayCtx.lineCap = 'round';
              this.overlayCtx.lineJoin = 'round';

              for (const [p1, p2] of this.SKELETON_PAIRS) {
                const pt1 = kpts[p1];
                const pt2 = kpts[p2];
                if (pt1 && pt2 && pt1[2] > 0.20 && pt2[2] > 0.20) {
                  const isHead = (p1 < 5 || p2 < 5);
                  this.overlayCtx.lineWidth = isHead ? 2.2 : 3.5;
                  this.overlayCtx.strokeStyle = isHead ? '#ffd166' : theme;
                  this.overlayCtx.beginPath();
                  this.overlayCtx.moveTo(pt1[0] * sx, pt1[1] * sy);
                  this.overlayCtx.lineTo(pt2[0] * sx, pt2[1] * sy);
                  this.overlayCtx.stroke();
                }
              }

              kpts.forEach((pt, idx) => {
                if (pt[2] > 0.20) {
                  const jx = pt[0] * sx;
                  const jy = pt[1] * sy;
                  const isHead = idx < 5;
                  const radius = isHead ? 3.5 : 4.8;
                  this.overlayCtx.beginPath();
                  this.overlayCtx.arc(jx, jy, radius, 0, 2 * Math.PI);
                  this.overlayCtx.fillStyle = isHead ? '#ffd166' : theme;
                  this.overlayCtx.fill();
                  this.overlayCtx.beginPath();
                  this.overlayCtx.arc(jx, jy, radius * 0.45, 0, 2 * Math.PI);
                  this.overlayCtx.fillStyle = '#ffffff';
                  this.overlayCtx.fill();
                }
              });
            }
          }
        }
      } catch (err) {
        console.error("Replay message parse error:", err);
      }
    };

    this.ws.onerror = (err) => {
      console.error("[EdgeVisionClient] Replay WS error:", err);
    };

    this.ws.onclose = () => {
      console.log("[EdgeVisionClient] Replay WS closed");
    };
  }

  stopCamera() {
    this.isStreaming = false;
    this.isServerReplay = false;
    this.waitingForResponse = false;
    this.closeWebSocket();

    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }

    if (this.videoEl) {
      this.videoEl.pause();
      this.videoEl.srcObject = null;
      this.videoEl.removeAttribute('src');
      this.videoEl.style.display = 'block';
    }

    if (this.serverFrameEl) {
      this.serverFrameEl.src = '';
      this.serverFrameEl.style.display = 'none';
    }

    if (this.overlayCtx && this.overlayCanvas) {
      this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    }

    const statusBadge = document.getElementById('camStatusBadge');
    if (statusBadge) {
      statusBadge.innerText = 'STREAM OFFLINE';
      statusBadge.style.color = '#7a847f';
    }
  }

  connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    let endpoint = '/ws/pipeline';
    if (this.mode === 'motion') endpoint = '/ws/motion';
    else if (this.mode === 'har') endpoint = '/ws/har';

    const wsUrl = `${protocol}//${location.host}${endpoint}`;

    console.log(`[EdgeVisionClient] Connecting WebSocket -> ${wsUrl}`);
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = () => {
      console.log(`[EdgeVisionClient] Connected to ${endpoint}`);
      const statusBadge = document.getElementById('camStatusBadge');
      if (statusBadge) {
        statusBadge.innerText = 'STREAM LIVE · RTX 3050 CUDA';
        statusBadge.style.color = '#5f806b';
      }
      this.waitingForResponse = false;
      this.sendCurrentFrame();
    };

    this.ws.onmessage = (event) => {
      this.waitingForResponse = false;
      try {
        const data = JSON.parse(event.data);
        if (this.mode === 'pipeline') {
          this.handlePipelineResult(data);
        } else if (this.mode === 'motion') {
          this.handleMotionResult(data);
        } else {
          this.handleHARResult(data);
        }
      } catch (e) {
        console.error("[EdgeVisionClient] Error parsing message", e);
      }

      this.requestNextFrame();
    };

    this.ws.onerror = (err) => {
      console.error("[EdgeVisionClient] WS Error", err);
      this.waitingForResponse = false;
    };

    this.ws.onclose = () => {
      console.log("[EdgeVisionClient] WS Disconnected");
      this.waitingForResponse = false;
    };
  }

  closeWebSocket() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  requestNextFrame() {
    if (!this.isStreaming) return;
    const now = performance.now();
    const elapsed = now - this.lastFrameTime;
    const minDelay = 40; // ~25 FPS smooth live detection on GPU

    if (elapsed < minDelay) {
      setTimeout(() => this.sendCurrentFrame(), minDelay - elapsed);
    } else {
      requestAnimationFrame(() => this.sendCurrentFrame());
    }
  }

  sendCurrentFrame() {
    if (!this.isStreaming || this.waitingForResponse) return;
    if (!this.videoEl || this.videoEl.readyState < 2) {
      setTimeout(() => this.sendCurrentFrame(), 50);
      return;
    }
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const vw = this.videoEl.videoWidth || 640;
    const vh = this.videoEl.videoHeight || 480;
    const targetW = 480;
    const targetH = Math.max(240, Math.round(targetW * (vh / vw)));

    this.canvas.width = targetW;
    this.canvas.height = targetH;
    this.ctx.drawImage(this.videoEl, 0, 0, targetW, targetH);
    const b64 = this.canvas.toDataURL('image/jpeg', 0.60);

    this.sentFrameWidth = targetW;
    this.sentFrameHeight = targetH;
    this.waitingForResponse = true;
    this.lastFrameTime = performance.now();
    this.ws.send(b64);
  }

  // ----------------- UNIFIED 1-PAGE PIPELINE HANDLER -----------------
  handlePipelineResult(data) {
    const s1 = data.stage1_presence;
    const s2 = data.stage2_posture;
    const s3 = data.stage3_event;

    // 1. Metric Strip Updates
    const mLatency = document.getElementById('mLatency');
    const mPresence = document.getElementById('mPresence');
    const mPosture = document.getElementById('mPosture');
    const mState = document.getElementById('mState');

    if (mLatency) mLatency.innerText = `${data.inference_ms} ms`;
    if (mPresence) mPresence.innerText = s1.person_present ? `Present (${s1.persons_count})` : 'Idle';
    if (mPosture) mPosture.innerText = s2.posture || 'Idle';
    if (mState) mState.innerText = s3.state;

    // 2. Stage 1: Person Presence & Movement Gate Card
    const pPresenceText = document.getElementById('pPresenceText');
    const pPresenceBadge = document.getElementById('pPresenceBadge');
    const pMotionFlux = document.getElementById('pMotionFlux');
    const pMotionFluxBar = document.getElementById('pMotionFluxBar');
    const pMotionActivity = document.getElementById('pMotionActivity');

    if (pPresenceText) pPresenceText.innerText = s1.status_text;
    if (pPresenceBadge) {
      pPresenceBadge.className = s1.person_present ? 'badge stable' : 'badge';
      pPresenceBadge.innerText = s1.person_present ? 'Person Detected' : 'Scene Idle';
    }
    if (pMotionFlux) pMotionFlux.innerText = `${s1.motion_flux_pct}%`;
    if (pMotionFluxBar) pMotionFluxBar.style.width = `${Math.min(100, s1.motion_flux_pct * 4)}%`;
    if (pMotionActivity) {
      pMotionActivity.innerText = s1.motion_active ? 'Active Movement' : 'Stationary Presence';
      pMotionActivity.style.color = s1.motion_active ? 'var(--amber)' : 'var(--sage)';
    }

    // 3. Stage 2: Biomechanical Posture Recognition Card
    const pPostureText = document.getElementById('pPostureText');
    const pPostureBadge = document.getElementById('pPostureBadge');
    const pTorsoAngle = document.getElementById('pTorsoAngle');
    const pAspectRatio = document.getElementById('pAspectRatio');
    const pPostureDetail = document.getElementById('pPostureDetail');

    if (pPostureText) pPostureText.innerText = s2.posture;
    if (pTorsoAngle) pTorsoAngle.innerText = `${s2.torso_angle_deg}°`;
    if (pAspectRatio) pAspectRatio.innerText = `${s2.aspect_ratio}`;
    if (pPostureDetail) pPostureDetail.innerText = s2.details;

    if (pPostureBadge) {
      if (s2.category === 'Standing') {
        pPostureBadge.className = 'badge stable';
        pPostureBadge.innerText = 'Standing (Upright)';
      } else if (s2.category === 'Sitting') {
        pPostureBadge.className = 'badge monitor';
        pPostureBadge.innerText = s2.posture.includes('Leaning') ? 'Sitting (Leaning)' : 'Sitting (Seated)';
      } else if (s2.category === 'Lying') {
        pPostureBadge.className = 'badge';
        pPostureBadge.style.background = '#ebe4f4';
        pPostureBadge.style.color = '#6b3ea3';
        pPostureBadge.innerText = 'Lying / Resting';
      } else {
        pPostureBadge.className = 'badge';
        pPostureBadge.innerText = s2.posture;
      }
    }

    // 4. Stage 3: Temporal Fall Event State Machine Card
    const pStateText = document.getElementById('pStateText');
    const pStateBadge = document.getElementById('pStateBadge');
    const pDropVelocity = document.getElementById('pDropVelocity');
    const pGroundDwell = document.getElementById('pGroundDwell');
    const pipelineAlertBanner = document.getElementById('pipelineAlertBanner');

    if (pStateText) pStateText.innerText = s3.state_label;
    if (pDropVelocity) pDropVelocity.innerText = `${s3.drop_velocity} px/s`;
    if (pGroundDwell) pGroundDwell.innerText = `${s3.ground_dwell_sec} s`;

    if (pStateBadge) {
      if (s3.state === 'FALL_CONFIRMED') {
        pStateBadge.className = 'badge review';
        pStateBadge.innerText = 'CRITICAL FALL';
      } else if (s3.state === 'VERIFYING') {
        pStateBadge.className = 'badge monitor';
        pStateBadge.innerText = 'VERIFYING';
      } else if (s3.state === 'RAPID_DROP') {
        pStateBadge.className = 'badge monitor';
        pStateBadge.innerText = 'DROP DETECTED';
      } else if (s3.state === 'RECOVERED') {
        pStateBadge.className = 'badge stable';
        pStateBadge.innerText = 'RECOVERED';
      } else {
        pStateBadge.className = 'badge stable';
        pStateBadge.innerText = 'STABLE';
      }
    }

    if (pipelineAlertBanner) {
      pipelineAlertBanner.style.display = s3.fall_confirmed ? 'flex' : 'none';
    }

    // 5. Central AI Server: VLM Verification, ChromaDB Memory & Escalation Decision
    if (data.central_server) {
      const c = data.central_server;
      const cDecisionBadge = document.getElementById('cDecisionBadge');
      const cActionText = document.getElementById('cActionText');
      const cRationaleText = document.getElementById('cRationaleText');
      const cVlmObs = document.getElementById('cVlmObs');
      const cVlmConf = document.getElementById('cVlmConf');
      const cMemoryDoc = document.getElementById('cMemoryDoc');
      const cMemoryScore = document.getElementById('cMemoryScore');

      if (cDecisionBadge) {
        if (c.escalation_decision.includes('EMERGENCY')) {
          cDecisionBadge.className = 'badge review';
          cDecisionBadge.innerText = 'EMERGENCY / ALERT';
        } else if (c.escalation_decision.includes('WARNING')) {
          cDecisionBadge.className = 'badge monitor';
          cDecisionBadge.innerText = 'WARNING / REVIEW';
        } else {
          cDecisionBadge.className = 'badge stable';
          cDecisionBadge.innerText = 'NORMAL / LOG';
        }
      }

      if (cActionText) cActionText.innerText = c.recommended_action;
      if (cRationaleText) cRationaleText.innerText = c.agent_rationale;
      if (cVlmObs) cVlmObs.innerText = c.vlm_verification.scene_observation;
      if (cVlmConf) cVlmConf.innerText = `${Math.round(c.vlm_verification.confidence * 100)}%`;

      if (c.chroma_memory_matches && c.chroma_memory_matches.length > 0) {
        const topMatch = c.chroma_memory_matches[0];
        if (cMemoryDoc) cMemoryDoc.innerText = topMatch.document;
        if (cMemoryScore) cMemoryScore.innerText = `Similarity: ${(topMatch.similarity_score * 100).toFixed(1)}%`;
      }
    }

    // 6. Push to Dynamic Alert Feed
    if (s3.active_alert && window.recordAlertEvent) {
      window.recordAlertEvent(s3.active_alert);
    }

    // 6. Canvas Overlays (Bounding Box + Skeleton with Smoothing & True Aspect Ratio)
    if (this.overlayCanvas && this.videoEl && this.videoEl.videoWidth > 0) {
      this.overlayCanvas.width = this.videoEl.videoWidth;
      this.overlayCanvas.height = this.videoEl.videoHeight;
      this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

      const sentW = this.sentFrameWidth || 480.0;
      const sentH = this.sentFrameHeight || (sentW * (this.videoEl.videoHeight / this.videoEl.videoWidth));
      const sx = this.overlayCanvas.width / sentW;
      const sy = this.overlayCanvas.height / sentH;

      // Color palette based on posture & risk state
      let themeColor = '#00f5d4'; // Vivid Emerald Cyan
      if (s2.category === 'Sitting') themeColor = '#4cc9f0'; // Clean Blue
      if (s2.category === 'Lying') themeColor = '#b5179e'; // Vivid Purple
      if (s3.state === 'RAPID_DROP' || s3.state === 'VERIFYING') themeColor = '#ff9f1c'; // Amber Warning
      if (s3.state === 'FALL_CONFIRMED') themeColor = '#e71d36'; // Red Alert

      // A. Draw Presence Bounding Box
      s1.boxes.forEach(b => {
        const bx = b.x1 * sx;
        const by = b.y1 * sy;
        const bw = (b.x2 - b.x1) * sx;
        const bh = (b.y2 - b.y1) * sy;

        this.overlayCtx.lineWidth = 2.0;
        this.overlayCtx.strokeStyle = themeColor;
        this.overlayCtx.fillStyle = 'rgba(0, 245, 212, 0.04)';
        this.overlayCtx.strokeRect(bx, by, bw, bh);
        this.overlayCtx.fillRect(bx, by, bw, bh);
      });

      // B. Smooth Keypoint Interpolation (EMA)
      const rawKpts = s2.keypoints;
      if (rawKpts && rawKpts.length >= 17) {
        if (!this.smoothedKpts || this.smoothedKpts.length !== rawKpts.length) {
          this.smoothedKpts = rawKpts.map(p => [p[0], p[1], p[2]]);
        } else {
          const alpha = 0.65; // Highly responsive smoothing
          for (let i = 0; i < rawKpts.length; i++) {
            if (rawKpts[i][2] > 0.15) {
              this.smoothedKpts[i][0] = alpha * rawKpts[i][0] + (1 - alpha) * this.smoothedKpts[i][0];
              this.smoothedKpts[i][1] = alpha * rawKpts[i][1] + (1 - alpha) * this.smoothedKpts[i][1];
              this.smoothedKpts[i][2] = rawKpts[i][2];
            } else {
              this.smoothedKpts[i][2] *= 0.85;
            }
          }
        }

        const kpts = this.smoothedKpts;
        this.overlayCtx.lineCap = 'round';
        this.overlayCtx.lineJoin = 'round';

        // Draw Bones (Skeleton pairs)
        this.SKELETON_PAIRS.forEach(([i, j]) => {
          if (kpts[i][2] > 0.20 && kpts[j][2] > 0.20) {
            const isHead = (i < 5 || j < 5);
            this.overlayCtx.lineWidth = isHead ? 2.2 : 3.5;
            this.overlayCtx.strokeStyle = isHead ? '#ffd166' : themeColor;
            this.overlayCtx.beginPath();
            this.overlayCtx.moveTo(kpts[i][0] * sx, kpts[i][1] * sy);
            this.overlayCtx.lineTo(kpts[j][0] * sx, kpts[j][1] * sy);
            this.overlayCtx.stroke();
          }
        });

        // Draw Joints (Dual-ring glowing dots)
        kpts.forEach((pt, idx) => {
          if (pt[2] > 0.20) {
            const jx = pt[0] * sx;
            const jy = pt[1] * sy;
            const isHead = (idx < 5);
            const radius = isHead ? 3.5 : 4.8;

            // Outer ring
            this.overlayCtx.beginPath();
            this.overlayCtx.arc(jx, jy, radius, 0, 2 * Math.PI);
            this.overlayCtx.fillStyle = isHead ? '#ffd166' : themeColor;
            this.overlayCtx.fill();

            // Inner white core
            this.overlayCtx.beginPath();
            this.overlayCtx.arc(jx, jy, radius * 0.45, 0, 2 * Math.PI);
            this.overlayCtx.fillStyle = '#ffffff';
            this.overlayCtx.fill();
          }
        });

        // Header Badge over subject
        const headPt = kpts[0];
        if (headPt && headPt[2] > 0.15) {
          const hx = headPt[0] * sx;
          const hy = Math.max(28, (headPt[1] * sy) - 18);
          this.overlayCtx.save();
          this.overlayCtx.translate(hx, hy);
          this.overlayCtx.scale(-1, 1); // Compensates for CSS scaleX(-1) so text reads left-to-right
          this.overlayCtx.fillStyle = themeColor;
          this.overlayCtx.font = "600 13px Inter, system-ui, sans-serif";
          this.overlayCtx.textAlign = "center";
          this.overlayCtx.fillText(`${s2.posture} · [${s3.state}]`, 0, 0);
          this.overlayCtx.restore();
        }
      }
    }
  }

  handleMotionResult(data) {
    // Kept for backward compatibility
  }

  handleHARResult(data) {
    // Kept for backward compatibility
  }
}

// Global instance
window.edgeVision = new EdgeVisionClient();
