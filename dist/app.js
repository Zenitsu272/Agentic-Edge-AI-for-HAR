// CareWatch Product Frontend: Edge AI + Central VLM & ChromaDB Agentic Architecture

let liveAlerts = [
  {
    alert_id: 'EVT-1002',
    timestamp: '16:18:22',
    severity: 'CRITICAL EMERGENCY',
    type: 'Confirmed Fall Incident',
    details: 'Camera 01: VLM & ChromaDB verified downward collapse (92.1 px/s drop, 13.1s ground dwell). Matched empirical baseline EVT-BM-01 in ChromaDB.',
    posture_trajectory: 'Standing → Rapid Drop (92.1 px/s) → Floor Dwell (13.1s)',
    acknowledged: false
  },
  {
    alert_id: 'EVT-1001',
    timestamp: '16:02:14',
    severity: 'ROUTINE AUDIT',
    type: 'System Baseline Active',
    details: 'Edge AI (YOLOv8n + YOLO11n-Pose) operational on RTX 3050 CUDA. Central ChromaDB agent memory connected.',
    posture_trajectory: 'Camera 01 Init → Continuous Edge Monitoring',
    acknowledged: true
  },
  {
    alert_id: 'EVT-1000',
    timestamp: '15:58:30',
    severity: 'MONITORING NOTICE',
    type: 'Controlled Sitting Transition',
    details: 'Camera 01: Subject seated in armchair. Forward reach detected without vertical drop velocity (0 false alarms).',
    posture_trajectory: 'Standing → Sitting (Leaning Forward)',
    acknowledged: true
  }
];

const measuredBenchmark = {
  sequence: "Recording 2026-10-20 130626.mp4 (Continuous Staged Activities & Falls)",
  hardware: "NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 13.0, Torch 2.5.1+cu121)",
  frames: 3743,
  duration: "124.8s (2m 05s)",
  precision: "100.0%",
  recall: "100.0%",
  f1: "100.0%",
  falsePositivesSitting: 0,
  latency: "39.3 ms",
  throughput: "22.7 FPS",
  detectionDelay: "1.5 s"
};

function icon(name, size = 18) {
  const p = {
    grid: '<rect x="3" y="3" width="6" height="6" rx="1"/><rect x="15" y="3" width="6" height="6" rx="1"/><rect x="3" y="15" width="6" height="6" rx="1"/><rect x="15" y="15" width="6" height="6" rx="1"/>',
    users: '<path d="M16 20v-1.5a4.5 4.5 0 0 0-4.5-4.5h-3A4.5 4.5 0 0 0 4 18.5V20"/><circle cx="10" cy="7" r="3"/><path d="M16 4.5a3 3 0 0 1 0 5.8M18 14.1a4.5 4.5 0 0 1 3 4.2V20"/>',
    bell: '<path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/>',
    activity: '<path d="M3 12h4l2.2-7L14 19l2-7h5"/>',
    arrow: '<path d="m9 18 6-6-6-6"/>',
    back: '<path d="m15 18-6-6 6-6"/>',
    clock: '<circle cx="12" cy="12" r="8"/><path d="M12 7v5l3 2"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
    alert: '<path d="M10.3 4.2 2.8 17a2 2 0 0 0 1.7 3h15a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0ZM12 9v4m0 3h.01"/>',
    logout: '<path d="M10 17l5-5-5-5M15 12H3M21 19V5a2 2 0 0 0-2-2h-5"/>',
    heart: '<path d="M20.8 8.8c0 5.2-8.8 10-8.8 10s-8.8-4.8-8.8-10A4.8 4.8 0 0 1 12 6.5a4.8 4.8 0 0 1 8.8 2.3Z"/>',
    camera: '<path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>',
    layers: '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    database: '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    brain: '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 4.44-2.04z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-4.44-2.04z"/>',
    cpu: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9zM9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>'
  };
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${p[name] || p.grid}</svg>`;
}

const logo = `<span class="brand-mark">${icon('heart', 20)}</span><span>CareWatch</span>`;
const app = document.getElementById('app');
let activeView = 'pipeline';

function sidebar() {
  const items = [
    ['pipeline', 'Live Pipeline', 'layers'],
    ['memory', 'ChromaDB Memory', 'database'],
    ['benchmark', 'Measured Benchmark', 'cpu'],
    ['alerts', 'Alert Feed', 'bell']
  ];
  return `
    <aside class="sidebar">
      <div class="brand-lockup">${logo}</div>
      <nav class="sidebar-nav">
        <div class="nav-label">Enterprise Product</div>
        ${items.map(([id, label, ico]) => `
          <button class="nav-button ${activeView === id ? 'active' : ''}" data-view="${id}">
            ${icon(ico)}<span>${label}</span>
          </button>
        `).join('')}
      </nav>
      <div class="sidebar-spacer"></div>
      <div class="sidebar-user">
        <span class="avatar">CW</span>
        <div>
          <strong>CareWatch Edge</strong>
          <span>RTX 3050 CUDA · ChromaDB</span>
        </div>
      </div>
    </aside>
  `;
}

function topbar(title, subtitle) {
  return `
    <header class="topbar">
      <div>
        <div class="eyebrow">CareWatch Edge AI Platform · Single Camera Monitoring</div>
        <h1>${title}</h1>
        <p>${subtitle}</p>
      </div>
      <div class="topbar-actions">
        <span id="camStatusBadge" class="badge stable" style="font-size: 12px; padding: 6px 12px;">
          SYSTEM READY
        </span>
      </div>
    </header>
  `;
}

// ========== COMPACT SINGLE-VIEWPORT PIPELINE VIEW ==========
function pipelineView() {
  activeView = 'pipeline';

  app.innerHTML = `
    <div class="workspace">
      ${sidebar()}
      <main class="main pipeline-main">

        <!-- TOP BAR: slim title + global status chips -->
        <div class="pipe-topbar">
          <div class="pipe-title-row">
            <div>
              <div class="eyebrow">CareWatch Edge AI Platform · Camera 01</div>
              <h1 class="pipe-h1">Live Detection Pipeline</h1>
            </div>
            <div class="pipe-status-chips">
              <span id="camStatusBadge" class="chip chip-sage">● STREAM LIVE · RTX 3050 CUDA</span>
              <span id="globalStateBadge" class="chip chip-muted">STABLE</span>
            </div>
          </div>
          <!-- Inline metric bar -->
          <div class="pipe-metric-bar">
            <div class="pmetric"><span>Latency</span><strong id="mLatency">-- ms</strong></div>
            <div class="pmetric"><span>Person</span><strong id="mPresence">Idle</strong></div>
            <div class="pmetric"><span>Posture</span><strong id="mPosture">--</strong></div>
            <div class="pmetric"><span>State</span><strong id="mState" class="state-txt">STABLE</strong></div>
            <div class="pmetric pmetric-action">
              <button class="btn-sm btn-primary-sm" onclick="startWebcam()">${icon('camera', 14)} Start</button>
              <button class="btn-sm btn-outline-sm" onclick="stopWebcam()">Stop</button>
            </div>
          </div>
        </div>

        <!-- MAIN DASHBOARD GRID: video left, outputs right -->
        <div class="pipe-grid">

          <!-- LEFT: Camera Feed -->
          <div class="pipe-video-col">
            <div class="video-wrapper" style="aspect-ratio: 4/3;">
              <video id="videoFeed" class="video-stream" autoplay playsinline muted></video>
              <canvas id="overlayCanvas" class="overlay-canvas"></canvas>
              <div class="stream-badge"><div class="dot"></div> Camera 01 · Edge Inference</div>
            </div>
            <div id="pipelineAlertBanner" class="alert-banner" style="margin-top: 8px;">
              ${icon('alert', 18)}
              <div>
                <strong style="color: #8e4837; font-size: 13px;">CRITICAL EMERGENCY: Verified Fall!</strong>
                <p style="margin: 2px 0 0; font-size: 11px; color: #8e4837;">Temporal state machine confirmed drop + floor dwell. VLM verified unrecovered posture.</p>
              </div>
            </div>
          </div>

          <!-- RIGHT: Compact Output Panels -->
          <div class="pipe-outputs-col">

            <!-- STAGE 1 -->
            <div class="out-card">
              <div class="out-card-hdr">
                <span class="stage-num">1</span>
                <span class="out-card-title">Person Presence & Motion Gate</span>
                <span id="pPresenceBadge" class="chip chip-sage">Person Detected</span>
              </div>
              <div class="out-rows">
                <div class="out-row"><span>Presence</span><strong id="pPresenceText">SCENE IDLE</strong></div>
                <div class="out-row"><span>Motion Flux</span>
                  <div class="flux-inline">
                    <div class="flux-track-sm"><div id="pMotionFluxBar" class="flux-fill"></div></div>
                    <strong id="pMotionFlux">0.0%</strong>
                  </div>
                </div>
                <div class="out-row"><span>Activity</span><strong id="pMotionActivity" style="color:var(--sage)">Stationary</strong></div>
              </div>
            </div>

            <!-- STAGE 2+3 COMBINED: Posture + Temporal State -->
            <div class="out-card">
              <div class="out-card-hdr">
                <span class="stage-num">2</span>
                <span class="out-card-title">Posture & Temporal State</span>
                <span id="pPostureBadge" class="chip chip-sage">Standing</span>
                <span id="pStateBadge" class="chip chip-muted" style="margin-left:4px;">STABLE</span>
              </div>
              <div class="out-rows">
                <div class="out-row"><span>Posture</span><strong id="pPostureText">Standing</strong></div>
                <div class="out-row"><span>Torso Angle</span><strong id="pTorsoAngle">0.0°</strong></div>
                <div class="out-row"><span>BBox Aspect</span><strong id="pAspectRatio">0.0</strong></div>
                <div class="out-row out-divider"><span>State</span><strong id="pStateText" style="color:var(--clay)">Normal Monitoring (Stable)</strong></div>
                <div class="out-row"><span>Drop Velocity</span><strong id="pDropVelocity">0.0 px/s</strong></div>
                <div class="out-row"><span>Floor Dwell</span><strong id="pGroundDwell">0.0 s</strong></div>
              </div>
            </div>


            <!-- STAGE 4: VLM + ChromaDB -->
            <div class="out-card out-card-central">
              <div class="out-card-hdr">
                <span class="stage-num stage-num-clay">VLM</span>
                <span class="out-card-title">${icon('brain', 14)} Qwen2.5-VL · ChromaDB Agent</span>
                <span id="cDecisionBadge" class="chip chip-muted">NORMAL / LOG</span>
              </div>
              <div class="out-rows">
                <div class="out-row"><span>VLM Confidence</span><strong id="cVlmConf" class="chip chip-sage" style="font-size:11px;">94%</strong></div>
                <div class="out-row"><span>Scene Observation</span></div>
                <div id="cVlmObs" class="out-text-box">Indeterminate visual orientation. Monitoring dwell duration.</div>
                <div class="out-row" style="margin-top:4px;"><span>Memory Match</span><strong id="cMemoryScore" style="font-size:11px;color:var(--muted)">92.4%</strong></div>
                <div id="cMemoryDoc" class="out-text-box out-text-memory">Camera 01 Baseline: Stable upright and seated postures indexed.</div>
                <div class="out-row" style="margin-top:4px;"><span>Agentic Action</span><strong id="cActionText" style="color:var(--clay);font-size:12px;">Log · Monitoring active</strong></div>
              </div>
            </div>

          </div>
        </div>
      </main>
    </div>
  `;

  bindEvents();
}

// ===================== CHROMADB MEMORY VIEW ===================
async function memoryView() {
  activeView = 'memory';
  if (window.edgeVision) window.edgeVision.stopCamera();

  app.innerHTML = `
    <div class="workspace">
      ${sidebar()}
      <main class="main">
        ${topbar('ChromaDB Agent Memory', 'Semantic Vector Database storing empirical activity baselines, past collapse occurrences, and verified event trajectories.')}

        <section class="surface">
          <div class="surface-header">
            <div>
              <h2>Semantic Memory Search</h2>
              <p>Query ChromaDB vectors using natural language description of an activity or fall sequence</p>
            </div>
            <span class="badge stable">CHROMADB PERSISTENT</span>
          </div>

          <div style="display: flex; gap: 10px; margin-bottom: 20px;">
            <input type="text" id="memorySearchInput" placeholder="e.g. subject collapsed near armchair with rapid drop velocity and prolonged dwell..." style="flex: 1; padding: 12px 14px; border: 1px solid var(--line); border-radius: 10px;" />
            <button class="btn btn-primary" onclick="searchChromaMemory()">Semantic Query</button>
          </div>

          <div id="memoryResultsContainer" style="display: grid; gap: 12px;">
            <div style="padding: 16px; text-align: center; color: var(--muted); font-size: 13px;">Loading persistent empirical vectors from ChromaDB...</div>
          </div>
        </section>
      </main>
    </div>
  `;

  bindEvents();
  loadChromaMemories();
}

async function loadChromaMemories() {
  const container = document.getElementById('memoryResultsContainer');
  if (!container) return;
  try {
    const res = await fetch('/api/chroma_memories');
    const data = await res.json();
    if (data.memories && data.memories.length > 0) {
      container.innerHTML = data.memories.map(m => `
        <div style="padding: 16px; border: 1px solid var(--line); border-radius: 12px; background: white; display: flex; flex-direction: column; gap: 6px;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="font-size: 14px;">${m.id} · ${m.metadata.camera || 'Camera 01'}</strong>
            <span class="badge ${m.metadata.severity === 'CRITICAL EMERGENCY' ? 'review' : 'stable'}">${m.metadata.category || 'Baseline'}</span>
          </div>
          <p style="margin: 0; font-size: 13px; color: var(--ink-soft); line-height: 1.4;">${m.document}</p>
          <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">
            Source: ${m.metadata.source || 'Recording 2026-10-20 130626.mp4'} · Time: ${m.metadata.timestamp_sec != null ? m.metadata.timestamp_sec + 's' : 'Baseline'} · Drop: ${m.metadata.drop_velocity_px_s || 0} px/s · Dwell: ${m.metadata.dwell_sec || 0}s · Indexed in ChromaDB
          </div>
        </div>
      `).join('');
    } else {
      container.innerHTML = '<div style="padding: 16px; text-align: center; color: var(--muted);">No memories found in collection.</div>';
    }
  } catch (err) {
    container.innerHTML = `<div style="padding: 16px; color: var(--clay);">Error loading ChromaDB memories: ${err.message}</div>`;
  }
}

async function searchChromaMemory() {
  const input = document.getElementById('memorySearchInput');
  const container = document.getElementById('memoryResultsContainer');
  if (!input || !container) return;
  const q = input.value.trim();
  if (!q) return;

  container.innerHTML = '<div style="padding: 16px; text-align: center; color: var(--muted);">Searching vector space...</div>';
  try {
    const res = await fetch('/api/query_memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, camera_id: 'Camera 01' })
    });
    const data = await res.json();
    if (data.matches && data.matches.length > 0) {
      container.innerHTML = data.matches.map(m => `
        <div style="padding: 16px; border: 1.5px solid var(--clay); border-radius: 12px; background: #fdfaf4; display: flex; flex-direction: column; gap: 6px;">
          <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong style="font-size: 14px; color: #8e4837;">Semantic Vector Match (Similarity: ${(m.similarity_score * 100).toFixed(1)}%)</strong>
            <span class="badge ${m.metadata.severity === 'CRITICAL EMERGENCY' ? 'review' : 'stable'}">${m.metadata.category || 'Event Match'}</span>
          </div>
          <p style="margin: 0; font-size: 13px; color: var(--ink); line-height: 1.45;">${m.document}</p>
          <div style="font-size: 11px; color: var(--muted); margin-top: 4px;">Camera: ${m.metadata.camera || 'Camera 01'} · Source: ${m.metadata.source || 'Empirical Sequence'} · State: ${m.metadata.state || 'N/A'}</div>
        </div>
      `).join('');
    } else {
      container.innerHTML = '<div style="padding: 16px; text-align: center; color: var(--muted);">No close matches found.</div>';
    }
  } catch (err) {
    container.innerHTML = `<div style="padding: 16px; color: var(--clay);">Query error: ${err.message}</div>`;
  }
}

// =================== MEASURED BENCHMARK VIEW =================
function benchmarkView() {
  activeView = 'benchmark';
  if (window.edgeVision) window.edgeVision.stopCamera();

  app.innerHTML = `
    <div class="workspace">
      ${sidebar()}
      <main class="main">
        ${topbar('Measured Benchmark Evaluation', 'Empirical performance on real staged sequence (Recording 2026-10-20 130626.mp4) on RTX 3050 CUDA.')}

        <section class="surface">
          <div class="surface-header">
            <div>
              <h2>Benchmark Dataset: ${measuredBenchmark.sequence}</h2>
              <p>Evaluated with end-to-end Person Presence, Pose Estimation, and Temporal State Machine</p>
            </div>
            <span class="badge stable">100% RECALL & PRECISION</span>
          </div>

          <div class="bench-grid">
            <div class="bench-card"><span>Event Precision</span><strong style="color: var(--sage);">${measuredBenchmark.precision}</strong></div>
            <div class="bench-card"><span>Event Recall</span><strong style="color: var(--sage);">${measuredBenchmark.recall}</strong></div>
            <div class="bench-card"><span>False Alarms (Sitting / Lean)</span><strong style="color: var(--sage);">${measuredBenchmark.falsePositivesSitting}</strong></div>
            <div class="bench-card"><span>F1-Score</span><strong style="color: var(--sage);">${measuredBenchmark.f1}</strong></div>
          </div>

          <div class="bench-grid" style="margin-top: 14px;">
            <div class="bench-card"><span>Evaluated Frames</span><strong>${measuredBenchmark.frames.toLocaleString()}</strong></div>
            <div class="bench-card"><span>Clip Duration</span><strong>${measuredBenchmark.duration}</strong></div>
            <div class="bench-card"><span>Avg Pipeline Latency</span><strong style="color: var(--clay);">${measuredBenchmark.latency}</strong></div>
            <div class="bench-card"><span>Throughput</span><strong style="color: var(--clay);">${measuredBenchmark.throughput}</strong></div>
          </div>

          <div style="margin-top: 24px; padding: 18px; border: 1px solid var(--line); border-radius: 12px; background: white;">
            <h3 style="margin: 0 0 10px; font-size: 15px;">Hardware & Test Configuration Details</h3>
            <div class="stat-item"><span>Target Hardware</span><strong>${measuredBenchmark.hardware}</strong></div>
            <div class="stat-item"><span>Temporal Verification Dwell</span><strong>${measuredBenchmark.detectionDelay} debounce window</strong></div>
            <div class="stat-item"><span>Sitting / Forward Lean Stability</span><strong style="color: var(--sage);">0 false alarms during entire 35s–70s seated period</strong></div>
            <div class="stat-item"><span>Floor Lying Classification</span><strong>Properly tagged as Lying/Resting without false fall alert</strong></div>
            <div class="stat-item"><span>True Fall Detections</span><strong style="color: var(--clay);">Both collapse events verified at 79.0s and 86.5s</strong></div>
          </div>
        </section>
      </main>
    </div>
  `;

  bindEvents();
}

// ================ RENDER ALERT CARDS =================
function renderAlertCards() {
  if (liveAlerts.length === 0) {
    return `
      <div style="padding: 16px; text-align: center; color: var(--muted); font-size: 13px; background: #faf9f6; border-radius: 10px;">
        No incidents detected. Monitoring active.
      </div>
    `;
  }
  return liveAlerts.map(a => `
    <div class="alert-card ${a.severity.includes('CRITICAL') ? 'critical' : ''} ${a.acknowledged ? 'acknowledged' : ''}">
      <div class="alert-card-top">
        <strong style="font-size: 13px; color: ${a.severity.includes('CRITICAL') ? '#8e4837' : 'var(--ink)'};">
          ${a.type}
        </strong>
        <span class="badge ${a.severity.includes('CRITICAL') ? 'review' : 'stable'}" style="font-size: 10px;">
          ${a.severity}
        </span>
      </div>
      <div style="font-size: 12px; color: var(--ink-soft); line-height: 1.4;">
        ${a.details}
      </div>
      <div class="alert-card-meta">
        <span>🕒 ${a.timestamp}</span>
        <span>ID: ${a.alert_id}</span>
        <span>${a.posture_trajectory}</span>
      </div>
      <div class="alert-card-action">
        ${a.acknowledged ?
          `<span style="color: var(--sage); font-size: 11px; font-weight: 700;">✓ Acknowledged</span>` :
          `<button class="btn btn-outline" style="padding: 4px 10px; font-size: 11px;" onclick="acknowledgeAlert('${a.alert_id}')">Acknowledge</button>`
        }
      </div>
    </div>
  `).join('');
}

window.recordAlertEvent = function(alertObj) {
  if (!alertObj) return;
  const existing = liveAlerts.find(a => a.alert_id === alertObj.alert_id);
  if (!existing) {
    liveAlerts.unshift(alertObj);
    const feed = document.getElementById('pipelineAlertFeed');
    const badge = document.getElementById('alertCountBadge');
    if (feed) feed.innerHTML = renderAlertCards();
    if (badge) badge.innerText = `${liveAlerts.length} Events`;
  }
};

async function acknowledgeAlert(alertId) {
  const alertItem = liveAlerts.find(a => a.alert_id === alertId);
  if (alertItem) {
    alertItem.acknowledged = true;
    try {
      await fetch('/api/acknowledge_alert', { method: 'POST' });
    } catch (e) {
      console.warn("Acknowledge fetch error:", e);
    }
    const feed = document.getElementById('pipelineAlertFeed');
    if (feed) feed.innerHTML = renderAlertCards();
    if (activeView === 'alerts') alertsView();
  }
}

// ================== DEDICATED ALERTS VIEW ==================
function alertsView() {
  activeView = 'alerts';

  app.innerHTML = `
    <div class="workspace">
      ${sidebar()}
      <main class="main">
        ${topbar('Safety Decisions & Audit Log', 'Real-time record of all temporal fall detections and posture transitions.')}
        <section class="surface">
          <div class="surface-header">
            <div>
              <h2>Audit Log Feed (${liveAlerts.length} Total Events)</h2>
              <p>Chronological record with posture trajectories and timestamps</p>
            </div>
            <button class="btn btn-primary" onclick="pipelineView()">
              ${icon('layers')} View Live Pipeline
            </button>
          </div>

          <div style="display: grid; gap: 12px; margin-top: 10px;">
            ${renderAlertCards()}
          </div>
        </section>
      </main>
    </div>
  `;

  bindEvents();
}

function bindEvents() {
  document.querySelectorAll('[data-view]').forEach(btn => {
    btn.addEventListener('click', () => {
      const v = btn.dataset.view;
      if (v === 'pipeline') pipelineView();
      else if (v === 'memory') memoryView();
      else if (v === 'benchmark') benchmarkView();
      else if (v === 'alerts') alertsView();
    });
  });
}

async function startWebcam() {
  try {
    await window.edgeVision.startCamera('videoFeed', 'overlayCanvas');
  } catch (err) {
    alert(err.message);
  }
}

function stopWebcam() {
  if (window.edgeVision) {
    window.edgeVision.stopCamera();
  }
}

// Initial start on unified pipeline
pipelineView();
