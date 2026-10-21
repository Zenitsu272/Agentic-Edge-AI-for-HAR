// Camera, landmarks and live activity.
document.getElementById('app').innerHTML = `
<div class="workspace">
<aside class="sidebar">
<a class="brand" href="/"><span class="brand-icon">♡</span>CareWatch</a>
<p class="nav-caption">EDGE ACTIVITY MONITOR</p>
<nav aria-label="Main navigation"><button class="nav-item active" data-page="live">◉ <span>Live pipeline</span></button><button class="nav-item" data-page="history">◷ <span>Activity & alerts</span></button><button class="nav-item" data-page="memory">▤ <span>ChromaDB memory</span></button><button class="nav-item" data-page="benchmark">▥ <span>Measured benchmark</span></button></nav>
</aside>
<main class="dashboard">
<header class="dashboard-header"><div><span class="eyebrow">CAREWATCH EDGE AI PLATFORM · CAMERA 01</span><h1 id="pageHeading">Live detection pipeline</h1><p id="pageSubtitle">Person tracking, body keypoints and activity recognition.</p></div><span id="systemStatus" class="pill">Connecting to server</span></header>
<div id="livePage" class="page-panel">
<div class="summary-grid"><div class="summary-card"><span>Standing</span><strong id="standingCount">0</strong><small>Upright posture</small></div><div class="summary-card"><span>Sitting</span><strong id="sittingCount">0</strong><small>Seated geometry</small></div><div class="summary-card fall-summary"><span>Fall</span><strong id="fallCount">0</strong><small>Rapid descent + sustained down pose</small></div><div class="summary-card"><span>Inference</span><strong id="inferenceMetric">—</strong><small id="modelMetric">Loading model status</small></div></div>
<div class="live-grid"><div class="video-column">
<section id="cameraView" class="camera-view" aria-label="Live camera and activity">
<video id="videoFeed" autoplay playsinline muted></video><img id="serverFrameFeed" alt="Video replay" hidden><canvas id="overlayCanvas" aria-label="Detected body keypoints"></canvas>
<div class="camera-toolbar"><span id="cameraStatus" class="camera-status">Camera off</span><div class="controls"><button id="startButton" class="primary">Start camera</button><button id="replayButton">Replay video</button><button id="stopButton" disabled>Stop</button><button id="fullscreenButton" aria-pressed="false">⛶ Full View</button><input type="file" id="replayInput" accept="video/*,.avi,.mkv,.mov" hidden></div></div>
<div id="emptyState" class="empty-state"><span class="camera-icon" aria-hidden="true">◎</span><h2>Ready to observe</h2><p>Start a camera or replay a video.<br>Keep shoulders, hips and knees in view.</p></div>
<div class="activity-bar"><div class="activity-main"><span class="eyebrow">Primary activity</span><strong id="activityLabel" aria-live="polite">Awaiting subject</strong><span id="activityHint">Standing, sitting and temporal fall detection.</span></div><div class="live-metrics"><span id="personCount">No person detected</span><span id="latencyLabel"></span></div></div>
<p id="cameraError" class="error-message" role="alert" hidden></p></section>
<div class="camera-caption"><span><i class="legend-dot"></i> Visible joints and skeleton</span><span>Each person is tracked independently</span></div>
<div id="fallBanner" class="fall-banner" role="alert" hidden><strong>Fall detected</strong><span id="fallBannerDetail">Sudden fall-like motion detected. Check the person.</span></div>
<section class="panel live-events"><div class="panel-heading"><h2>Recent activity</h2><span class="pill">This session</span></div><div id="sessionEvents" class="event-list"><p class="muted">Activity changes will appear here.</p></div></section>
</div>
<aside class="analysis-column">
<section class="panel"><div class="panel-heading"><h2><span class="step-number">1</span>Person presence & motion</h2><span id="presenceBadge" class="pill">Idle</span></div><div class="metric-row"><span>People tracked</span><strong id="trackedMetric">0</strong></div><div class="metric-row"><span>Movement</span><strong id="movementMetric">—</strong></div><div class="metric-row"><span>Motion flux</span><strong id="fluxMetric">—</strong></div></section>
<section class="panel"><div class="panel-heading"><h2><span class="step-number">2</span>Posture & temporal state</h2><span class="pill accent">Active</span></div><div id="personCards"><p class="muted">Waiting for visible body keypoints.</p></div></section>
<section class="panel"><div class="panel-heading"><h2>Fall observation</h2><span id="fallState" class="pill">Monitoring</span></div><div class="metric-row"><span>Torso descent</span><strong id="descentMetric">—</strong></div><div class="metric-row"><span>Down-pose duration</span><strong id="dwellMetric">—</strong></div><p class="panel-note">A person already lying down is not automatically classified as a fall. Missing joints are marked as limited evidence.</p></section>
<section class="panel paused-panel"><div class="panel-heading"><h2>VLM & central agent</h2><span class="pill">Paused</span></div><p class="panel-note">Pose-based recognition is active. Second-level VLM verification is disabled for now.</p></section>
</aside></div></div>
<section id="historyPage" class="page-panel panel" hidden><div class="panel-heading"><h2>Activity & alert feed</h2><button class="light-button" id="refreshHistory">Refresh</button></div><p class="panel-note">Persisted activity transitions and fall alerts. Older records retain their original detection method.</p><div id="historyList"></div></section>
<section id="memoryPage" class="page-panel panel" hidden><div class="panel-heading"><h2>ChromaDB semantic memory</h2><span class="pill">Stored history</span></div><p class="panel-note">Review existing memories. They do not influence the current pose classification.</p><form id="memoryForm" class="search-form"><input id="memoryQuery" placeholder="Search stored memories" aria-label="Search stored memories"><button class="light-button">Search</button></form><div id="memoryList"></div></section>
<section id="benchmarkPage" class="page-panel panel" hidden><div class="panel-heading"><h2>Measured benchmark</h2><span class="pill">Historical run</span></div><p class="notice">These saved measurements belong to the previous classifier. The new pose classifier needs a fresh labeled-video evaluation; these scores do not validate it.</p><div id="benchmarkContent"></div></section>
</main></div>`;
const cameraView = document.getElementById('cameraView');
const fullscreenButton = document.getElementById('fullscreenButton');
window.edgeVision = new EdgeVisionClient();
async function startSource(file) {
  document.getElementById('cameraError').hidden = true;
  document.getElementById('startButton').disabled = true;
  document.getElementById('replayButton').disabled = true;
  try {
    if (file) await window.edgeVision.startReplay(file);
    else await window.edgeVision.startCamera();
  } catch (error) { window.edgeVision.fail(error.message); }
  finally {
    document.getElementById('startButton').disabled = false;
    document.getElementById('replayButton').disabled = false;
  }
}
document.getElementById('startButton').addEventListener('click', () => startSource());
document.getElementById('stopButton').addEventListener('click', () => window.edgeVision.stop());
document.getElementById('replayButton').addEventListener('click', () => document.getElementById('replayInput').click());
document.getElementById('replayInput').addEventListener('change', event => {
  const file = event.target.files[0];
  if (file) startSource(file);
  event.target.value = '';
});
function updateFullscreen() {
  const expanded = document.fullscreenElement === cameraView || cameraView.classList.contains('expanded');
  fullscreenButton.textContent = expanded ? '⤢ Exit full screen' : '⛶ Full View';
  fullscreenButton.setAttribute('aria-pressed', String(expanded));
  window.edgeVision.drawOverlay();
}
fullscreenButton.addEventListener('click', async () => {
  if (document.fullscreenElement) await document.exitFullscreen();
  else if (cameraView.classList.contains('expanded')) cameraView.classList.remove('expanded');
  else {
    try {
      if (!cameraView.requestFullscreen) throw new Error('Fullscreen unavailable');
      await cameraView.requestFullscreen();
    } catch { cameraView.classList.add('expanded'); }
  }
  updateFullscreen();
});
document.addEventListener('fullscreenchange', updateFullscreen);
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') { cameraView.classList.remove('expanded'); updateFullscreen(); }
});
window.addEventListener('pagehide', () => window.edgeVision.stop());

// All server-derived text is rendered as text, never as HTML.
function text(id, value) { document.getElementById(id).textContent = value; }
function node(tag, className, value) { const el = document.createElement(tag); el.className = className; if (value !== undefined) el.textContent = value; return el; }
const recentStates = new Map();
let recentEvents = [];
window.updateDashboard = data => {
  const poses = data.poses || [], s1 = data.stage1_presence, s3 = data.stage3_event;
  const fallAlert = s3.fall_alert_active ?? s3.fall_confirmed;
  for (const category of ['Standing','Sitting','Fall']) text(category.toLowerCase()+'Count',category==='Fall' && fallAlert ? Math.max(1,poses.filter(p=>p.category===category).length) : poses.filter(p=>p.category===category).length);
  text('inferenceMetric', `${Math.round(data.inference_ms)} ms`);
  text('modelMetric', `${data.model || 'Pose model'} · ${data.device || 'Local'}`);
  text('trackedMetric', poses.length); text('presenceBadge', poses.length ? 'Detected' : 'Idle');
  text('movementMetric',s1.movement_status || '—'); text('fluxMetric',`${s1.motion_flux_pct || 0}%`);
  text('fallState',fallAlert ? (s3.fall_detection_level==='possible_partial' ? 'Possible / partial fall' : 'Fall detected') : s3.state==='VERIFYING' ? 'Checking motion' : 'Monitoring');
  text('descentMetric',`${s3.normalized_drop_speed || 0} torso lengths/s`);
  text('dwellMetric',`${s3.ground_dwell_sec || 0} s`);
  document.getElementById('fallBanner').hidden = !fallAlert;
  text('fallBannerDetail',s3.fall_detection_level==='possible_partial' ? 'Possible / partial fall after sudden motion from standing or sitting. Alert stays visible for 10 seconds.' : 'Sustained collapse confirmed by the pose checks. Check the person.');
  const cards = document.getElementById('personCards'); cards.replaceChildren();
  if (!poses.length) cards.append(node('p','muted','Waiting for visible body keypoints.'));
  for (const p of poses) {
    const card = node('div','person-card');
    const heading = node('div','person-heading'); heading.append(node('strong','',`Person ${p.track_id}`),node('span',`activity-tag ${p.category.toLowerCase()}`,p.posture));
    card.append(heading,node('p','person-reason',p.reason));
    const f = p.features || {};
    const rows = [['Visible joints',`${f.visible_keypoints || 0} / 17`],['Torso tilt',f.torso_angle == null ? 'Unavailable' : `${f.torso_angle}°`],['Knee angle',f.knee_angle == null ? 'Not visible' : `${f.knee_angle}°`],['Evidence',p.provisional ? 'Limited · provisional' : 'Observed geometry']];
    for (const [label,value] of rows) { const row=node('div','metric-row'); row.append(node('span','',label),node('strong','',value)); card.append(row); }
    cards.append(card);
    if (recentStates.get(p.track_id)!==p.posture) {
      recentEvents.unshift({time:new Date().toLocaleTimeString(),person:p.track_id,label:p.posture,limited:p.provisional});
      recentEvents=recentEvents.slice(0,5); recentStates.set(p.track_id,p.posture);
    }
  }
  const events = document.getElementById('sessionEvents');
  if (recentEvents.length) { events.replaceChildren(); for (const e of recentEvents) { const row=node('div','session-event'); row.append(node('time','',e.time),node('strong','',`Person ${e.person}`),node('span','',e.label+(e.limited?' · limited view':''))); events.append(row); } }
};
window.resetDashboard = () => {
  recentStates.clear(); recentEvents=[];
  window.updateDashboard({poses:[],inference_ms:0,stage1_presence:{},stage3_event:{}});
  text('inferenceMetric','—'); text('sessionEvents','Activity changes will appear here.');
};
async function getJSON(url,options) { const r=await fetch(url,options); if(!r.ok) throw new Error(`Request failed (${r.status})`); return r.json(); }
async function loadHistory() {
  const list=document.getElementById('historyList'); text('historyList','Loading records…');
  try {
    const data=await getJSON('/api/alerts?limit=50'); list.replaceChildren();
    if(!data.alerts?.length) { list.append(node('p','muted','No activity records yet. Start the camera to begin.')); return; }
    for(const event of data.alerts) {
      const row=node('article','history-row');
      const content=node('div',''); content.append(node('strong','',`${event.type} · ${event.track_id}`),node('p','',event.details),node('small','muted',`${event.full_timestamp} · ${event.metadata?.classifier_version || 'Previous pipeline'}`)); row.append(content);
      if(event.severity!=='NORMAL'&&!event.acknowledged) { const btn=node('button','light-button','Acknowledge'); btn.onclick=async()=>{ try {await getJSON(`/api/alerts/${encodeURIComponent(event.alert_id)}/acknowledge`,{method:'POST'}); await loadHistory();} catch(e){btn.textContent='Retry acknowledgement';} }; row.append(btn); }
      else row.append(node('span','pill',event.acknowledged?'Acknowledged':event.severity));
      list.append(row);
    }
  } catch(e){text('historyList',e.message);}
}
async function loadMemory(query='') {
  const list=document.getElementById('memoryList'); text('memoryList','Loading memories…');
  try {
    const data=query ? await getJSON('/api/query_memory',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({query})}) : await getJSON('/api/chroma_memories');
    list.replaceChildren(); const memories=data.memories || data.matches || [];
    if(!memories.length) list.append(node('p','muted','No stored memories found.'));
    for(const m of memories) { const card=node('article','memory-card'); card.append(node('p','',m.document || m.text || ''),node('small','muted',m.id || m.metadata?.source || 'Stored memory')); list.append(card); }
  } catch(e){text('memoryList',e.message);}
}
async function loadBenchmark() {
  try {
    const {benchmark:b}=await getJSON('/api/benchmark'); const container=document.getElementById('benchmarkContent'); container.replaceChildren();
    if(!b) {container.append(node('p','muted','No saved benchmark.'));return;}
    container.append(node('h3','',b.dataset_name),node('p','muted',b.dataset_description));
    for(const [label,value] of [['Frames evaluated',b.total_frames_evaluated],['Event precision',`${b.event_level_precision_pct}%`],['Event recall',`${b.event_level_recall_pct}%`],['Average inference',`${b.avg_inference_latency_ms} ms`],['P95 latency',`${b.p95_tail_latency_ms} ms`]]) {const row=node('div','metric-row');row.append(node('span','',label),node('strong','',value));container.append(row);}
  } catch(e){text('benchmarkContent',e.message);}
}
const pageTitles={live:['Live detection pipeline','Person tracking, body keypoints and activity recognition.'],history:['Activity & alerts','Review and acknowledge recorded events.'],memory:['ChromaDB memory','Existing semantic memories and context.'],benchmark:['Measured benchmark','Saved evaluation results and their scope.']};
document.querySelectorAll('[data-page]').forEach(button=>button.addEventListener('click',()=>{
  const page=button.dataset.page;
  document.querySelectorAll('.page-panel').forEach(panel=>panel.hidden=panel.id!==page+'Page');
  document.querySelectorAll('[data-page]').forEach(b=>b.classList.toggle('active',b===button));
  text('pageHeading',pageTitles[page][0]);text('pageSubtitle',pageTitles[page][1]);
  if(page==='history')loadHistory();if(page==='memory')loadMemory();if(page==='benchmark')loadBenchmark();
}));
document.getElementById('refreshHistory').onclick=loadHistory;
document.getElementById('memoryForm').onsubmit=e=>{e.preventDefault();loadMemory(document.getElementById('memoryQuery').value.trim());};
getJSON('/api/health').then(h=>{text('systemStatus',`Pose engine online · ${h.hardware.yolo_device}`);text('modelMetric',h.components.pose_estimator.model);}).catch(()=>text('systemStatus','Server unavailable'));
