// One transform for video and landmarks, including letterboxing and webcam mirroring.
class EdgeVisionClient {
  constructor() {
    this.video = document.getElementById('videoFeed');
    this.image = document.getElementById('serverFrameFeed');
    this.overlay = document.getElementById('overlayCanvas');
    this.ctx = this.overlay.getContext('2d');
    this.capture = document.createElement('canvas');
    this.captureCtx = this.capture.getContext('2d');
    this.generation = 0;
    this.running = false;
    this.poses = [];
    this.pairs = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
    new ResizeObserver(() => this.drawOverlay()).observe(this.overlay.parentElement);
  }
  setStatus(text) { document.getElementById('cameraStatus').textContent = text; }
  setAlertAppearance(active) {
    document.body.classList.toggle('fall-alert-active', Boolean(active));
    document.getElementById('cameraView').classList.toggle('fall-alert-active', Boolean(active));
  }
  async startCamera() {
    this.stop();
    const token = this.generation;
    this.setStatus('Opening camera…');
    const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 1280}, height: {ideal: 720}}, audio: false});
    if (token !== this.generation) { stream.getTracks().forEach(track => track.stop()); return; }
    this.stream = stream;
    this.video.srcObject = stream;
    this.mirrored = true;
    this.video.classList.add('mirrored');
    await this.video.play();
    if (token !== this.generation) return;
    this.connect('/ws/pipeline');
  }
  async startReplay(file) {
    this.stop();
    const token = this.generation;
    this.setStatus('Opening video…');
    this.objectURL = URL.createObjectURL(file);
    this.video.src = this.objectURL;
    this.video.loop = true;
    try {
      await this.video.play();
      if (token !== this.generation) return;
      this.connect('/ws/pipeline');
    } catch {
      if (token !== this.generation) return;
      this.video.pause();
      this.video.hidden = true;
      this.image.hidden = false;
      this.setStatus('Preparing video…');
      const body = new FormData();
      body.append('file', file);
      const response = await fetch('/api/upload_video', {method: 'POST', body});
      const data = await response.json();
      if (token !== this.generation) return;
      if (!response.ok || data.status !== 'success') throw new Error(data.message || 'Could not open this video.');
      this.serverReplay = true;
      this.connect(`/ws/replay_stream?video=${encodeURIComponent(data.filename)}`);
    }
  }
  connect(endpoint) {
    const token = this.generation;
    this.running = true;
    this.setStatus('Connecting…');
    document.getElementById('stopButton').disabled = false;
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}${endpoint}`);
    this.ws = ws;
    ws.onopen = () => {
      if (token !== this.generation) return;
      this.setStatus(this.stream ? 'Camera live' : 'Video replay');
      document.getElementById('cameraView').classList.add('is-live');
      document.getElementById('emptyState').hidden = true;
      document.getElementById('activityLabel').textContent = 'Looking for a person…';
      if (!this.serverReplay) this.sendFrame();
    };
    ws.onmessage = async event => {
      if (token !== this.generation) return;
      clearTimeout(this.responseTimer);
      try {
        const data = JSON.parse(event.data);
        if (data.error) throw new Error(data.error);
        if (data.rendered_frame) {
          this.image.src = data.rendered_frame;
          await this.image.decode();
          if (token !== this.generation) return;
        }
        this.handleResult(data);
        if (!this.serverReplay) this.frameTimer = setTimeout(() => this.sendFrame(), 35);
      } catch (error) { if (token === this.generation) this.fail(error.message); }
    };
    ws.onerror = () => { if (token === this.generation) this.fail('Cannot connect to the detection server. Try Start camera again.'); };
    ws.onclose = () => { if (token === this.generation && this.running) this.fail('Connection lost. Start the camera again to reconnect.'); };
    this.responseTimer = setTimeout(() => this.fail('The detection server did not respond. Try again.'), 20000);
  }
  sendFrame() {
    if (!this.running || this.ws?.readyState !== WebSocket.OPEN) return;
    if (this.video.readyState < 2) { this.frameTimer = setTimeout(() => this.sendFrame(), 50); return; }
    this.frameWidth = Math.min(960, this.video.videoWidth);
    this.frameHeight = Math.max(1, Math.round(this.frameWidth * this.video.videoHeight / this.video.videoWidth));
    this.capture.width = this.frameWidth;
    this.capture.height = this.frameHeight;
    this.captureCtx.drawImage(this.video, 0, 0, this.frameWidth, this.frameHeight);
    this.ws.send(JSON.stringify({frame: this.capture.toDataURL('image/jpeg', .85), timestamp: this.stream ? null : this.video.currentTime}));
    clearTimeout(this.responseTimer);
    this.responseTimer = setTimeout(() => this.fail('Detection timed out. Start the camera again to reconnect.'), 20000);
  }
  handleResult(data) {
    const presence = data.stage1_presence;
    const pose = data.stage2_posture;
    this.frameWidth = data.frame_width || this.frameWidth;
    this.frameHeight = data.frame_height || this.frameHeight;
    this.poses = data.poses || (pose.keypoints?.length ? [{keypoints: pose.keypoints, posture: pose.posture, box: pose.box}] : []);
    const fallAlert = data.stage3_event?.fall_alert_active ?? data.stage3_event?.fall_confirmed;
    this.setAlertAppearance(fallAlert);
    const activity = fallAlert ? 'Fall detected' : (!presence.person_present ? 'No person detected' : pose.posture);
    document.getElementById('activityLabel').textContent = activity;
    document.getElementById('personCount').textContent = presence.person_present ? `${presence.persons_count} ${presence.persons_count === 1 ? 'person' : 'people'} detected` : 'No person detected';
    document.getElementById('latencyLabel').textContent = `${Math.round(data.inference_ms)} ms`;
    document.getElementById('activityHint').textContent = fallAlert
      ? (data.stage3_event.fall_detection_level === 'possible_partial' ? 'Possible / partial fall after sudden motion. Check the person.' : 'Fall alert active. Check the person.')
      : (!presence.person_present ? 'Step into the camera view to begin.' : pose.details);
    window.updateDashboard?.(data);
    this.drawOverlay();
  }
  drawOverlay() {
    const width = this.overlay.clientWidth, height = this.overlay.clientHeight;
    const dpr = window.devicePixelRatio || 1;
    this.overlay.width = Math.round(width * dpr);
    this.overlay.height = Math.round(height * dpr);
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!this.frameWidth || !this.frameHeight || !this.running) return;
    const scale = Math.min(width / this.frameWidth, height / this.frameHeight);
    const ox = (width - this.frameWidth * scale) / 2, oy = (height - this.frameHeight * scale) / 2;
    const point = pt => [ox + (this.mirrored ? this.frameWidth - pt[0] : pt[0]) * scale, oy + pt[1] * scale];
    const visible = pt => pt && pt[2] >= .35 && pt[0] > 1 && pt[1] > 1 && pt[0] < this.frameWidth - 1 && pt[1] < this.frameHeight - 1;
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    for (const pose of this.poses) {
      const points = pose.keypoints;
      for (const [a,b] of this.pairs) {
        if (!visible(points[a]) || !visible(points[b])) continue;
        ctx.beginPath(); ctx.moveTo(...point(points[a])); ctx.lineTo(...point(points[b]));
        ctx.lineWidth = 6; ctx.strokeStyle = '#071a13b0'; ctx.stroke();
        ctx.lineWidth = 3; ctx.strokeStyle = a < 5 ? '#ffe08a' : '#63ffd0'; ctx.stroke();
      }
      points.forEach((pt, i) => {
        if (!visible(pt)) return;
        ctx.beginPath(); ctx.arc(...point(pt), i < 5 ? 4 : 5, 0, 2 * Math.PI);
        ctx.fillStyle = i < 5 ? '#ffe08a' : '#63ffd0'; ctx.fill();
        ctx.lineWidth = 1.5; ctx.strokeStyle = '#122e24'; ctx.stroke();
      });
      if (pose.box) {
        const [x,y] = point([this.mirrored ? pose.box[2] : pose.box[0], pose.box[1]]);
        const label = `#${pose.track_id || ''} ${pose.posture || 'Person'}${pose.provisional ? ' · limited view' : ''}`;
        ctx.font = '600 14px system-ui';
        const labelWidth = ctx.measureText(label).width + 20;
        const lx = Math.max(8, Math.min(width - labelWidth - 8, x));
        const ly = Math.max(100, Math.min(height - 140, y - 34));
        ctx.fillStyle = pose.category === 'Fall' ? '#7c3025eb' : '#10271ee8'; ctx.fillRect(lx, ly, labelWidth, 28);
        ctx.fillStyle = '#b8ffe2'; ctx.fillText(label, lx + 10, ly + 19);
      }
    }
  }
  fail(message) {
    this.stop();
    const error = document.getElementById('cameraError');
    error.textContent = message; error.hidden = false;
  }
  stop() {
    this.generation++; this.running = false;
    this.setAlertAppearance(false);
    window.resetDashboard?.();
    clearTimeout(this.frameTimer); clearTimeout(this.responseTimer);
    if (this.ws) { this.ws.onclose = null; this.ws.close(); this.ws = null; }
    if (this.stream) this.stream.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.pause(); this.video.srcObject = null; this.video.removeAttribute('src'); this.video.load();
    if (this.objectURL) URL.revokeObjectURL(this.objectURL);
    this.objectURL = null;
    this.image.removeAttribute('src'); this.image.hidden = true; this.video.hidden = false;
    this.video.classList.remove('mirrored');
    this.mirrored = false; this.serverReplay = false; this.poses = [];
    this.drawOverlay(); this.setStatus('Camera off');
    document.getElementById('cameraView').classList.remove('is-live');
    document.getElementById('emptyState').hidden = false;
    document.getElementById('stopButton').disabled = true;
    document.getElementById('activityLabel').textContent = 'Ready when you are';
    document.getElementById('activityHint').textContent = 'Keep your full body in view for better recognition.';
    document.getElementById('personCount').textContent = 'No person detected';
    document.getElementById('latencyLabel').textContent = '';
  }
}
