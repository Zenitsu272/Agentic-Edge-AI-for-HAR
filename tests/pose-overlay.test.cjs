const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function client(width, height, overrides={}) {
  const dots = [];
  const context = new Proxy({arc: (x, y) => dots.push([x, y]), measureText: () => ({width: 40})}, {get: (o,k) => o[k] || (() => {})});
  const elements = new Map();
  const classes = () => {
    const values = new Set();
    return {add: value => values.add(value), remove: value => values.delete(value),
      toggle: (value, on) => on ? values.add(value) : values.delete(value), contains: value => values.has(value)};
  };
  const element = id => {
    if (!elements.has(id)) elements.set(id, {clientWidth: width, clientHeight: height, getContext: () => context, classList: classes(), parentElement: {}, pause(){}, load(){}, removeAttribute(){}});
    return elements.get(id);
  };
  const sandbox = {document: {body: element('body'), getElementById: element, createElement: () => element('capture')}, window: {devicePixelRatio: 2}, ResizeObserver: class {observe(){}}, clearTimeout, URL,
    setTimeout:(callback,delay)=>{const timer=setTimeout(callback,delay);timer.unref();return timer;}};
  Object.assign(sandbox,overrides);
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync('dist/edge-vision.js','utf8') + '\nthis.client = new EdgeVisionClient();', sandbox);
  const c = sandbox.client;
  c.running = true; c.frameWidth = 640; c.frameHeight = 480;
  c.poses = [{keypoints: [[100,100,.9]]}];
  return {c, dots, elements, sandbox};
}

test('landmarks account for letterboxing, pixel density and fullscreen resizing', () => {
  const {c, dots} = client(1920,1080);
  c.drawOverlay();
  assert.deepEqual(dots[0], [465,225]);
  assert.equal(c.overlay.width, 3840);
  c.overlay.clientWidth = 640; c.overlay.clientHeight = 480;
  c.drawOverlay();
  assert.deepEqual(dots[1], [100,100]);
});

test('video selection uploads for single-pass server analysis', async () => {
  let uploaded,endpoint;
  const {c,sandbox}=client(640,480,{
    AbortController,FormData:class {append(name,file){uploaded=file;}},
    fetch:async(url,options)=>({ok:true,json:async()=>({status:'success',filename:'upload.avi',total_frames:4})})
  });
  let began=false,ready=false;
  sandbox.window.beginVideoCheck=()=>{began=true;};sandbox.window.videoUploadReady=()=>{ready=true;};
  c.connect=value=>{endpoint=value;};
  const file={name:'clip.avi'};await c.startReplay(file);
  assert.equal(uploaded,file);assert.ok(began && ready);
  assert.equal(endpoint,'/ws/replay_stream?video=upload.avi&loop=false');
  assert.equal(c.serverReplay,true);assert.equal(c.mirrored,false);
});

test('a completed video keeps its preview without a connection-lost error',async()=>{
  let socket,completed=false;
  class WS {constructor(){socket=this;}close(){}}
  const {c,elements,sandbox}=client(640,480,{WebSocket:WS,location:{protocol:'http:',host:'localhost'},setTimeout});
  sandbox.window.finishVideoCheck=()=>{completed=true;};
  c.serverReplay=true;c.connect('/ws/replay_stream');socket.onopen();
  await socket.onmessage({data:JSON.stringify({type:'video_complete',processed_frames:4,incidents:[]})});
  socket.onclose();
  assert.ok(completed && c.completed);
  assert.equal(elements.get('cameraStatus').textContent,'Video checked');
  assert.equal(c.running,true);c.stop();
});

test('stopping an upload aborts it and prevents a late video connection',async()=>{
  let signal,connections=0;
  const {c}=client(640,480,{AbortController,FormData:class {append(){}},
    fetch:async(url,options)=>new Promise((resolve,reject)=>{signal=options.signal;signal.addEventListener('abort',()=>reject(new Error('aborted')));})});
  c.connect=()=>connections++;
  const upload=c.startReplay({name:'clip.mp4'});c.stop();await upload;
  assert.ok(signal.aborted);assert.equal(connections,0);
});

test('only camera landmarks are mirrored, and low-confidence joints are hidden', () => {
  const {c,dots} = client(1920,1080);
  c.mirrored = true;
  c.poses[0].keypoints.push([200,200,.1]);
  c.drawOverlay();
  assert.deepEqual(dots, [[1455,225]]);
});

test('empty scene clears skeleton and activity; stop releases camera', () => {
  const {c, elements} = client(640,480);
  c.handleResult({frame_width:640, frame_height:480, poses:[], inference_ms:20, stage1_presence:{person_present:false}, stage2_posture:{category:'Unknown'}});
  assert.equal(c.poses.length,0);
  assert.equal(elements.get('activityLabel').textContent,'No person detected');
  let stopped = false;
  c.stream = {getTracks: () => [{stop: () => {stopped = true;}}]};
  c.stop();
  assert.ok(stopped);
  assert.equal(c.running, false);
});

test('held fall remains the main label when the person leaves the frame', () => {
  const {c, elements} = client(640,480);
  const result = {frame_width:640, frame_height:480, poses:[], inference_ms:20,
    stage1_presence:{person_present:false}, stage2_posture:{category:'Unknown'},
    stage3_event:{fall_confirmed:true,fall_alert_held:true}};
  c.handleResult(result);
  assert.equal(elements.get('activityLabel').textContent,'Fall detected');
  assert.ok(elements.get('body').classList.contains('fall-alert-active'));
  assert.ok(elements.get('cameraView').classList.contains('fall-alert-active'));
  result.stage3_event.fall_confirmed=false;
  c.handleResult(result);
  assert.equal(elements.get('activityLabel').textContent,'No person detected');
  c.stop();
  assert.ok(!elements.get('body').classList.contains('fall-alert-active'));
  assert.ok(!elements.get('cameraView').classList.contains('fall-alert-active'));
});

function alertClock() {
  let now=0,next=0;const timers=new Map();
  return {Date:{now:()=>now},setTimeout:(callback,delay)=>{timers.set(++next,{at:now+delay,callback});return next;},
    clearTimeout:id=>timers.delete(id),tick:ms=>{
      now+=ms;
      for(const [id,timer] of [...timers])if(timer.at<=now){timers.delete(id);timer.callback();}
    }};
}
function fallPacket(id='incident-1') {
  return {frame_width:640,frame_height:480,poses:[],inference_ms:20,
    stage1_presence:{person_present:true,persons_count:1},
    stage2_posture:{posture:'Fall detected',category:'Fall'},
    stage3_event:{fall_alert_active:true,fall_confirmed:false,active_alert:{alert_id:id},
      active_incidents:[{alert_id:id}]}};
}

test('red border and banner expire at ten seconds despite repeated detections and VLM updates',()=>{
  const clock=alertClock();const {c,elements}=client(640,480,clock);
  const data=fallPacket();c.handleResult(data);
  clock.tick(9000);data.stage3_event.fall_confirmed=true;
  data.verification={status:'CONFIRMED_FALL',scene_observation:'Visible collapse.'};c.handleResult(data);
  assert.ok(elements.get('body').classList.contains('fall-alert-active'));
  clock.tick(999);assert.ok(elements.get('cameraView').classList.contains('fall-alert-active'));
  clock.tick(1);
  assert.ok(!elements.get('body').classList.contains('fall-alert-active'));
  assert.ok(!elements.get('cameraView').classList.contains('fall-alert-active'));
  assert.equal(elements.get('fallBanner').hidden,true);
  c.handleResult(data);
  assert.equal(data.visual_fall_alert_active,false);
  assert.equal(elements.get('fallBanner').hidden,true);
  assert.equal(elements.get('activityLabel').textContent,'Fall confirmed · VLM');
});

test('red alert clears without another video frame and a different incident can alert again',()=>{
  const clock=alertClock();const {c,elements}=client(640,480,clock);
  c.handleResult(fallPacket());clock.tick(10000);
  assert.ok(!elements.get('body').classList.contains('fall-alert-active'));
  c.handleResult(fallPacket('incident-2'));
  assert.ok(elements.get('body').classList.contains('fall-alert-active'));
  clock.tick(10000);assert.equal(elements.get('fallBanner').hidden,true);
  c.handleResult(fallPacket('incident-1'));
  assert.ok(!elements.get('body').classList.contains('fall-alert-active'));
});

test('stop cancels the previous visual alert before a new stream starts',()=>{
  const clock=alertClock();const {c,elements}=client(640,480,clock);
  c.handleResult(fallPacket());clock.tick(5000);c.stop();
  assert.ok(!elements.get('body').classList.contains('fall-alert-active'));
  c.handleResult(fallPacket());clock.tick(5000);
  assert.ok(elements.get('body').classList.contains('fall-alert-active'));
  clock.tick(5000);assert.equal(elements.get('fallBanner').hidden,true);
});

test('a partial fall alert is displayed before strict confirmation', () => {
  const {c,elements} = client(640,480);
  c.handleResult({frame_width:640,frame_height:480,poses:[],inference_ms:20,
    stage1_presence:{person_present:true,persons_count:1},
    stage2_posture:{posture:'Standing',category:'Standing'},
    stage3_event:{fall_alert_active:true,fall_confirmed:false,fall_detection_level:'possible_partial'}});
  assert.equal(elements.get('activityLabel').textContent,'Fall detected');
  assert.match(elements.get('activityHint').textContent,/partial fall/);
  assert.ok(elements.get('cameraView').classList.contains('fall-alert-active'));
  c.stop();
  assert.ok(!elements.get('body').classList.contains('fall-alert-active'));
});

test('a negative VLM result preserves the independent pose alert for review', () => {
  const {c,elements}=client(640,480);
  c.handleResult({frame_width:640,frame_height:480,poses:[],inference_ms:20,
    stage1_presence:{person_present:true,persons_count:1},
    stage2_posture:{posture:'Fall detected',category:'Fall'},
    stage3_event:{fall_alert_active:true,fall_confirmed:true},
    verification:{status:'NOT_FALL',scene_observation:'Controlled movement.'}});
  assert.equal(elements.get('activityLabel').textContent,'Fall detected');
  assert.match(elements.get('activityHint').textContent,/retained for review/);
  assert.ok(elements.get('cameraView').classList.contains('fall-alert-active'));
});
