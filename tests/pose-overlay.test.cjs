const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function client(width, height) {
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
  const sandbox = {document: {body: element('body'), getElementById: element, createElement: () => element('capture')}, window: {devicePixelRatio: 2}, ResizeObserver: class {observe(){}}, clearTimeout, URL};
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync('dist/edge-vision.js','utf8') + '\nthis.client = new EdgeVisionClient();', sandbox);
  const c = sandbox.client;
  c.running = true; c.frameWidth = 640; c.frameHeight = 480;
  c.poses = [{keypoints: [[100,100,.9]]}];
  return {c, dots, elements};
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
  assert.ok(!elements.get('body').classList.contains('fall-alert-active'));
  assert.ok(!elements.get('cameraView').classList.contains('fall-alert-active'));
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
