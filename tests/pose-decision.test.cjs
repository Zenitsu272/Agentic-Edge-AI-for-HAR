const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');

function dashboard(){
  const elements=new Map();
  const node=(tag,cls,text='')=>({textContent:text,children:[],dataset:{},append(...children){this.children.push(...children);},replaceChildren(){this.children=[];},removeAttribute(){}});
  const element=id=>{if(!elements.has(id))elements.set(id,node());return elements.get(id);};
  const sandbox={document:{getElementById:element},text:(id,value)=>element(id).textContent=value,node,setTimeout:()=>1,clearTimeout(){},getJSON(){}};
  vm.createContext(sandbox);
  const source=fs.readFileSync('dist/app.js','utf8');
  vm.runInContext(source.slice(source.indexOf('const verificationLabels='),source.indexOf('window.beginVideoCheck=')),sandbox);
  return {sandbox,elements};
}

for(const status of ['NOT_FALL','CONFIRMED_FALL','INCONCLUSIVE','VERIFYING','UNAVAILABLE']){
  test(`pose fall stays the final dashboard decision for VLM ${status}`,()=>{
    const {sandbox,elements}=dashboard();
    sandbox.record={event_id:'pose-fall',status,scene_observation:'Independent scene note.'};
    vm.runInContext('updateVerification(record)',sandbox);
    assert.equal(elements.get('verificationVerdict').textContent,'Fall detected · pose model');
    assert.equal(elements.get('cameraVerification').textContent,'Fall detected · pose model');
    assert.doesNotMatch(elements.get('verificationBadge').textContent,/False alarm/);
    assert.match(elements.get('verificationReason').textContent,/Independent scene note/);
  });
}

test('completed video counts rejected and unavailable VLM reviews as pose-detected falls',()=>{
  const {sandbox,elements}=dashboard();
  vm.runInContext(`videoCheck={name:'test.mp4',phase:'complete',percent:100,frames:100,total:100,time:4,duration:4,
    incidents:new Map([['a',{event_id:'a',status:'NOT_FALL',trigger_time_sec:1}],['b',{event_id:'b',status:'UNAVAILABLE',trigger_time_sec:2}]])};renderVideoCheck();`,sandbox);
  assert.equal(elements.get('videoResultCount').textContent,'2 falls detected');
  assert.match(elements.get('videoResultSummary').textContent,/2 falls detected by the pose model/);
  for(const row of elements.get('videoIncidentList').children)assert.match(row.children[0].textContent,/Fall detected · pose model/);
});
