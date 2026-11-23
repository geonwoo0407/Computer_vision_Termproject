const $ = id => document.getElementById(id);
const role = document.body.dataset.role;
let state, selected = 1, moving = null, pending = false, offset = 0, live;
const desks = new Map();
function element(tag, text, cls) {
  const el = document.createElement(tag);
  el.textContent = text;
  if (cls) el.className = cls;
  return el;
}
function control(container, label, click, disabled = false) {
  const b = element('button', label);
  b.disabled = disabled || pending || !connected;
  b.onclick = click;
  container.append(b);
  return b;
}
function elapsed(t) {
  const sec = Math.max(0, Math.floor((Date.now() + offset) / 1000 - t));
  return [Math.floor(sec / 3600), Math.floor(sec / 60) % 60, sec % 60]
    .map(n => String(n).padStart(2, '0')).join(':');
}
function tick() {
  for (const node of document.querySelectorAll('[data-since]'))
    node.textContent = elapsed(Number(node.dataset.since));
}
function drawMap() {
  for (const s of state.seats) {
    let desk = desks.get(s.id);
    if (!desk) {
      desk = element('button', '');
      desk.onclick = () => {
        selected = s.id;
        draw();
        $('detail').scrollIntoView({behavior:'smooth',block:'nearest'});
      };
      $(s.id <= 6 ? 'west' : 'east').append(desk);
      desks.set(s.id, desk);
    }
    desk.className = 'desk ' + s.color;
    desk.setAttribute('aria-pressed', String(selected === s.id));
    desk.replaceChildren(element('strong', String(s.id).padStart(2,'0')),
      element('span', (s.mine ? '내 자리 · ' : '') + s.status));
    if (s.away) {
      const clock = element('small', '');
      clock.dataset.since = s.away_since;
      desk.append(clock);
    }
  }
}
function drawDetail() {
  const s = state.seats.find(seat => seat.id === selected);
  const panel = $('detail');
  panel.replaceChildren(element('h2', selected + '번 좌석 · ' + s.status),
    element('p', '사람 ' + (s.occupied?'감지':'없음') + ' · 소지품 ' + (s.belongings?'있음':'없음')));
  const actions = element('div','','actions');
  panel.append(actions);
  const action = (name, extra={}) => send({action:name, seat_id:s.id, ...extra});
  if (role === 'student') {
    if (moving !== null) {
      panel.prepend(element('p', moving + '번에서 이동할 빈 좌석을 배치도에서 선택하세요.'));
      if (s.color === 'empty')
        control(actions, selected+'번으로 이동', () => action('move',{seat_id:moving,target_id:s.id}));
      control(actions, '이동 취소', () => { selected=moving; moving=null; draw(); });
    } else if (s.mine) {
      if (!s.used) control(actions,'예약 취소',()=>action('cancel'));
      control(actions,'반납하기',()=>{if(confirm(s.id+'번 좌석을 반납할까요?')) action('return');});
      control(actions,'자리 이동',()=>{moving=s.id; draw(); $('seats').scrollIntoView({block:'center'});});
    } else if (s.color === 'empty') {
      const mine = state.seats.find(item=>item.mine);
      if (mine) control(actions,'이 자리로 이동',()=>action('move',{seat_id:mine.id,target_id:s.id}));
      else control(actions,'예약하기',()=>action('reserve'));
    }
  }
  function owner(name) {
    const id=prompt('학생 화면에 표시된 이용자 ID를 입력하세요.',state.viewer_id);
    if(id?.trim()) action(name,{owner_id:id.trim()});
  }
  if(role==='admin') {
    if(s.reserved) {
      panel.append(element('small','예약자 · '+(s.owner_id || '미지정')));
      control(actions,'예약자 지정',()=>owner('assign_owner'));
      control(actions,'예약 권한 해제',()=>action('revoke'));
    } else control(actions,'예약 권한 부여',()=>owner('grant'));
  }
  if(role==='demo') {
    for(const [label,occupied,belongings] of [
      ['착석 + 짐',true,true],['착석 · 짐 없음',true,false],
      ['사람 없음 + 짐',false,true],['사람·짐 없음',false,false]])
      control(actions,label,()=>action('detect',{source:'demo',occupied,belongings}),state.mode!=='demo');
    control(actions,s.reserved?'예약 해제':'예약 부여',()=>action(s.reserved?'revoke':'grant'));
  }
  if(role!=='student' && s.updated) panel.append(element('p','마지막 감지 · '+new Date(s.updated*1000).toLocaleTimeString()));
  if(s.warning) panel.append(element('p','미예약 이용 또는 소지품 확인 필요','warning'));
}
function draw() {
  if(moving!==null && !state.seats.some(s=>s.id===moving && s.mine)) moving=null;
  $('identity').textContent = '내 이용자 ID · '+state.viewer_id;
  $('mode').textContent = state.mode==='demo'?'시연 모드':'카메라 API 모드';
  const counts = [['전체 좌석',state.seats.length],['예약 가능',state.seats.filter(s=>s.color==='empty').length],
    ['자리 비움',state.seats.filter(s=>s.away).length],['미예약 경고',state.seats.filter(s=>s.warning).length]];
  $('summary').replaceChildren(...counts.map(([label,n])=>{
    const box=element('div',''); box.append(element('span',label),element('strong',n+'석')); return box;
  }));
  drawMap(); drawDetail(); tick();
  if(role==='demo') {
    const box=$('controls');box.hidden=false;box.replaceChildren(element('h2','시연 설정'));
    for(const [mode,label] of [['demo','시연 모드'],['camera','카메라 API 모드']])
      control(box,label,()=>send({action:'mode',mode}),state.mode===mode);
    control(box,'전체 초기화',()=>{if(confirm('전체 좌석을 초기화할까요?'))send({action:'reset'});},state.mode!=='demo');
  }
}
function receive(data) {
  if(state && data.viewer_id === state.viewer_id && data.revision < state.revision) return;
  state=data;offset=data.server_time*1000-Date.now();draw();
}
let connected = false, retry = 1000, timer, generation = 0;
const prefix = 'seat-request:';
function status(ok, message) {
  connected = ok;
  $('connection').textContent = message;
  if (state) draw();
}
async function api(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(()=>controller.abort(), 8000);
  try {
    const response = await fetch(path, {method:body?'POST':'GET', cache:'no-store',
      headers:body?{'Content-Type':'application/json'}:{}, body:body && JSON.stringify(body),
      signal:controller.signal});
    const data = await response.json();
    if(response.status >= 500) throw Error();
    return {response, data};
  } finally {clearTimeout(timeout);}
}
async function send(body) {
  if(pending || !connected) return;
  pending=true;
  draw();
  const token=Array.from(crypto.getRandomValues(new Uint8Array(16)), n=>n.toString(16).padStart(2,'0')).join('');
  const key=prefix+state.viewer_id+':'+token;
  $('message').textContent='요청 처리 중…';
  try {
    localStorage.setItem(key, token);
    const {response,data}=await api('/api/action',{...body,request_id:token});
    localStorage.removeItem(key);
    if(response.ok) {
      if(body.action==='move'){selected=body.target_id;moving=null;}
      receive(data);
    }
    $('message').textContent=response.ok?'변경이 반영되었습니다.':data.error;
  } catch {
    $('message').textContent='요청 결과 확인 중 · 연결 복구 후 서버에서 확인합니다.';
    restart();
  } finally {pending=false;if(state) draw();}
}
function restart(delay=0) {
  const version=++generation;
  clearTimeout(timer);
  if(live) live.close();
  status(false,'연결 확인 중 · 표시된 좌석은 최신 상태가 아닐 수 있습니다.');
  timer=setTimeout(()=>sync(version),delay);
}
async function sync(version) {
  try {
    const {response,data}=await api('/api/seats');
    if(version!==generation) return;
    if(!response.ok) throw Error();
    receive(data);
    for(const key of Object.keys(localStorage).filter(k=>k.startsWith(prefix+data.viewer_id+':'))) {
      if(pending) break;
      const result=await api('/api/result',{request_id:localStorage.getItem(key)});
      if(version!==generation) return;
      if(!result.response.ok) throw Error();
      receive(result.data.state);
      $('message').textContent=result.data.error || '서버에서 변경 완료를 확인했습니다.';
      localStorage.removeItem(key);
    }
    if(!window.EventSource) {
      status(true,'서버 연결됨 · 주기적으로 갱신 중');
      retry=1000;
      timer=setTimeout(()=>sync(version),2000);
      return;
    }
    live=new EventSource('/api/live');
    const watchdog=()=>{
      clearTimeout(timer);
      timer=setTimeout(fail,15000);
    };
    live.onmessage=e=>{
      if(version!==generation) return;
      receive(JSON.parse(e.data));
      status(true,'실시간 연결됨');
      retry=1000;
      watchdog();
    };
    live.addEventListener('heartbeat',watchdog);
    live.onerror=fail;
    watchdog();
  } catch {fail();}
  function fail() {
    if(version!==generation) return;
    restart(retry);
    retry=Math.min(30000,retry*2);
  }
}
window.addEventListener('online',()=>restart());
window.addEventListener('offline',()=>restart());
window.addEventListener('pageshow',()=>restart());
document.addEventListener('visibilitychange',()=>{if(!document.hidden)restart();});
setInterval(tick,1000);
restart();
