const $ = id => document.getElementById(id);
const role = document.body.dataset.role;
let state, selected = role === 'student' ? null : 1, pending = false, offset = 0, live;
const floor = $('seats');
const desks = new Map(Array.from(floor.querySelectorAll('[data-seat-id]'),
  button => [Number(button.dataset.seatId), button]));
// Touch and trackpad scrolling are native. Add grab-to-scroll for a mouse.
let mouseDrag = null;
floor.addEventListener('pointerdown', event => {
  if (event.pointerType === 'mouse' && event.button === 0 && event.target.closest('.seat-banks')) {
    mouseDrag = {startX: event.clientX, startY: event.clientY, scrollLeft: floor.scrollLeft, scrollTop: floor.scrollTop};
  }
});
floor.addEventListener('pointermove', event => {
  if (!mouseDrag || event.pointerType !== 'mouse') return;
  const dx = event.clientX - mouseDrag.startX;
  const dy = event.clientY - mouseDrag.startY;
  if (!floor.hasPointerCapture(event.pointerId)) {
    if (Math.hypot(dx, dy) < 6) return;
    floor.setPointerCapture(event.pointerId);
    floor.classList.add('dragging');
  }
  floor.scrollLeft = mouseDrag.scrollLeft - dx;
  floor.scrollTop = mouseDrag.scrollTop - dy;
});
function endMouseDrag() {
  mouseDrag = null;
  floor.classList.remove('dragging');
}
window.addEventListener('pointerup', endMouseDrag);
floor.addEventListener('pointercancel', endMouseDrag);
window.addEventListener('blur', endMouseDrag);
floor.addEventListener('click', event => {
  const desk = event.target.closest('[data-seat-id]');
  if (!desk || desk.disabled) return;
  selected = Number(desk.dataset.seatId);
  draw();
  if (role === 'student') { showStudentDialog('seat'); return; }
  $('detail').scrollIntoView({behavior:'smooth',block:'nearest'});
});

let mapScale = 1;
const minScale = 0.5, maxScale = 2;
function pointOnMap(x, y) {
  return {x: (floor.scrollLeft + x) / mapScale, y: (floor.scrollTop + y) / mapScale};
}
function zoomMap(scale, x = floor.clientWidth / 2, y = floor.clientHeight / 2, anchor = pointOnMap(x, y)) {
  mapScale = Math.max(minScale, Math.min(maxScale, scale));
  floor.style.setProperty('--map-scale', mapScale);
  floor.scrollLeft = anchor.x * mapScale - x;
  floor.scrollTop = anchor.y * mapScale - y;
  $('zoom-reset').textContent = Math.round(mapScale * 100) + '%';
  $('zoom-out').disabled = mapScale <= minScale;
  $('zoom-in').disabled = mapScale >= maxScale;
}
$('zoom-out').onclick = () => zoomMap(mapScale - 0.25);
$('zoom-in').onclick = () => zoomMap(mapScale + 0.25);
$('zoom-reset').onclick = () => zoomMap(1);
floor.addEventListener('wheel', event => {
  if (!event.deltaY || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
  event.preventDefault();
  const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? floor.clientHeight : 1;
  const factor = Math.exp(-event.deltaY * unit * 0.002);
  const rect = floor.getBoundingClientRect();
  zoomMap(mapScale * factor, event.clientX - rect.left - floor.clientLeft,
          event.clientY - rect.top - floor.clientTop);
}, {passive:false});

let pinch = null;
function touchGeometry(touches) {
  const [a, b] = touches;
  const rect = floor.getBoundingClientRect();
  return {x: (a.clientX + b.clientX) / 2 - rect.left - floor.clientLeft,
          y: (a.clientY + b.clientY) / 2 - rect.top - floor.clientTop,
          distance: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)};
}
floor.addEventListener('touchstart', event => {
  if (event.touches.length !== 2) { pinch = null; return; }
  const {x, y, distance} = touchGeometry(event.touches);
  if (distance > 0) {
    event.preventDefault();
    pinch = {distance, scale: mapScale, anchor: pointOnMap(x, y)};
  }
}, {passive:false});
floor.addEventListener('touchmove', event => {
  if (!pinch || event.touches.length !== 2) return;
  event.preventDefault();
  const {x, y, distance} = touchGeometry(event.touches);
  zoomMap(pinch.scale * distance / pinch.distance, x, y, pinch.anchor);
}, {passive:false});
floor.addEventListener('touchend', () => {pinch = null;});
floor.addEventListener('touchcancel', () => {pinch = null;});
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
function seatLabel(seat) {
  if (role === 'student') return seat.color === 'empty' ? '예약 가능' : '이용 중';
  return seat.status;
}
function drawMap() {
  for (const s of state.seats) {
    const desk = desks.get(s.id);
    if (!desk) continue;
    desk.disabled = false;
    desk.className = 'desk ' + (role === 'student' && s.color !== 'empty' ? 'reserved' : s.color);
    desk.setAttribute('aria-pressed', String(selected === s.id));
    desk.replaceChildren(element('strong', String(s.id).padStart(2,'0')),
      element('span', seatLabel(s)));
    if (role === 'admin' && (s.reserved || s.occupied)) {
      desk.append(element('small', s.owner_name || '예약자 미확인', 'owner-name'));
    }
    if (role !== 'student' && s.away) {
      const clock = element('small', '');
      clock.dataset.since = s.away_since;
      desk.append(clock);
    }
  }
}
function drawDetail() {
  const s = state.seats.find(seat => seat.id === selected);
  const panel = $('detail');
  panel.replaceChildren(element('h2', selected + '번 좌석 · ' + seatLabel(s)));
  if (role !== 'student') panel.append(element('p', '사람 ' + (s.occupied?'감지':'없음') + ' · 소지품 ' + (s.belongings?'있음':'없음')));
  const actions = element('div','','actions');
  panel.append(actions);
  const action = (name, extra={}) => send({action:name, seat_id:s.id, ...extra});
  function owner(name) {
    const id=prompt('예약할 학생의 로그인 아이디를 입력하세요.');
    if(id?.trim()) action(name,{login_id:id.trim()});
  }
  if(role==='admin') {
    if(s.reserved) {
      panel.append(element('small','예약자 · '+(s.owner_name || '미확인')));
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
  if(role!=='student' && s.warning) panel.append(element('p','미예약 이용 또는 소지품 확인 필요','warning'));
}
function draw() {
  if (role === 'student') { drawStudent(); return; }
  $('mode').textContent = state.mode==='demo'?'시연 모드':'카메라 API 모드';
  const available = state.seats.filter(s=>s.color==='empty').length;
  const counts = [['전체 좌석',state.seats.length],['예약 가능',available]];
  counts.push(['자리 비움',state.seats.filter(s=>s.away).length],['미예약 경고',state.seats.filter(s=>s.warning).length]);
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
  $('connection').dataset.online = String(ok);
  if (state) draw();
}
async function api(path, body) {
  const controller = new AbortController();
  const timeout = setTimeout(()=>controller.abort(), 8000);
  try {
    const response = await fetch(role === 'admin' ? path + '?view=admin' : path, {method:body?'POST':'GET', cache:'no-store',
      headers:body?{'Content-Type':'application/json', 'X-Seat-Viewer':state.viewer_id}:{}, body:body && JSON.stringify(body),
      signal:controller.signal});
    const data = await response.json();
    if (data.identity_changed) location.reload();
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
    localStorage.setItem(key, body.action);
    const {response,data}=await api('/api/action',{...body,request_id:token});
    localStorage.removeItem(key);
    if(response.ok) {
      receive(data);
    }
    if (role === 'student') studentActionResult(body.action, response.ok ? null : data.error);
    $('message').textContent=response.ok && role === 'student' ? '' : response.ok?'변경이 반영되었습니다.':data.error;
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
      const savedAction = localStorage.getItem(key);
      const result=await api('/api/result',{request_id:key.slice(key.lastIndexOf(':')+1)});
      if(version!==generation) return;
      if(!result.response.ok) throw Error();
      receive(result.data.state);
      $('message').textContent=result.data.error || '서버에서 변경 완료를 확인했습니다.';
      if (role === 'student') studentActionResult(savedAction, result.data.error);
      localStorage.removeItem(key);
    }
    if(!window.EventSource) {
      status(true,'서버 연결됨 · 주기적으로 갱신 중');
      retry=1000;
      timer=setTimeout(()=>sync(version),2000);
      return;
    }
    live=new EventSource(role === 'admin' ? '/api/live?view=admin' : '/api/live');
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
window.addEventListener('storage', event => {
  if (event.key === 'seat-account-change') location.reload();
});
if (role === 'student') $('logout').onclick = async () => {
  try {
    const {response, data} = await api('/api/student/logout', {});
    if (!response.ok) { $('message').textContent = data.error; return; }
    localStorage.setItem('seat-account-change', String(Date.now()));
    location.replace('/');
  } catch { $('message').textContent = '로그아웃하지 못했습니다. 연결을 확인하세요.'; }
};
window.addEventListener('offline',()=>restart());
window.addEventListener('pageshow',()=>{floor.scrollTo(0,0);restart();});
document.addEventListener('visibilitychange',()=>{if(!document.hidden)restart();});
if (role === 'student') initStudent();
else setInterval(tick,1000);
restart();
