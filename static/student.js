// Navigation is local so closing a seat dialog preserves the map's scroll and zoom.
const roomName = '2F 일반열람실2';
const studentPages = {
  home: ['열람실 좌석배정', '⌂', '홈으로', 'home'],
  rooms: ['좌석예약', '←', '뒤로', 'home'],
  map: [roomName, '←', '열람실 선택으로', 'rooms'],
  ticket: ['좌석배정확정', '×', '닫기', 'home']
};
let studentPage, dialogKind, displayedTicket, requestedSeat, dialogError = '';
const ticketDate = new Intl.DateTimeFormat('sv-SE', {
  timeZone:'Asia/Seoul', year:'numeric', month:'2-digit', day:'2-digit',
  hour:'2-digit', minute:'2-digit', second:'2-digit', hourCycle:'h23'
});

function studentRoute() {
  const key = location.hash.slice(1);
  const seatLink = /^seat-(\d+)$/.exec(key);
  const page = seatLink ? 'map' : Object.hasOwn(studentPages, key) ? key : 'home';
  const [title, icon, label, back] = studentPages[page];
  closeStudentDialog();
  for (const name of Object.keys(studentPages)) $(name).hidden = name !== page;
  $('back').href = '#' + back;
  $('back').textContent = icon;
  $('back').setAttribute('aria-label', label);
  $('page-title').textContent = title;
  if (studentPage !== page) {
    window.scrollTo(0, 0);
    if (studentPage) $('page-title').focus({preventScroll:true});
    if (page === 'map') floor.scrollTo(0, 0);
  }
  studentPage = page;
  $('message').textContent = '';
  requestedSeat = seatLink ? Number(seatLink[1]) : null;
  if (seatLink) history.replaceState(null, '', '#map');
  if (state) draw();
}
function closeStudentDialog() {
  dialogKind = null;
  displayedTicket = null;
  dialogError = '';
  if ($('seat-dialog').open) $('seat-dialog').close();
}
function showStudentDialog(kind) {
  dialogKind = kind;
  dialogError = '';
  if (kind === 'return' || kind === 'cancel') displayedTicket = state.my_reservation;
  updateStudentDialog();
  if (!$('seat-dialog').open) $('seat-dialog').showModal();
}
function updateStudentDialog() {
  if (!dialogKind) return;
  const go = $('dialog-go'), back = $('dialog-back');
  const seat = state.seats.find(s => s.id === selected);
  const mine = state.my_reservation;
  let title, prompt, detail = '', statusText = '', label, confirmAction, needsServer = false;
  let dismiss = '취소';
  if (dialogKind === 'seat') {
    title = '배정좌석확인';
    detail = roomName + ' / ' + seat.id + '번 좌석';
    statusText = seatLabel(seat);
    if (seat.mine) {
      prompt = '내가 배정받은 좌석입니다.';
      label = '좌석배정표'; confirmAction = () => { closeStudentDialog(); location.hash = 'ticket'; };
    } else if (seat.color !== 'empty') {
      prompt = '현재 이용 중인 좌석입니다.'; label = '확인'; dismiss = ''; confirmAction = closeStudentDialog;
    } else if (!state.viewer_name) {
      prompt = '좌석을 배정받으려면 학생 계정으로 로그인하세요.';
      label = '로그인'; confirmAction = () => location.assign('/login#seat-' + seat.id);
    } else if (mine) {
      prompt = mine.seat_id + '번에서 아래 좌석으로 이동하시겠습니까?';
      label = '자리 이동'; needsServer = true;
      confirmAction = () => send({action:'move', seat_id:mine.seat_id, reservation_id:mine.id, target_id:seat.id});
    } else {
      prompt = '아래의 좌석으로 배정하시겠습니까?'; label = '배정'; needsServer = true;
      confirmAction = () => send({action:'reserve', seat_id:seat.id});
    }
  } else if (dialogKind === 'returned') {
    title = '반납처리'; prompt = '정상적으로 반납되었습니다.'; dismiss = ''; label = 'OK';
    confirmAction = () => { closeStudentDialog(); draw(); };
  } else {
    const returning = dialogKind === 'return';
    title = returning ? '반납처리' : '예약 취소';
    prompt = returning ? '이용 중인 좌석을 반납하시겠습니까?' : '이 좌석의 예약을 취소하시겠습니까?';
    dismiss = '아니오'; label = '예'; needsServer = true;
    const ticket = displayedTicket;
    if (!mine || mine.id !== ticket.id || mine.seat_id !== ticket.seat_id) {
      if (!pending) statusText = '배정 정보가 변경되었습니다.';
      needsServer = false; confirmAction = null;
    } else {
      confirmAction = () => send({action:returning ? 'return' : 'cancel', seat_id:ticket.seat_id, reservation_id:ticket.id});
    }
  }
  $('dialog-title').textContent = title;
  $('dialog-prompt').textContent = prompt;
  $('dialog-seat').textContent = detail;
  $('dialog-status').textContent = dialogError || (pending ? '요청 처리 중…' : needsServer && !connected ? '서버에 연결 중입니다.' : statusText);
  back.textContent = dismiss; back.hidden = !dismiss; back.disabled = pending;
  go.textContent = label; go.disabled = pending || !confirmAction || (needsServer && !connected);
  go.onclick = confirmAction;
}
function drawStudent() {
  const signedIn = Boolean(state.viewer_name || state.is_staff);
  $('identity').textContent = state.viewer_name ? state.viewer_name + '님' : '';
  $('identity').hidden = !state.viewer_name;
  $('staff-home').hidden = !state.is_staff;
  $('nav-login').hidden = signedIn;
  $('logout').hidden = !signedIn;
  if (studentPage === 'rooms') {
    const count = state.seats.filter(s => s.color === 'empty').length;
    $('available-count').textContent = count;
    $('availability-ring').style.setProperty('--available', count / state.seats.length * 100 + '%');
    const favorites = state.favorites.map(id => {
      const link = element('a', id + '번 좌석'); link.href = '#seat-' + id; return link;
    });
    $('favorite-list').replaceChildren(...(favorites.length ? favorites : ['등록된 선호좌석이 없습니다.']));
  }
  if (studentPage === 'map') {
    drawMap();
    if (requestedSeat !== null) {
      const desk = desks.get(requestedSeat);
      if (desk) {
        selected = requestedSeat;
        desk.scrollIntoView({block:'nearest', inline:'nearest'});
        desk.focus({preventScroll:true});
        showStudentDialog('seat');
      }
      requestedSeat = null;
    }
  }
  if (studentPage === 'ticket') {
    const ticket = displayedTicket || state.my_reservation;
    $('ticket-info').hidden = !ticket;
    $('ticket-empty').hidden = Boolean(ticket);
    if (ticket) {
      $('ticket-seat').textContent = roomName + ' / ' + ticket.seat_id + '번';
      $('ticket-time').textContent = ticket.created == null ? '기록 없음' : ticketDate.format(ticket.created * 1000);
      const seat = state.seats.find(s => s.id === ticket.seat_id);
      $('cancel-reservation').hidden = seat.used;
      $('favorite-seat').textContent = state.favorites.includes(ticket.seat_id) ? '★ 선호좌석 해제' : '☆ 선호좌석';
      for (const id of ['return-seat', 'favorite-seat', 'cancel-reservation']) $(id).disabled = pending || !connected;
    } else if (state.viewer_name) {
      $('ticket-empty').textContent = '좌석배정 정보가 없습니다.';
    } else {
      const link = element('a', '로그인'); link.href = '/login#ticket';
      $('ticket-empty').replaceChildren('내 배정표를 확인하려면 로그인하세요.', link);
    }
  }
  updateStudentDialog();
}
function studentActionResult(action, error) {
  if (error) {
    dialogError = error;
    return;
  }
  if (['reserve','move','return','cancel'].includes(action)) {
    if (studentPage !== 'ticket') {
      history.pushState(null, '', '#ticket');
      studentRoute();
    }
    if (action === 'return') showStudentDialog('returned');
    else closeStudentDialog();
  }
}
function initStudent() {
  $('dialog-back').onclick = () => { closeStudentDialog(); draw(); };
  $('seat-dialog').addEventListener('cancel', event => {
    event.preventDefault();
    if (!pending) { closeStudentDialog(); draw(); }
  });
  $('return-seat').onclick = () => showStudentDialog('return');
  $('cancel-reservation').onclick = () => showStudentDialog('cancel');
  $('favorite-seat').onclick = () => {
    const id = state.my_reservation.seat_id;
    send({action:state.favorites.includes(id) ? 'unfavorite' : 'favorite', seat_id:id});
  };
  window.addEventListener('hashchange', studentRoute);
  studentRoute();
}
