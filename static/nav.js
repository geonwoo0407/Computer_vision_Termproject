window.addEventListener('storage', event => {
  if (event.key === 'seat-account-change') location.reload();
});
window.addEventListener('pageshow', event => {
  if (event.persisted) location.reload();
});
const logoutButton = document.getElementById('logout');
if (logoutButton) logoutButton.addEventListener('click', async () => {
  logoutButton.disabled = true;
  try {
    const response = await fetch('/api/student/logout', {
      method:'POST', headers:{'Content-Type':'application/json'}, body:'{}'
    });
    if (!response.ok) throw Error();
    localStorage.setItem('seat-account-change', String(Date.now()));
    location.replace('/');
  } catch {
    logoutButton.disabled = false;
    alert('로그아웃하지 못했습니다. 연결을 확인해 주세요.');
  }
});
