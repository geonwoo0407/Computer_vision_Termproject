const form = document.getElementById('student-account');
form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = form.querySelector('button');
  if (button.disabled) return;
  button.disabled = true;
  const error = document.getElementById('account-error');
  error.hidden = true;
  try {
    const response = await fetch(form.action, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(Object.fromEntries(new FormData(form)))
    });
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.error || '요청을 처리할 수 없습니다.');
    }
    localStorage.setItem('seat-account-change', String(Date.now()));
    location.replace('/');
  } catch (reason) {
    error.textContent = reason instanceof TypeError ? '서버에 연결할 수 없습니다.' : reason.message;
    error.hidden = false;
    button.disabled = false;
  }
});
