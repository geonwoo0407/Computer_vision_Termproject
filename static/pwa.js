let installPrompt = null;
const installButton = document.getElementById('install-app');
if ('serviceWorker' in navigator && (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')) {
  window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
}
window.addEventListener('beforeinstallprompt', event => {
  event.preventDefault(); installPrompt = event;
  if (installButton) installButton.hidden = false;
});
window.addEventListener('appinstalled', () => {
  installPrompt = null; if (installButton) installButton.hidden = true;
});
if (installButton) installButton.addEventListener('click', async () => {
  if (!installPrompt) {
    alert('브라우저 메뉴에서 “홈 화면에 추가”를 선택하세요. 설치 메뉴가 없다면 HTTPS 주소로 접속해야 합니다.');
    return;
  }
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
});
