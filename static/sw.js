const VERSION = 'seat-shell-v7';
const SHELL = ['/', '/static/style.css', '/static/app.js', '/static/student.js', '/static/student.css', '/static/icons/seoultech-emblem.gif', '/static/pwa.js', '/static/icons/seat-192.png', '/static/icons/seat-512.png', '/static/icons/seat-maskable-512.png'];
self.addEventListener('install', event => {
  event.waitUntil(caches.open(VERSION).then(cache => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', event => {
  event.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(key => key.startsWith('seat-shell-') && key !== VERSION).map(key => caches.delete(key)))).then(() => self.clients.claim()));
});
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || url.origin !== self.location.origin || url.pathname.startsWith('/api/')) return;
  if (url.pathname === '/' || url.pathname.startsWith('/static/')) {
    event.respondWith(fetch(event.request).then(response => {
      if (response.ok) { const copy = response.clone(); event.waitUntil(caches.open(VERSION).then(cache => cache.put(event.request, copy))); }
      return response;
    }).catch(() => caches.match(event.request)));
  }
});
