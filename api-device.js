// Per-browser device id: a random id kept in localStorage, not fingerprinting and not a security
// boundary. Used for the Pro/Max device limit (/session/heartbeat) and, because many students
// share one carrier IP, for per-device rate limits and the guest daily budget (server.py's
// rate_limit and guest_device_usage_log) -- the server only uses it to split an IP's allowance
// between the students behind it.
//
// Adds X-Device-Id to every fetch this page makes to the NEET-AI API, and only there -- never to
// Supabase or any other origin. Load it before any script that calls the API or getDeviceId().
(function () {
  const API_ORIGIN = 'https://neet-ai-production.up.railway.app';
  let cached = null;

  // Same shape the app and older pages generate; cached so a browser that blocks localStorage
  // still keeps one id for the whole page load.
  function getDeviceId() {
    if (cached) return cached;
    try { cached = localStorage.getItem('deviceId'); } catch (e) { /* storage blocked */ }
    if (!cached) {
      cached = crypto.randomUUID ? crypto.randomUUID() : 'dev-' + Date.now() + '-' + Math.random().toString(36).slice(2);
      try { localStorage.setItem('deviceId', cached); } catch (e) { /* storage blocked */ }
    }
    return cached;
  }
  window.getDeviceId = getDeviceId;

  const nativeFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input instanceof Request ? input.url : String(input));
    if (!url.startsWith(API_ORIGIN + '/')) return nativeFetch(input, init);
    const headers = new Headers((init && init.headers) || (input instanceof Request ? input.headers : undefined));
    if (!headers.has('X-Device-Id')) headers.set('X-Device-Id', getDeviceId());
    return nativeFetch(input, { ...init, headers });
  };
})();
