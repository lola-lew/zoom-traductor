/**
 * Offscreen document: MediaRecorder + WebSocket puro hacia /coordinator_ws.
 * Mantiene la conexión viva con pings cada 20 s y reconecta con backoff de 3 s.
 * Prepend del init segment en cada chunk para que Whisper reciba webm válido.
 */

let _recorder    = null;
let _stream      = null;
let _ws          = null;
let _wsUrl       = null;
let _active      = false;
let _initSegment = null;   // primer chunk = cabecera EBML — se prepend a todos los demás
let _pingTimer   = null;

// ── Mensajes desde background ─────────────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === 'offscreen_start') {
    startCapture(msg.streamId, msg.serverUrl)
      .then(() => sendResponse({ ok: true }))
      .catch(e => { console.error('[offscreen] start error:', e); sendResponse({ ok: false, error: e.message }); });
    return true;
  }

  if (msg.action === 'offscreen_stop') {
    stopCapture();
    sendResponse({ ok: true });
  }
});

// ── Captura ───────────────────────────────────────────────────────────────────

async function startCapture(streamId, serverUrl) {
  _active  = true;
  _wsUrl   = serverUrl.replace(/^https?/, 'wss').replace(/\/+$/, '') + '/coordinator_ws';

  connectWS();

  _stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource:   'tab',
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  // Notificar al background si el stream se interrumpe (pestaña cerrada, etc.)
  _stream.getAudioTracks()[0]?.addEventListener('ended', () => {
    console.warn('[offscreen] audio track ended — notificando al background');
    chrome.runtime.sendMessage({ action: 'offscreen_error', error: 'Audio track interrumpido' });
    stopCapture();
  });

  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus' : '';
  _recorder = new MediaRecorder(_stream, mimeType ? { mimeType } : {});

  let isFirstChunk = true;
  _recorder.ondataavailable = async (e) => {
    if (e.data.size === 0) return;
    const buf = await e.data.arrayBuffer();

    if (isFirstChunk) {
      // Guardar init segment (cabecera EBML) — se prepend a cada chunk siguiente
      _initSegment = buf;
      isFirstChunk = false;
      console.log('[offscreen] init segment guardado — %d bytes', buf.byteLength);
      // No enviar el init segment solo; esperar al primer chunk real
      return;
    }

    sendChunk(buf);
  };

  _recorder.onerror = (e) => {
    console.error('[offscreen] recorder error:', e.error?.message);
    chrome.runtime.sendMessage({ action: 'offscreen_error', error: e.error?.message || 'MediaRecorder error' });
  };

  _recorder.start(5000); // chunk cada 5 s
  console.log('[offscreen] MediaRecorder started — mimeType:', _recorder.mimeType);
}

function stopCapture() {
  _active = false;
  clearInterval(_pingTimer);
  _pingTimer = null;

  if (_recorder && _recorder.state !== 'inactive') _recorder.stop();
  if (_stream) _stream.getTracks().forEach(t => t.stop());
  if (_ws)     { _ws.close(); _ws = null; }

  _recorder    = null;
  _stream      = null;
  _initSegment = null;
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

function connectWS() {
  if (!_active) return;

  console.log('[offscreen] conectando a', _wsUrl);
  _ws = new WebSocket(_wsUrl);
  _ws.binaryType = 'arraybuffer';

  _ws.onopen = () => {
    console.log('[offscreen] WS conectado');
    // Ping cada 20 s para mantener la conexión viva en Railway
    clearInterval(_pingTimer);
    _pingTimer = setInterval(() => {
      if (_ws && _ws.readyState === WebSocket.OPEN) {
        _ws.send('ping');
      }
    }, 20_000);
  };

  _ws.onclose = () => {
    clearInterval(_pingTimer);
    _pingTimer = null;
    if (_active) {
      console.log('[offscreen] WS cerrado — reconectando en 3 s');
      setTimeout(connectWS, 3000);
    }
  };

  _ws.onerror = () => console.warn('[offscreen] WS error');
}

// ── Envío de chunks ───────────────────────────────────────────────────────────

function sendChunk(buf) {
  if (!_ws || _ws.readyState !== WebSocket.OPEN) {
    console.warn('[offscreen] WS no listo — chunk descartado (%d bytes)', buf.byteLength);
    return;
  }

  // Prepend init segment para que cada chunk sea un webm válido standalone
  let payload = buf;
  if (_initSegment) {
    const combined = new Uint8Array(_initSegment.byteLength + buf.byteLength);
    combined.set(new Uint8Array(_initSegment), 0);
    combined.set(new Uint8Array(buf), _initSegment.byteLength);
    payload = combined.buffer;
  }

  _ws.send(payload);
}
