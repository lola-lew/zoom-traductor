/**
 * Offscreen document: corre el MediaRecorder y el cliente SocketIO.
 * Vive mientras la captura está activa; el background lo destruye al detener.
 *
 * Protocolo: SocketIO v4 / Engine.IO v4 implementado manualmente sobre WebSocket
 * (sin dependencias externas). Envía chunks de audio como binary events.
 */

let _recorder   = null;
let _stream     = null;
let _ws         = null;
let _wsReady    = false;
let _active     = false;
let _serverUrl  = null;

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
  _active    = true;
  _serverUrl = serverUrl;

  connectWS(serverUrl);

  // getUserMedia con el stream ID obtenido por tabCapture en el background
  _stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      mandatory: {
        chromeMediaSource:   'tab',
        chromeMediaSourceId: streamId,
      },
    },
    video: false,
  });

  const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
    ? 'audio/webm;codecs=opus' : '';
  _recorder = new MediaRecorder(_stream, mimeType ? { mimeType } : {});

  _recorder.ondataavailable = async (e) => {
    if (e.data.size === 0) return;
    const buf = await e.data.arrayBuffer();
    sendAudioChunk(buf);
  };

  _recorder.onerror = (e) => console.error('[offscreen] recorder error:', e.error?.message);
  _recorder.start(3000); // chunk cada 3 s
  console.log('[offscreen] MediaRecorder started — mimeType:', _recorder.mimeType);
}

function stopCapture() {
  _active = false;
  if (_recorder && _recorder.state !== 'inactive') _recorder.stop();
  if (_stream) _stream.getTracks().forEach(t => t.stop());
  if (_ws)     { _ws.close(); _ws = null; }
  _recorder = null;
  _stream   = null;
  _wsReady  = false;
}

// ── Mini cliente SocketIO v4 (Engine.IO v4) ───────────────────────────────────
//
// SocketIO v4 sobre WebSocket — protocolo en texto + frames binarios:
//   EIO tipos: 0=open  2=ping  3=pong  4=message  (todo texto excepto attachment)
//   SIO tipos (dentro de EIO message "4"): 0=connect  2=event  5=binary_event
//
// Para emitir un binary event con 1 attachment:
//   → Frame texto: "451-["coordinator_audio",{"_placeholder":true,"num":0}]"
//   → Frame binario: <ArrayBuffer>
//
// El servidor Flask-SocketIO (python-socketio ≥5 / EIO=4) entiende este formato.

function connectWS(serverUrl) {
  if (!_active) return;

  const wsUrl = serverUrl.replace(/^https?/, 'wss').replace(/\/+$/, '')
    + '/socket.io/?EIO=4&transport=websocket';

  console.log('[offscreen] conectando a', wsUrl);
  _ws = new WebSocket(wsUrl);
  _ws.binaryType = 'arraybuffer';

  _ws.onopen = () => console.log('[offscreen] WS abierto');

  _ws.onmessage = (e) => {
    if (typeof e.data !== 'string') return;
    const pkt = e.data;

    if (pkt.startsWith('0')) {
      // EIO open → responder con SIO connect
      _ws.send('40');
    } else if (pkt === '2') {
      // EIO ping → pong
      _ws.send('3');
    } else if (pkt.startsWith('40')) {
      // SIO connected al namespace /
      _wsReady = true;
      console.log('[offscreen] SocketIO conectado');
    } else if (pkt.startsWith('41')) {
      _wsReady = false;
    }
  };

  _ws.onclose = () => {
    _wsReady = false;
    if (_active) {
      console.log('[offscreen] WS cerrado — reconectando en 3 s');
      setTimeout(() => connectWS(_serverUrl), 3000);
    }
  };

  _ws.onerror = () => console.warn('[offscreen] WS error');
}

function sendAudioChunk(buffer) {
  if (!_wsReady || !_ws || _ws.readyState !== WebSocket.OPEN) {
    console.warn('[offscreen] WS no listo — chunk descartado (%d bytes)', buffer.byteLength);
    return;
  }
  // Binary event con 1 attachment
  _ws.send('451-["coordinator_audio",{"_placeholder":true,"num":0}]');
  _ws.send(buffer);
}
