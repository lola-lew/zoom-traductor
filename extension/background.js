let _capturing  = false;
let _serverUrl  = null;
let _lastError  = null;  // error pendiente de mostrar en el popup

// ── Mensajes desde popup y offscreen ─────────────────────────────────────────

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === 'status') {
    const err = _lastError;
    _lastError = null;           // consumir el error al leerlo
    sendResponse({ capturing: _capturing, error: err });
    return;
  }

  // Notificación de error desde offscreen (stream interrumpido, recorder error)
  if (msg.action === 'offscreen_error') {
    console.error('[bg] error desde offscreen:', msg.error);
    _lastError  = msg.error || 'Error de captura';
    _capturing  = false;
    _serverUrl  = null;
    closeOffscreen().catch(() => {});
    return;
  }

  if (msg.action === 'start') {
    startCapture(msg.tabId, msg.serverUrl)
      .then(() => sendResponse({ ok: true }))
      .catch(e => { console.error('[bg] start error:', e); sendResponse({ ok: false, error: e.message }); });
    return true; // async
  }

  if (msg.action === 'stop') {
    stopCapture()
      .then(() => sendResponse({ ok: true }))
      .catch(e => sendResponse({ ok: false, error: e.message }));
    return true;
  }
});

// ── Captura ───────────────────────────────────────────────────────────────────

async function startCapture(tabId, serverUrl) {
  if (_capturing) await stopCapture();

  // Obtener stream ID de la pestaña (funciona desde service worker)
  const streamId = await new Promise((resolve, reject) => {
    chrome.tabCapture.getMediaStreamId({ targetTabId: tabId }, (id) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(id);
      }
    });
  });

  // Crear offscreen document (donde corre MediaRecorder + WebSocket)
  await ensureOffscreen();

  // Enviar streamId al offscreen (con reintentos por si aún no está listo)
  await sendToOffscreen({ action: 'offscreen_start', streamId, serverUrl });

  _capturing = true;
  _serverUrl = serverUrl;
}

async function stopCapture() {
  if (!_capturing) return;
  _capturing = false;
  _serverUrl = null;

  try { await sendToOffscreen({ action: 'offscreen_stop' }); } catch (_) {}
  await closeOffscreen();
}

// ── Offscreen ─────────────────────────────────────────────────────────────────

async function ensureOffscreen() {
  const contexts = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
  if (contexts.length > 0) return;

  await chrome.offscreen.createDocument({
    url: 'offscreen.html',
    reasons: ['USER_MEDIA'],
    justification: 'Captura de audio de pestaña para traducción en tiempo real',
  });
}

async function closeOffscreen() {
  const contexts = await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'] });
  if (contexts.length > 0) await chrome.offscreen.closeDocument();
}

// Envía mensaje al offscreen con reintentos (puede no estar listo de inmediato)
async function sendToOffscreen(msg, maxRetries = 8, delayMs = 150) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await chrome.runtime.sendMessage(msg);
    } catch (e) {
      if (i < maxRetries - 1) {
        await new Promise(r => setTimeout(r, delayMs));
      } else {
        throw e;
      }
    }
  }
}
