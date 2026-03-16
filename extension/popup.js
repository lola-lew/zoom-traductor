let capturing = false;

document.getElementById('btn').addEventListener('click', toggle);

// Restaurar URL guardada
chrome.storage.local.get('serverUrl', ({ serverUrl }) => {
  if (serverUrl) document.getElementById('serverUrl').value = serverUrl;
});

// Sincronizar estado con el background al abrir el popup
chrome.runtime.sendMessage({ action: 'status' }, (resp) => {
  if (chrome.runtime.lastError) return;
  if (resp?.capturing) setCapturing(true, 'Capturando...');
  if (resp?.error)     setStatus('Error: ' + resp.error, 'err');
});

function toggle() {
  capturing ? stopCapture() : startCapture();
}

function startCapture() {
  const serverUrl = document.getElementById('serverUrl').value.trim();
  if (!serverUrl) { setStatus('Ingresá la URL del servidor', 'err'); return; }

  chrome.storage.local.set({ serverUrl });
  setBtn(true);

  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    const tabId = tabs[0]?.id;
    if (!tabId) { setStatus('No hay pestaña activa', 'err'); setBtn(false); return; }

    chrome.runtime.sendMessage({ action: 'start', tabId, serverUrl }, (resp) => {
      setBtn(false);
      if (chrome.runtime.lastError) {
        setStatus('Error: ' + chrome.runtime.lastError.message, 'err');
        return;
      }
      if (resp?.ok) {
        setCapturing(true, 'Capturando...');
      } else {
        setStatus('Error: ' + (resp?.error || 'desconocido'), 'err');
      }
    });
  });
}

function stopCapture() {
  setBtn(true);
  chrome.runtime.sendMessage({ action: 'stop' }, () => {
    setCapturing(false, 'Detenido');
    setBtn(false);
  });
}

function setCapturing(state, msg) {
  capturing = state;
  const btn = document.getElementById('btn');
  btn.textContent = state ? 'Detener captura' : 'Iniciar captura';
  btn.classList.toggle('stop', state);
  if (msg) setStatus(msg, state ? 'ok' : '');
}

function setBtn(disabled) {
  document.getElementById('btn').disabled = disabled;
}

function setStatus(msg, type = '') {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.className = type;
}
