"""
Zoom Tradutor — Cliente de captura de audio para el coordinador.

Captura el audio del speaker (WASAPI loopback) a 16 kHz mono,
descarta silencio (RMS < 100) y envía chunks PCM int16 de 3 s
via WebSocket a wss://web-production-a6d81.up.railway.app/coordinator_ws.

Compilar como .exe:
    pyinstaller --onefile --windowed --name ZoomTradutor capture_client.py
"""

import math
import queue
import struct
import threading
import tkinter as tk
from tkinter import ttk

import soundcard as sc
import websocket  # websocket-client

# ── Constantes ────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 16000
CHANNELS      = 1
CHUNK_SECONDS = 3
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_SECONDS   # 48 000 muestras por chunk
RMS_THRESHOLD = 100                           # chunks más silenciosos → descartar
DEFAULT_URL   = 'wss://web-production-a6d81.up.railway.app/coordinator_ws'
RECONNECT_DELAY = 3                           # segundos entre reintentos de WS

# ── Estado compartido ─────────────────────────────────────────────────────────

_running   = False
_ws_app    = None          # WebSocketApp activo
_audio_q:  queue.Queue = queue.Queue(maxsize=40)
_status_cb = None          # función para actualizar la UI


def _rms(samples: bytes) -> float:
    """RMS de samples PCM int16 (little-endian)."""
    n = len(samples) // 2
    if n == 0:
        return 0.0
    total = sum(s * s for s in struct.unpack_from(f'<{n}h', samples))
    return math.sqrt(total / n)


# ── Hilo de captura de audio ──────────────────────────────────────────────────

def _capture_thread():
    """Captura audio del speaker por defecto (WASAPI loopback) y llena _audio_q."""
    mic = sc.get_microphone(id=str(sc.default_speaker().name), include_loopback=True)
    buf = b''

    with mic.recorder(samplerate=SAMPLE_RATE, channels=CHANNELS, blocksize=1024) as rec:
        _set_status('Capturando...', 'green')
        while _running:
            frame = rec.record(numframes=1024)
            # float32 → int16
            samples_int16 = (frame[:, 0] * 32767).clip(-32768, 32767).astype('int16')
            buf += samples_int16.tobytes()

            if len(buf) >= CHUNK_SAMPLES * 2:
                chunk = buf[:CHUNK_SAMPLES * 2]
                buf   = buf[CHUNK_SAMPLES * 2:]

                rms = _rms(chunk)
                if rms < RMS_THRESHOLD:
                    continue  # silencio — descartar

                try:
                    _audio_q.put_nowait(chunk)
                except queue.Full:
                    pass  # WS lento — descartar chunk más viejo


# ── Hilo WebSocket ────────────────────────────────────────────────────────────

def _ws_thread(url: str):
    """Conecta al servidor Railway y envía chunks desde _audio_q."""
    global _ws_app

    def on_open(ws):
        _set_status('Conectado — enviando audio', 'green')
        # Hilo de envío: drena _audio_q mientras la conexión esté abierta
        def _sender():
            while _running and ws.sock and ws.sock.connected:
                try:
                    chunk = _audio_q.get(timeout=0.5)
                    ws.send_binary(chunk)
                except queue.Empty:
                    pass
                except Exception:
                    break
        threading.Thread(target=_sender, daemon=True).start()

    def on_close(ws, code, msg):
        if _running:
            _set_status(f'Desconectado — reconectando en {RECONNECT_DELAY} s...', 'orange')

    def on_error(ws, err):
        _set_status(f'Error WS: {err}', 'red')

    while _running:
        _ws_app = websocket.WebSocketApp(
            url,
            on_open=on_open,
            on_close=on_close,
            on_error=on_error,
        )
        _ws_app.run_forever(ping_interval=20, ping_timeout=10)

        if _running:
            import time; time.sleep(RECONNECT_DELAY)

    _set_status('Detenido', 'gray')


# ── UI ────────────────────────────────────────────────────────────────────────

def _set_status(msg: str, color: str = 'gray'):
    if _status_cb:
        _status_cb(msg, color)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Zoom Tradutor — Captura')
        self.resizable(False, False)
        self.configure(bg='#0d1117')
        self._build_ui()
        self.protocol('WM_DELETE_WINDOW', self._on_close)

    def _build_ui(self):
        pad = {'padx': 14, 'pady': 6}

        tk.Label(self, text='Servidor WebSocket', bg='#0d1117', fg='#8b949e',
                 font=('Segoe UI', 8)).pack(anchor='w', **pad)

        self._url_var = tk.StringVar(value=DEFAULT_URL)
        url_entry = tk.Entry(self, textvariable=self._url_var, width=52,
                             bg='#161b22', fg='#e6edf3', insertbackground='white',
                             relief='flat', font=('Segoe UI', 9))
        url_entry.pack(fill='x', padx=14, pady=(0, 8))

        self._btn = tk.Button(self, text='Iniciar captura', command=self._toggle,
                              bg='#2f81f7', fg='white', activebackground='#388bfd',
                              activeforeground='white', relief='flat',
                              font=('Segoe UI', 10, 'bold'), cursor='hand2',
                              padx=10, pady=6)
        self._btn.pack(fill='x', padx=14, pady=(0, 8))

        self._status_lbl = tk.Label(self, text='Listo', bg='#0d1117', fg='#8b949e',
                                    font=('Segoe UI', 9))
        self._status_lbl.pack(anchor='w', **pad)

        # Exponer callback para hilos
        global _status_cb
        _status_cb = self._set_status_safe

    def _set_status_safe(self, msg: str, color: str):
        color_map = {
            'green':  '#3fb950',
            'orange': '#d29922',
            'red':    '#f85149',
            'gray':   '#8b949e',
        }
        fg = color_map.get(color, '#8b949e')
        self.after(0, lambda: self._status_lbl.config(text=msg, fg=fg))

    def _toggle(self):
        global _running
        if _running:
            self._stop()
        else:
            self._start()

    def _start(self):
        global _running
        url = self._url_var.get().strip()
        if not url:
            self._set_status_safe('Ingresá la URL del servidor', 'red')
            return
        _running = True
        self._btn.config(text='Detener captura', bg='#f85149', activebackground='#ff6b6b')
        self._set_status_safe('Iniciando...', 'orange')

        threading.Thread(target=_capture_thread, daemon=True).start()
        threading.Thread(target=_ws_thread, args=(url,), daemon=True).start()

    def _stop(self):
        global _running, _ws_app
        _running = False
        if _ws_app:
            try: _ws_app.close()
            except Exception: pass
            _ws_app = None
        self._btn.config(text='Iniciar captura', bg='#2f81f7', activebackground='#388bfd')
        self._set_status_safe('Detenido', 'gray')

    def _on_close(self):
        self._stop()
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
