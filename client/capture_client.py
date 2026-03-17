"""
Zoom Tradutor — Cliente de captura de audio para el coordinador.

Captura el audio del speaker (WASAPI loopback) con pyaudiowpatch,
resamplea a 16 kHz mono, descarta silencio (RMS float32 < 0.01) y
envía chunks PCM int16 de 3 s via WebSocket a Railway.

Compilar como .exe:
    pyinstaller --onefile --windowed --name ZoomTradutor capture_client.py
"""

import queue
import threading
import tkinter as tk

import numpy as np
import pyaudio
import websocket  # websocket-client

# ── Constantes ────────────────────────────────────────────────────────────────

TARGET_RATE     = 16000
CHUNK_SECONDS   = 3
CHUNK_SAMPLES   = TARGET_RATE * CHUNK_SECONDS   # 48 000 muestras por chunk
RMS_THRESHOLD   = 0.01                          # float32; < umbral → silencio, descartar
DEFAULT_URL     = 'wss://web-production-a6d81.up.railway.app/coordinator_ws'
RECONNECT_DELAY = 3                             # segundos entre reintentos WS

# ── Estado compartido ─────────────────────────────────────────────────────────

_running  = False
_ws_app   = None
_audio_q: queue.Queue = queue.Queue(maxsize=40)
_status_cb = None   # callback UI → texto de estado
_rms_cb    = None   # callback UI → nivel RMS


def _set_status(msg: str, color: str = 'gray'):
    if _status_cb:
        _status_cb(msg, color)


def _set_rms(rms: float):
    if _rms_cb:
        _rms_cb(rms)


def _rms_f32(arr: np.ndarray) -> float:
    """RMS de array float32 normalizado (0.0–1.0)."""
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr.astype(np.float64) ** 2)))


def _find_loopback_device(p: pyaudio.PyAudio):
    """
    Retorna (device_index, device_info) del dispositivo WASAPI loopback
    que corresponde al speaker por defecto.
    Lanza RuntimeError si no se encuentra.
    """
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError:
        raise RuntimeError('WASAPI no disponible en este sistema')

    default_idx = wasapi['defaultOutputDevice']
    default_name = p.get_device_info_by_index(default_idx)['name']

    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get('hostApi') != wasapi['index']:
            continue
        if not dev.get('isLoopbackDevice', False):
            continue
        # Coincidencia por nombre base (el loopback añade " [Loopback]" al nombre)
        if default_name in dev['name'] or dev['name'] in default_name:
            return i, dev

    raise RuntimeError(f"No se encontró dispositivo loopback para '{default_name}'")


# ── Hilo de captura de audio ──────────────────────────────────────────────────

def _capture_thread():
    """Captura WASAPI loopback, resamplea a 16 kHz mono y llena _audio_q."""
    p = pyaudio.PyAudio()
    try:
        dev_idx, dev_info = _find_loopback_device(p)
    except RuntimeError as e:
        _set_status(str(e), 'red')
        p.terminate()
        return

    native_rate = int(dev_info['defaultSampleRate'])
    native_ch   = max(1, dev_info.get('maxInputChannels', 2))
    frames_per_read = int(native_rate * 0.05)   # bloques de ~50 ms

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=native_ch,
        rate=native_rate,
        input=True,
        input_device_index=dev_idx,
        frames_per_buffer=frames_per_read,
    )

    _set_status('Capturando...', 'green')
    buf = np.empty(0, dtype=np.float32)

    try:
        while _running:
            raw  = stream.read(frames_per_read, exception_on_overflow=False)
            frame = np.frombuffer(raw, dtype=np.float32).reshape(-1, native_ch)

            # Mezclar canales → mono
            mono = frame.mean(axis=1)

            # Resamplear native_rate → 16 kHz
            if native_rate != TARGET_RATE:
                n_out = int(len(mono) * TARGET_RATE / native_rate)
                mono = np.interp(
                    np.linspace(0, len(mono), n_out, endpoint=False),
                    np.arange(len(mono)),
                    mono,
                ).astype(np.float32)

            buf = np.concatenate([buf, mono])

            while len(buf) >= CHUNK_SAMPLES:
                chunk_f32 = buf[:CHUNK_SAMPLES]
                buf        = buf[CHUNK_SAMPLES:]

                rms = _rms_f32(chunk_f32)
                _set_rms(rms)

                if rms < RMS_THRESHOLD:
                    continue  # silencio — descartar

                chunk_i16 = (chunk_f32 * 32767).clip(-32768, 32767).astype(np.int16)
                try:
                    _audio_q.put_nowait(chunk_i16.tobytes())
                except queue.Full:
                    pass  # WS lento — descartar chunk más viejo
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


# ── Hilo WebSocket ────────────────────────────────────────────────────────────

def _ws_thread(url: str):
    """Conecta al servidor Railway y envía chunks desde _audio_q."""
    global _ws_app

    def on_open(ws):
        _set_status('Conectado — enviando audio', 'green')

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
        tk.Entry(self, textvariable=self._url_var, width=52,
                 bg='#161b22', fg='#e6edf3', insertbackground='white',
                 relief='flat', font=('Segoe UI', 9)).pack(fill='x', padx=14, pady=(0, 8))

        self._btn = tk.Button(self, text='Iniciar captura', command=self._toggle,
                              bg='#2f81f7', fg='white', activebackground='#388bfd',
                              activeforeground='white', relief='flat',
                              font=('Segoe UI', 10, 'bold'), cursor='hand2',
                              padx=10, pady=6)
        self._btn.pack(fill='x', padx=14, pady=(0, 8))

        self._status_lbl = tk.Label(self, text='Listo', bg='#0d1117', fg='#8b949e',
                                    font=('Segoe UI', 9))
        self._status_lbl.pack(anchor='w', **pad)

        tk.Label(self, text='Nivel de audio (RMS)', bg='#0d1117', fg='#8b949e',
                 font=('Segoe UI', 8)).pack(anchor='w', padx=14)
        self._rms_lbl = tk.Label(self, text='RMS: —', bg='#0d1117', fg='#8b949e',
                                  font=('Segoe UI', 9, 'bold'))
        self._rms_lbl.pack(anchor='w', padx=14, pady=(0, 10))

        global _status_cb, _rms_cb
        _status_cb = self._set_status_safe
        _rms_cb    = self._set_rms_safe

    def _set_status_safe(self, msg: str, color: str):
        color_map = {
            'green':  '#3fb950',
            'orange': '#d29922',
            'red':    '#f85149',
            'gray':   '#8b949e',
        }
        fg = color_map.get(color, '#8b949e')
        self.after(0, lambda: self._status_lbl.config(text=msg, fg=fg))

    def _set_rms_safe(self, rms: float):
        color = '#3fb950' if rms >= RMS_THRESHOLD else '#8b949e'
        self.after(0, lambda: self._rms_lbl.config(text=f'RMS: {rms:.4f}', fg=color))

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
