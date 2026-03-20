"""
Zoom Tradutor — Cliente de captura de audio para el coordinador.

Captura el audio del speaker (WASAPI loopback) con pyaudiowpatch,
resamplea a 16 kHz mono y usa VAD basado en energía (frames de 30 ms)
para detectar pausas naturales del habla. Envía cada enunciado completo
como un chunk PCM int16 via WebSocket a Railway.

Lógica VAD:
  - Procesa el audio en frames de 30 ms
  - Acumula frames mientras hay voz (RMS >= RMS_THRESHOLD)
  - Cuando detecta 1 s de silencio tras voz → envía el enunciado
  - Mínimo 0.5 s de voz para enviar; máximo 12 s (forzado)

Compilar como .exe:
    pyinstaller --onefile --windowed --name ZoomTradutor capture_client.py
"""

import queue
import threading
import tkinter as tk

import numpy as np
import pyaudiowpatch as pyaudio
import websocket  # websocket-client
from websocket import ABNF

# ── Constantes ────────────────────────────────────────────────────────────────

TARGET_RATE     = 16000
RMS_THRESHOLD   = 0.015          # float32; < umbral → silencio
DEFAULT_URL     = 'wss://web-production-a6d81.up.railway.app/coordinator_ws'
RECONNECT_DELAY = 3              # segundos entre reintentos WS

# VAD — detección de pausas naturales del habla
FRAME_MS           = 30                                      # duración de cada frame
FRAME_SAMPLES      = int(TARGET_RATE * FRAME_MS / 1000)      # 480 muestras @ 16 kHz
SILENCE_FRAMES_END = int(0.7 / (FRAME_MS / 1000))            # 23 frames ≈ 700 ms de silencio → enviar
                                                             # Reducido de 1 s: menos latencia percibida entre
                                                             # frases. 700 ms es suficiente para separar
                                                             # enunciados sin cortar pausas internas del habla.
MIN_SPEECH_SAMPLES = int(TARGET_RATE * 0.5)                  # mínimo 0.5 s de voz para enviar
MAX_SPEECH_SAMPLES = int(TARGET_RATE * 12)                   # máximo 12 s → enviar aunque no haya pausa
PADDING_SAMPLES    = int(TARGET_RATE * 0.3)                  # 300 ms de silencio al inicio/fin del enunciado
                                                             # Whisper pierde fonemas cuando el audio empieza
                                                             # o termina de forma abrupta sin contexto previo

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

def _enqueue_utterance(speech_buf: np.ndarray, label: str) -> None:
    """Convierte un array float32 de un enunciado completo a PCM int16 y lo encola.

    Agrega 300 ms de silencio al inicio y fin del enunciado. Whisper pierde los
    primeros y últimos fonemas cuando el audio empieza/termina de forma abrupta —
    el padding le da el contexto acústico necesario para una transcripción completa.
    """
    rms = _rms_f32(speech_buf)
    padding = np.zeros(PADDING_SAMPLES, dtype=np.float32)
    padded_buf = np.concatenate([padding, speech_buf, padding])
    chunk_i16   = (padded_buf * 32767).clip(-32768, 32767).astype(np.int16)
    chunk_bytes = chunk_i16.tobytes()
    try:
        _audio_q.put_nowait(chunk_bytes)
        dur_speech = len(speech_buf) / TARGET_RATE
        dur_total  = len(padded_buf) / TARGET_RATE
        print(f'[Capture] {label} — RMS={rms:.4f} voz={dur_speech:.2f}s total={dur_total:.2f}s '
              f'(+{PADDING_SAMPLES*2/TARGET_RATE:.1f}s padding) qsize={_audio_q.qsize()}')
    except queue.Full:
        print(f'[Capture] COLA LLENA — {label} descartado, qsize={_audio_q.qsize()}')


def _capture_thread():
    """
    Captura WASAPI loopback, resamplea a 16 kHz mono y aplica VAD basado en energía.

    Procesa el audio en frames de 30 ms. Acumula frames de voz en speech_buf.
    Cuando detecta SILENCE_FRAMES_END frames consecutivos de silencio tras voz,
    envía el enunciado acumulado como un chunk completo. Esto alinea los chunks
    con las pausas naturales del habla en lugar de cortar cada 3 s fijos.
    """
    p = pyaudio.PyAudio()
    try:
        dev_idx, dev_info = _find_loopback_device(p)
    except RuntimeError as e:
        _set_status(str(e), 'red')
        p.terminate()
        return

    native_rate     = int(dev_info['defaultSampleRate'])
    native_ch       = max(1, dev_info.get('maxInputChannels', 2))
    frames_per_read = int(native_rate * FRAME_MS / 1000)   # bloques de 30 ms nativos

    stream = p.open(
        format=pyaudio.paFloat32,
        channels=native_ch,
        rate=native_rate,
        input=True,
        input_device_index=dev_idx,
        frames_per_buffer=frames_per_read,
    )

    _set_status('Capturando...', 'green')

    # Buffer de resampleo (puede acumular fracciones de frame por diferencia de rates)
    resample_buf = np.empty(0, dtype=np.float32)

    # Estado VAD
    speech_buf     = np.empty(0, dtype=np.float32)   # enunciado en construcción
    in_speech      = False
    silence_frames = 0

    try:
        while _running:
            raw   = stream.read(frames_per_read, exception_on_overflow=False)
            frame = np.frombuffer(raw, dtype=np.float32).reshape(-1, native_ch)

            # Mezclar canales → mono
            mono = frame.mean(axis=1)

            # Resamplear native_rate → 16 kHz si hace falta
            if native_rate != TARGET_RATE:
                n_out = int(len(mono) * TARGET_RATE / native_rate)
                mono = np.interp(
                    np.linspace(0, len(mono), n_out, endpoint=False),
                    np.arange(len(mono)),
                    mono,
                ).astype(np.float32)

            resample_buf = np.concatenate([resample_buf, mono])

            # Procesar en frames de FRAME_SAMPLES para VAD
            while len(resample_buf) >= FRAME_SAMPLES:
                vad_frame    = resample_buf[:FRAME_SAMPLES]
                resample_buf = resample_buf[FRAME_SAMPLES:]

                rms = _rms_f32(vad_frame)
                _set_rms(rms)

                is_voice = rms >= RMS_THRESHOLD

                if is_voice:
                    # Frame de voz → acumular
                    speech_buf     = np.concatenate([speech_buf, vad_frame])
                    silence_frames = 0
                    in_speech      = True

                    # Seguridad: si el enunciado es demasiado largo, enviarlo de todos modos
                    if len(speech_buf) >= MAX_SPEECH_SAMPLES:
                        _enqueue_utterance(speech_buf, 'utterance[MAX]')
                        speech_buf     = np.empty(0, dtype=np.float32)
                        in_speech      = False
                        silence_frames = 0

                else:
                    # Frame de silencio
                    if in_speech:
                        # Incluir el silencio en el buffer para contexto de Whisper
                        speech_buf     = np.concatenate([speech_buf, vad_frame])
                        silence_frames += 1

                        if silence_frames >= SILENCE_FRAMES_END:
                            # 1 s de silencio tras voz → fin de enunciado
                            if len(speech_buf) >= MIN_SPEECH_SAMPLES:
                                _enqueue_utterance(speech_buf, 'utterance[VAD]')
                            else:
                                print(f'[Capture] enunciado muy corto ({len(speech_buf)/TARGET_RATE:.2f}s) — descartado')
                            speech_buf     = np.empty(0, dtype=np.float32)
                            in_speech      = False
                            silence_frames = 0
                    # si no estábamos en voz, simplemente ignorar el frame de silencio
    finally:
        # Enviar lo que quedaba acumulado si hay suficiente
        if in_speech and len(speech_buf) >= MIN_SPEECH_SAMPLES:
            _enqueue_utterance(speech_buf, 'utterance[FLUSH]')
        stream.stop_stream()
        stream.close()
        p.terminate()


# ── Hilo WebSocket ────────────────────────────────────────────────────────────

def _ws_thread(url: str):
    """Conecta al servidor Railway y envía chunks desde _audio_q."""
    global _ws_app

    def on_open(ws):
        _set_status('Conectado — enviando audio', 'green')

        # Vaciar cola de audio viejo capturado antes de conectar
        # (evita burst de 40 chunks stale al entrar al meeting)
        drained = 0
        while not _audio_q.empty():
            try:
                _audio_q.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained:
            print(f'[WS] cola vaciada al conectar — {drained} chunks descartados')

        _send_count = 0
        def _sender():
            nonlocal _send_count
            while _running:
                try:
                    chunk = _audio_q.get(timeout=0.5)
                    ws.send(chunk, ABNF.OPCODE_BINARY)
                    _send_count += 1
                    print(f'[WS] chunk #{_send_count} enviado — {len(chunk)} bytes')
                except queue.Empty:
                    pass
                except Exception as e:
                    print(f'[WS] error en sender: {e}')
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
