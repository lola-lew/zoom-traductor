"""
Servidor principal: Flask + Flask-SocketIO + WebSocket interno de audio.

Flujo:
  Zoom (Playwright) → ws://localhost:8765 (PCM/webm raw)
       → TranslatorPipeline (Whisper + GPT + TTS)
       → SocketIO broadcast → clientes /listen (Web Audio API)
"""

import asyncio
import datetime
import logging
import os
import re
import threading
import uuid
from pathlib import Path
from typing import Optional

import websockets
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request
from flask_sock import Sock
from flask_socketio import SocketIO, emit
from playwright.async_api import async_playwright

from translator import LANGUAGES, TranslatorPipeline
from zoom_bot import CONNECTING, DISCONNECTED, IN_MEETING, ZoomBot

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'zoom-traductor-dev')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
sock = Sock(app)

# ── Configuración ─────────────────────────────────────────────────────────────

AUDIO_WS_PORT = int(os.environ.get('AUDIO_WS_PORT', 8765))
SESSIONS_DIR  = Path('sessions')
SESSIONS_DIR.mkdir(exist_ok=True)

# ── Estado global ─────────────────────────────────────────────────────────────

_state: dict = {
    'running':     False,
    'bot_state':   'idle',       # idle | connecting | waiting_room | in_meeting | disconnected | error
    'meeting_id':  None,
    'target_lang': 'pt',
}

translator    = TranslatorPipeline()
_zoom_bot:    Optional[ZoomBot] = None
_session_log: Optional['SessionLog'] = None

_HEADLESS          = os.environ.get('HEADLESS', '0') == '1'
_shared_playwright = None
_shared_browser    = None

# Loop asyncio compartido (Playwright + WebSocket interno)
_async_loop:   Optional[asyncio.AbstractEventLoop] = None
_async_thread: Optional[threading.Thread]          = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _async_loop, _async_thread
    if _async_loop is None or not _async_loop.is_running():
        _async_loop = asyncio.new_event_loop()
        _async_thread = threading.Thread(
            target=_async_loop.run_forever, daemon=True, name='async-loop'
        )
        _async_thread.start()
        import time
        for _ in range(20):
            if _async_loop.is_running():
                break
            time.sleep(0.05)
    return _async_loop


def _run_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _get_loop())


# ── Logger de sesión ──────────────────────────────────────────────────────────

class SessionLog:
    """Guarda traducciones con timestamp en un archivo .txt por sesión."""

    def __init__(self, meeting_id: str, target_lang: str):
        ts       = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'{ts}__{meeting_id}__{target_lang}.txt'
        self.path = SESSIONS_DIR / filename
        self._lock = threading.Lock()
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(f'Sesión: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            f.write(f'Reunión: {meeting_id}  |  Idioma destino: {target_lang}\n')
            f.write('=' * 70 + '\n\n')
        logger.info('Log de sesión: %s', self.path)

    def write(self, original: str, translated: str) -> None:
        ts   = datetime.datetime.now().strftime('%H:%M:%S')
        line = f'[{ts}] {original}\n       → {translated}\n\n'
        with self._lock:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(line)

    def close(self) -> None:
        with self._lock:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(f'\n[FIN] {datetime.datetime.now().strftime("%H:%M:%S")}\n')


# ── WebSocket interno (PCM/webm del bot de Zoom) ──────────────────────────────

_audio_chunk_count = 0

async def _audio_ws_handler(websocket):
    global _audio_chunk_count
    logger.info('[AudioWS] bridge conectado: %s', websocket.remote_address)
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue

            _audio_chunk_count += 1

            if _state.get('bot_state') != IN_MEETING:
                if _audio_chunk_count % 200 == 1:
                    logger.debug('[AudioWS] chunk descartado — bot_state=%s (esperando IN_MEETING)',
                                 _state.get('bot_state'))
                continue

            if _audio_chunk_count % 50 == 1:
                logger.info('[AudioWS] chunk #%d — %d bytes', _audio_chunk_count, len(message))

            translator.feed_audio(message)
    except Exception as exc:
        logger.warning('[AudioWS] bridge desconectado: %s', exc)


async def _audio_ws_serve():
    while True:
        try:
            async with websockets.serve(
                _audio_ws_handler,
                'localhost',
                AUDIO_WS_PORT,
                ping_interval=None,
                max_size=2 ** 20,
            ):
                logger.info('[AudioWS] servidor escuchando en ws://localhost:%d', AUDIO_WS_PORT)
                stop_evt = asyncio.Event()
                await stop_evt.wait()

        except asyncio.CancelledError:
            logger.info('[AudioWS] servidor cancelado — saliendo')
            return

        except OSError as e:
            logger.error('[AudioWS] OSError: %s — reintentando en 3 s', e)
            await asyncio.sleep(3)

        except Exception as e:
            logger.error('[AudioWS] error inesperado: %s — reintentando en 2 s', e)
            await asyncio.sleep(2)


# ── Callbacks del pipeline ────────────────────────────────────────────────────

def _on_translation(original: str, translated: str, audio_b64: str) -> None:
    socketio.emit('translation', {
        'original':   original,
        'translated': translated,
        'lang':       _state['target_lang'],
    })
    socketio.emit('audio', {'data': audio_b64})
    if _session_log:
        _session_log.write(original, translated)


def _on_pipeline_error(message: str) -> None:
    logger.error('Pipeline error: %s', message)
    socketio.emit('pipeline_error', {'message': message})


translator.on_translation = _on_translation
translator.on_error       = _on_pipeline_error

# ── Callbacks del bot de Zoom ─────────────────────────────────────────────────

def _on_bot_log(msg: str) -> None:
    socketio.emit('bot_status', {'message': msg})


def _on_bot_state_change(bot_state: str, msg: str) -> None:
    _state['bot_state'] = bot_state
    if bot_state == DISCONNECTED:
        _state['running'] = False
        translator.stop()
        if _session_log:
            _session_log.close()
    socketio.emit('status_change', {'state': bot_state, 'message': msg})
    socketio.emit('bot_status',    {'message': msg})
    # Tras informar DISCONNECTED, resetear a idle para que la UI
    # vuelva a mostrar el botón "Iniciar" y permita reiniciar.
    if bot_state == DISCONNECTED:
        _state['bot_state'] = 'idle'
        socketio.emit('status_change', {'state': 'idle', 'message': msg})


# ── Rutas HTTP ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', languages=LANGUAGES)


@app.route('/listen')
def listen():
    return render_template('listen.html')


@app.route('/ping')
def ping():
    return jsonify({'pong': True, 'state': _state['bot_state']})


@app.route('/start', methods=['POST'])
def start():
    global _zoom_bot, _session_log

    raw_body = request.get_data(as_text=True)
    print(f'[/start] recibido — body: {raw_body[:300]}', flush=True)
    logger.info('/start recibido — body: %s', raw_body[:300])

    if _state['running']:
        logger.warning('/start rechazado — ya está en ejecución')
        return jsonify({'error': 'Ya está en ejecución'}), 400

    try:
        import json as _json
        data = _json.loads(raw_body) if raw_body.strip() else {}
    except Exception:
        data = request.get_json(force=True, silent=True) or {}

    raw_input   = str(data.get('meeting_url') or data.get('meeting_id') or '')
    target_lang = data.get('target_lang', 'pt')

    meeting_id = re.sub(r'\D', '', re.search(r'\d+', raw_input).group()) \
                 if re.search(r'\d+', raw_input) else ''

    meeting_url_or_id = raw_input.strip() if raw_input.strip().startswith('http') else meeting_id

    logger.info('/start — input: %r  id extraído: %r  lang: %s', raw_input, meeting_id, target_lang)

    if not meeting_id:
        logger.warning('/start rechazado — sin meeting ID. raw=%r data=%r', raw_input, data)
        return jsonify({'error': 'Ingresá el link o el ID de la reunión Zoom'}), 400

    _state.update({
        'running':     True,
        'bot_state':   CONNECTING,
        'meeting_id':  meeting_id,
        'target_lang': target_lang,
    })
    translator.start(target_lang)
    _session_log = SessionLog(meeting_id, target_lang)

    _zoom_bot = ZoomBot(audio_ws_url=f'ws://localhost:{AUDIO_WS_PORT}')
    _zoom_bot.on_status        = _on_bot_log
    _zoom_bot.on_status_change = _on_bot_state_change

    async def _start_bots():
        global _shared_playwright, _shared_browser
        try:
            _shared_playwright = await async_playwright().start()
            _shared_browser = await _shared_playwright.chromium.launch(
                headless=_HEADLESS,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--mute-audio',
                    '--use-fake-ui-for-media-stream',
                    '--use-fake-device-for-media-stream',
                    '--autoplay-policy=no-user-gesture-required',
                    '--allow-running-insecure-content',
                    '--disable-features=ExternalProtocolDialog,WebRtcHideLocalIpsWithMdns,PrivateNetworkAccessChecks',
                    f'--unsafely-treat-insecure-origin-as-secure=http://localhost:{AUDIO_WS_PORT},ws://localhost:{AUDIO_WS_PORT}',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                ],
            )
            logger.info('[App] browser iniciado')
            await _zoom_bot.join(meeting_url_or_id, _browser=_shared_browser)
        except Exception as exc:
            logger.error('Error al iniciar bots: %s', exc)
            _state.update({'running': False, 'bot_state': 'error'})
            translator.stop()
            if _session_log:
                _session_log.close()
            socketio.emit('status_change', {'state': 'error', 'message': str(exc)})

    _run_async(_start_bots())

    return jsonify({'status': 'started', 'meeting_id': meeting_id})


@app.route('/stop', methods=['POST'])
def stop():
    if not _state['running']:
        return jsonify({'error': 'No está en ejecución'}), 400

    translator.stop()
    if _session_log:
        _session_log.close()

    async def _stop_bots():
        global _shared_playwright, _shared_browser
        if _zoom_bot:
            await _zoom_bot.leave()
        if _shared_browser:
            try:
                await _shared_browser.close()
            except Exception:
                pass
            _shared_browser = None
        if _shared_playwright:
            try:
                await _shared_playwright.stop()
            except Exception:
                pass
            _shared_playwright = None

    future = _run_async(_stop_bots())
    try:
        future.result(timeout=12)
    except Exception as exc:
        logger.error('Error al detener bots: %s', exc)

    _state.update({'running': False, 'bot_state': 'idle', 'meeting_id': None})
    socketio.emit('status_change', {'state': 'idle', 'message': 'Sesión detenida'})
    return jsonify({'status': 'stopped'})


@app.route('/download_audio')
def download_audio():
    data = translator.get_audio_bytes()
    if not data:
        return jsonify({'error': 'No hay audio grabado en esta sesión'}), 404
    return Response(
        data,
        mimetype='audio/mpeg',
        headers={'Content-Disposition': 'attachment; filename="sesion_traducida.mp3"'},
    )


@app.route('/status')
def status():
    return jsonify({
        **_state,
        'languages': {k: v['name'] for k, v in LANGUAGES.items()},
    })


# ── SocketIO ──────────────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    emit('status',        _state)
    emit('status_change', {'state': _state['bot_state'], 'message': ''})


@sock.route('/coordinator_ws')
def coordinator_ws(ws):
    """WebSocket puro: recibe chunks webm/opus desde la extensión Chrome."""
    chunk_count = 0
    logger.info('[CoordinatorWS] coordinador conectado — bot_state=%s running=%s',
                _state.get('bot_state'), _state.get('running'))
    try:
        while True:
            data = ws.receive()
            if data is None:
                logger.info('[CoordinatorWS] receive() retornó None — cerrando')
                break
            if not isinstance(data, bytes):
                logger.info('[CoordinatorWS] DIAGNÓSTICO: recibido tipo=%s len=%d primeros_bytes=%r',
                            type(data).__name__, len(data) if data else 0,
                            data[:80] if data else '')
                continue  # ping de keepalive (texto) — ignorar
            chunk_count += 1
            if chunk_count == 1:
                logger.info('[CoordinatorWS] PRIMER chunk binario recibido — %d bytes, bot_state=%s',
                            len(data), _state.get('bot_state'))
            if _state.get('bot_state') != IN_MEETING:
                if chunk_count % 20 == 1:
                    logger.debug('[CoordinatorWS] chunk descartado — bot_state=%s',
                                 _state.get('bot_state'))
                continue
            if chunk_count % 10 == 1:
                logger.info('[CoordinatorWS] chunk #%d — %d bytes', chunk_count, len(data))
            translator.feed_audio(data)
    except Exception as exc:
        logger.warning('[CoordinatorWS] desconectado: %s', exc)
    logger.info('[CoordinatorWS] coordinador desconectado — %d chunks procesados', chunk_count)


# ── Arranque ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    HTTP_PORT = int(os.environ.get('PORT', 5000))

    print('=' * 60, flush=True)
    print('  Zoom Tradutor — iniciando servidor...', flush=True)

    loop = _get_loop()
    _audio_ws_future = asyncio.run_coroutine_threadsafe(_audio_ws_serve(), loop)  # noqa: F841
    print(f'  Audio WS interno: ws://localhost:{AUDIO_WS_PORT}', flush=True)
    print(f'  Flask-SocketIO:   http://0.0.0.0:{HTTP_PORT}', flush=True)
    print('=' * 60, flush=True)
    logger.info('Servidor listo en http://0.0.0.0:%d', HTTP_PORT)

    socketio.run(
        app,
        host='0.0.0.0',
        port=HTTP_PORT,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
