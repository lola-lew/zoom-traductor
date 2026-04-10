"""
ZoomBot: Playwright que entra a Zoom Web Client como oyente silencioso.

Requisitos cubiertos:
 - Stealth anti-detección (webdriver, plugins, chrome obj, zoommtg://)
 - Aceptar cookies (OneTrust)
 - Cerrar popup "Abrir app de escritorio"
 - Formulario "Enter Meeting Info": nombre, sin passcode, click Join
 - Sala de espera: espera hasta ser admitido
 - "Join with Computer Audio": aceptar automáticamente
 - Silenciar mic y desactivar cámara antes de entrar
 - Monitorear fin de reunión / expulsión y notificar a la UI
 - Cierre limpio de browser sin procesos huérfanos
 - Captura de audio: sounddevice desde VB-Cable (índice 17) → WS interno
"""

import asyncio
import json
import logging
import os
import platform as _platform
import queue as stdlib_queue
import re
from typing import Callable, Optional
from urllib.parse import quote

import numpy as np
import sounddevice as sd
import websockets as ws_lib
from scipy.signal import resample_poly
from playwright.async_api import BrowserContext, Page, async_playwright

# En producción (Railway) setear HEADLESS=1 — localmente corre visible.
_HEADLESS = os.environ.get('HEADLESS', '0') == '1'

logger = logging.getLogger(__name__)

# ── Estados del bot ──────────────────────────────────────────────────────────
CONNECTING   = 'connecting'
WAITING_ROOM = 'waiting_room'
IN_MEETING   = 'in_meeting'
DISCONNECTED = 'disconnected'
ERROR        = 'error'

# ── Script de stealth ────────────────────────────────────────────────────────
# Inyectado ANTES de cualquier código de la página (add_init_script).
_STEALTH_SCRIPT = r"""
() => {
  // 1. Eliminar navigator.webdriver
  Object.defineProperty(navigator, 'webdriver', { get: () => false, configurable: true });

  // 2. Simular plugins de Chrome real
  const _pl = [
    { name: 'Chrome PDF Plugin',   filename: 'internal-pdf-viewer',             description: 'Portable Document Format' },
    { name: 'Chrome PDF Viewer',   filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai', description: '' },
    { name: 'Native Client',       filename: 'internal-nacl-plugin',             description: '' },
  ];
  _pl.item = i => _pl[i]; _pl.namedItem = n => _pl.find(p => p.name === n) ?? null; _pl.refresh = () => {};
  Object.defineProperty(navigator, 'plugins',   { get: () => _pl,            configurable: true });
  Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'], configurable: true });

  // 3. Objeto window.chrome (ausente en Chromium headless)
  if (!window.chrome) window.chrome = { runtime: {}, loadTimes: () => {}, csi: () => {}, app: {} };

  // 4. Permissions API
  try {
    const _q = navigator.permissions.query.bind(navigator.permissions);
    navigator.permissions.query = p =>
      p.name === 'notifications' ? Promise.resolve({ state: Notification.permission }) : _q(p);
  } catch (_) {}

  // 5. Bloquear redirección a la app de escritorio (deep link zoommtg://)
  const _origOpen = window.open;
  window.open = function (url, ...args) {
    if (typeof url === 'string' && url.startsWith('zoommtg://')) {
      console.log('[ZoomBot] zoommtg:// bloqueado');
      return null;
    }
    return _origOpen.call(this, url, ...args);
  };

  // 6. Bloquear location.assign / location.replace a zoommtg://
  const _origAssign  = location.assign.bind(location);
  const _origReplace = location.replace.bind(location);
  location.assign  = url => url.startsWith('zoommtg://') ? null : _origAssign(url);
  location.replace = url => url.startsWith('zoommtg://') ? null : _origReplace(url);
}
"""

# ── Script de captura de audio via Web Audio API (Linux/Railway) ─────────────
# Inyectado como init_script ANTES de que Zoom inicialice (solo en Linux).
# Intercepta RTCPeerConnection → obtiene el audio track de los participantes →
# AudioContext (16 kHz) + ScriptProcessorNode → VAD por energía idéntico al
# capture_client.py → envía utterances completos como PCM int16 via WebSocket.
#
# Al usar PCM int16 (no webm), el servidor reutiliza _feed_pcm() con el mismo
# pipeline VAD → Whisper → GPT → TTS que ya funciona. Sin duplicar lógica.
_CAPTURE_SCRIPT = r"""
(wsUrl) => {

  // ── Constantes VAD (espejo de capture_client.py) ──────────────────────
  const TARGET_RATE        = 16000;
  const FRAME_MS           = 30;
  const FRAME_SAMPLES      = Math.floor(TARGET_RATE * FRAME_MS / 1000); // 480
  const SILENCE_FRAMES_END = Math.floor(0.3  / (FRAME_MS / 1000));      // 10 ≈ 300 ms
  const MIN_SPEECH_SAMPLES = Math.floor(TARGET_RATE * 0.5);             // 8 000
  const MAX_SPEECH_SAMPLES = Math.floor(TARGET_RATE * 5);               // 80 000
  const PADDING_SAMPLES    = Math.floor(TARGET_RATE * 0.15);            // 2 400 (150 ms)
  const RMS_THRESHOLD      = 0.015;

  // ── WebSocket ─────────────────────────────────────────────────────────
  let ws      = null;
  let wsReady = false;
  const pending = [];

  // ── Watchdog ──────────────────────────────────────────────────────────
  let lastSentTime    = null;   // timestamp (ms) del último sendUtterance()
  let currentAudioCtx = null;   // AudioContext activo (para cerrar en restart)

  function connectWS() {
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => {
      wsReady = true;
      console.log('[Capture] WS conectado:', wsUrl);
      while (pending.length) ws.send(pending.shift());
    };
    ws.onclose = () => {
      wsReady  = false;
      console.warn('[Capture] WS cerrado — reconectando en 2 s');
      setTimeout(connectWS, 2000);
    };
    ws.onerror = (e) => console.warn('[Capture] WS error:', e.type);
  }
  connectWS();

  // ── Envío de utterance ────────────────────────────────────────────────
  function sendUtterance(speechArr, label) {
    // Añadir 300 ms de silencio al inicio y fin (mismo padding que capture_client.py)
    const pad    = new Float32Array(PADDING_SAMPLES); // ceros
    const total  = new Float32Array(PADDING_SAMPLES + speechArr.length + PADDING_SAMPLES);
    total.set(pad, 0);
    total.set(speechArr, PADDING_SAMPLES);
    total.set(pad, PADDING_SAMPLES + speechArr.length);

    // Convertir float32 → int16 little-endian
    const i16 = new Int16Array(total.length);
    for (let i = 0; i < total.length; i++) {
      i16[i] = Math.max(-32768, Math.min(32767, Math.round(total[i] * 32767)));
    }

    lastSentTime = Date.now();
    const dur = speechArr.length / TARGET_RATE;
    console.log('[Capture]', label, '— voz=', dur.toFixed(2) + 's',
                'bytes=', i16.buffer.byteLength);
    if (ws && ws.readyState === 1) {
      ws.send(i16.buffer);
    } else if (pending.length < 10) {
      pending.push(i16.buffer);
    }
  }

  // ── VAD (Voice Activity Detection) ───────────────────────────────────
  let speechBuf    = [];   // samples float32 acumulados
  let inSpeech     = false;
  let silenceFrames = 0;
  let frameBuf     = new Float32Array(0);  // buffer de resampleo

  function processVADFrame(frame) {
    // RMS del frame
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    const rms     = Math.sqrt(sum / frame.length);
    const isVoice = rms >= RMS_THRESHOLD;

    if (isVoice && !processVADFrame._loggedVoice) {
      processVADFrame._loggedVoice = true;
      console.log('[Capture] primera voz detectada — rms:', rms.toFixed(5),
                  'threshold:', RMS_THRESHOLD);
    }

    if (isVoice) {
      for (let i = 0; i < frame.length; i++) speechBuf.push(frame[i]);
      silenceFrames = 0;
      inSpeech      = true;
      if (speechBuf.length >= MAX_SPEECH_SAMPLES) {
        sendUtterance(new Float32Array(speechBuf), 'utterance[MAX]');
        speechBuf     = [];
        inSpeech      = false;
        silenceFrames = 0;
      }
    } else {
      if (inSpeech) {
        for (let i = 0; i < frame.length; i++) speechBuf.push(frame[i]);
        silenceFrames++;
        if (silenceFrames >= SILENCE_FRAMES_END) {
          if (speechBuf.length >= MIN_SPEECH_SAMPLES) {
            sendUtterance(new Float32Array(speechBuf), 'utterance[VAD]');
          } else {
            console.log('[Capture] utterance muy corto —', (speechBuf.length/TARGET_RATE).toFixed(2) + 's descartado');
          }
          speechBuf     = [];
          inSpeech      = false;
          silenceFrames = 0;
        }
      }
    }
  }

  // ── Captura del track WebRTC via AudioContext ─────────────────────────
  function startCapture(track) {
    // Cerrar AudioContext anterior si existe (watchdog restart)
    if (currentAudioCtx) {
      try { currentAudioCtx.close(); } catch(e) {}
      currentAudioCtx = null;
    }
    const stream   = new MediaStream([track]);
    // AudioContext a TARGET_RATE — Chrome lo soporta y resamplea internamente
    const audioCtx = new AudioContext({ sampleRate: TARGET_RATE });
    currentAudioCtx = audioCtx;
    const source   = audioCtx.createMediaStreamSource(stream);
    // ScriptProcessorNode: buffer 4096 para estabilidad, VAD interno procesa en FRAME_SAMPLES
    const processor = audioCtx.createScriptProcessor(4096, 1, 1);

    let frameCount = 0;
    processor.onaudioprocess = (e) => {
      const input = e.inputBuffer.getChannelData(0);
      frameCount++;
      if (frameCount % 100 === 0) {
        let s = 0; for (let i = 0; i < input.length; i++) s += input[i] * input[i];
        const rms = Math.sqrt(s / input.length);
        console.log('[Capture] frames:', frameCount, 'rms:', rms.toFixed(5),
                    'audioCtx.state:', audioCtx.state);
      }
      // Acumular en frameBuf y procesar en chunks de FRAME_SAMPLES
      const merged = new Float32Array(frameBuf.length + input.length);
      merged.set(frameBuf);
      merged.set(input, frameBuf.length);
      frameBuf = merged;
      while (frameBuf.length >= FRAME_SAMPLES) {
        processVADFrame(frameBuf.slice(0, FRAME_SAMPLES));
        frameBuf = frameBuf.slice(FRAME_SAMPLES);
      }
    };

    source.connect(processor);
    processor.connect(audioCtx.destination);  // necesario para que el grafo procese

    // audioCtx puede iniciarse en estado 'suspended' en Chrome headless sin
    // interacción de usuario. resume() lo activa explícitamente.
    audioCtx.resume()
      .then(() => console.log('[Capture] audioCtx.resume() OK — state:', audioCtx.state))
      .catch(e => console.warn('[Capture] audioCtx.resume() error:', e));
    console.log('[Capture] AudioContext+VAD iniciado — sampleRate:', audioCtx.sampleRate,
                'state:', audioCtx.state);
  }

  // ── Intercepción de RTCPeerConnection ────────────────────────────────
  const connectedTracks = new Set();
  let pendingTrack  = null;
  let captureArmed  = false;
  let captureStarted = false;

  function attachTrack(track) {
    if (connectedTracks.has(track.id)) return;
    connectedTracks.add(track.id);
    console.log('[Capture] track detectado — kind:', track.kind, 'id:', track.id,
                'armed:', captureArmed,
                'readyState:', track.readyState, 'enabled:', track.enabled);
    track.onended  = () => console.warn('[Capture] track.onended  id:', track.id);
    track.onmute   = () => console.warn('[Capture] track.onmute   id:', track.id);
    track.onunmute = () => console.log('[Capture] track.onunmute id:', track.id);
    pendingTrack = track;
    if (captureArmed && !captureStarted) {
      captureStarted = true;
      startCapture(track);
    }
  }

  window.__startCapture = () => {
    if (captureArmed) return 'already_armed';
    captureArmed = true;
    console.log('[Capture] __startCapture() — pendingTrack:', !!pendingTrack);
    if (pendingTrack && !captureStarted) {
      captureStarted = true;
      startCapture(pendingTrack);
      return 'started';
    }
    return 'armed_waiting_track';
  };

  const _OrigPC = window.RTCPeerConnection;
  window.RTCPeerConnection = function(...args) {
    const pc = new _OrigPC(...args);
    pc.addEventListener('track', (event) => {
      if (event.track.kind === 'audio') attachTrack(event.track);
    });
    return pc;
  };
  Object.assign(window.RTCPeerConnection, _OrigPC);

  // ── Watchdog: reiniciar captura si lleva >30 s sin enviar audio ────────
  setInterval(() => {
    if (lastSentTime === null) return;   // aún no se envió ningún utterance
    const elapsed = Date.now() - lastSentTime;
    if (elapsed < 30000) return;
    if (pendingTrack && pendingTrack.readyState === 'live') {
      console.warn('[Capture] watchdog — sin audio por', (elapsed/1000).toFixed(0) + 's, reiniciando captura');
      // Resetear estado VAD para empezar limpio
      speechBuf    = [];
      inSpeech     = false;
      silenceFrames = 0;
      frameBuf     = new Float32Array(0);
      startCapture(pendingTrack);
      lastSentTime = Date.now();   // evitar re-trigger inmediato
    } else {
      console.warn('[Capture] track ended — sin audio');
    }
  }, 30000);

  console.log('[Capture] script VAD iniciado — wsUrl:', wsUrl);
}
"""

# ── Captura de audio vía sounddevice (VB-Cable) ───────────────────────────────
# Zoom reproduce audio por VB-Audio Virtual Cable. sounddevice captura ese
# dispositivo como entrada y envía los frames PCM int16 a 16 kHz al WS interno.
#
# VB-Cable soporta 48 kHz pero NO 16 kHz. Se captura a 48 kHz y se resamplea
# 3:1 (promedio de grupos de 3) a 16 kHz antes de enviar al pipeline Whisper.
_CAPTURE_DEVICE_CANDIDATES = [2, 9, 17]    # índices VB-Cable a intentar en orden
_CAPTURE_RATE_DEVICE  = 48000              # Hz nativo VB-Cable (Internal SR = 48000 Hz)
_CAPTURE_RATE_WHISPER = 16000              # Hz requerido por Whisper
# 48000 → 16000: ratio racional 1/3 (gcd=16000); se usa resample_poly
_CAPTURE_RESAMPLE_UP  = 1                  # 16000 / 16000
_CAPTURE_RESAMPLE_DOWN = 3                 # 48000 / 16000
_CAPTURE_CHANNELS     = 1                  # mono
_CAPTURE_BLOCKSIZE    = 4800               # 100 ms a 48000 Hz → 1600 muestras a 16 kHz


class ZoomBot:
    def __init__(self, audio_ws_url: str = 'ws://localhost:8765'):
        self.audio_ws_url  = audio_ws_url
        self._playwright   = None
        self._browser      = None
        self._owns_browser = True   # False si el browser fue provisto externamente
        self._context: Optional[BrowserContext] = None
        self._page:    Optional[Page]           = None
        self._monitoring   = False

        # Callbacks: asignar desde app.py antes de llamar a join()
        self.on_status:        Optional[Callable[[str], None]]        = None  # mensaje de log
        self.on_status_change: Optional[Callable[[str, str], None]]   = None  # (estado, mensaje)

    # ── Helpers de estado ──────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        logger.info(msg)
        if self.on_status:
            self.on_status(msg)

    def _set_state(self, state: str, msg: str) -> None:
        logger.info('[%s] %s', state.upper(), msg)
        if self.on_status:
            self.on_status(msg)
        if self.on_status_change:
            self.on_status_change(state, msg)

    # ── Entrada pública ────────────────────────────────────────────────────

    @staticmethod
    def _build_url(meeting_url_or_id: str, display_name: str, mute_mic: bool = True) -> str:
        """Construye la URL del web client directo, evitando la página /j/ intermedia.

        La página /j/ muestra el dialog nativo "¿Abrir Zoom Meetings?" y el botón
        "Join from browser". Ir directo a app.zoom.us/wc/ID/join salta todo eso.

        Acepta:
          - URL completa:  https://us02web.zoom.us/j/87818853738[?pwd=abc]
          - URL web client: https://zoom.us/wc/87818853738/join
          - Solo dígitos:  87818853738
        """
        # mv=0, video=0 → cámara desactivada
        # av=0  → micrófono muteado al entrar (solo Windows: en Linux puede bloquear
        #          la recepción de audio de otros participantes via WebRTC)
        # audio=0 eliminado — desactiva también el audio ENTRANTE.
        av_param = '&av=0' if mute_mic else ''
        params = f'prefer=0&name={quote(display_name)}{av_param}&mv=0&video=0'

        if meeting_url_or_id.startswith('http'):
            # Extraer el ID numérico de /j/ID o /wc/ID/
            m = re.search(r'/(?:j|wc)/(\d+)', meeting_url_or_id)
            meeting_id = m.group(1) if m else re.sub(r'\D', '', meeting_url_or_id)
            # Preservar el pwd si viene en la URL original
            pwd = re.search(r'[?&]pwd=([^&]+)', meeting_url_or_id)
            if pwd:
                params += f'&pwd={pwd.group(1)}'
        else:
            meeting_id = re.sub(r'\D', '', meeting_url_or_id)

        # Ir directo al web client — sin page /j/, sin dialog, sin "Join from browser"
        return f'https://app.zoom.us/wc/{meeting_id}/join?{params}'

    async def join(self, meeting_url_or_id: str, display_name: str = 'tradutor-zoom',
                   *, _browser=None) -> None:
        """Entra a la reunión Zoom.

        Args:
            _browser: Browser de Playwright ya creado (compartido). Si se provee,
                      este bot NO lo cierra al hacer leave(). Si es None, crea el suyo.
        """
        label = meeting_url_or_id if meeting_url_or_id.startswith('http') \
                else re.sub(r'\D', '', meeting_url_or_id)
        self._set_state(CONNECTING, f'Iniciando browser (reunión {label})')

        if _browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=_HEADLESS,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--use-fake-ui-for-media-stream',
                    '--use-fake-device-for-media-stream',
                    '--autoplay-policy=no-user-gesture-required',
                    '--allow-running-insecure-content',
                    '--disable-features=ExternalProtocolDialog,WebRtcHideLocalIpsWithMdns,PrivateNetworkAccessChecks',
                    '--unsafely-treat-insecure-origin-as-secure=http://localhost:8765,ws://localhost:8765',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                ],
            )
            self._owns_browser = True
        else:
            # Browser compartido: este contexto es EXCLUSIVO de Zoom.
            # _CAPTURE_SCRIPT se registra aquí y SOLO aquí — VDO.Ninja
            # usa otro new_context() donde ese script nunca se instala.
            self._playwright = None
            self._browser = _browser
            self._owns_browser = False
            logger.info('[ZoomBot] usando browser compartido — contexto Zoom aislado de VDO.Ninja')

        self._context = await self._browser.new_context(
            user_agent=(
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/124.0.0.0 Safari/537.36'
            ),
            viewport={'width': 1280, 'height': 720},
            # Permisos necesarios para que WebRTC funcione correctamente.
            # El bot NO transmite su propio audio/video — eso se controla con
            # &av=0&mv=0 en la URL de Zoom, no bloqueando permisos del browser.
            # Denegar microphone aquí impide que Zoom inicialice su subsistema
            # de audio completo, lo que resulta en 0 receivers y 0 tracks.
            permissions=['microphone', 'camera', 'notifications'],
        )

        await self._context.add_init_script(f'({_STEALTH_SCRIPT})()')

        # En Linux (Railway): inyectar _CAPTURE_SCRIPT como init_script para que
        # intercepte RTCPeerConnection ANTES de que Zoom lo inicialice.
        # El script captura los tracks de audio de los participantes via WebRTC
        # y los envía como webm/opus al WebSocket interno del servidor.
        if _platform.system() != 'Windows':
            logger.info('[ZoomBot] Linux detectado — inyectando _CAPTURE_SCRIPT para captura WebRTC')
            await self._context.add_init_script(
                f'({_CAPTURE_SCRIPT})({repr(self.audio_ws_url)})'
            )

        self._page = await self._context.new_page()

        # Reenviar consola del browser a Python (especialmente para diagnóstico
        # de _CAPTURE_SCRIPT en Linux donde la captura ocurre en JS)
        def _fwd_console(msg):
            lvl = {'error': logger.error, 'warning': logger.warning}.get(msg.type, logger.info)
            lvl('[ZoomBot-browser] %s', msg.text)
        self._page.on('console', _fwd_console)

        # Registrar handler ANTES de goto() para capturar cualquier dialog
        # JS (alert/confirm/prompt) que aparezca durante la navegación
        self._page.on('dialog', lambda d: asyncio.ensure_future(d.dismiss()))

        # En Linux no usar &av=0 — puede bloquear el audio entrante de otros
        # participantes via WebRTC (tracks de audio que necesita _CAPTURE_SCRIPT)
        url = self._build_url(meeting_url_or_id, display_name,
                              mute_mic=(_platform.system() == 'Windows'))
        self._log(f'Navegando directo al web client: {url}')
        await self._page.goto(url, wait_until='domcontentloaded', timeout=30_000)
        self._log(f'URL actual: {self._page.url} | Título: {await self._page.title()}')

        # ── Pipeline: web client directo, sin page /j/ intermedia ──────────
        # Al ir a app.zoom.us/wc/ID/join no aparece el dialog "Abrir Zoom"
        # ni el botón "Join from browser" — se va directo al formulario.
        await self._dismiss_cookies()          # cookie banner (si aparece)
        await self._fill_join_form(display_name)
        await self._handle_post_join()

    # ── Paso 1: dialog "¿Abrir Zoom Meetings?" ────────────────────────────
    #
    # Puede ser un dialog nativo del browser (protocolo zoommtg://) o un modal
    # DOM de Zoom. El nativo ya está cubierto por page.on('dialog').
    # Este método cubre el caso del modal DOM, con timeout corto para no bloquear.

    async def _dismiss_open_app_popup(self) -> None:
        p = self._page
        try:
            await p.screenshot(path='debug_01_landing.png')
            self._log('Screenshot landing: debug_01_landing.png')
        except Exception:
            pass

        for sel in (
            'button:has-text("Cancelar")',
            'button:has-text("Cancel")',
            'a:has-text("Cancelar")',
            'a:has-text("Cancel")',
        ):
            try:
                # Timeout corto: si no aparece en 3s, probablemente no existe
                await p.wait_for_selector(sel, state='visible', timeout=3_000)
                await p.click(sel)
                self._log(f'Dialog "Abrir Zoom" cerrado ({sel!r})')
                await asyncio.sleep(0.5)
                return
            except Exception:
                continue

        self._log('Sin dialog "Abrir Zoom" — continuando')

    # ── Paso 2: banner de cookies "ACCEPT COOKIES" ─────────────────────────
    #
    # El banner de OneTrust en zoom.us muestra un botón "ACCEPT COOKIES"
    # (texto en mayúsculas). Debe cerrarse ANTES de interactuar con la página.

    async def _dismiss_cookies(self) -> None:
        p = self._page
        # Selector exacto visible en el screenshot: botón "ACCEPT COOKIES"
        # Probamos en orden de certeza: ID de OneTrust primero, luego texto exacto.
        for sel in (
            '#onetrust-accept-btn-handler',
            'button:has-text("ACCEPT COOKIES")',
            'button:has-text("Accept All Cookies")',
            'button:has-text("Accept Cookies")',
            'button:has-text("Accept All")',
        ):
            try:
                await p.wait_for_selector(sel, state='visible', timeout=6_000)
                await p.click(sel)
                self._log(f'Cookies aceptadas ({sel!r})')
                # Esperar a que el banner desaparezca
                try:
                    await p.wait_for_selector('#onetrust-banner-sdk', state='hidden', timeout=5_000)
                except Exception:
                    try:
                        await p.wait_for_selector('#onetrust-consent-sdk', state='hidden', timeout=3_000)
                    except Exception:
                        await asyncio.sleep(1.5)
                try:
                    await p.screenshot(path='debug_02_cookies_dismissed.png')
                    self._log('Screenshot post-cookies: debug_02_cookies_dismissed.png')
                except Exception:
                    pass
                return
            except Exception:
                continue
        self._log('Sin banner de cookies')

    # ── Paso 3: botón "Join from browser" ──────────────────────────────────
    #
    # Selector exacto visible en el screenshot: button con texto "Join from browser"

    async def _click_join_from_browser(self) -> None:
        p = self._page
        try:
            await p.screenshot(path='debug_03_before_join_browser.png')
            self._log('Screenshot pre-join-browser: debug_03_before_join_browser.png')
        except Exception:
            pass

        # Hay DOS botones "Join from browser" en la página — necesitamos el SEGUNDO (.nth(1))
        for text in (
            'Join from browser',
            'Join from Browser',
            'Join from Your Browser',
        ):
            try:
                loc = p.locator(f'button:has-text("{text}")')
                count = await loc.count()
                self._log(f'Botones "{text}" encontrados: {count}')
                # Usar el segundo si hay 2+, el primero si solo hay 1
                target = loc.nth(1) if count >= 2 else loc.first
                await target.wait_for(state='visible', timeout=8_000)
                await target.click()
                self._log(f'Click "Join from browser" (índice {"1" if count >= 2 else "0"}, texto: {text!r})')
                # Esperar que la navegación al web client complete
                await p.wait_for_load_state('domcontentloaded', timeout=20_000)
                self._log(f'Web client URL: {p.url}')
                try:
                    await p.screenshot(path='debug_04_web_client.png')
                    self._log('Screenshot web client: debug_04_web_client.png')
                except Exception:
                    pass
                # El web client puede mostrar otro banner de cookies
                await self._dismiss_cookies()
                return
            except Exception:
                continue

        self._log('WARN: "Join from browser" no encontrado')
        try:
            await p.screenshot(path='debug_03_no_join_browser.png')
        except Exception:
            pass

    # ── Paso 3: formulario "Enter Meeting Info" ────────────────────────────

    # Selectores del campo nombre (orden de especificidad descendente)
    _NAME_SELS = (
        '#inputname',
        'input[placeholder="Your Name"]',
        'input[placeholder="Your name"]',
        'input[placeholder="Name"]',
        'input[placeholder*="Your Name" i]',
        'input[placeholder*="name" i][type="text"]',
        '.preview-join-info input[type="text"]',
        '.preview-join-info input',
        'input[type="text"]',
    )

    # Selectores del campo passcode (para limpiarlo explícitamente)
    _PASS_SELS = (
        '#input-for-pwd',
        'input[placeholder*="Passcode" i]',
        'input[placeholder*="Password" i]',
        'input[type="password"]',
        'input[id*="pwd" i]',
        'input[id*="pass" i]',
    )

    # Selectores del botón Join
    _JOIN_SELS = (
        'button.preview-join-button',
        'button[class*="join-btn"]',
        'button[class*="joinBtn"]',
        'button[class*="btn-join"]',
        'button[class*="join"]',
        '.preview-join-info button',
        '.preview-join-info .btn',
    )

    async def _fill_join_form(self, display_name: str) -> None:
        p = self._page
        self._log('Buscando formulario "Enter Meeting Info"...')
        self._log(f'URL: {p.url} | Título: {await p.title()}')

        # Screenshot de diagnóstico: captura el estado del formulario
        try:
            await p.screenshot(path='debug_form.png')
            self._log('Screenshot guardado: debug_form.png')
        except Exception as e:
            self._log(f'Screenshot fallido: {e}')

        # ── 0. Inspeccionar HTML real del formulario ─────────────────────────
        # Loguear atributos exactos de los inputs para diagnóstico
        try:
            form_info = await p.evaluate('''() => {
                const inputs = Array.from(document.querySelectorAll("input"));
                return inputs.map(el => ({
                    id: el.id,
                    name: el.name,
                    type: el.type,
                    placeholder: el.placeholder,
                    value: el.value,
                    disabled: el.disabled,
                    className: el.className.slice(0, 80),
                    ariaLabel: el.getAttribute("aria-label"),
                    tabIndex: el.tabIndex,
                }));
            }''')
            self._log(f'Inputs en el formulario: {form_info}')
        except Exception as exc:
            self._log(f'Inspect inputs falló: {exc}')

        # ── 1. Campo nombre ──────────────────────────────────────────────────
        # Zoom usa React — fill() y keyboard.type() no actualizan el estado
        # interno. Estrategia en cascada:
        #   A) role="textbox" + name="Your Name" (Playwright locator semántico)
        #   B) Tab desde Passcode hasta llegar al campo nombre + keyboard.type()
        #   C) Click directo en Join (Zoom toma el nombre de &name= en la URL)
        name_found = False

        # ── Intento A: locator por role + accessible name ──────────────────
        for role_name in ('Your Name', 'Your name', 'Name', 'Nombre'):
            try:
                loc = p.get_by_role('textbox', name=role_name)
                await loc.wait_for(state='visible', timeout=8_000)
                await loc.click()
                await p.keyboard.press('Control+a')
                await p.keyboard.press('Delete')
                await p.keyboard.type(display_name, delay=60)
                await asyncio.sleep(0.3)
                val = await loc.input_value()
                if val.strip():
                    self._log(f'Nombre con locator role=textbox name={role_name!r}: "{val}"')
                    name_found = True
                    break
                # Si sigue vacío, intento React nativeInputValueSetter
                set_result = await p.evaluate(
                    '''([roleName, value]) => {
                        const el = [...document.querySelectorAll("input")]
                            .find(i => (i.placeholder || i.getAttribute("aria-label") || "")
                                        .toLowerCase().includes(roleName.toLowerCase()));
                        if (!el) return "not_found:" + roleName;
                        const nativeSet = Object.getOwnPropertyDescriptor(
                            window.HTMLInputElement.prototype, "value"
                        ).set;
                        nativeSet.call(el, value);
                        el.dispatchEvent(new Event("input",  { bubbles: true }));
                        el.dispatchEvent(new Event("change", { bubbles: true }));
                        el.dispatchEvent(new InputEvent("input", { bubbles: true, data: value }));
                        return el.value;
                    }''',
                    [role_name, display_name],
                )
                await asyncio.sleep(0.3)
                val = await loc.input_value()
                if val.strip():
                    self._log(f'Nombre con React setter (role={role_name!r}): "{val}"')
                    name_found = True
                    break
                self._log(f'A falló para role_name={role_name!r}, set_result={set_result!r}')
            except Exception as exc:
                self._log(f'Intento A role={role_name!r}: {exc}')
                continue

        # ── Intento B: Tab desde Passcode ──────────────────────────────────
        if not name_found:
            self._log('Intento B: Tab desde Passcode hacia Your Name')
            try:
                # Hacer click en el campo passcode (primer input de tipo password o con "pwd")
                pass_sel = None
                for sel in self._PASS_SELS:
                    el = await p.query_selector(sel)
                    if el:
                        pass_sel = sel
                        break
                if pass_sel:
                    await p.click(pass_sel)
                    self._log(f'Click en Passcode ({pass_sel!r}) — navegando con Tab')
                    # Tab hacia atrás (Shift+Tab) suele ir al campo nombre
                    await p.keyboard.press('Shift+Tab')
                    await asyncio.sleep(0.2)
                    # Verificar qué elemento tiene foco
                    focused_tag = await p.evaluate(
                        '() => ({ tag: document.activeElement.tagName, '
                        'id: document.activeElement.id, '
                        'placeholder: document.activeElement.placeholder })'
                    )
                    self._log(f'Foco tras Shift+Tab: {focused_tag}')
                    # Escribir en el elemento enfocado
                    await p.keyboard.press('Control+a')
                    await p.keyboard.press('Delete')
                    await p.keyboard.type(display_name, delay=60)
                    await asyncio.sleep(0.3)
                    val = await p.evaluate('() => document.activeElement.value')
                    if str(val).strip():
                        self._log(f'Nombre con Tab+type(): "{val}"')
                        name_found = True
                    else:
                        self._log(f'Tab+type() sigue vacío: "{val}"')
            except Exception as exc:
                self._log(f'Intento B (Tab) falló: {exc}')

        try:
            await p.screenshot(path='debug_form_after_name.png')
            self._log('Screenshot post-nombre: debug_form_after_name.png')
        except Exception:
            pass

        if not name_found:
            self._log('Todos los métodos fallaron — intentando Join directo (nombre en &name= URL)')

        # ── 2. Campo passcode — LIMPIAR explícitamente ───────────────────────
        # Zoom a veces pre-rellena el passcode desde localStorage o URL params.
        # Triple-click selecciona todo; Delete borra; fill('') garantiza vacío.
        for sel in self._PASS_SELS:
            try:
                el = await p.query_selector(sel)
                if el is None:
                    continue
                await p.click(sel, click_count=3)   # seleccionar todo el texto
                await p.keyboard.press('Delete')     # borrar
                await p.fill(sel, '')                # confirmar vacío con fill
                self._log(f'Passcode limpiado ({sel!r})')
                break
            except Exception:
                continue

        # ── 3. Botón Join ────────────────────────────────────────────────────
        # Si el nombre no pudo escribirse, intentar Join de todas formas:
        # Zoom a veces acepta el &name= de la URL y habilita el botón.
        await self._click_join(name_found=name_found)

        # ── 4. Manejar error de passcode incorrecto (retry) ──────────────────
        await self._retry_if_passcode_error()

    # Selectores del botón Join habilitado (sin :disabled)
    _JOIN_ENABLED_SELS = (
        # Clases específicas de Zoom
        'button.preview-join-button:not([disabled])',
        'button[class*="join-btn"]:not([disabled])',
        'button[class*="joinBtn"]:not([disabled])',
        'button[class*="btn-join"]:not([disabled])',
        'button[class*="join"]:not([disabled])',
        # Contenedor del formulario
        '.preview-join-info button:not([disabled])',
    )

    async def _click_join(self, name_found: bool = True) -> bool:
        """Espera a que el botón Join esté habilitado, toma screenshot y hace click."""
        p = self._page
        self._log(f'URL antes de Join: {p.url} | name_found={name_found}')

        # Buscar el primer botón Join habilitado
        found_sel = None
        for sel in self._JOIN_ENABLED_SELS:
            try:
                await p.wait_for_selector(sel, state='visible', timeout=6_000)
                found_sel = sel
                break
            except Exception:
                continue

        # Fallback: button:has-text("Join") habilitado
        if found_sel is None:
            try:
                loc = p.locator('button:has-text("Join"):not([disabled])')
                await loc.first.wait_for(state='visible', timeout=4_000)
                found_sel = 'button:has-text("Join"):not([disabled])'
            except Exception:
                pass

        # Fallback: get_by_role con texto exacto
        if found_sel is None:
            for text in ('Join', 'Unirse', 'Join Meeting', 'Unirse a la reunión'):
                try:
                    btn = p.get_by_role('button', name=text)
                    await btn.wait_for(state='visible', timeout=3_000)
                    # Verificar que no esté deshabilitado
                    disabled = await btn.get_attribute('disabled')
                    if disabled is None:
                        found_sel = f'role=button[name="{text}"]'
                        break
                except Exception:
                    continue

        if found_sel is None and not name_found:
            # Nombre en URL (&name=...) — intentar con cualquier botón Join visible,
            # aunque esté deshabilitado (Zoom puede habilitarlo al detectar el nombre de la URL)
            self._log('WARN: Join habilitado no encontrado — intentando click en cualquier botón Join visible')
            for sel in self._JOIN_SELS:
                try:
                    await p.wait_for_selector(sel, state='visible', timeout=3_000)
                    await p.click(sel, force=True)   # force=True ignora disabled
                    self._log(f'Join forzado ({sel!r})')
                    try:
                        await p.screenshot(path='debug_after_forced_join.png')
                    except Exception:
                        pass
                    return True
                except Exception:
                    continue

        if found_sel is None:
            self._log('WARN: botón Join habilitado no encontrado — screenshot de diagnóstico')
            try:
                await p.screenshot(path='debug_nojoin.png')
            except Exception:
                pass
            return False

        # Screenshot justo antes de hacer click para confirmar que el nombre está escrito
        try:
            await p.screenshot(path='debug_before_join.png')
            self._log('Screenshot pre-join guardado: debug_before_join.png')
        except Exception:
            pass

        # Hacer click
        try:
            if found_sel.startswith('role='):
                name = found_sel.split('"')[1]
                await p.get_by_role('button', name=name).click(timeout=4_000)
            elif found_sel == 'button:has-text("Join"):not([disabled])':
                await p.locator('button:has-text("Join"):not([disabled])').first.click(timeout=4_000)
            else:
                await p.click(found_sel, timeout=4_000)
            self._log(f'Join presionado ({found_sel!r})')
            await asyncio.sleep(0.5)
            try:
                await p.screenshot(path='debug_after_join.png')
                self._log('Screenshot post-join: debug_after_join.png')
            except Exception:
                pass
            return True
        except Exception as e:
            self._log(f'WARN: click Join fallido ({found_sel!r}): {e}')
            try:
                await p.screenshot(path='debug_join_error.png')
            except Exception:
                pass
            return False

    async def _retry_if_passcode_error(self) -> None:
        """Si Zoom muestra error de passcode incorrecto, limpia el campo y reintenta."""
        p = self._page
        error_sels = (
            '[class*="error"]:text("passcode")',
            '[class*="error"]:text("password")',
            '[class*="error"]:text("incorrect")',
            'p:text-is("Please enter the meeting passcode")',
            'p:text("Incorrect passcode")',
            'p:text("Wrong passcode")',
            '[class*="error-msg"]',
        )

        # Esperar brevemente — si no hay error, continuar normalmente
        error_found = False
        for sel in error_sels:
            try:
                await p.wait_for_selector(sel, timeout=3_000)
                self._log(f'Error de passcode detectado ({sel!r}) — limpiando y reintentando')
                error_found = True
                break
            except Exception:
                continue

        if not error_found:
            return

        # Limpiar passcode con todos los métodos posibles y reintentar
        for sel in self._PASS_SELS:
            try:
                el = await p.query_selector(sel)
                if el is None:
                    continue
                await p.click(sel, click_count=3)
                await p.keyboard.press('Delete')
                await p.fill(sel, '')
                self._log(f'Passcode re-limpiado en retry ({sel!r})')
                break
            except Exception:
                continue

        await asyncio.sleep(0.3)
        await self._click_join()

    # ── Paso 4: después del Join ───────────────────────────────────────────

    async def _handle_post_join(self) -> None:
        p = self._page
        try:
            title = await p.title()
        except Exception:
            title = '?'
        self._log(f'Post-join URL: {p.url} | Título: {title}')

        # — Sala de espera —
        wr_sel = (
            '[class*="waiting-room"], [class*="waitingRoom"], '
            'p:text-is("Please wait"), p:text("waiting for the host")'
        )
        try:
            await p.wait_for_selector(wr_sel, timeout=5_000)
            self._set_state(WAITING_ROOM, 'En sala de espera — aguardando al anfitrión...')
            await p.wait_for_selector(wr_sel, state='hidden', timeout=300_000)
            self._log('Anfitrión admitió al bot')
        except Exception:
            pass  # no hay sala de espera

        # — "Join with Computer Audio" —
        audio_joined = False
        for sel in (
            'button[class*="join-audio-by-voip"]',
            'button[class*="joinAudioByVoip"]',
        ):
            try:
                await p.wait_for_selector(sel, timeout=10_000)
                await p.click(sel)
                self._log(f'Audio activado (clase: {sel!r})')
                audio_joined = True
                break
            except Exception:
                continue

        if not audio_joined:
            for text in (
                'Join with Computer Audio',
                'Join Audio by Computer',
                'Unirse con audio de computadora',
                'Unirse al audio de la computadora',
            ):
                try:
                    await p.get_by_role('button', name=text).click(timeout=4_000)
                    self._log(f'Audio activado (texto: {text!r})')
                    audio_joined = True
                    break
                except Exception:
                    continue

        if not audio_joined:
            self._log('WARN: diálogo de audio no encontrado')

        try:
            title2 = await p.title()
        except Exception:
            title2 = '?'
        self._log(f'URL tras audio join: {p.url} | Título: {title2}')

        # Esperar que la UI de la reunión termine de renderizar
        await asyncio.sleep(3)

        # — Silenciar micrófono: el bot NO debe transmitir audio a Zoom ────────
        # &av=0 en la URL ya solicita entrar silenciado, pero Zoom puede ignorarlo.
        # Alt+A es el atajo de teclado nativo de Zoom para mute/unmute.
        # Hacemos dos intentos y verificamos el estado del botón de micrófono.
        muted = False
        for attempt in range(2):
            try:
                await p.keyboard.press('Alt+a')
                await asyncio.sleep(0.5)
                # Verificar si el botón de mic muestra estado "muted"
                mic_state = await p.evaluate('''() => {
                    const btn = document.querySelector(
                        '[class*="audio-button"], [aria-label*="mute" i], [aria-label*="mic" i]'
                    );
                    if (!btn) return "btn_not_found";
                    return btn.getAttribute("aria-label") || btn.getAttribute("title") || "found";
                }''')
                self._log(f'Mic mute intento {attempt+1}: botón aria-label={mic_state!r}')
                # Si el botón dice "unmute" o "start audio" → ya está muteado
                ml = mic_state.lower()
                if any(w in ml for w in ('unmute', 'start audio', 'unsilence', 'silenciado')):
                    muted = True
                    break
            except Exception as e:
                self._log(f'Mic mute intento {attempt+1} falló: {e}')
        self._log(f'Estado del micrófono: {"MUTEADO ✓" if muted else "estado desconocido (probablemente muteado por &av=0)"}')

        # — Iniciar captura + monitor + heartbeat ────────────────────────────
        if _platform.system() == 'Windows':
            self._set_state(IN_MEETING, 'Dentro de la reunión — iniciando captura VB-Cable')
            self._monitoring = True
            asyncio.create_task(self._capture_audio())
        else:
            self._set_state(IN_MEETING, 'Dentro de la reunión — audio via WebRTC (servidor)')
            self._monitoring = True
            asyncio.create_task(self._capture_audio_playwright())
        asyncio.create_task(self._monitor_meeting())
        asyncio.create_task(self._heartbeat())

    # ── Captura de audio (sounddevice → WebSocket interno) ────────────────

    async def _capture_audio(self) -> None:
        """Captura PCM de VB-Cable a 48 kHz, resamplea 3:1 a 16 kHz y envía al WS.

        Flujo:
          sounddevice callback (audio thread) →  stdlib Queue (thread-safe)
          loop asyncio → drena Queue → resamplea 48k→16k → envía bytes al WS

        Fallback de dispositivo: prueba _CAPTURE_DEVICE_CANDIDATES en orden;
        usa el primero que acepte 48 kHz de entrada.
        """
        # ── Seleccionar dispositivo VB-Cable (Windows) ────────────────────
        device_idx  = None
        device_name = '?'
        for candidate in _CAPTURE_DEVICE_CANDIDATES:
            try:
                sd.check_input_settings(
                    device=candidate,
                    channels=_CAPTURE_CHANNELS,
                    dtype='float32',
                    samplerate=_CAPTURE_RATE_DEVICE,
                )
                device_idx  = candidate
                device_name = sd.query_devices(candidate)['name']
                logger.info('[Capture] dispositivo seleccionado: %d — %s', candidate, device_name)
                break
            except Exception as e:
                logger.warning('[Capture] dispositivo %d no disponible (%s) — probando siguiente',
                               candidate, e)

        if device_idx is None:
            logger.error('[Capture] ningún dispositivo VB-Cable disponible en índices %s — abortando',
                         _CAPTURE_DEVICE_CANDIDATES)
            return

        raw_q: stdlib_queue.Queue = stdlib_queue.Queue(maxsize=400)

        def _sd_callback(indata, frames, time_info, status):
            if status:
                logger.warning('[Capture] sounddevice status: %s', status)
            mono = indata[:, 0].copy()
            # Resamplear 48000 → 16000 Hz con anti-aliasing correcto (ratio 1/3)
            down = resample_poly(mono, _CAPTURE_RESAMPLE_UP, _CAPTURE_RESAMPLE_DOWN)
            # Amplificar ×10 para compensar el bajo volumen de VB-Cable
            down = down * 10.0
            i16  = (np.clip(down, -1.0, 1.0) * 32767).astype(np.int16)
            if not raw_q.full():
                raw_q.put_nowait(i16.tobytes())

        ws_conn    = None
        frame_count = 0
        rms_log_acc = 0   # acumulador para log periódico de RMS (diagnóstico VB-Cable)

        try:
            with sd.InputStream(
                device=device_idx,
                samplerate=_CAPTURE_RATE_DEVICE,
                channels=_CAPTURE_CHANNELS,
                dtype='float32',
                callback=_sd_callback,
                blocksize=_CAPTURE_BLOCKSIZE,
            ):
                logger.info('[Capture] InputStream abierto ✓ — %dHz→%dHz (up=%d/down=%d) block=%d',
                            _CAPTURE_RATE_DEVICE, _CAPTURE_RATE_WHISPER,
                            _CAPTURE_RESAMPLE_UP, _CAPTURE_RESAMPLE_DOWN, _CAPTURE_BLOCKSIZE)
                logger.info('[Capture] IMPORTANTE: verificá que Zoom tiene el audio '
                            'activo y la salida de audio apunta a VB-Cable')

                while self._monitoring:
                    # Conectar / reconectar WebSocket
                    if ws_conn is None:
                        try:
                            ws_conn = await ws_lib.connect(
                                self.audio_ws_url, ping_interval=None
                            )
                            logger.info('[Capture] WS conectado: %s', self.audio_ws_url)
                        except Exception as e:
                            logger.warning('[Capture] WS no disponible: %s — reintentando en 2 s', e)
                            await asyncio.sleep(2)
                            continue

                    # Drenar la queue y enviar frames al WS
                    while not raw_q.empty() and self._monitoring:
                        try:
                            data = raw_q.get_nowait()
                        except stdlib_queue.Empty:
                            break
                        try:
                            await ws_conn.send(data)
                            frame_count += 1
                            # Log periódico: frame count + RMS para diagnosticar si
                            # VB-Cable está recibiendo audio de Zoom
                            if frame_count % 10 == 1:
                                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                                rms = float(np.sqrt(np.mean(samples ** 2)))
                                logger.info('[Capture] frame #%d — %d bytes | RMS=%.0f | queue=%d',
                                            frame_count, len(data), rms, raw_q.qsize())
                                if rms < 50:
                                    logger.warning('[Capture] RMS muy bajo (%.0f) — '
                                                   'verificá que Zoom reproduce audio por VB-Cable '
                                                   'y que el volumen no está en cero', rms)
                        except Exception as e:
                            logger.warning('[Capture] WS send error: %s — reconectando', e)
                            try:
                                await ws_conn.close()
                            except Exception:
                                pass
                            ws_conn = None
                            break

                    await asyncio.sleep(0.05)   # 50 ms entre drains

        except Exception as e:
            logger.error('[Capture] error fatal: %s', e, exc_info=True)
        finally:
            if ws_conn:
                try:
                    await ws_conn.close()
                except Exception:
                    pass
            logger.info('[Capture] detenido — %d frames enviados', frame_count)

    async def _capture_audio_playwright(self) -> None:
        """Linux/Railway: arma el VAD AudioContext en la página Zoom vía __startCapture().

        __startCapture() activa el _CAPTURE_SCRIPT inyectado como init_script.
        El script intercepta el RTCPeerConnection, toma el track de audio entrante,
        aplica VAD (idéntico a capture_client.py) y envía utterances PCM int16 al
        WebSocket interno (port 8765). Solo se arma aquí, una vez confirmado IN_MEETING,
        para no capturar audio del join flow ni de la sala de espera.
        """
        logger.info('[Capture] IN_MEETING confirmado — llamando __startCapture() en la página')
        try:
            if self._page and not self._page.is_closed():
                result = await self._page.evaluate(
                    '() => window.__startCapture && window.__startCapture()'
                )
                logger.info('[Capture] __startCapture() resultado: %s', result)
        except Exception as e:
            logger.warning('[Capture] __startCapture() error: %s', e)

        while self._monitoring:
            await asyncio.sleep(10)
        logger.info('[Capture] tarea Web Audio API finalizada')

    # ── Heartbeat ──────────────────────────────────────────────────────────

    async def _heartbeat(self) -> None:
        """Mueve el mouse cada 30 s para que Zoom no detecte inactividad.

        Zoom cierra la sesión web si no detecta actividad del usuario durante
        un tiempo (lanza 'context or browser has been closed').
        Un micro-movimiento del mouse es suficiente para reactivar el timer.
        """
        p = self._page
        x, y = 640, 360
        while self._monitoring:
            await asyncio.sleep(30)
            if not self._monitoring or p is None or p.is_closed():
                break
            try:
                # Micro-movimiento: 2 px en diagonal y vuelta, no interacciona con la UI
                await p.mouse.move(x + 2, y + 2)
                await asyncio.sleep(0.1)
                await p.mouse.move(x, y)
                logger.debug('[Heartbeat] mouse move OK')
            except Exception as e:
                logger.warning('[Heartbeat] error (ignorado): %s', e)

    # ── Monitor de reunión ─────────────────────────────────────────────────

    # Selectores de elementos DOM VISIBLES que Zoom muestra al finalizar la reunión.
    # NO usar p.content() porque el bundle JS de Zoom contiene estas frases como
    # strings de mensajes de error aunque la reunión esté perfectamente activa,
    # lo que provoca falsos positivos a los 5 segundos.
    _END_SELECTORS = (
        '[class*="meeting-end-dialog"]',
        '[class*="mtg-end"]',
        '[class*="meeting-over"]',
        # Textos exactos del overlay de fin de reunión
        'div:text-is("This meeting has been ended by the host")',
        'div:text-is("This meeting has ended")',
        'div:text-is("La reunión ha finalizado")',
        'h3:text-is("Meeting Over")',
        'p:text-is("You have been removed from the meeting")',
        'p:text-is("The host has ended the meeting")',
    )

    async def _monitor_meeting(self) -> None:
        """Detecta fin de reunión o expulsión y notifica vía callback.

        Usa contadores de gracia para evitar falsos positivos por errores
        transitorios de Playwright (p.ej. 'context or browser has been closed'
        durante una recarga momentánea de Zoom):
          - closed_count: 3 checks consecutivos con is_closed()=True → DISCONNECTED
          - url_outside_wc_count: 3 checks consecutivos fuera de /wc/ → DISCONNECTED
        Intervalo: 10 s (menos agresivo que 5 s).
        """
        p = self._page
        check_count = 0
        closed_count = 0
        url_outside_wc_count = 0

        while self._monitoring:
            await asyncio.sleep(10)
            check_count += 1

            if p is None:
                self._set_state(DISCONNECTED, 'El browser se cerró inesperadamente')
                self._monitoring = False
                break

            # ── Gracia por page cerrada (transitorio vs. definitivo) ─────────
            if p.is_closed():
                closed_count += 1
                logger.warning('[Monitor] page.is_closed()=True (count=%d/3)', closed_count)
                if closed_count >= 3:
                    self._set_state(DISCONNECTED, 'El browser se cerró inesperadamente')
                    self._monitoring = False
                    break
                continue  # esperar siguiente tick; no hacer más checks esta vuelta
            else:
                closed_count = 0  # reset si la página volvió a estar disponible

            try:
                # ── 1. Verificar elementos VISIBLES de fin de reunión ────────────
                # Se busca en el DOM un elemento que Zoom renderiza SOLO cuando la
                # reunión realmente terminó. Esto evita falsos positivos por strings
                # embebidos en el bundle JS.
                ended = False
                for sel in self._END_SELECTORS:
                    try:
                        el = await p.query_selector(sel)
                        if el and await el.is_visible():
                            self._set_state(DISCONNECTED, f'Reunión finalizada — elemento visible: {sel!r}')
                            self._monitoring = False
                            ended = True
                            break
                    except Exception:
                        continue
                if ended:
                    return

                # ── 2. Verificar redirección fuera del web client ────────────────
                # Solo si la URL es de zoom.us pero NO está dentro de /wc/ (web client).
                # Aplicar gracia de 3 checks consecutivos para evitar falsos positivos
                # por recargas momentáneas.
                url = p.url
                if (url and 'zoom.us' in url
                        and '/wc/' not in url
                        and 'join' not in url
                        and url not in ('about:blank', '')):
                    url_outside_wc_count += 1
                    logger.warning('[Monitor] URL fuera de /wc/ (count=%d/3): %s',
                                   url_outside_wc_count, url)
                    if url_outside_wc_count >= 3:
                        self._set_state(DISCONNECTED, f'Redireccionado fuera del WC: {url}')
                        self._monitoring = False
                        return
                else:
                    url_outside_wc_count = 0

                # ── 3. Log periódico de estado ───────────────────────────────────
                if check_count % 6 == 0:  # cada ~60 s (6 × 10 s)
                    try:
                        status = await p.evaluate(
                            '() => window.__zbStatus ? window.__zbStatus() : {}'
                        )
                        logger.info('[Monitor] activo — %s | url=%s', status, p.url)
                    except Exception:
                        pass

            except Exception as e:
                logger.warning('[Monitor] error ignorado (transitorio): %s', e)

    # ── Salida limpia ──────────────────────────────────────────────────────

    async def leave(self) -> None:
        """Cierra page → context → browser en orden estricto, sin dejar tabs zombie.

        El orden importa: Playwright mantiene referencias internas que se liberan
        correctamente solo si se cierran de más específico a más general.
        El context se cierra siempre (es exclusivo de este bot), incluso cuando
        el browser fue provisto externamente (_owns_browser=False).
        """
        self._monitoring = False

        # 1. Page — cerrar explícitamente antes del contexto
        if self._page is not None:
            try:
                if not self._page.is_closed():
                    await self._page.close()
                    logger.info('[ZoomBot] página cerrada')
            except Exception as exc:
                logger.error('Error cerrando página: %s', exc)
            finally:
                self._page = None

        # 2. Context — siempre cerrar, incluso si el browser es compartido.
        # Es el contexto Zoom aislado: cerrarlo desconecta el tab de WebRTC.
        if self._context is not None:
            try:
                await self._context.close()
                logger.info('[ZoomBot] contexto cerrado')
            except Exception as exc:
                logger.error('Error cerrando contexto: %s', exc)
            finally:
                self._context = None

        # 3. Browser y Playwright — solo si los creamos nosotros
        if self._owns_browser:
            if self._browser is not None:
                try:
                    await self._browser.close()
                    logger.info('[ZoomBot] browser cerrado')
                except Exception as exc:
                    logger.error('Error cerrando browser: %s', exc)
                finally:
                    self._browser = None

            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                    logger.info('[ZoomBot] playwright detenido')
                except Exception as exc:
                    logger.error('Error deteniendo playwright: %s', exc)
                finally:
                    self._playwright = None

        self._set_state(DISCONNECTED, 'ZoomBot desconectado')
