"""
Bot Playwright que publica audio traducido en VDO.Ninja.

Estrategia de inyección de audio (dos capas para máxima fiabilidad):

  Capa 1 — getUserMedia:
    Interceptamos navigator.mediaDevices.getUserMedia ANTES de que VDO.Ninja
    lo llame. Devolvemos dest.stream (MediaStreamDestination) como si fuera
    el micrófono real. VDO.Ninja añade este track al RTCPeerConnection.

  Capa 2 — replaceTrack (fallback garantizado):
    Interceptamos RTCPeerConnection para capturar el objeto de VDO.Ninja.
    Una vez conectado, buscamos el RTCRtpSender de audio y llamamos a
    replaceTrack(dest.stream.getAudioTracks()[0]).
    Esto garantiza que nuestro track está en el sender aunque VDO.Ninja
    hubiera obtenido otro stream por getUserMedia.

  Reproducción de audio:
    Cuando llega un MP3 base64, AudioContext.decodeAudioData() lo decodifica
    y un AudioBufferSourceNode se conecta a dest. El audio fluye a través del
    track ya registrado en el RTCPeerConnection de VDO.Ninja hacia los viewers.

URL para estudiantes:
  https://vdo.ninja/?view=STREAM_ID&autoplay
"""

import asyncio
import logging
import os
from typing import Optional

from playwright.async_api import BrowserContext, Page, async_playwright

_HEADLESS = os.environ.get('HEADLESS', '0') == '1'

logger = logging.getLogger(__name__)

# ── Script inyectado ANTES de que VDO.Ninja inicialice (add_init_script) ─────
_INIT_SCRIPT = r"""
() => {
  // ── 1. AudioContext + MediaStreamDestination ──────────────────────────────
  const ctx  = new (window.AudioContext || window.webkitAudioContext)();
  const dest = ctx.createMediaStreamDestination();

  window.__vdoCtx   = ctx;
  window.__vdoDest  = dest;
  window.__vdoGumCalled  = false;
  window.__vdoPCs   = [];   // RTCPeerConnections de VDO.Ninja

  // Intentar resume inmediato (puede fallar sin gesto; se reintenta tras click)
  ctx.resume().catch(() => {});

  // ── 2. Interceptar getUserMedia ───────────────────────────────────────────
  // VDO.Ninja llama getUserMedia({audio: true}) con &miconly.
  // Devolvemos dest.stream: el track de este stream es el que VDO.Ninja
  // añadirá al RTCPeerConnection.
  const _origGUM = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
  navigator.mediaDevices.getUserMedia = async (constraints) => {
    if (constraints && constraints.audio) {
      window.__vdoGumCalled = true;
      const t = dest.stream.getAudioTracks()[0];
      console.log('[VdoBot] getUserMedia interceptado',
        JSON.stringify(constraints),
        '→ track:', t ? t.id : 'NONE',
        '| enabled:', t ? t.enabled : '-',
        '| readyState:', t ? t.readyState : '-',
        '| ctx:', ctx.state);
      return dest.stream;
    }
    return _origGUM(constraints);
  };

  // ── 3. Interceptar RTCPeerConnection ─────────────────────────────────────
  // Capturamos el PC de VDO.Ninja para poder llamar replaceTrack() después.
  const _OrigPC = window.RTCPeerConnection;
  window.RTCPeerConnection = function(...args) {
    const pc = new _OrigPC(...args);
    window.__vdoPCs.push(pc);
    pc.addEventListener('connectionstatechange', () => {
      console.log('[VdoBot] PC connectionState:', pc.connectionState,
        '| senders:', pc.getSenders().length);
      if (pc.connectionState === 'connected') {
        // Auto-replaceTrack: asegurar que nuestro dest.stream track está en el sender
        // cuando el PC llega a connected (cubre viewers que conectan tarde).
        const track = dest.stream.getAudioTracks()[0];
        if (!track) return;
        const audioSenders = pc.getSenders().filter(s => s.track && s.track.kind === 'audio');
        if (audioSenders.length > 0) {
          audioSenders.forEach(s => {
            s.replaceTrack(track).then(() => {
              console.log('[VdoBot] auto-replaceTrack OK en connected — trackId:', track.id);
            }).catch(e => {
              console.warn('[VdoBot] auto-replaceTrack error:', e.message);
            });
          });
        } else {
          // No hay senders de audio todavía — añadir el track directamente
          try {
            pc.addTrack(track, dest.stream);
            console.log('[VdoBot] auto-addTrack en connected (no había sender de audio)');
          } catch(e) {
            console.warn('[VdoBot] auto-addTrack error:', e.message);
          }
        }
      }
    });
    return pc;
  };
  Object.assign(window.RTCPeerConnection, _OrigPC);

  // ── 4. API de diagnóstico ─────────────────────────────────────────────────
  window.__vdoStatus = () => {
    const tracks = dest.stream.getAudioTracks();
    const pcs = window.__vdoPCs.map(pc => ({
      connState: pc.connectionState,
      iceState:  pc.iceConnectionState,
      sigState:  pc.signalingState,
      senders: pc.getSenders().map(s => ({
        kind:       s.track ? s.track.kind : null,
        trackId:    s.track ? s.track.id : null,
        enabled:    s.track ? s.track.enabled : null,
        readyState: s.track ? s.track.readyState : null,
        ourTrack:   s.track && tracks.length > 0 ? s.track.id === tracks[0].id : false,
      })),
    }));
    return {
      ctxState:   ctx.state,
      sampleRate: ctx.sampleRate,
      gumCalled:  window.__vdoGumCalled,
      tracks:     tracks.length,
      trackEnabled: tracks.length > 0 ? tracks[0].enabled : false,
      trackState:   tracks.length > 0 ? tracks[0].readyState : 'none',
      pcs,
    };
  };

  // ── 5. replaceTrack: asigna nuestro track en todos los senders de audio ──
  window.__vdoReplaceTrack = async () => {
    const track = dest.stream.getAudioTracks()[0];
    if (!track) return {error: 'no_dest_track'};

    let replaced = 0;
    for (const pc of window.__vdoPCs) {
      for (const sender of pc.getSenders()) {
        if (sender.track && sender.track.kind === 'audio') {
          try {
            await sender.replaceTrack(track);
            replaced++;
            console.log('[VdoBot] replaceTrack OK en PC', pc.connectionState,
              '— trackId:', track.id);
          } catch(e) {
            console.warn('[VdoBot] replaceTrack error:', e.message);
          }
        }
      }
    }
    // Si no había senders de audio, añadir el track directamente
    if (replaced === 0 && window.__vdoPCs.length > 0) {
      const pc = window.__vdoPCs[0];
      try {
        pc.addTrack(track, dest.stream);
        replaced = -1;  // señal de addTrack
        console.log('[VdoBot] addTrack (no había sender de audio)');
      } catch(e) {
        console.warn('[VdoBot] addTrack error:', e.message);
      }
    }
    return {replaced, track: track.id, ctx: ctx.state};
  };

  console.log('[VdoBot] init OK — ctx:', ctx.state, '| dest tracks:', dest.stream.getAudioTracks().length);
}
"""

# ── Reproducción de audio MP3 (evaluado en la página) ────────────────────────
_PLAY_AUDIO_JS = r"""
async (b64) => {
  const ctx  = window.__vdoCtx;
  const dest = window.__vdoDest;
  if (!ctx || !dest) return 'no_ctx';

  // Asegurar AudioContext activo
  if (ctx.state !== 'running') {
    try { await ctx.resume(); } catch(e) {}
  }

  const binary = atob(b64);
  const bytes  = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

  try {
    const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));
    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(dest);   // fluye hacia dest.stream → RTCRtpSender → viewer
    source.start();
    return 'ok:' + audioBuffer.duration.toFixed(2) + 's ctx:' + ctx.state;
  } catch (e) {
    console.error('[VdoBot] decodeAudioData error:', e.message);
    return 'error:' + e.message;
  }
}
"""


class VdoNinjaBot:
    def __init__(self, stream_id: str):
        self.stream_id     = stream_id
        self._playwright   = None
        self._browser      = None
        self._owns_browser = True   # False si el browser fue provisto externamente
        self._context: Optional[BrowserContext] = None
        self._page:    Optional[Page]           = None
        self._ready = False

    @property
    def view_url(self) -> str:
        return f'https://vdo.ninja/?view={self.stream_id}&autoplay'

    # ──────────────────────────────────────────────────────────────────────────

    async def start(self, *, _browser=None) -> None:
        """Inicia el bot de VDO.Ninja.

        Args:
            _browser: Browser de Playwright ya creado (compartido). Si se provee,
                      este bot crea su propio new_context() en ese browser pero NO
                      lo cierra al hacer stop(). Si es None, crea el suyo.

        Aislamiento de contextos:
            Este contexto NUNCA registra _CAPTURE_SCRIPT (el script de captura de
            audio de Zoom). Eso garantiza que el audio de VDO.Ninja no puede ser
            capturado por el pipeline de Whisper, eliminando el riesgo de loop.
        """
        if _browser is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=_HEADLESS,
                args=[
                    '--use-fake-ui-for-media-stream',
                    '--use-fake-device-for-media-stream',
                    '--autoplay-policy=no-user-gesture-required',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                ],
            )
            self._owns_browser = True
        else:
            self._playwright = None
            self._browser = _browser
            self._owns_browser = False
            logger.info('[VdoBot] usando browser compartido — contexto VDO.Ninja aislado de Zoom')

        # Contexto exclusivo de VDO.Ninja: _CAPTURE_SCRIPT de Zoom NO se registra aquí
        self._context = await self._browser.new_context(
            permissions=['microphone'],
        )
        await self._context.grant_permissions(['microphone'])
        await self._context.add_init_script(f'({_INIT_SCRIPT})()')

        self._page = await self._context.new_page()

        # Reenviar consola del browser a los logs de Python para diagnóstico
        def _fwd_console(msg):
            lvl = {'error': logger.error, 'warning': logger.warning}.get(msg.type, logger.debug)
            lvl('[VdoBot-browser] %s', msg.text)
        self._page.on('console', _fwd_console)

        push_url = (
            f'https://vdo.ninja/?push={self.stream_id}'
            '&label=Tradutor&autostart&noisegate=0&denoise=0&comp=0&miconly&muted'
        )
        logger.info('[VdoBot] abriendo push URL: %s', push_url)
        await self._page.goto(push_url, wait_until='domcontentloaded', timeout=30_000)

        # ── Desbloquear AudioContext con gesto de usuario real ─────────────────
        # Chrome requiere un gesto real para que AudioContext.resume() funcione.
        # page.mouse.click() genera un InputEvent reconocido por Chrome.
        await asyncio.sleep(1)
        try:
            await self._page.mouse.click(640, 360)
            logger.info('[VdoBot] gesto click simulado → AudioContext desbloqueado')
        except Exception as e:
            logger.warning('[VdoBot] gesto click falló: %s', e)

        # Dar tiempo a VDO.Ninja para llamar getUserMedia y negociar WebRTC
        logger.info('[VdoBot] esperando WebRTC setup de VDO.Ninja...')
        await asyncio.sleep(7)

        # Estado final del setup
        await asyncio.sleep(1)
        try:
            status = await self._page.evaluate('() => window.__vdoStatus && window.__vdoStatus()')
            if status:
                pcs    = status.get('pcs', [])
                ctx_st = status.get('ctxState', '?')
                gum    = status.get('gumCalled', False)
                tracks = status.get('tracks', 0)
                # Verificar que al menos un sender tiene nuestro track
                our_track_in_sender = any(
                    s.get('ourTrack') for pc in pcs for s in pc.get('senders', [])
                )
                logger.info(
                    '[VdoBot] SETUP FINAL: ctx=%s | PCs=%d | gumCalled=%s | '
                    'destTracks=%d | ourTrackInSender=%s',
                    ctx_st, len(pcs), gum, tracks, our_track_in_sender,
                )
                if not our_track_in_sender:
                    logger.warning('[VdoBot] nuestro track NO está en ningún sender — '
                                   'el audio puede no llegar al viewer')
        except Exception as e:
            logger.warning('[VdoBot] status final falló: %s', e)

        self._ready = True
        logger.info('[VdoBot] listo ✓ — viewer: %s', self.view_url)

    async def play_audio(self, audio_b64: str) -> None:
        """Inyecta audio MP3 (base64) en el stream de VDO.Ninja."""
        if not self._ready or not self._page or self._page.is_closed():
            logger.debug('[VdoBot] play_audio ignorado (not ready)')
            return
        try:
            result = await self._page.evaluate(_PLAY_AUDIO_JS, audio_b64)
            logger.info('[VdoBot] audio inyectado → %s (b64=%d chars)', result, len(audio_b64))
        except Exception as exc:
            logger.error('[VdoBot] play_audio error: %s', exc)

    async def stop(self) -> None:
        self._ready = False
        try:
            if self._page and not self._page.is_closed():
                await self._page.close()
            if self._context:
                await self._context.close()
            # Solo cerramos browser/playwright si los creamos nosotros
            if self._owns_browser:
                if self._browser:
                    await self._browser.close()
                if self._playwright:
                    await self._playwright.stop()
        except Exception as exc:
            logger.error('[VdoBot] stop error: %s', exc)
        logger.info('[VdoBot] detenido')
