"""
Pipeline: PCM audio → Whisper → GPT-4o-mini → TTS nova

Mejoras respecto a versión anterior:
 - Detección de silencio: chunks con RMS < umbral no se envían a Whisper
 - Reintento automático 3 veces (2 s / 4 s / 8 s) en errores transitorios de OpenAI
 - Log de silencio prolongado (> 30 s) sin spam de API
"""

import base64
import io
import logging
import struct
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np

from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE    = 16000
CHANNELS       = 1
CHUNK_SECONDS  = 3
CHUNK_SAMPLES  = SAMPLE_RATE * CHUNK_SECONDS   # 48 000 muestras

# Magic bytes del init segment webm/EBML — identifica chunks de MediaRecorder (Linux)
_WEBM_MAGIC = b'\x1a\x45\xdf\xa3'

# Alucinaciones conocidas de Whisper con audio silencioso o ruido de fondo
_WHISPER_HALLUCINATIONS = {
    'you', 'thank you', 'thanks for watching', 'thanks for watching!',
    'bye', 'bye-bye', 'bye bye', 'thanks', 'thank you!', 'thank you.',
    'obrigado', 'tchau', 'você', 'subtitles by', 'subtitles',
    'transcribed by', 'www.youtube.com', '♪', '...',
}

import re as _re
_PUNCT_RE = _re.compile(r'[^\w\s]', _re.UNICODE)

def _is_hallucination(text: str) -> bool:
    """Retorna True si el texto es una alucinación conocida de Whisper."""
    normalized = _PUNCT_RE.sub('', text.lower()).strip()
    return normalized in _WHISPER_HALLUCINATIONS

# Umbral de silencio para chunks webm — RMS < 200 → descartar sin llamar a Whisper
# (Whisper alucina con audio silencioso: "you", "thank you", "thanks for watching")
_WEBM_SILENCE_RMS = 200

# Detección de silencio
# RMS int16: silencio puro ≈ 0, ruido/fake device ≈ 100-500, voz normal ≈ 500+
# Umbral conservador para filtrar silencio sin cortar voz real.
SILENCE_THRESHOLD = 50
SILENCE_LOG_S     = 30   # loguear (pero no enviar) si hay silencio continuo

# Reintentos en errores transitorios de OpenAI
MAX_RETRIES   = 3
RETRY_DELAYS  = (2, 4, 8)   # segundos entre intentos

# Idiomas disponibles: clave → {name, prompt}
LANGUAGES: dict[str, dict] = {
    'pt': {
        'name': 'Português (Brasil)',
        'prompt': (
            'Eres un traductor profesional. Traduce el texto al portugués de Brasil '
            '(pt-BR) de forma natural y fluida. Devuelve únicamente la traducción, '
            'sin explicaciones ni notas.'
        ),
    },
    'en': {
        'name': 'English',
        'prompt': (
            'You are a professional translator. Translate the text to English '
            'naturally and fluently. Return only the translation, no explanations.'
        ),
    },
    'es': {
        'name': 'Español',
        'prompt': (
            'Eres un traductor profesional. Traduce el texto al español de forma '
            'natural y fluida. Devuelve únicamente la traducción, sin explicaciones.'
        ),
    },
    'fr': {
        'name': 'Français',
        'prompt': (
            'Tu es un traducteur professionnel. Traduis le texte en français de '
            'manière naturelle et fluide. Retourne uniquement la traduction, '
            'sans explications.'
        ),
    },
    'de': {
        'name': 'Deutsch',
        'prompt': (
            'Du bist ein professioneller Übersetzer. Übersetze den Text natürlich '
            'und flüssig ins Deutsche. Gib nur die Übersetzung zurück, '
            'keine Erklärungen.'
        ),
    },
}

_TRANSIENT_ERRORS = (RateLimitError, APITimeoutError, APIConnectionError)


class TranslatorPipeline:
    """Acumula PCM int16, procesa chunks y dispara callbacks con el resultado."""

    def __init__(self):
        self._client   = OpenAI()
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix='translator')
        self._buffer: list[int] = []
        self._lock     = threading.Lock()
        self._running  = False
        self.target_lang = 'pt'

        # Contadores de silencio (accedidos solo desde threads del executor)
        self._silent_samples_acc = 0   # muestras silenciosas acumuladas

        # Callbacks
        self.on_translation: Optional[Callable[[str, str, str], None]] = None
        self.on_error:       Optional[Callable[[str], None]]            = None

    # ── Control ───────────────────────────────────────────────────────────

    def start(self, target_lang: str = 'pt') -> None:
        self.target_lang = target_lang
        with self._lock:
            self._buffer = []
        self._silent_samples_acc = 0
        self._running = True
        logger.info('TranslatorPipeline iniciado (lang=%s)', target_lang)

    def stop(self) -> None:
        self._running = False
        logger.info('TranslatorPipeline detenido')

    # ── Feed de audio (llamado desde el WebSocket interno) ────────────────

    def feed_audio(self, data: bytes) -> None:
        """Recibe bytes de audio — PCM int16 (Windows) o webm/opus (Linux vía MediaRecorder)."""
        if not self._running:
            return
        if data[:4] == _WEBM_MAGIC:
            # Ruta Linux: webm/opus de MediaRecorder → directo a Whisper sin buffer
            fut = self._executor.submit(self._process_webm, data)
            fut.add_done_callback(_log_future_error)
        else:
            self._feed_pcm(data)

    def _feed_pcm(self, pcm_bytes: bytes) -> None:
        """Acumula PCM int16 little-endian y encola chunks completos de 3 s."""
        count   = len(pcm_bytes) // 2
        samples = struct.unpack(f'<{count}h', pcm_bytes)

        with self._lock:
            self._buffer.extend(samples)

            while len(self._buffer) >= CHUNK_SAMPLES:
                chunk = self._buffer[:CHUNK_SAMPLES]
                self._buffer = self._buffer[CHUNK_SAMPLES:]
                rms_preview = _rms(chunk)
                logger.info(
                    '[Pipeline] chunk %.1f s encolado — RMS=%.0f umbral=%d buffer_restante=%d',
                    CHUNK_SECONDS, rms_preview, SILENCE_THRESHOLD, len(self._buffer),
                )
                fut = self._executor.submit(self._process_chunk, list(chunk))
                fut.add_done_callback(_log_future_error)

    # ── Procesamiento por chunk (ThreadPoolExecutor) ──────────────────────

    def _process_chunk(self, samples: list[int]) -> None:
        # Log de confirmación: si esta línea NO aparece, el worker no arrancó
        rms = _rms(samples)
        logger.info('[Pipeline] WORKER inicio — RMS=%.0f muestras=%d umbral=%d',
                    rms, len(samples), SILENCE_THRESHOLD)

        try:
            # ── Detección de silencio ─────────────────────────────────────
            if rms < SILENCE_THRESHOLD:
                self._silent_samples_acc += len(samples)
                silent_s = self._silent_samples_acc / SAMPLE_RATE
                if silent_s >= SILENCE_LOG_S and (self._silent_samples_acc % (SAMPLE_RATE * 10)) < len(samples):
                    logger.info('[Pipeline] silencio continuo %.0f s (RMS=%.0f) — chunk descartado',
                                silent_s, rms)
                else:
                    logger.info('[Pipeline] chunk descartado por silencio — RMS=%.0f < umbral=%d (acum=%.0f s)',
                                rms, SILENCE_THRESHOLD, self._silent_samples_acc / SAMPLE_RATE)
                return  # No enviar silencio a Whisper

            # Audio con señal detectada → resetear contador de silencio
            self._silent_samples_acc = 0
            logger.info('[Pipeline] RMS=%.0f ≥ umbral=%d → enviando a Whisper', rms, SILENCE_THRESHOLD)

            wav_data   = _samples_to_wav(samples)
            logger.info('[Pipeline] WAV preparado (%d bytes) — llamando Whisper API...', len(wav_data))
            original, detected_lang = self._transcribe(wav_data)
            if not original:
                logger.info('[Pipeline] Whisper devolvió texto vacío — chunk ignorado')
                return
            if _is_hallucination(original):
                logger.info('[Pipeline] alucinación descartada: %r', original)
                return

            # Filtro anti-loop: si Whisper detecta el idioma destino (pt), es audio
            # del TTS que llegó a VB-Cable — descartarlo silenciosamente.
            if detected_lang == self.target_lang:
                logger.info('[Pipeline] chunk descartado — idioma detectado=%r coincide con target_lang=%r (loop)',
                            detected_lang, self.target_lang)
                return

            logger.info('[Pipeline] Whisper OK: %r (lang=%s)', original, detected_lang)
            translated = self._translate(original)
            logger.info('[Pipeline] GPT OK: %r', translated)
            audio_b64  = self._tts(translated)
            logger.info('[Pipeline] TTS OK: %d bytes MP3 (b64 len=%d)',
                        len(audio_b64) * 3 // 4, len(audio_b64))

            if self.on_translation:
                self.on_translation(original, translated, audio_b64)

        except Exception as exc:
            logger.error('[Pipeline] WORKER error: %s', exc, exc_info=True)
            if self.on_error:
                self.on_error(str(exc))

    def _rms_webm(self, webm_data: bytes) -> float:
        """Decodifica webm a PCM con pydub y calcula RMS. Retorna 0.0 si falla."""
        try:
            from pydub import AudioSegment  # import lazy — solo si está instalado
            seg = AudioSegment.from_file(io.BytesIO(webm_data), format='webm')
            samples = np.array(seg.get_array_of_samples(), dtype=np.float64)
            if samples.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(samples ** 2)))
        except Exception as e:
            logger.debug('[Pipeline] rms_webm error (asumiendo audio): %s', e)
            return float('inf')  # si no se puede decodificar, dejar pasar

    def _process_webm(self, webm_data: bytes) -> None:
        """Ruta webm/opus: procesa un chunk de MediaRecorder."""
        rms = self._rms_webm(webm_data)
        if rms < _WEBM_SILENCE_RMS:
            logger.info('[Pipeline] WEBM silencioso (RMS=%.0f) — descartado', rms)
            return
        logger.info('[Pipeline] WEBM chunk — %d bytes, RMS=%.0f — llamando Whisper...', len(webm_data), rms)
        try:
            original, detected_lang = self._transcribe_webm(webm_data)
            if not original:
                logger.info('[Pipeline] Whisper devolvió texto vacío — chunk ignorado')
                return
            if _is_hallucination(original):
                logger.info('[Pipeline] alucinación descartada: %r', original)
                return
            if detected_lang == self.target_lang:
                logger.info('[Pipeline] chunk descartado — loop detectado (lang=%s)', detected_lang)
                return
            logger.info('[Pipeline] Whisper OK: %r (lang=%s)', original, detected_lang)
            translated = self._translate(original)
            logger.info('[Pipeline] GPT OK: %r', translated)
            audio_b64  = self._tts(translated)
            logger.info('[Pipeline] TTS OK: %d bytes MP3', len(audio_b64) * 3 // 4)
            if self.on_translation:
                self.on_translation(original, translated, audio_b64)
        except Exception as exc:
            logger.error('[Pipeline] WORKER error (webm): %s', exc, exc_info=True)
            if self.on_error:
                self.on_error(str(exc))

    # ── Llamadas a OpenAI con reintento automático ────────────────────────

    def _transcribe_webm(self, webm_data: bytes) -> tuple[str, str]:
        """Transcribe audio webm/opus de MediaRecorder. Devuelve (text, language)."""
        def _call():
            t = self._client.audio.transcriptions.create(
                model='whisper-1',
                file=('chunk.webm', webm_data, 'audio/webm'),
                response_format='verbose_json',
            )
            return t.text.strip(), (t.language or '').lower()
        return _retry(_call, 'Whisper')

    def _transcribe(self, wav_data: bytes) -> tuple[str, str]:
        """Devuelve (text, language). language es el código ISO detectado por Whisper."""
        def _call():
            t = self._client.audio.transcriptions.create(
                model='whisper-1',
                file=('chunk.wav', wav_data, 'audio/wav'),
                response_format='verbose_json',
            )
            return t.text.strip(), (t.language or '').lower()
        return _retry(_call, 'Whisper')

    def _translate(self, text: str) -> str:
        lang = LANGUAGES.get(self.target_lang, LANGUAGES['pt'])
        def _call():
            resp = self._client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[
                    {'role': 'system', 'content': lang['prompt']},
                    {'role': 'user',   'content': text},
                ],
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        return _retry(_call, 'GPT')

    def _tts(self, text: str) -> str:
        def _call():
            tts = self._client.audio.speech.create(
                model='tts-1',
                voice='nova',
                input=text,
                response_format='mp3',
            )
            return base64.b64encode(tts.content).decode()
        return _retry(_call, 'TTS')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_future_error(fut):
    """Callback para Future del executor — loguea si el worker terminó con excepción."""
    exc = fut.exception()
    if exc is not None:
        logger.error('[Pipeline] WORKER excepción no capturada: %s', exc, exc_info=exc)


def _rms(samples: list[int]) -> float:
    """Root Mean Square de muestras int16 (0–32767)."""
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def _retry(func: Callable, label: str = ''):
    """Llama func hasta MAX_RETRIES veces en errores transitorios de OpenAI."""
    last_exc: Optional[Exception] = None
    for attempt, delay in enumerate(RETRY_DELAYS):
        try:
            return func()
        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            logger.warning(
                '%s error transitorio (intento %d/%d): %s — reintentando en %d s',
                label, attempt + 1, MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
        except Exception:
            raise   # Error no transitorio → propagar directamente

    # Último intento sin capturar
    try:
        return func()
    except Exception as exc:
        raise exc from last_exc


def _samples_to_wav(samples: list[int]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(struct.pack(f'<{len(samples)}h', *samples))
    return buf.getvalue()
