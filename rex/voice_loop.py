"""Async voice assistant loop orchestrating wake word, STT, LLM, and TTS.

RELATIONSHIP NOTE — two voice_loop files exist in this repo:
- ``rex/voice_loop.py`` (this file, package): canonical implementation.
  ``rex_loop.py`` imports ``build_voice_loop`` from here and this is the
  authoritative voice loop used when Rex starts.
- ``voice_loop.py`` (repo root): legacy implementation containing
  ``AsyncRexAssistant``. Kept for backward compatibility only. Changes here
  do NOT affect the ``rex_loop.py`` startup path.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import warnings
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from rex.wake_acknowledgment import ensure_wake_acknowledgment_sound

from .assistant_errors import (
    AudioDeviceError,
    AudioFormatError,
    SpeechToTextError,
    TextToSpeechError,
)
from .config import settings
from .memory import (
    extract_voice_reference,
    load_all_profiles,
    load_users_map,
    resolve_user_key,
)
from .tts_utils import chunk_text_for_xtts

# Suppress torio FFmpeg extension warnings — FFmpeg is not required for audio
# capture/playback (sounddevice handles that).  It is only used internally by
# Coqui XTTS; the XTTS fallback path handles the case where it is absent.
warnings.filterwarnings("ignore", message=".*FFmpeg extension.*")
warnings.filterwarnings("ignore", message=".*libtorio.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torio")


def _import_optional(module_name: str):
    module = sys.modules.get(module_name)
    if module is not None:
        return module
    if find_spec(module_name) is None:
        return None
    return import_module(module_name)


def _lazy_import_numpy():
    return _import_optional("numpy")


np = _lazy_import_numpy()


def _lazy_import_simpleaudio():
    return _import_optional("simpleaudio")


sa = _lazy_import_simpleaudio()
sd = None


def _lazy_import_whisper():
    return _import_optional("whisper")


def _lazy_import_tts():
    # Only check availability - do NOT import TTS yet (it triggers
    # internal transformers imports that need the shim first).
    if find_spec("TTS") is None:
        return None
    from rex.compat import ensure_transformers_compatibility

    ensure_transformers_compatibility()
    return import_module("TTS.api").TTS


def _lazy_import_soundfile():
    return _import_optional("soundfile")


def _load_sounddevice():
    global sd
    if sd is not None:
        return sd
    sd = _import_optional("sounddevice")
    return sd


def _require_numpy():
    if np is None:
        raise AudioDeviceError("numpy is required for audio processing")
    return np


def _require_sounddevice():
    module = _load_sounddevice()
    if module is None:
        raise AudioDeviceError("sounddevice is not installed")
    return module


def _device_name(device: Any) -> str:
    if isinstance(device, dict):
        return str(device.get("name", "<unknown>"))
    return str(getattr(device, "name", "<unknown>"))


def _max_input_channels(device: Any) -> int:
    if isinstance(device, dict):
        value = device.get("max_input_channels", 0)
    else:
        value = getattr(device, "max_input_channels", 0)
    return int(value or 0)


def _available_input_devices(devices: Any) -> list[str]:
    available: list[str] = []
    for index, device in enumerate(devices):
        if _max_input_channels(device) > 0:
            available.append(f"{index}: {_device_name(device)}")
    return available


def _validate_input_device_index(device_index: int | None) -> int | None:
    if device_index is None:
        return None

    sd_module = _require_sounddevice()
    try:
        devices = sd_module.query_devices()
    except Exception as exc:
        raise AudioDeviceError(str(exc)) from exc

    available_devices = _available_input_devices(devices)
    available_list = ", ".join(available_devices) if available_devices else "none"

    try:
        device = devices[device_index]
    except (IndexError, KeyError, TypeError):
        raise AudioDeviceError(
            f"Input device {device_index} not found. Available: {available_list}"
        ) from None

    if _max_input_channels(device) <= 0:
        raise AudioDeviceError(
            f"Input device {device_index} not found. Available: {available_list}"
        )

    return device_index


def _detect_audio_format(audio_buffer: bytes) -> str:
    header = audio_buffer[:4]
    if not header:
        return "empty"
    if header.startswith(b"ID3"):
        return "ID3"
    text = header.decode("ascii", errors="ignore")
    text = "".join(char for char in text if char.isprintable()).strip()
    return text or header.hex()


def _to_wav_buffer(audio: AudioArray | bytes | bytearray | memoryview, sample_rate: int) -> bytes:
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)

    numpy = _require_numpy()
    samples = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
    samples = numpy.clip(samples, -1.0, 1.0)
    pcm16 = (samples * 32767).astype(numpy.int16)

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm16.tobytes())
        return buffer.getvalue()


_STT_AUTO_GAIN_TARGET_PEAK = 0.45
_STT_AUTO_GAIN_MAX_GAIN = 12.0
_STT_AUTO_GAIN_MIN_RMS = 0.0005


def _audio_level(samples: AudioArray) -> tuple[float, float]:
    numpy = _require_numpy()
    if samples.size == 0:
        return 0.0, 0.0
    abs_samples = numpy.abs(samples)
    return (
        float(numpy.sqrt(numpy.mean(samples * samples))),
        float(numpy.max(abs_samples)),
    )


def _apply_stt_auto_gain(samples: AudioArray) -> AudioArray:
    numpy = _require_numpy()
    if not bool(getattr(settings, "stt_auto_gain", True)):
        return samples

    target_peak = float(getattr(settings, "stt_target_peak", _STT_AUTO_GAIN_TARGET_PEAK))
    max_gain = float(getattr(settings, "stt_max_gain", _STT_AUTO_GAIN_MAX_GAIN))
    min_rms = float(getattr(settings, "stt_min_rms_for_gain", _STT_AUTO_GAIN_MIN_RMS))
    rms, peak = _audio_level(samples)
    if rms < min_rms or peak <= 0.0 or peak >= target_peak:
        return samples

    gain = min(max_gain, target_peak / peak)
    boosted = numpy.clip(samples * gain, -1.0, 1.0)
    logger.info(
        "[STT] Applied input auto-gain",
        extra=_voice_log_extra(
            event="stt_audio_auto_gain",
            audio_rms_before=round(rms, 6),
            audio_peak_before=round(peak, 6),
            applied_gain=round(gain, 3),
            target_peak=round(target_peak, 3),
            max_gain=round(max_gain, 3),
        ),
    )
    return cast(AudioArray, boosted)


def _prepare_audio_for_stt(
    audio: AudioArray | bytes | bytearray | memoryview,
) -> AudioArray | bytes:
    """Return STT input with non-finite values removed and amplitude clamped.

    Whisper on CUDA can fail with NaN logits if the captured microphone buffer
    already contains NaN/inf values. Sanitize the audio before inference while
    preserving the preferred GPU execution path.
    """
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)

    numpy = _require_numpy()
    prepared: AudioArray = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
    if prepared.size == 0:
        return prepared
    prepared = numpy.nan_to_num(prepared, nan=0.0, posinf=1.0, neginf=-1.0)
    prepared = numpy.clip(prepared, -1.0, 1.0)
    return _apply_stt_auto_gain(prepared)


def _audio_quality_summary(
    audio: AudioArray | bytes | bytearray | memoryview,
    sample_rate: int,
) -> dict[str, object]:
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return {
            "audio_input_kind": "bytes",
            "audio_bytes": len(bytes(audio)),
            "sample_rate": sample_rate,
        }

    numpy = _require_numpy()
    samples = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
    if samples.size == 0:
        return {
            "audio_input_kind": "array",
            "audio_samples": 0,
            "audio_duration_s": 0.0,
            "audio_rms": 0.0,
            "audio_peak": 0.0,
            "audio_clipped_samples": 0,
            "sample_rate": sample_rate,
        }

    abs_samples = numpy.abs(samples)
    return {
        "audio_input_kind": "array",
        "audio_samples": int(samples.size),
        "audio_duration_s": round(samples.size / sample_rate, 3) if sample_rate > 0 else None,
        "audio_rms": round(float(numpy.sqrt(numpy.mean(samples * samples))), 6),
        "audio_peak": round(float(numpy.max(abs_samples)), 6),
        "audio_clipped_samples": int(numpy.count_nonzero(abs_samples >= 0.999)),
        "sample_rate": sample_rate,
    }


logger = logging.getLogger(__name__)
_VOICE_INTERACTION_ID: ContextVar[int | None] = ContextVar(
    "rex_voice_interaction_id",
    default=None,
)


def _voice_log_extra(**extra: object) -> dict[str, object]:
    interaction_id = _VOICE_INTERACTION_ID.get()
    if interaction_id is not None and "interaction_id" not in extra:
        extra["interaction_id"] = interaction_id
    return extra


_USE_CONFIG_LANGUAGE = object()

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray: TypeAlias = NDArray[Any]
else:
    AudioArray: TypeAlias = Any

RecorderCallable = Callable[[float], Awaitable[AudioArray] | AudioArray]
IdentifySpeakerCallable = Callable[[AudioArray], str | None] | Callable[[], str | None]

# Backwards-compatible runtime alias used by optional-import tests.
_NDArray = np.ndarray if np is not None else Any


# Sentence-boundary pattern for streaming TTS sentence splitting.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Short phrase used to pre-warm the TTS engine on startup.
_WARMUP_PHRASE = "."

# Common single-word abbreviations that should not trigger sentence boundaries.
# Matched as whole words (case-insensitive) followed by "." and whitespace.
_ABBREV_WORDS: frozenset[str] = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "al",
        "st",
        "fig",
        "dept",
        "est",
        "approx",
        "cf",
        "rev",
        "gen",
        "col",
        "lt",
        "sgt",
        "capt",
        "gov",
        "sen",
        "rep",
        "no",
        "vol",
        "ave",
        "blvd",
    ]
)

# Abbreviations containing internal dots (e.g., i.e.) followed by "." and whitespace.
_ABBREV_DOT: frozenset[str] = frozenset(["e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "u.n"])

# Placeholder character used to protect abbreviation periods during splitting.
_ABBREV_PLACEHOLDER = "\x00"

_LOW_VALUE_TRANSCRIPT_WORDS: frozenset[str] = frozenset(
    {
        "ah",
        "alright",
        "annably",
        "good",
        "hm",
        "hmm",
        "much",
        "nope",
        "nowey",
        "ok",
        "okay",
        "please",
        "thanks",
        "thank",
        "uh",
        "um",
        "very",
        "yeah",
        "yep",
        "yes",
        "you",
    }
)
_LOW_VALUE_TRANSCRIPT_EXACT: frozenset[str] = frozenset(
    {
        "alright",
        "good",
        "ok",
        "okay",
        "thank you",
        "thank you very much",
        "thanks",
        "thanks a lot",
        "you are welcome",
        "youre welcome",
        "you're welcome",
    }
)
_LOW_VALUE_TRANSCRIPT_PHRASES: tuple[str, ...] = (
    "thanks for watching",
    "thank you for watching",
    "if there is anything else",
    "if there's anything else",
)
_WEAK_TRANSCRIPT_WORDS: frozenset[str] = frozenset(
    {
        "again",
        "eh",
        "huh",
        "hm",
        "hmm",
        "pardon",
        "repeat",
        "sorry",
        "uh",
        "um",
        "what",
    }
)
_WEAK_TRANSCRIPT_EXACT: frozenset[str] = frozenset(
    {
        "again",
        "eh",
        "huh",
        "hm",
        "hmm",
        "pardon",
        "repeat",
        "sorry",
        "uh",
        "um",
        "what",
        "what was that",
    }
)
_WEAK_TRANSCRIPT_RETRY_PROMPT = "I only caught part of that. Please repeat the question."
_SUSPICIOUS_TRANSCRIPT_RETRY_PROMPT = "I may have misheard that. What did you need?"
_SUSPICIOUS_NEED_LEAD_WORDS: frozenset[str] = frozenset({"neutral"})
_SUSPICIOUS_NEED_TOKENS: frozenset[str] = frozenset({"kate"})
_CLARIFICATION_REPLY_MARKERS: tuple[str, ...] = (
    "could you clarify",
    "can you clarify",
    "please clarify",
    "what do you need",
    "what would you like",
    "what kind",
    "which one",
    "which",
    "can you tell me more",
    "please repeat",
)
_ACTION_TRANSCRIPT_RE = re.compile(
    r"\b(?:"
    r"alarm|battery|brightness|calendar|close|cpu|date|disk|email|"
    r"find|forecast|google|launch|light|lights|look|memory|message|"
    r"open|pause|play|power|remind|resume|run|search|send|set|skip|"
    r"sms|start|stop|temperature|text|time|timer|turn|volume|weather"
    r")\b",
    re.IGNORECASE,
)
_TRANSCRIPT_WORD_RE = re.compile(r"[a-z0-9']+")
_WAKE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:hey|hi)\s+)?(?:jarvis|rex)(?:[\s,;:.-]+|$)",
    re.IGNORECASE,
)
_MIN_WAKE_PREROLL_SOURCE_SECONDS = 0.2
_COMMAND_CAPTURE_CHUNK_SECONDS = 0.25
_COMMAND_CAPTURE_MIN_SECONDS = 3.0
_COMMAND_CAPTURE_MAX_SECONDS = 10.0
_COMMAND_CAPTURE_END_SILENCE_SECONDS = 0.9
_COMMAND_CAPTURE_RMS_THRESHOLD = 0.003
_DEFAULT_STT_INITIAL_PROMPT = (
    "The audio is an English voice command to Rex, a home assistant. "
    "It may ask for time, date, weather, recipes, reminders, smart home control, "
    "or general help."
)


def _protect_abbreviations(text: str) -> str:
    """Replace trailing periods in known abbreviations with a placeholder.

    This prevents *_SENTENCE_BOUNDARY* from treating abbreviations like
    "Dr.", "Mr.", or "e.g." as sentence-ending punctuation.  Original
    casing is preserved via a capturing group in each substitution.
    """
    protected = text
    # Single-word abbreviations: word-boundary + abbr + "." + whitespace.
    # Group 1 captures the original-cased abbreviation so it is preserved.
    for abbr in _ABBREV_WORDS:
        protected = re.sub(
            rf"(?<!\w)({re.escape(abbr)})\.\s",
            r"\1" + _ABBREV_PLACEHOLDER + " ",
            protected,
            flags=re.IGNORECASE,
        )
    # Dot-internal abbreviations: abbr + "." + whitespace
    for abbr in _ABBREV_DOT:
        protected = re.sub(
            rf"({re.escape(abbr)})\.\s",
            r"\1" + _ABBREV_PLACEHOLDER + " ",
            protected,
            flags=re.IGNORECASE,
        )
    return protected


def _normalize_transcript_for_guard(text: str) -> str:
    text = text.lower().replace("’", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def _is_low_value_transcript(transcript: str) -> bool:
    """Return True for likely Whisper filler/hallucination with no user command."""
    normalized = _normalize_transcript_for_guard(transcript)
    if not normalized:
        return True

    if _ACTION_TRANSCRIPT_RE.search(normalized):
        return False

    if normalized in _LOW_VALUE_TRANSCRIPT_EXACT:
        return True

    if any(phrase in normalized for phrase in _LOW_VALUE_TRANSCRIPT_PHRASES):
        return True

    if normalized.count("thank you") >= 2:
        return True

    words = _TRANSCRIPT_WORD_RE.findall(normalized)
    if not words:
        return True

    low_value_words = sum(1 for word in words if word in _LOW_VALUE_TRANSCRIPT_WORDS)
    if len(words) <= 4 and low_value_words == len(words):
        return True

    return len(words) >= 8 and low_value_words / len(words) >= 0.45


def _is_weak_transcript_fragment(transcript: str) -> bool:
    """Return True for fragments too ambiguous to route to the assistant."""
    normalized = _normalize_transcript_for_guard(transcript)
    if not normalized:
        return True

    if _ACTION_TRANSCRIPT_RE.search(normalized):
        return False

    if normalized in _WEAK_TRANSCRIPT_EXACT:
        return True

    words = _TRANSCRIPT_WORD_RE.findall(normalized)
    if not words:
        return True

    return len(words) <= 2 and all(word in _WEAK_TRANSCRIPT_WORDS for word in words)


def _is_suspicious_voice_transcript(transcript: str) -> bool:
    """Return True for plausible-looking ASR corruption that needs confirmation."""
    normalized = _normalize_transcript_for_guard(transcript)
    if not normalized:
        return False

    words = _TRANSCRIPT_WORD_RE.findall(normalized)
    if not words:
        return False

    if len(words) <= 6 and words[0] in _SUSPICIOUS_NEED_LEAD_WORDS and "need" in words:
        return True

    return (
        len(words) <= 8
        and "need" in words
        and bool(_SUSPICIOUS_NEED_TOKENS.intersection(words))
        and "recipe" not in words
    )


def _looks_like_clarification_reply(reply: str, transcript: str) -> bool:
    normalized_reply = _normalize_transcript_for_guard(reply)
    if any(marker in normalized_reply for marker in _CLARIFICATION_REPLY_MARKERS):
        return True

    transcript_words = _TRANSCRIPT_WORD_RE.findall(_normalize_transcript_for_guard(transcript))
    return len(transcript_words) <= 3 and reply.strip().endswith("?")


def _combine_followup_transcript(first: str, followup: str) -> str:
    return f"{first.rstrip(' .?!')} {followup.lstrip()}".strip()


def _strip_wake_prefix(transcript: str) -> str:
    """Remove a wake phrase that leaked into STT from wake-frame pre-roll."""
    stripped = transcript.strip()
    return _WAKE_PREFIX_RE.sub("", stripped).strip()


def _split_into_sentences(text: str) -> list[str]:
    """Split *text* into sentence-sized chunks for streaming TTS.

    Uses NLTK ``sent_tokenize`` when available; otherwise falls back to an
    abbreviation-aware regex splitter that does not break on common titles
    (Dr., Mr.) or abbreviations (e.g., etc.).
    """
    stripped = text.strip()
    if not stripped:
        return []

    # Try NLTK sent_tokenize first (handles abbreviations natively).
    if find_spec("nltk") is not None:
        try:
            nltk = _import_optional("nltk")
            if nltk is None:
                raise ImportError("nltk is not available")

            sentences = nltk.sent_tokenize(stripped)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # punkt tokenizer not downloaded or other NLTK error — fall through.
            pass

    # Abbreviation-aware regex fallback.
    protected = _protect_abbreviations(stripped)
    parts = _SENTENCE_BOUNDARY.split(protected)
    return [s.replace(_ABBREV_PLACEHOLDER, ".").strip() for s in parts if s.strip()]


async def _sentence_stream(text: str) -> AsyncIterator[str]:
    """Yield sentences from *text* as an async iterator."""
    for sentence in _split_into_sentences(text):
        yield sentence


def _extract_completed_sentences(buffer: str) -> tuple[list[str], str]:
    """Return completed sentences and the remaining partial buffer."""
    protected = _protect_abbreviations(buffer)
    matches = list(_SENTENCE_BOUNDARY.finditer(protected))
    if not matches:
        return [], buffer

    split_index = matches[-1].end()
    completed_text = buffer[:split_index]
    remainder = buffer[split_index:]
    return _split_into_sentences(completed_text), remainder


async def _sentence_buffer_stream(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Convert a token stream into sentence chunks for streaming TTS."""
    buffer = ""
    async for token in tokens:
        if not token:
            continue
        buffer += token
        sentences, buffer = _extract_completed_sentences(buffer)
        for sentence in sentences:
            yield sentence

    for sentence in _split_into_sentences(buffer):
        yield sentence


@dataclass
class SynthesizedAudio:
    """Container for synthesized audio data."""

    data: AudioArray
    sample_rate: int


class AsyncMicrophone:
    """Async microphone recording."""

    def __init__(
        self,
        *,
        sample_rate: int,
        detection_seconds: float,
        capture_seconds: float,
        detection_hop_seconds: float | None = None,
        device_index: int | None = None,
        recorder: RecorderCallable | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self._detection_seconds = detection_seconds
        self._capture_seconds = capture_seconds
        self._detection_hop_seconds = detection_hop_seconds or detection_seconds
        self._device_index = device_index
        self._recorder = recorder
        self._detection_buffer: AudioArray | None = None
        self._detection_buffer_filled_samples = 0

    async def detection_frame(self) -> AudioArray:
        """Record a short frame for wake word detection."""
        if self._detection_hop_seconds >= self._detection_seconds:
            return await self._record(self._detection_seconds)

        np = _require_numpy()
        chunk = await self._record(self._detection_hop_seconds)
        window_samples = max(int(self.sample_rate * self._detection_seconds), 1)

        previous = (
            self._detection_buffer
            if self._detection_buffer is not None
            else np.zeros(0, dtype=np.float32)
        )
        combined = np.concatenate([previous, chunk])
        filled_samples = min(int(previous.size + chunk.size), window_samples)
        self._detection_buffer_filled_samples = filled_samples
        self._detection_buffer = combined[-window_samples:]

        if combined.size < window_samples:
            frame = np.pad(combined, (window_samples - combined.size, 0))
        else:
            frame = combined[-window_samples:]

        zero_pad_samples = max(window_samples - filled_samples, 0)
        chunk_rms, chunk_peak = _audio_level(cast(AudioArray, chunk))
        frame_rms, frame_peak = _audio_level(cast(AudioArray, frame))
        logger.debug(
            (
                "MIC DEBUG: overlapping detection frame window=%.2fs hop=%.2fs "
                "samples=%d fill=%.2f"
            ),
            self._detection_seconds,
            self._detection_hop_seconds,
            len(frame),
            filled_samples / window_samples,
            extra=_voice_log_extra(
                event="audio_detection_frame",
                window_s=self._detection_seconds,
                hop_s=self._detection_hop_seconds,
                audio_samples=len(frame),
                buffer_filled_samples=filled_samples,
                buffer_fill_ratio=round(filled_samples / window_samples, 3),
                zero_pad_samples=zero_pad_samples,
                priming=zero_pad_samples > 0,
                chunk_audio_rms=round(chunk_rms, 6),
                chunk_audio_peak=round(chunk_peak, 6),
                frame_audio_rms=round(frame_rms, 6),
                frame_audio_peak=round(frame_peak, 6),
            ),
        )
        return cast(AudioArray, np.asarray(frame, dtype=np.float32))

    async def prime_detection_buffer(self, *, reason: str = "manual") -> None:
        """Fill the rolling wake-detection window before reporting wake readiness."""
        if self._detection_hop_seconds >= self._detection_seconds:
            logger.info(
                "[Audio] Detection buffer priming skipped for non-overlap mode",
                extra=_voice_log_extra(
                    event="audio_detection_buffer_prime_skipped",
                    reason=reason,
                    window_s=self._detection_seconds,
                    hop_s=self._detection_hop_seconds,
                ),
            )
            return

        window_samples = max(int(self.sample_rate * self._detection_seconds), 1)
        if self._detection_buffer_filled_samples >= window_samples:
            logger.info(
                "[Audio] Detection buffer already primed",
                extra=_voice_log_extra(
                    event="audio_detection_buffer_primed",
                    reason=reason,
                    frames_recorded=0,
                    duration_s=0.0,
                    buffer_filled_samples=self._detection_buffer_filled_samples,
                    window_samples=window_samples,
                    ready=True,
                ),
            )
            return

        timeout_s = max(2.0, self._detection_seconds * 2.0)
        started_at = time.perf_counter()
        frames_recorded = 0
        logger.info(
            "[Audio] Priming wake detection buffer",
            extra=_voice_log_extra(
                event="audio_detection_buffer_prime_start",
                reason=reason,
                window_s=self._detection_seconds,
                hop_s=self._detection_hop_seconds,
                buffer_filled_samples=self._detection_buffer_filled_samples,
                window_samples=window_samples,
                timeout_s=timeout_s,
            ),
        )

        while self._detection_buffer_filled_samples < window_samples:
            if time.perf_counter() - started_at >= timeout_s:
                break
            before = self._detection_buffer_filled_samples
            await self.detection_frame()
            frames_recorded += 1
            if self._detection_buffer_filled_samples <= before:
                break

        duration_s = round(time.perf_counter() - started_at, 3)
        ready = self._detection_buffer_filled_samples >= window_samples
        log = logger.info if ready else logger.warning
        message = (
            "[Audio] Wake detection buffer primed"
            if ready
            else "[Audio] Wake detection buffer prime incomplete"
        )
        log(
            message,
            extra=_voice_log_extra(
                event="audio_detection_buffer_primed",
                reason=reason,
                frames_recorded=frames_recorded,
                duration_s=duration_s,
                buffer_filled_samples=self._detection_buffer_filled_samples,
                window_samples=window_samples,
                ready=ready,
            ),
        )

    def reset_detection_buffer(self, *, reason: str = "manual") -> None:
        self._detection_buffer = None
        self._detection_buffer_filled_samples = 0
        logger.debug(
            "MIC DEBUG: detection overlap buffer reset",
            extra=_voice_log_extra(event="audio_detection_buffer_reset", reason=reason),
        )

    async def record_phrase(self, duration: float | None = None) -> AudioArray:
        """Record user speech after wake word."""
        if duration is not None:
            return await self._record(duration)
        if not bool(getattr(settings, "command_adaptive_capture_enabled", True)):
            return await self._record(self._capture_seconds)
        return await self._record_adaptive_phrase()

    async def _record_adaptive_phrase(self) -> AudioArray:
        """Record until end-of-speech, bounded by configured safety limits."""
        np = _require_numpy()
        base_duration = max(float(self._capture_seconds), 0.1)
        min_duration = max(
            float(getattr(settings, "command_min_capture_seconds", _COMMAND_CAPTURE_MIN_SECONDS)),
            base_duration,
        )
        max_duration = max(
            float(getattr(settings, "command_max_capture_seconds", _COMMAND_CAPTURE_MAX_SECONDS)),
            min_duration,
        )
        silence_seconds = max(
            float(
                getattr(
                    settings,
                    "command_end_silence_seconds",
                    _COMMAND_CAPTURE_END_SILENCE_SECONDS,
                )
            ),
            _COMMAND_CAPTURE_CHUNK_SECONDS,
        )
        rms_threshold = max(
            float(getattr(settings, "command_vad_rms_threshold", _COMMAND_CAPTURE_RMS_THRESHOLD)),
            0.0,
        )
        chunk_seconds = min(_COMMAND_CAPTURE_CHUNK_SECONDS, max_duration)

        logger.info(
            "[Audio] Adaptive command capture starting",
            extra=_voice_log_extra(
                event="audio_adaptive_capture_start",
                base_duration_s=base_duration,
                min_duration_s=min_duration,
                max_duration_s=max_duration,
                end_silence_s=silence_seconds,
                rms_threshold=rms_threshold,
            ),
        )

        chunks: list[AudioArray] = []
        total_duration = 0.0
        speech_started = False
        last_voice_at: float | None = None
        peak_rms = 0.0
        stop_reason = "max_duration"

        while total_duration < max_duration:
            remaining = max_duration - total_duration
            chunk_duration = min(chunk_seconds, remaining)
            chunk = await self._record(chunk_duration)
            chunk = cast(AudioArray, np.asarray(chunk, dtype=np.float32).reshape(-1))
            chunks.append(chunk)
            total_duration += chunk_duration

            rms = 0.0
            if chunk.size:
                rms = float(np.sqrt(np.mean(chunk * chunk)))
                peak_rms = max(peak_rms, rms)
            if rms >= rms_threshold:
                speech_started = True
                last_voice_at = total_duration

            if total_duration < min_duration:
                continue

            if speech_started and last_voice_at is not None:
                trailing_silence = total_duration - last_voice_at
                if trailing_silence >= silence_seconds:
                    stop_reason = "end_silence"
                    break
            elif total_duration >= base_duration:
                stop_reason = "no_speech"
                break

        audio = (
            np.concatenate(chunks).astype(np.float32, copy=False)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )
        logger.info(
            "[Audio] Adaptive command capture complete",
            extra=_voice_log_extra(
                event="audio_adaptive_capture_complete",
                audio_duration_s=round(total_duration, 3),
                audio_samples=int(audio.size),
                speech_started=speech_started,
                peak_rms=round(peak_rms, 6),
                stop_reason=stop_reason,
            ),
        )
        return cast(AudioArray, audio)

    async def _record(self, duration: float) -> AudioArray:
        """Internal recording method."""
        np = _require_numpy()
        if duration <= 0:
            raise AudioDeviceError("Recording duration must be positive")

        if self._recorder is not None:
            result = self._recorder(duration)
            if asyncio.iscoroutine(result):
                result = await result
            if result is None:
                raise AudioDeviceError("Audio recorder returned no audio")
            return cast(AudioArray, np.asarray(result, dtype=np.float32).reshape(-1))

        sd = _require_sounddevice()

        frames = max(int(self.sample_rate * duration), 1)

        def _capture() -> np.ndarray:  # type: ignore[name-defined]
            start = time.perf_counter()
            logger.debug("MIC DEBUG: _record start duration=%.2f frames=%d", duration, frames)

            recording = sd.rec(
                frames,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self._device_index,
                blocking=True,
            )

            end = time.perf_counter()
            logger.debug("MIC DEBUG: blocking sd.rec returned after %.3fs total", end - start)

            return recording.reshape(-1)

        try:
            data = await asyncio.to_thread(_capture)
        except Exception as exc:
            raise AudioDeviceError(str(exc)) from exc
        return cast(AudioArray, np.asarray(data, dtype=np.float32))


class WakeAcknowledgement:
    """Play acknowledgement sound when wake word is detected."""

    def __init__(
        self,
        sound_path: Path | None = None,
        *,
        filler_phrase: str | None = None,
        is_speaking: Callable[[], bool] | None = None,
        filler_speak: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        default_path = Path(__file__).resolve().parents[1] / "assets" / "wake_acknowledgment.wav"
        self._sound_path = Path(sound_path) if sound_path else default_path
        self._filler_phrase = filler_phrase
        self._is_speaking = is_speaking
        self._filler_speak = filler_speak
        if not filler_phrase and not self._sound_path.exists():
            try:
                ensure_wake_acknowledgment_sound(path=str(self._sound_path))
            except Exception as exc:
                logger.warning("Failed to generate wake acknowledgment sound: %s", exc)

    async def play(self) -> None:
        """Play the wake acknowledgement sound or spoken filler phrase."""
        if self._is_speaking is not None and self._is_speaking():
            logger.debug("TTS is speaking; skipping wake acknowledgment")
            return

        if self._filler_phrase and self._filler_speak is not None:
            try:
                await self._filler_speak(self._filler_phrase)
            except Exception as exc:
                logger.warning("Filler phrase acknowledgment failed: %s", exc)
            return

        if not self._sound_path.exists():
            return

        def _play() -> None:
            if sa is None and _load_sounddevice() is None:
                logger.warning("No audio playback backend available for wake acknowledgment.")
                return
            if sa is not None:
                wave_obj = sa.WaveObject.from_wave_file(str(self._sound_path))
                play_obj = wave_obj.play()
                play_obj.wait_done()
                return
            sd = _require_sounddevice()
            sf = _lazy_import_soundfile()
            if sf is None:
                raise AudioDeviceError("soundfile is required for wake acknowledgement playback")
            data, rate = sf.read(str(self._sound_path), dtype="float32")
            sd.play(data, rate)
            sd.wait()

        try:
            await asyncio.to_thread(_play)
        except Exception as exc:
            logger.warning("Wake acknowledgement failed: %s", exc)


class SpeechToText:
    """Speech-to-text using Whisper."""

    def __init__(
        self,
        model_name: str,
        device: str,
        language: str | None | object = _USE_CONFIG_LANGUAGE,
        *,
        async_load: bool = False,
    ) -> None:
        whisper_module = _lazy_import_whisper()
        if whisper_module is None:
            raise SpeechToTextError("openai-whisper is not installed")

        if language is _USE_CONFIG_LANGUAGE:
            language = getattr(settings, "whisper_language", "en")
        # Normalise "auto" and "" to None so Whisper uses its built-in auto-detect.
        if language in ("auto", ""):
            language = None
        self._language = cast(str | None, language)
        configured_prompt = getattr(settings, "whisper_initial_prompt", None)
        self._initial_prompt = (
            str(configured_prompt).strip()
            if configured_prompt
            else (_DEFAULT_STT_INITIAL_PROMPT if self._language in (None, "en") else "")
        )

        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self._device = device
        self._model_name = model_name
        self._whisper_module = whisper_module
        self._model: Any = None
        self._load_event = threading.Event()
        self._load_error: str | None = None

        if async_load:
            t = threading.Thread(target=self._load_model, daemon=True, name="stt-warmup")
            t.start()
        else:
            self._load_model()
            if self._load_error is not None:
                raise SpeechToTextError(self._load_error)

    def _load_model(self) -> None:
        """Load the Whisper model; called synchronously or from a background thread."""
        try:
            self._model = self._whisper_module.load_model(self._model_name, device=self._device)
            logger.info("[STT] Model '%s' loaded on %s", self._model_name, self._device)
        except Exception as exc:
            self._load_error = str(exc)
            logger.error("[STT] Model load failed: %s", exc)
        finally:
            self._load_event.set()

    def is_loaded(self) -> bool:
        """Return True when the Whisper model has finished loading without error."""
        return self._load_event.is_set() and self._load_error is None

    async def transcribe(
        self,
        audio: AudioArray | bytes | bytearray | memoryview,
        sample_rate: int,
    ) -> str:
        """Transcribe audio to text."""

        model_name = getattr(self, "_model_name", "whisper")
        device = getattr(self, "_device", "unknown")
        initial_prompt = getattr(self, "_initial_prompt", "")

        load_event = getattr(self, "_load_event", None)
        if load_event is not None and not load_event.is_set():
            logger.info("[STT] Waiting for model warm-up to complete...")
            await asyncio.to_thread(load_event.wait)

        if self._load_error is not None:
            raise SpeechToTextError(f"Model failed to load: {self._load_error}")

        prepared_audio = _prepare_audio_for_stt(audio)
        logger.info(
            "[STT] Audio input prepared",
            extra=_voice_log_extra(
                event="stt_audio_quality",
                model=model_name,
                device=device,
                language=self._language or "auto",
                condition_on_previous_text=False,
                initial_prompt_enabled=bool(initial_prompt),
                **_audio_quality_summary(prepared_audio, sample_rate),
            ),
        )
        audio_buffer = _to_wav_buffer(prepared_audio, sample_rate)
        if audio_buffer[:4] != b"RIFF":
            detected_format = _detect_audio_format(audio_buffer)
            raise AudioFormatError(f"Expected WAV, got {detected_format}")

        def _transcribe() -> str:
            def run_transcribe(language: str | None) -> dict[str, Any]:
                kwargs: dict[str, Any] = {
                    "language": language,
                    "fp16": False,
                    "condition_on_previous_text": False,
                }
                if initial_prompt:
                    kwargs["initial_prompt"] = initial_prompt
                try:
                    return cast(dict[str, Any], self._model.transcribe(prepared_audio, **kwargs))
                except TypeError:
                    logger.debug(
                        "[STT] Whisper version does not support all transcription options; "
                        "retrying with basic options",
                    )
                    return cast(
                        dict[str, Any],
                        self._model.transcribe(
                            prepared_audio,
                            language=language,
                            fp16=False,
                        ),
                    )

            try:
                result = run_transcribe(self._language)
            except Exception:
                if self._language is None:
                    logger.warning("[STT] Auto-detect not supported; falling back to language='en'")
                    result = run_transcribe("en")
                else:
                    raise
            return str(result.get("text", "")).strip()

        try:
            return await asyncio.to_thread(_transcribe)
        except Exception as exc:
            logger.error("[STT] Whisper failed: %s", exc, exc_info=True)
            raise SpeechToTextError(str(exc)) from exc


class TextToSpeech:
    """Text-to-speech synthesis."""

    def __init__(self, *, language: str, default_speaker: str | None = None) -> None:
        self._language = language
        self._default_speaker = default_speaker
        self._tts_speed = getattr(settings, "tts_speed", 1.08)

        # Get TTS settings from config (defaults: xtts provider, en-US-AndrewNeural voice)
        self._provider = getattr(settings, "tts_provider", "xtts").lower()
        if self._provider == "edge-tts":
            self._provider = "edge"

        self._edge_voice = getattr(settings, "tts_voice", None) or "en-US-AndrewNeural"

        # Smart speaker output device name (US-SP-002); None → local audio
        self._tts_output_device: str | None = getattr(settings, "tts_output_device", None)

        self._tts = None
        self._xtts_init_error: str | None = None
        self._speaking = threading.Event()
        if self._provider == "xtts":
            self._initialize_xtts()

    def _current_edge_voice(self) -> str:
        """Return the active edge-tts voice, re-reading rex_config.json for hot-swap support."""
        try:
            from rex.config_manager import load_config as _load_json_config

            raw = _load_json_config()
            voice = str(raw.get("models", {}).get("tts_voice", "") or "")
            return voice or self._edge_voice
        except Exception:
            return self._edge_voice

    def is_speaking(self) -> bool:
        """Return True while TTS audio playback is in progress."""
        return self._speaking.is_set()

    def _initialize_xtts(self) -> bool:
        """Initialize XTTS model, storing diagnostics on failure."""
        if self._tts is not None:
            return True

        tts_class = _lazy_import_tts()
        if tts_class is None:
            self._xtts_init_error = "Coqui XTTS is not installed"
            logger.warning("XTTS init skipped: %s", self._xtts_init_error)
            return False

        try:
            from rex.tts_utils import apply_xtts_safe_globals

            apply_xtts_safe_globals()
            torch = import_module("torch")
            self._tts = tts_class(
                model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                progress_bar=False,
            )
            if torch.cuda.is_available() and self._tts is not None:
                self._tts.to("cuda")
            self._xtts_init_error = None
            return True
        except Exception as exc:
            self._xtts_init_error = str(exc)
            logger.warning("XTTS init failed: %s", exc)
            return False

    @staticmethod
    def _settings_int(name: str, default: int) -> int:
        value = getattr(settings, name, default)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float, str)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default
        return default

    async def speak(
        self,
        text: str,
        *,
        speaker_wav: str | None = None,
        prefer_fast: bool = False,
    ) -> dict[str, object]:
        """Synthesize and play text as speech."""
        if not text:
            return {}

        original_text = text
        text = self._clean_text(text)
        if not text:
            return {}

        if text.lstrip().startswith("TOOL_REQUEST:"):
            trailing_answer = self._strip_tool_request_prefix(text)
            if trailing_answer:
                logger.warning(
                    "[TTS] Stripped raw TOOL_REQUEST prefix before speech: %r",
                    text[:200],
                )
                text = trailing_answer
            else:
                logger.error("[TTS] Suppressing raw TOOL_REQUEST from speech: %r", text[:200])
                return {
                    "configured_provider": self._provider,
                    "path_used": "suppressed_tool_request",
                    "fast_short_candidate": False,
                    "fast_short_used": False,
                    "fallback_used": False,
                    "speech_start_delay_s": None,
                }

        max_spoken_chars = self._settings_int("tts_max_spoken_chars", 120)
        fast_short_candidate = (
            prefer_fast
            and self._provider == "edge"
            and os.name == "nt"
            and len(text)
            <= self._settings_int("tts_fast_short_reply_max_chars", 140)
        )
        logger.info(
            "[TTS] Spoken text prepared",
            extra=_voice_log_extra(
                event="tts_spoken_text_prepared",
                provider=self._provider,
                original_text_chars=len(original_text.strip()),
                spoken_text_chars=len(text),
                compact_speech_used=len(text) < len(original_text.strip()),
                max_spoken_chars=max_spoken_chars,
                fast_short_candidate=fast_short_candidate,
            ),
        )
        run_metrics: dict[str, object] = {
            "configured_provider": self._provider,
            "fast_short_candidate": fast_short_candidate,
            "fast_short_used": False,
            "fallback_used": False,
            "speech_start_delay_s": None,
            "spoken_text_chars": len(text),
        }
        self._speaking.set()
        started_at = time.perf_counter()
        logger.info(
            "[TTS] Request started",
            extra=_voice_log_extra(
                event="tts_request_start",
                provider=self._provider,
                text_chars=len(text),
            ),
        )
        try:
            if fast_short_candidate:
                try:
                    logger.info(
                        "[TTS] Using fast local short-reply path",
                        extra=_voice_log_extra(
                            event="tts_fast_short_path_selected",
                            configured_provider=self._provider,
                            text_chars=len(text),
                        ),
                    )
                    run_metrics.update(
                        await self._speak_windows_direct(
                            text,
                            reason="fast_short_reply",
                            request_started_at=started_at,
                        )
                    )
                    run_metrics["fast_short_used"] = True
                    return run_metrics
                except Exception as exc:
                    logger.warning(
                        "[TTS] Fast local short-reply path failed; falling back to %s: %s",
                        self._provider,
                        exc,
                        extra=_voice_log_extra(
                            event="tts_fast_short_path_failed",
                            configured_provider=self._provider,
                            error=str(exc),
                        ),
                    )
                    run_metrics["fallback_used"] = True
                    run_metrics["fast_short_failure"] = str(exc)

            if self._provider == "xtts":
                run_metrics.update(
                    await self._speak_xtts(text, speaker_wav, request_started_at=started_at)
                )
            elif self._provider == "edge":
                run_metrics.update(await self._speak_edge(text, request_started_at=started_at))
            elif self._provider == "windows":
                run_metrics.update(
                    await self._speak_windows(text, request_started_at=started_at)
                )
            else:
                run_metrics["path_used"] = "stdout"
                run_metrics["speech_start_delay_s"] = 0.0
                print(f"Rex: {text}")
        except Exception as exc:
            if self._provider == "xtts" and self._xtts_init_error:
                reason = f"XTTS not initialized ({self._xtts_init_error})"
                logger.error("[TTS] Failed: %s", reason)
            else:
                logger.error("[TTS] Failed: %s", exc)
            run_metrics["fallback_used"] = True
            run_metrics["path_used"] = "stdout_fallback"
            print(f"Rex: {text}")
        finally:
            self._speaking.clear()
            run_metrics.setdefault("total_duration_s", round(time.perf_counter() - started_at, 3))
            logger.info(
                "[TTS] Request finished",
                extra=_voice_log_extra(
                    event="tts_request_end",
                    provider=self._provider,
                    duration_s=round(time.perf_counter() - started_at, 3),
                ),
            )
        return run_metrics

    def _clean_text(self, text: str) -> str:
        """Clean text for TTS."""
        original_text = text
        if "Additional info:" in text:
            text = text.split("Additional info:")[0].strip()
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"uddg=\S+", "", text)
        text = re.sub(r"\[.*?\]", "", text)
        sentences = _split_into_sentences(text)
        text = " ".join(sentences[:2]) if sentences else text.strip()
        max_chars = self._settings_int("tts_max_spoken_chars", 120)
        if max_chars > 0 and len(text) > max_chars and len(text) > 80:
            selected: list[str] = []
            current_len = 0
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                next_len = len(sentence) if not selected else current_len + 1 + len(sentence)
                if next_len > max_chars:
                    break
                selected.append(sentence)
                current_len = next_len

            if selected:
                text = " ".join(selected)
            else:
                text = (
                    original_text.strip()
                    if len(original_text.strip()) <= 80
                    else "I have a longer answer ready. Please check the transcript for the details."
                )
            logger.info(
                "[TTS] Shortened spoken response for voice latency",
                extra=_voice_log_extra(
                    event="tts_spoken_text_shortened",
                    original_chars=len(original_text),
                    spoken_chars=len(text),
                    max_chars=max_chars,
                    sentence_safe=True,
                ),
            )
        return text if text.endswith((".", "!", "?")) else text + "."

    def _strip_tool_request_prefix(self, text: str) -> str:
        """Return natural trailing text after a leading TOOL_REQUEST, if present."""
        stripped = text.lstrip()
        if not stripped.startswith("TOOL_REQUEST:"):
            return text

        payload = stripped[len("TOOL_REQUEST:") :].strip()
        try:
            _, end = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError:
            return ""

        trailing = payload[end:].strip()
        if trailing in {"", ".", "!", "?"}:
            return ""
        return trailing.lstrip(" .!?,;:-")

    def _edge_rate(self) -> str:
        """Convert tts_speed into the rate string expected by edge-tts."""
        try:
            speed = float(self._tts_speed or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        percent = max(-50, min(100, int(round((speed - 1.0) * 100))))
        return f"{percent:+d}%"

    def _trim_pcm_silence(
        self,
        pcm_data: AudioArray,
        sample_rate: int,
        *,
        threshold: int = 180,
        padding_ms: int = 80,
    ) -> AudioArray:
        """Trim leading/trailing near-silence from decoded TTS PCM."""
        numpy = _require_numpy()
        samples = numpy.asarray(pcm_data)
        if samples.size == 0:
            return pcm_data

        mono = samples
        if mono.ndim > 1:
            mono = numpy.max(numpy.abs(mono), axis=1)
        else:
            mono = numpy.abs(mono)

        active = numpy.flatnonzero(mono > threshold)
        if active.size == 0:
            return pcm_data

        padding = max(0, int(round(sample_rate * padding_ms / 1000)))
        start = max(0, int(active[0]) - padding)
        end = min(int(samples.shape[0]), int(active[-1]) + padding + 1)
        if start == 0 and end == int(samples.shape[0]):
            return pcm_data

        trimmed = samples[start:end]
        dropped_frames = int(samples.shape[0]) - int(trimmed.shape[0])
        logger.info(
            "[TTS] Trimmed decoded edge-tts silence",
            extra=_voice_log_extra(
                event="tts_audio_silence_trimmed",
                original_frames=int(samples.shape[0]),
                trimmed_frames=int(trimmed.shape[0]),
                dropped_frames=dropped_frames,
                dropped_s=round(dropped_frames / sample_rate, 3) if sample_rate else 0.0,
            ),
        )
        return cast(AudioArray, numpy.ascontiguousarray(trimmed))

    def _try_smart_speaker(self, wav_path: str) -> bool:
        """Attempt to play *wav_path* on the configured smart speaker.

        Returns ``True`` if the audio was routed successfully so the caller
        can skip local playback.  Returns ``False`` if no smart speaker is
        configured or playback failed (caller should fall back to local audio).
        """
        tts_output_device = getattr(self, "_tts_output_device", None)
        if not tts_output_device:
            return False
        try:
            from rex.audio.smart_speaker_output import get_smart_speaker_output
            from rex.audio.speaker_discovery import get_speaker_discovery

            cached = get_speaker_discovery().get_cached_speakers()
            target = next(
                (s for s in cached if s.name == tts_output_device),
                None,
            )
            if target is None:
                logger.warning(
                    "[TTS] Smart speaker %r not found in cached speakers; falling back to local.",
                    tts_output_device,
                )
                return False
            return get_smart_speaker_output().play_wav(
                wav_path, provider=target.provider, ip=target.ip
            )
        except Exception as exc:
            logger.warning("[TTS] Smart speaker routing failed: %s", exc)
            return False

    async def _speak_xtts(
        self,
        text: str,
        speaker_wav: str | None,
        *,
        request_started_at: float | None = None,
    ) -> dict[str, object]:
        """Synthesize speech using XTTS, playing each chunk immediately."""
        if request_started_at is None:
            request_started_at = time.perf_counter()
        if self._tts is None and not self._initialize_xtts():
            reason = (
                f"XTTS not initialized "
                f"({self._xtts_init_error or 'unknown initialization error'})"
            )
            logger.error("[TTS] Failed: %s", reason)
            logger.warning("XTTS initialization failed; falling back to edge-tts")
            try:
                metrics = await self._speak_edge(text, request_started_at=request_started_at)
            except TypeError as exc:
                if "request_started_at" not in str(exc):
                    raise
                fallback_result = await self._speak_edge(text)  # type: ignore[call-arg]
                metrics = fallback_result if isinstance(fallback_result, dict) else {}
            metrics["fallback_used"] = True
            metrics["path_requested"] = "xtts"
            return metrics
        sf = _lazy_import_soundfile()
        if sf is None:
            raise TextToSpeechError("soundfile is required for XTTS output")
        chunks = chunk_text_for_xtts(text, max_tokens=300)
        if not chunks:
            return {"path_used": "xtts", "speech_start_delay_s": None}

        first_chunk_started_at: float | None = None
        for chunk in chunks:
            if first_chunk_started_at is None:
                first_chunk_started_at = time.perf_counter()
            await self._synthesize_and_play_chunk(chunk, speaker_wav, sf)
        return {
            "path_used": "xtts",
            "speech_start_delay_s": round(
                (first_chunk_started_at or time.perf_counter()) - request_started_at,
                3,
            ),
        }

    async def _synthesize_and_play_chunk(
        self, chunk: str, speaker_wav: str | None, sf: Any
    ) -> None:
        """Synthesize a single text chunk and play it immediately."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            chunk_path = tmp.name

        try:

            tts_engine = self._tts
            if tts_engine is None:
                raise TextToSpeechError("XTTS not initialized")

            def _synthesize(_chunk=chunk, _chunk_path=chunk_path) -> None:
                tts_engine.tts_to_file(
                    text=_chunk,
                    speaker_wav=speaker_wav or self._default_speaker,
                    language=self._language,
                    file_path=_chunk_path,
                    speed=self._tts_speed,
                )

            await asyncio.to_thread(_synthesize)

            if Path(chunk_path).exists():
                routed = await asyncio.to_thread(self._try_smart_speaker, chunk_path)
                if not routed and sa is not None:

                    def _play(_path=chunk_path) -> None:
                        wave_obj = sa.WaveObject.from_wave_file(_path)
                        play_obj = wave_obj.play()
                        play_obj.wait_done()

                    await asyncio.to_thread(_play)
        finally:
            try:
                os.unlink(chunk_path)
            except (OSError, PermissionError) as exc:
                logger.warning("Failed to remove temp file %s: %s", chunk_path, exc)

    async def warmup(self, *, speaker_wav: str | None = None) -> None:
        """Pre-warm the TTS engine by synthesizing a short phrase in the background.

        Call via ``asyncio.create_task(tts.warmup())`` so it does not block startup.
        """
        try:
            logger.info("[TTS] Pre-warming engine...")
            await self.speak(_WARMUP_PHRASE, speaker_wav=speaker_wav)
            logger.info("[TTS] Pre-warm complete.")
        except Exception as exc:
            logger.warning("[TTS] Pre-warm failed (non-fatal): %s", exc)

    async def speak_streaming(
        self,
        sentences: AsyncIterator[str],
        *,
        speaker_wav: str | None = None,
    ) -> None:
        """Speak each sentence from an async iterator as soon as it arrives.

        This enables first audio to begin playing before the full response is
        available, reducing perceived latency.
        """
        try:
            async for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    await self.speak(sentence, speaker_wav=speaker_wav)
                except Exception as exc:
                    logger.error("[TTS streaming] chunk failed: %s", exc)
        except Exception as exc:
            logger.error("[TTS streaming] failed: %s", exc)

    async def _speak_edge(self, text: str, *, request_started_at: float) -> dict[str, object]:
        """Synthesize speech using Edge TTS."""
        try:
            import edge_tts
        except ImportError:
            raise TextToSpeechError("edge-tts is not installed")

        numpy = _require_numpy()
        sf = _lazy_import_soundfile()
        if sf is None:
            raise TextToSpeechError("soundfile is required for Edge TTS playback")

        voice = self._current_edge_voice()
        rate = self._edge_rate()
        edge_started_at = time.perf_counter()
        logger.info(
            "[TTS:edge] Synthesis request started",
            extra=_voice_log_extra(
                event="tts_edge_synthesis_start",
                voice=voice,
                rate=rate,
                simpleaudio_available=sa is not None,
                text_chars=len(text),
            ),
        )
        logger.debug(
            "EDGE DEBUG: entered _speak_edge voice=%s sa=%s text=%r",
            voice,
            sa is not None,
            text[:120],
        )

        audio_bytes = bytearray()
        used_streaming = True
        first_audio_chunk_s: float | None = None
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        try:
            stream = communicate.stream()
            if inspect.isawaitable(stream):
                stream = await stream
            async for chunk in stream:
                if chunk.get("type") != "audio":
                    continue
                data = chunk.get("data", b"")
                if not isinstance(data, bytes):
                    continue
                if not audio_bytes:
                    first_audio_chunk_s = round(time.perf_counter() - request_started_at, 3)
                    logger.info(
                        "[TTS:edge] First synthesis audio chunk received",
                        extra=_voice_log_extra(
                            event="tts_edge_first_audio_chunk",
                            duration_s=round(time.perf_counter() - edge_started_at, 3),
                            bytes=len(data),
                        ),
                    )
                audio_bytes.extend(data)
        except Exception as exc:
            used_streaming = False
            logger.warning(
                "[TTS:edge] Streaming synthesis failed; falling back to file save: %s",
                exc,
                extra=_voice_log_extra(event="tts_edge_stream_failed", error=str(exc)),
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                output_path = tmp.name
            try:
                await communicate.save(output_path)
                audio_bytes.extend(Path(output_path).read_bytes())
            finally:
                try:
                    os.unlink(output_path)
                except (OSError, PermissionError) as unlink_exc:
                    logger.warning("Failed to remove temp file %s: %s", output_path, unlink_exc)

        if not audio_bytes:
            raise TextToSpeechError("Edge TTS returned no audio data")

        logger.info(
            "[TTS:edge] Synthesis audio ready",
            extra=_voice_log_extra(
                event="tts_edge_synthesis_ready",
                duration_s=round(time.perf_counter() - edge_started_at, 3),
                audio_bytes=len(audio_bytes),
                streaming=used_streaming,
            ),
        )
        synthesis_ready_s = round(time.perf_counter() - request_started_at, 3)

        def _decode_from_bytes(_audio=bytes(audio_bytes)):
            try:
                return sf.read(io.BytesIO(_audio), dtype="int16", always_2d=True)
            except Exception:
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp.write(_audio)
                    fallback_path = tmp.name
                try:
                    return sf.read(fallback_path, dtype="int16", always_2d=True)
                finally:
                    try:
                        os.unlink(fallback_path)
                    except (OSError, PermissionError) as exc:
                        logger.warning("Failed to remove temp file %s: %s", fallback_path, exc)

        decode_started_at = time.perf_counter()
        pcm_data, sample_rate = await asyncio.to_thread(_decode_from_bytes)
        pcm_data = numpy.ascontiguousarray(pcm_data)
        pcm_data = self._trim_pcm_silence(pcm_data, int(sample_rate))
        channel_count = int(pcm_data.shape[1]) if pcm_data.ndim > 1 else 1
        audio_duration_s = (
            round(int(pcm_data.shape[0]) / int(sample_rate), 3) if int(sample_rate) else 0.0
        )
        logger.info(
            "[TTS:edge] Audio decoded for local playback",
            extra=_voice_log_extra(
                event="tts_edge_audio_ready",
                duration_s=round(time.perf_counter() - decode_started_at, 3),
                sample_rate=sample_rate,
                channels=channel_count,
                frames=int(pcm_data.shape[0]),
                audio_duration_s=audio_duration_s,
            ),
        )

        if sa is None:
            logger.error("LOCAL PLAYBACK ERROR: simpleaudio is not available in _speak_edge")
            return

        def _play(_pcm=pcm_data, _sample_rate=sample_rate, _channels=channel_count) -> None:
            play_obj = sa.play_buffer(
                _pcm.tobytes(),
                num_channels=_channels,
                bytes_per_sample=2,
                sample_rate=_sample_rate,
            )
            play_obj.wait_done()

        playback_started_at = time.perf_counter()
        logger.info(
            "[TTS:edge] Local playback started",
            extra=_voice_log_extra(
                event="tts_playback_start",
                sample_rate=sample_rate,
                channels=channel_count,
                frames=int(pcm_data.shape[0]),
                audio_duration_s=audio_duration_s,
                speech_start_delay_s=round(time.perf_counter() - request_started_at, 3),
            ),
        )
        logger.debug(
            "EDGE DEBUG: about to play PCM buffer locally sr=%s channels=%s frames=%s",
            sample_rate,
            channel_count,
            int(pcm_data.shape[0]),
        )
        await asyncio.to_thread(_play)
        playback_duration_s = round(time.perf_counter() - playback_started_at, 3)
        logger.info(
            "[TTS:edge] Local playback finished",
            extra=_voice_log_extra(
                event="tts_playback_end",
                duration_s=playback_duration_s,
                audio_duration_s=audio_duration_s,
            ),
        )
        return {
            "path_used": "edge",
            "voice": voice,
            "first_audio_chunk_s": first_audio_chunk_s,
            "synthesis_ready_s": synthesis_ready_s,
            "speech_start_delay_s": round(playback_started_at - request_started_at, 3),
            "playback_duration_s": playback_duration_s,
            "audio_duration_s": audio_duration_s,
        }

    def _windows_sapi_rate(self) -> int:
        try:
            speed = float(self._tts_speed or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        return max(-10, min(10, int(round((speed - 1.0) * 10))))

    async def _speak_windows_direct(
        self,
        text: str,
        *,
        reason: str = "windows_provider",
        request_started_at: float | None = None,
    ) -> dict[str, object]:
        """Speak through native Windows SAPI without Edge network synthesis."""
        if os.name != "nt":
            raise TextToSpeechError("Windows SAPI is only available on Windows")

        base_rate = self._windows_sapi_rate()
        rate_boost = 2 if reason == "fast_short_reply" else 0
        rate = max(-10, min(10, base_rate + rate_boost))
        started_at = time.perf_counter()
        speech_start_delay_s: float | None = None
        logger.info(
            "[TTS:windows] Local SAPI playback started",
            extra=_voice_log_extra(
                event="tts_windows_sapi_start",
                reason=reason,
                text_chars=len(text),
                rate=rate,
                rate_boost=rate_boost,
            ),
        )

        def _speak() -> None:
            nonlocal speech_start_delay_s
            try:
                import pythoncom
                import win32com.client
            except ImportError as exc:
                raise TextToSpeechError("pywin32 is required for Windows SAPI") from exc

            initialized = False
            try:
                try:
                    pythoncom.CoInitialize()
                    initialized = True
                except Exception:
                    initialized = False
                voice = win32com.client.Dispatch("SAPI.SpVoice")
                voice.Rate = rate
                if request_started_at is not None:
                    speech_start_delay_s = round(time.perf_counter() - request_started_at, 3)
                voice.Speak(text)
            finally:
                if initialized:
                    pythoncom.CoUninitialize()

        await asyncio.to_thread(_speak)
        logger.info(
            "[TTS:windows] Local SAPI playback finished",
            extra=_voice_log_extra(
                event="tts_windows_sapi_end",
                reason=reason,
                duration_s=round(time.perf_counter() - started_at, 3),
                text_chars=len(text),
                rate=rate,
                speech_start_delay_s=speech_start_delay_s,
                rate_boost=rate_boost,
            ),
        )
        return {
            "path_used": "windows_sapi",
            "speech_start_delay_s": speech_start_delay_s,
            "playback_duration_s": round(time.perf_counter() - started_at, 3),
            "rate": rate,
            "rate_boost": rate_boost,
        }

    async def _speak_windows(
        self,
        text: str,
        *,
        request_started_at: float,
    ) -> dict[str, object]:
        """Synthesize speech using the pyttsx3 Windows provider."""

        try:
            import pyttsx3
        except ImportError:
            return await self._speak_windows_direct(
                text,
                request_started_at=request_started_at,
            )

        speech_start_delay_s: float | None = None

        def _speak() -> None:
            nonlocal speech_start_delay_s
            engine = pyttsx3.init()
            speech_start_delay_s = round(time.perf_counter() - request_started_at, 3)
            engine.say(text)
            engine.runAndWait()

        started_at = time.perf_counter()
        await asyncio.to_thread(_speak)
        return {
            "path_used": "pyttsx3",
            "speech_start_delay_s": speech_start_delay_s,
            "playback_duration_s": round(time.perf_counter() - started_at, 3),
        }


class VoiceLoop:
    """Main voice assistant loop coordinating wake word, STT, LLM, and TTS."""

    def __init__(
        self,
        assistant,
        *,
        wake_listener,
        detection_source: Callable[[], Awaitable[np.ndarray]],  # type: ignore[name-defined]
        record_phrase: Callable[[], Awaitable[np.ndarray]],  # type: ignore[name-defined]
        transcribe: Callable[[np.ndarray], Awaitable[str]],  # type: ignore[name-defined]
        speak: Callable[[str], Awaitable[None]],
        speak_streaming: Callable[[AsyncIterator[str]], Awaitable[None]] | None = None,
        warmup: Callable[[], Awaitable[None]] | None = None,
        acknowledge: Callable[[], Awaitable[None]] | None = None,
        post_stt_acknowledge: Callable[[], Awaitable[None]] | None = None,
        identify_speaker: IdentifySpeakerCallable | None = None,
        state_callback: Callable[[str], None] | None = None,
        sample_rate: int = 16000,
        stt_timeout: float = 30.0,
        llm_timeout: float = 60.0,
        tts_timeout: float = 30.0,
        post_interaction_cooldown: float = 0.75,
        post_wake_preroll_seconds: float = 0.35,
    ) -> None:
        self._assistant = assistant
        if getattr(settings, "use_openclaw_voice_backend", False):
            from rex.openclaw.http_client import get_openclaw_client
            from rex.openclaw.voice_bridge import VoiceBridge

            # Fail-fast: verify the gateway is reachable before committing to the backend.
            client = get_openclaw_client(settings)
            if client is None:
                gateway_url = getattr(settings, "openclaw_gateway_url", "<not set>") or "<not set>"
                raise RuntimeError(
                    f"OpenClaw voice backend is enabled (use_openclaw_voice_backend=true) "
                    f"but no gateway URL is configured (openclaw_gateway_url={gateway_url!r}). "
                    "Set openclaw_gateway_url in your config or disable the voice backend."
                )
            try:
                client.get("/health")
            except Exception as exc:
                gateway_url = getattr(settings, "openclaw_gateway_url", "<unknown>")
                raise RuntimeError(
                    f"OpenClaw voice backend is enabled but the gateway is unreachable "
                    f"at {gateway_url!r}. Ensure the OpenClaw service is running. "
                    f"Detail: {exc}"
                ) from exc

            self._assistant = VoiceBridge()
            logger.info("Voice loop using OpenClaw VoiceBridge backend")

        self._wake_listener = wake_listener
        self._detection_source = detection_source
        self._record_phrase = record_phrase
        self._transcribe = transcribe
        self._speak = speak
        self._speak_streaming = speak_streaming
        self._warmup = warmup
        self._acknowledge = acknowledge
        self._post_stt_acknowledge = post_stt_acknowledge
        self._identify_speaker = identify_speaker
        self._state_callback = state_callback
        self._identify_speaker_accepts_audio = self._resolve_identify_speaker_signature(
            identify_speaker
        )
        self._sample_rate = sample_rate
        self._stt_timeout = stt_timeout
        self._llm_timeout = llm_timeout
        self._tts_timeout = tts_timeout
        self._post_interaction_cooldown = max(0.0, post_interaction_cooldown)
        self._post_wake_preroll_seconds = max(0.0, post_wake_preroll_seconds)
        self._interaction_id = 0

    @staticmethod
    def _resolve_identify_speaker_signature(
        identify_speaker: IdentifySpeakerCallable | None,
    ) -> bool:
        """Return True when identify_speaker accepts an audio argument."""
        if identify_speaker is None:
            return False
        try:
            signature = inspect.signature(identify_speaker)
        except (TypeError, ValueError):
            return False

        for parameter in signature.parameters.values():
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                return True

        return False

    async def _safe_acknowledge(self, *, interaction_id: int = 0, requested_at: float | None = None) -> None:
        try:
            ack = self._acknowledge
            if ack is not None:
                await ack()
        except Exception as exc:
            logger.warning("[Ack] Acknowledgement tone failed (non-fatal): %s", exc)

    async def _settle_wake_acknowledgement(
        self,
        task: asyncio.Task[None] | None,
        *,
        interaction_id: int,
    ) -> None:
        """Finish or cancel a wake acknowledgement task before later audio stages."""
        if task is None:
            return
        if task.done():
            try:
                await task
            except asyncio.CancelledError:
                pass
            return
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.5)
        except TimeoutError:
            logger.warning(
                "[Ack] Wake acknowledgement was still pending after phrase capture; "
                "cancelling stale acknowledgement task",
                extra={"event": "wake_ack_stale_cancelled", "interaction_id": interaction_id},
            )
            task.cancel()
        except asyncio.CancelledError:
            pass

    async def _post_interaction_reset(self, *, interaction_id: int) -> None:
        """Reset wake state and wait briefly so speaker audio cannot re-trigger wake."""
        token = _VOICE_INTERACTION_ID.set(interaction_id)
        reset = getattr(self._wake_listener, "reset", None)
        try:
            if callable(reset):
                reset(reason="post_interaction")

            source_owner = getattr(self._detection_source, "__self__", None)
            reset_detection_buffer = getattr(source_owner, "reset_detection_buffer", None)
            if callable(reset_detection_buffer):
                reset_detection_buffer(reason="post_interaction")

            if self._post_interaction_cooldown > 0:
                logger.info(
                    "[Wake] Post-interaction cooldown before listening resumes",
                    extra=_voice_log_extra(
                        event="wakeword_cooldown_start",
                        cooldown_s=self._post_interaction_cooldown,
                    ),
                )
                await asyncio.sleep(self._post_interaction_cooldown)
                logger.info(
                    "[Wake] Post-interaction cooldown complete; detector ready",
                    extra=_voice_log_extra(
                        event="wakeword_cooldown_end",
                        detector_ready=True,
                    ),
                )

            await self._prime_wake_detection(reason="post_interaction_reset")
        finally:
            _VOICE_INTERACTION_ID.reset(token)

    async def _prime_wake_detection(self, *, reason: str) -> None:
        """Prime source-side wake audio before marking the listener as ready."""
        source_owner = getattr(self._detection_source, "__self__", None)
        prime_detection_buffer = getattr(source_owner, "prime_detection_buffer", None)
        if not callable(prime_detection_buffer):
            return
        logger.info(
            "[Wake] Priming wake listener audio window",
            extra=_voice_log_extra(event="wake_listen_prime_start", reason=reason),
        )
        try:
            await prime_detection_buffer(reason=reason)
        except Exception as exc:
            logger.warning(
                "[Wake] Wake listener audio priming failed; continuing unprimed: %s",
                exc,
                extra=_voice_log_extra(
                    event="wake_listen_prime_failed",
                    reason=reason,
                    error=str(exc),
                ),
            )
            return
        logger.info(
            "[Wake] Wake listener audio window primed",
            extra=_voice_log_extra(event="wake_listen_prime_complete", reason=reason),
        )

    async def _safe_post_stt_acknowledge(self) -> None:
        """Play post-STT acknowledgement (after transcription, before LLM), suppressing errors."""
        started_at = time.perf_counter()
        try:
            if self._post_stt_acknowledge is not None:
                logger.info(
                    "[Ack] Post-STT acknowledgement starting",
                    extra=_voice_log_extra(event="post_stt_ack_start"),
                )
                await self._post_stt_acknowledge()
                logger.info(
                    "[Ack] Post-STT acknowledgement finished",
                    extra=_voice_log_extra(
                        event="post_stt_ack_end",
                        duration_s=round(time.perf_counter() - started_at, 3),
                    ),
                )
        except Exception as exc:
            logger.warning(
                "[Ack] Post-STT acknowledgement failed (non-fatal): %s",
                exc,
                extra=_voice_log_extra(
                    event="post_stt_ack_failed",
                    duration_s=round(time.perf_counter() - started_at, 3),
                    error=str(exc),
                ),
            )

    def _prepend_wake_preroll(
        self,
        wake_frame: AudioArray,
        audio: AudioArray,
        *,
        interaction_id: int,
    ) -> AudioArray:
        """Prepend the end of the wake frame so no-pause commands are not clipped."""
        if isinstance(audio, (bytes, bytearray, memoryview)) or isinstance(
            wake_frame, (bytes, bytearray, memoryview)
        ):
            return audio
        if wake_frame is None:
            return audio
        if self._post_wake_preroll_seconds <= 0 or self._sample_rate <= 0:
            return audio

        numpy = _require_numpy()
        wake_samples = numpy.asarray(wake_frame, dtype=numpy.float32).reshape(-1)
        phrase_samples = numpy.asarray(audio, dtype=numpy.float32).reshape(-1)
        wake_duration_s = wake_samples.size / self._sample_rate
        if wake_duration_s < _MIN_WAKE_PREROLL_SOURCE_SECONDS:
            return audio

        preroll_samples = min(
            wake_samples.size,
            max(0, int(round(self._sample_rate * self._post_wake_preroll_seconds))),
        )
        if preroll_samples <= 0:
            return audio

        combined = numpy.concatenate([wake_samples[-preroll_samples:], phrase_samples])
        logger.info(
            "[Audio] Wake-frame pre-roll prepended before STT",
            extra={
                "event": "post_wake_preroll_applied",
                "interaction_id": interaction_id,
                "preroll_s": round(preroll_samples / self._sample_rate, 3),
                "preroll_samples": int(preroll_samples),
                "phrase_samples": int(phrase_samples.size),
                "combined_samples": int(combined.size),
            },
        )
        return cast(AudioArray, combined)

    async def _capture_followup_transcript(
        self,
        *,
        interaction_id: int,
        reason: str,
        emit_state: Callable[[str], None],
    ) -> str:
        logger.info(
            "[Voice] Immediate follow-up capture starting",
            extra=_voice_log_extra(
                event="voice_followup_capture_start",
                interaction_id=interaction_id,
                reason=reason,
            ),
        )
        emit_state("followup_listening")
        capture_started_at = time.perf_counter()
        audio = await self._record_phrase()
        audio_samples = len(audio) if hasattr(audio, "__len__") else 0
        audio_duration_s = audio_samples / self._sample_rate if self._sample_rate > 0 else 0.0
        logger.info(
            "[Voice] Immediate follow-up audio capture complete",
            extra=_voice_log_extra(
                event="voice_followup_capture_complete",
                interaction_id=interaction_id,
                reason=reason,
                audio_duration_s=audio_duration_s,
                audio_samples=audio_samples,
                capture_elapsed_s=round(time.perf_counter() - capture_started_at, 3),
            ),
        )

        emit_state("processing")
        try:
            transcript = await asyncio.wait_for(self._transcribe(audio), timeout=self._stt_timeout)
        except TimeoutError:
            logger.error(
                "Follow-up STT stage timed out after %.0fs",
                self._stt_timeout,
                extra=_voice_log_extra(
                    event="pipeline_timeout",
                    interaction_id=interaction_id,
                    stage="followup_stt",
                ),
            )
            return ""

        raw_transcript = transcript.strip()
        transcript = _strip_wake_prefix(raw_transcript)
        logger.info(
            "[Voice] Immediate follow-up transcript: %r",
            transcript,
            extra=_voice_log_extra(
                event="voice_followup_transcript",
                interaction_id=interaction_id,
                reason=reason,
                raw_transcript=raw_transcript,
                transcript=transcript,
            ),
        )
        return transcript

    async def warmup(self) -> None:
        """Pre-warm TTS in the background.

        Schedule as a fire-and-forget task::

            asyncio.create_task(voice_loop.warmup())
        """
        if self._warmup is not None:
            await self._warmup()

    async def run(self, max_interactions: int | None = None) -> None:
        """Run the voice loop for a specified number of interactions."""
        from .voice_latency import VoiceLatencyTracker  # noqa: PLC0415

        def _emit(status: str) -> None:
            """Emit a status event (best-effort, never raises)."""
            try:
                from rex.dashboard.sse import emit_status  # noqa: PLC0415

                emit_status(status)
            except Exception:
                pass
            if self._state_callback is not None:
                try:
                    self._state_callback(status)
                except Exception:
                    pass

        def _emit_wake_listening(*, reason: str) -> None:
            mark_listening_started = getattr(
                self._wake_listener,
                "mark_listening_started",
                None,
            )
            if callable(mark_listening_started):
                mark_listening_started(reason=reason)
            logger.info(
                "[Wake] Wake listener armed",
                extra={"event": "wake_listen_armed", "reason": reason},
            )
            _emit("wake_listening")

        interactions = 0
        voice_mode_kwargs = dict(voice_mode=True)
        _speak_streaming = self._speak_streaming

        logger.info(
            "[Wake] Wake listen requested",
            extra={"event": "wake_listen_requested", "reason": "voice_loop_start"},
        )
        await self._prime_wake_detection(reason="voice_loop_start")
        _emit_wake_listening(reason="voice_loop_start")
        try:
            async for wake_frame in self._wake_listener.listen(self._detection_source):
                try:
                    self._interaction_id += 1
                    interaction_id = self._interaction_id
                    tracker = VoiceLatencyTracker()
                    logger.info(
                        "[Wake] Interaction accepted",
                        extra={"event": "wake_interaction_start", "interaction_id": interaction_id},
                    )
                    _emit("listening")

                    # Fire acknowledgment tone concurrently with recording so the
                    # microphone starts capturing immediately after wake word.
                    # Playback failure is suppressed to keep the pipeline running.
                    ack_task: asyncio.Task[None] | None = None
                    if self._acknowledge:
                        ack_task = asyncio.create_task(
                            self._safe_acknowledge(
                                interaction_id=interaction_id,
                                requested_at=time.monotonic(),
                            )
                        )

                    # Record user speech.  Keep the tail of the accepted wake frame
                    # because no-pause commands can otherwise be clipped before
                    # phrase capture begins.
                    capture_started_at = time.perf_counter()
                    logger.info(
                        "[Audio] Post-wake phrase capture starting",
                        extra={
                            "event": "post_wake_capture_start",
                            "interaction_id": interaction_id,
                        },
                    )
                    audio = await self._record_phrase()
                    audio = self._prepend_wake_preroll(
                        wake_frame,
                        audio,
                        interaction_id=interaction_id,
                    )
                    await self._settle_wake_acknowledgement(ack_task, interaction_id=interaction_id)

                    audio_samples = len(audio) if hasattr(audio, "__len__") else 0
                    audio_duration_s = (
                        audio_samples / self._sample_rate if self._sample_rate > 0 else 0.0
                    )
                    logger.info(
                        "Audio capture complete: %.2fs captured",
                        audio_duration_s,
                        extra={
                            "event": "audio_capture_complete",
                            "interaction_id": interaction_id,
                            "audio_duration_s": audio_duration_s,
                            "audio_samples": audio_samples,
                            "capture_elapsed_s": round(time.perf_counter() - capture_started_at, 3),
                        },
                    )

                    # Optionally identify the speaker from voice
                    if self._identify_speaker is not None:
                        try:
                            if self._identify_speaker_accepts_audio:
                                cast(Any, self._identify_speaker)(audio)
                            else:
                                cast(Any, self._identify_speaker)()
                        except Exception as exc:
                            logger.warning("Voice identity check failed: %s", exc)

                    # Transcribe to text
                    logger.debug(
                        "Handing audio buffer to STT engine (%d samples)",
                        audio_samples,
                        extra={
                            "event": "stt_handoff",
                            "interaction_id": interaction_id,
                            "audio_samples": audio_samples,
                        },
                    )
                    tracker.mark("stt_start")
                    try:
                        transcript = await asyncio.wait_for(
                            self._transcribe(audio), timeout=self._stt_timeout
                        )
                    except TimeoutError:
                        logger.error(
                            "STT stage timed out after %.0fs — resetting pipeline",
                            self._stt_timeout,
                            extra={
                                "event": "pipeline_timeout",
                                "interaction_id": interaction_id,
                                "stage": "stt",
                            },
                        )
                        continue
                    tracker.mark("stt_end")
                    raw_transcript = transcript.strip()
                    stripped_transcript = _strip_wake_prefix(raw_transcript)
                    transcript = (
                        stripped_transcript
                        if stripped_transcript != raw_transcript
                        else raw_transcript
                    )
                    if transcript != raw_transcript:
                        logger.info(
                            "[STT] Stripped leaked wake phrase from transcript",
                            extra={
                                "event": "stt_wake_prefix_stripped",
                                "interaction_id": interaction_id,
                                "raw_transcript": raw_transcript,
                                "transcript": transcript,
                            },
                        )
                    if not transcript:
                        logger.info("No speech detected")
                        _emit("cooldown")
                        await self._post_interaction_reset(interaction_id=interaction_id)
                        _emit_wake_listening(reason="no_speech_reset")
                        continue

                    logger.info(
                        "[STT] Transcript: %r",
                        transcript,
                        extra={
                            "event": "stt_transcript",
                            "interaction_id": interaction_id,
                            "transcript": transcript,
                        },
                    )
                    if _is_weak_transcript_fragment(transcript):
                        initial_fragment = transcript
                        logger.warning(
                            "[STT] Asking for repeat after weak transcript fragment: %r",
                            transcript,
                            extra={
                                "event": "stt_weak_transcript",
                                "interaction_id": interaction_id,
                                "transcript": transcript,
                            },
                        )
                        _emit("thinking")
                        token = _VOICE_INTERACTION_ID.set(interaction_id)
                        try:
                            await asyncio.wait_for(
                                self._speak(_WEAK_TRANSCRIPT_RETRY_PROMPT),
                                timeout=self._tts_timeout,
                            )
                        finally:
                            _VOICE_INTERACTION_ID.reset(token)
                        transcript = await self._capture_followup_transcript(
                            interaction_id=interaction_id,
                            reason="weak_transcript_retry",
                            emit_state=_emit,
                        )
                        if not transcript or _is_weak_transcript_fragment(transcript):
                            logger.warning(
                                "[STT] Follow-up after weak transcript was still unusable",
                                extra={
                                    "event": "stt_weak_transcript_followup_failed",
                                    "interaction_id": interaction_id,
                                    "initial_transcript": initial_fragment,
                                    "followup_transcript": transcript,
                                },
                            )
                            _emit("cooldown")
                            await self._post_interaction_reset(interaction_id=interaction_id)
                            _emit_wake_listening(reason="weak_transcript_reset")
                            continue
                        logger.info(
                            "[Voice] Continuing interaction with immediate follow-up transcript",
                            extra={
                                "event": "voice_followup_continued",
                                "interaction_id": interaction_id,
                                "initial_transcript": initial_fragment,
                                "followup_transcript": transcript,
                            },
                        )

                    if _is_suspicious_voice_transcript(transcript):
                        suspicious_transcript = transcript
                        logger.warning(
                            "[STT] Asking for confirmation after suspicious transcript: %r",
                            transcript,
                            extra={
                                "event": "stt_suspicious_transcript",
                                "interaction_id": interaction_id,
                                "transcript": transcript,
                            },
                        )
                        _emit("thinking")
                        token = _VOICE_INTERACTION_ID.set(interaction_id)
                        try:
                            await asyncio.wait_for(
                                self._speak(_SUSPICIOUS_TRANSCRIPT_RETRY_PROMPT),
                                timeout=self._tts_timeout,
                            )
                        finally:
                            _VOICE_INTERACTION_ID.reset(token)
                        transcript = await self._capture_followup_transcript(
                            interaction_id=interaction_id,
                            reason="suspicious_transcript_retry",
                            emit_state=_emit,
                        )
                        if (
                            not transcript
                            or _is_weak_transcript_fragment(transcript)
                            or _is_low_value_transcript(transcript)
                            or _is_suspicious_voice_transcript(transcript)
                        ):
                            logger.warning(
                                "[STT] Follow-up after suspicious transcript was unusable",
                                extra={
                                    "event": "stt_suspicious_transcript_followup_failed",
                                    "interaction_id": interaction_id,
                                    "initial_transcript": suspicious_transcript,
                                    "followup_transcript": transcript,
                                },
                            )
                            _emit("cooldown")
                            await self._post_interaction_reset(interaction_id=interaction_id)
                            _emit_wake_listening(reason="suspicious_transcript_reset")
                            continue
                        logger.info(
                            "[Voice] Continuing interaction with confirmed follow-up transcript",
                            extra={
                                "event": "voice_suspicious_transcript_continued",
                                "interaction_id": interaction_id,
                                "initial_transcript": suspicious_transcript,
                                "followup_transcript": transcript,
                            },
                        )

                    if _is_low_value_transcript(transcript):
                        logger.warning(
                            "[STT] Ignoring likely filler transcript: %r",
                            transcript,
                            extra={
                                "event": "stt_transcript_ignored",
                                "interaction_id": interaction_id,
                                "transcript": transcript,
                            },
                        )
                        _emit("cooldown")
                        await self._post_interaction_reset(interaction_id=interaction_id)
                        _emit_wake_listening(reason="ignored_transcript_reset")
                        continue

                    _emit("thinking")

                    # Post-STT acknowledgment: fires after transcription and before
                    # LLM processing, giving the user quick confirmation that their
                    # command was heard.  Runs inline (not as a background task) so
                    # the ack completes within the 500 ms budget before LLM starts.
                    if self._post_stt_acknowledge is not None:
                        await self._safe_post_stt_acknowledge()

                    stream_reply = getattr(self._assistant, "stream_reply", None)

                    # Get LLM response - voice_mode=True enables conciseness prompt
                    _emit("executing")
                    tracker.mark("llm_start")
                    llm_response: str | None = None
                    if _speak_streaming is not None and callable(stream_reply):
                        tracker.mark("tts_synthesis_start")
                        tracker.mark("tts_first_chunk")
                        try:
                            token = _VOICE_INTERACTION_ID.set(interaction_id)
                            try:
                                await asyncio.wait_for(
                                    _speak_streaming(
                                        _sentence_buffer_stream(
                                            stream_reply(transcript, **voice_mode_kwargs)
                                        )
                                    ),
                                    timeout=self._llm_timeout + self._tts_timeout,
                                )
                            finally:
                                _VOICE_INTERACTION_ID.reset(token)
                        except TimeoutError:
                            logger.error(
                                "LLM+TTS streaming stage timed out after %.0fs — resetting pipeline",
                                self._llm_timeout + self._tts_timeout,
                                extra={
                                    "event": "pipeline_timeout",
                                    "interaction_id": interaction_id,
                                    "stage": "llm_tts_streaming",
                                },
                            )
                            continue
                        tracker.mark("llm_end")
                    else:
                        try:
                            llm_response = await asyncio.wait_for(
                                self._assistant.generate_reply(transcript, **voice_mode_kwargs),
                                timeout=self._llm_timeout,
                            )
                        except TimeoutError:
                            logger.error(
                                "LLM stage timed out after %.0fs — resetting pipeline",
                                self._llm_timeout,
                                extra={
                                    "event": "pipeline_timeout",
                                    "interaction_id": interaction_id,
                                    "stage": "llm",
                                },
                            )
                            continue
                        tracker.mark("llm_end")

                        if not llm_response:
                            continue

                        if not llm_response.endswith((".", "!", "?")):
                            llm_response = llm_response + "."

                        try:
                            logger.info(
                                "[Voice] Text response ready; starting TTS",
                                extra={
                                    "event": "voice_text_response_ready",
                                    "interaction_id": interaction_id,
                                    "response_chars": len(llm_response),
                                },
                            )
                            tracker.mark("tts_synthesis_start")
                            token = _VOICE_INTERACTION_ID.set(interaction_id)
                            try:
                                await asyncio.wait_for(
                                    self._speak(llm_response), timeout=self._tts_timeout
                                )
                            finally:
                                _VOICE_INTERACTION_ID.reset(token)
                        except TimeoutError:
                            logger.error(
                                "TTS stage timed out after %.0fs — resetting pipeline",
                                self._tts_timeout,
                                extra={
                                    "event": "pipeline_timeout",
                                    "interaction_id": interaction_id,
                                    "stage": "tts",
                                    "llm_response": llm_response,
                                },
                            )
                            continue
                        if _looks_like_clarification_reply(llm_response, transcript):
                            followup_transcript = await self._capture_followup_transcript(
                                interaction_id=interaction_id,
                                reason="assistant_clarification",
                                emit_state=_emit,
                            )
                            if (
                                followup_transcript
                                and not _is_weak_transcript_fragment(followup_transcript)
                                and not _is_low_value_transcript(followup_transcript)
                                and not _is_suspicious_voice_transcript(followup_transcript)
                            ):
                                continued_transcript = _combine_followup_transcript(
                                    transcript,
                                    followup_transcript,
                                )
                                logger.info(
                                    "[Voice] Assistant clarification answered; generating follow-up reply",
                                    extra={
                                        "event": "voice_clarification_followup",
                                        "interaction_id": interaction_id,
                                        "initial_transcript": transcript,
                                        "followup_transcript": followup_transcript,
                                        "continued_transcript": continued_transcript,
                                    },
                                )
                                _emit("executing")
                                try:
                                    followup_response = await asyncio.wait_for(
                                        self._assistant.generate_reply(
                                            continued_transcript,
                                            voice_mode=True,
                                        ),
                                        timeout=self._llm_timeout,
                                    )
                                except TimeoutError:
                                    logger.error(
                                        "Clarification follow-up LLM stage timed out after %.0fs",
                                        self._llm_timeout,
                                        extra={
                                            "event": "pipeline_timeout",
                                            "interaction_id": interaction_id,
                                            "stage": "clarification_followup_llm",
                                        },
                                    )
                                    continue

                                if followup_response:
                                    if not followup_response.endswith((".", "!", "?")):
                                        followup_response = followup_response + "."
                                    logger.info(
                                        "[Voice] Clarification follow-up response ready; starting TTS",
                                        extra={
                                            "event": "voice_clarification_followup_response_ready",
                                            "interaction_id": interaction_id,
                                            "response_chars": len(followup_response),
                                        },
                                    )
                                    token = _VOICE_INTERACTION_ID.set(interaction_id)
                                    try:
                                        try:
                                            await asyncio.wait_for(
                                                self._speak(followup_response),
                                                timeout=self._tts_timeout,
                                            )
                                        except TimeoutError:
                                            logger.error(
                                                "Clarification follow-up TTS stage timed out after %.0fs",
                                                self._tts_timeout,
                                                extra={
                                                    "event": "pipeline_timeout",
                                                    "interaction_id": interaction_id,
                                                    "stage": "clarification_followup_tts",
                                                },
                                            )
                                            continue
                                    finally:
                                        _VOICE_INTERACTION_ID.reset(token)
                            else:
                                logger.info(
                                    "[Voice] No usable immediate answer to clarification",
                                    extra={
                                        "event": "voice_clarification_followup_empty",
                                        "interaction_id": interaction_id,
                                        "initial_transcript": transcript,
                                        "followup_transcript": followup_transcript,
                                    },
                                )
                    tracker.mark("tts_synthesis_end")
                    tracker.mark("playback_start")
                    tracker.log_summary()
                    _emit("cooldown")
                    await self._post_interaction_reset(interaction_id=interaction_id)
                    _emit_wake_listening(reason="post_interaction_reset")

                except SpeechToTextError as exc:
                    logger.error(
                        "STT error: %s — resetting pipeline",
                        exc,
                        exc_info=True,
                        extra={"event": "stt_error", "error": str(exc)},
                    )
                    _emit("error")
                    # Continue loop on transcription errors
                except TextToSpeechError as exc:
                    logger.error(
                        "TTS error: %s — resetting pipeline",
                        exc,
                        extra={
                            "event": "tts_error",
                            "error": str(exc),
                            "llm_response": llm_response,
                        },
                    )
                    _emit("error")
                    # Continue loop on TTS errors; text response preserved in log
                except AudioDeviceError as exc:
                    logger.error("Audio device error: %s", exc)
                    _emit("error")
                    break
                except Exception as exc:
                    logger.error("Unexpected error in voice loop: %s", exc)
                    _emit("error")

                interactions += 1
                if max_interactions is not None and interactions >= max_interactions:
                    break
        except AudioDeviceError as exc:
            logger.error(
                "Audio device error — pipeline halted: %s",
                exc,
                extra={"event": "pipeline_blocker", "stage": "audio_device", "error": str(exc)},
            )
            _emit("error")


def _build_voice_id_callback() -> IdentifySpeakerCallable | None:
    """Build an identify_speaker callback if voice identity is enabled.

    Reads the voice_identity config section, loads enrolled embeddings, and
    returns a callback that:
    - Converts a numpy audio array to PCM bytes
    - Generates an embedding via the configured backend
    - Runs recognition against all enrolled users
    - Calls resolve_speaker_identity() to update the session user

    Returns None when voice identity is disabled or no users are enrolled.
    All errors are caught and logged; the callback never raises.
    """
    try:
        from rex.config_manager import load_config as _load_json_config
        from rex.voice_identity.types import VoiceIdentityConfig

        raw_cfg = _load_json_config()
        vi_dict = raw_cfg.get("voice_identity", {})
        vi_cfg = VoiceIdentityConfig(
            enabled=vi_dict.get("enabled", False),
            accept_threshold=float(vi_dict.get("accept_threshold", 0.85)),
            review_threshold=float(vi_dict.get("review_threshold", 0.65)),
            embedding_dim=int(vi_dict.get("embedding_dim", 192)),
            model_id=str(vi_dict.get("model_id", "synthetic")),
        )
    except Exception as exc:
        logger.debug("Could not load voice_identity config: %s", exc)
        return None

    if not vi_cfg.enabled:
        return None

    try:
        from rex.voice_identity.embeddings_store import EmbeddingsStore
        from rex.voice_identity.optional_deps import get_embedding_backend
        from rex.voice_identity.recognizer import SpeakerRecognizer

        memory_dir = Path(__file__).resolve().parent.parent / "Memory"
        store = EmbeddingsStore(memory_dir)
        enrolled = store.load_all()

        if not enrolled:
            logger.info(
                "Voice identity enabled but no users are enrolled. "
                "Use 'rex voice-id enroll' to enroll users."
            )
            return None

        backend = get_embedding_backend(vi_cfg.model_id, dim=vi_cfg.embedding_dim)
        recognizer = SpeakerRecognizer(vi_cfg)

        logger.info(
            "Voice identity active: backend=%s, enrolled=%d user(s), " "accept=%.2f, review=%.2f",
            vi_cfg.model_id,
            len(enrolled),
            vi_cfg.accept_threshold,
            vi_cfg.review_threshold,
        )
    except ImportError as exc:
        logger.warning(
            "Voice identity backend unavailable: %s. "
            "Install optional extras: pip install '.[voice-id]'",
            exc,
        )
        return None
    except Exception as exc:
        logger.warning("Failed to initialise voice identity: %s", exc)
        return None

    def _identify(audio: AudioArray) -> str | None:
        try:
            # Convert numpy float32 array to raw bytes for the embedding backend
            np_mod = _lazy_import_numpy()
            if np_mod is not None:
                pcm_bytes = np_mod.asarray(audio, dtype=np_mod.float32).tobytes()
            else:
                # Fallback: use bytes() if numpy unavailable at call time
                pcm_bytes = bytes(audio)

            vector = backend.embed(pcm_bytes)
            result = recognizer.recognize(vector, enrolled)

            from rex.voice_identity.fallback_flow import resolve_speaker_identity

            resolved = resolve_speaker_identity(result)

            if result.decision.value == "recognized":
                logger.info(
                    "Voice recognized: user=%s score=%.3f",
                    result.best_user_id,
                    result.score,
                )
            elif result.decision.value == "review":
                logger.info(
                    "Voice uncertain (review): best_match=%s score=%.3f. "
                    "Run 'rex identify' to set user manually.",
                    result.best_user_id,
                    result.score,
                )

            return resolved
        except Exception as exc:
            logger.warning("Voice identity check failed: %s", exc)
            return None

    return _identify


def build_voice_loop(
    assistant,
    *,
    sample_rate: int = 16000,
    detection_seconds: float = 1.0,
    capture_seconds: float | None = None,
    whisper_model: str = "base",
    device: str = "auto",
    language: str = "en",
    speaker_wav: str | None = None,
    wake_sound_path: Path | None = None,
) -> VoiceLoop:
    """Build a VoiceLoop with default components.

    When ``voice_identity.enabled=true`` is set in ``config/rex_config.json``
    and at least one user is enrolled, an ``identify_speaker`` callback is
    built and wired into the voice loop automatically.
    """
    logger.info(
        "[Pipeline] Initialising voice pipeline stages...",
        extra={"event": "pipeline_stage_start", "stage": "audio_device"},
    )
    if capture_seconds is None:
        configured_capture = getattr(settings, "capture_seconds", None)
        if not isinstance(configured_capture, (int, float, str)) or configured_capture == "":
            configured_capture = getattr(settings, "command_duration", 5.0)
        if not isinstance(configured_capture, (int, float, str)) or configured_capture == "":
            configured_capture = 5.0
        capture_seconds = float(configured_capture)
    try:
        input_device_index = _validate_input_device_index(settings.audio_input_device)
    except AudioDeviceError as exc:
        logger.error(
            "[Pipeline] Audio device stage failed: %s",
            exc,
            extra={"event": "pipeline_stage_failed", "stage": "audio_device", "error": str(exc)},
        )
        raise
    logger.info(
        "[Pipeline] Audio device stage OK (index=%s)",
        input_device_index,
        extra={
            "event": "pipeline_stage_ok",
            "stage": "audio_device",
            "device_index": input_device_index,
        },
    )

    from .wakeword.listener import build_default_detector

    # Smart speaker microphone input (US-SP-003)
    smart_mic_recorder = None
    wake_word_device = getattr(settings, "wake_word_input_device", None)
    if wake_word_device and wake_word_device != "auto":
        try:
            from rex.audio.smart_speaker_mic import SmartSpeakerMic
            from rex.audio.speaker_discovery import get_speaker_discovery

            cached = get_speaker_discovery().get_cached_speakers()
            target = next((s for s in cached if s.name == wake_word_device), None)
            if target is not None:
                smart_mic = SmartSpeakerMic(
                    provider=target.provider,
                    ip=target.ip,
                    sample_rate=sample_rate,
                )
                if smart_mic.connect():
                    smart_mic_recorder = cast(RecorderCallable, smart_mic.read_frame)
                    logger.info(
                        "[voice] Wake word input routed to %r (%s).", target.name, target.ip
                    )
                else:
                    logger.warning(
                        "[voice] Smart speaker mic %r unavailable; falling back to local mic.",
                        wake_word_device,
                    )
            else:
                logger.warning(
                    "[voice] Wake word device %r not found in cached speakers; using local mic.",
                    wake_word_device,
                )
        except Exception as exc:
            logger.warning("[voice] Smart speaker mic setup failed: %s — using local mic.", exc)

    mic = AsyncMicrophone(
        sample_rate=sample_rate,
        detection_seconds=detection_seconds,
        capture_seconds=capture_seconds,
        device_index=input_device_index,
        recorder=smart_mic_recorder,
    )

    logger.info(
        "[Pipeline] Initialising wake-word detector...",
        extra={"event": "pipeline_stage_start", "stage": "wake_word"},
    )
    try:
        wake_listener = build_default_detector(
            sample_rate=sample_rate,
            chunk_duration=detection_seconds,
            threshold=getattr(settings, "wakeword_threshold", 0.1),
            poll_interval=getattr(settings, "wakeword_poll_interval", 0.01),
            keyword=getattr(settings, "wakeword_keyword", None)
            or getattr(settings, "wakeword", None),
            model_path=getattr(settings, "wakeword_model_path", None),
            embedding_path=getattr(settings, "wakeword_embedding_path", None),
            backend=getattr(settings, "wakeword_backend", None),
            fallback_to_builtin=getattr(settings, "wakeword_fallback_to_builtin", True),
            fallback_keyword=getattr(settings, "wakeword_fallback_keyword", "hey jarvis"),
        )
    except Exception as exc:
        logger.error(
            "[Pipeline] Wake-word stage failed: %s",
            exc,
            extra={"event": "pipeline_stage_failed", "stage": "wake_word", "error": str(exc)},
        )
        raise
    logger.info(
        "[Pipeline] Wake-word detector ready",
        extra={"event": "pipeline_stage_ok", "stage": "wake_word"},
    )

    logger.info(
        "[Pipeline] Initialising STT (model=%s, device=%s)...",
        whisper_model,
        device,
        extra={
            "event": "pipeline_stage_start",
            "stage": "stt",
            "model": whisper_model,
            "device": device,
        },
    )
    stt = SpeechToText(
        model_name=whisper_model,
        device=device,
        language=language,
        async_load=True,
    )
    logger.info(
        "[Pipeline] STT initialised (background model load in progress)",
        extra={"event": "pipeline_stage_ok", "stage": "stt"},
    )

    logger.info(
        "[Pipeline] Initialising TTS (language=%s)...",
        language,
        extra={"event": "pipeline_stage_start", "stage": "tts", "language": language},
    )
    tts = TextToSpeech(language=language, default_speaker=speaker_wav)
    logger.info(
        "[Pipeline] TTS initialised (provider=%s)",
        tts._provider,
        extra={"event": "pipeline_stage_ok", "stage": "tts", "provider": tts._provider},
    )

    # AC US-020 #1: warn at startup if FFmpeg is absent and XTTS is active.
    # XTTS relies on torio which uses FFmpeg for audio decoding; other providers
    # (edge-tts, pyttsx3) work without it so the warning is scoped to XTTS.
    if tts._provider == "xtts" and shutil.which("ffmpeg") is None:
        logger.warning(
            "[Pipeline] FFmpeg not found on PATH. XTTS requires FFmpeg for audio "
            "decoding. Install FFmpeg or switch to a different TTS provider. "
            "Windows: https://ffmpeg.org/download.html  "
            "macOS: brew install ffmpeg  "
            "Linux: sudo apt install ffmpeg",
            extra={"event": "ffmpeg_missing", "tts_provider": tts._provider},
        )

    ack_sound = getattr(settings, "acknowledgment_sound", "chime")
    if ack_sound and ack_sound != "chime" and not ack_sound.lower().endswith((".wav", ".mp3")):
        # Spoken filler phrase (e.g. "mm-hmm", "one moment")
        ack = WakeAcknowledgement(
            filler_phrase=ack_sound,
            is_speaking=tts.is_speaking,
            filler_speak=lambda text: tts.speak(text),
        )
    elif ack_sound and ack_sound != "chime":
        # Custom audio file path
        ack = WakeAcknowledgement(
            sound_path=Path(ack_sound),
            is_speaking=tts.is_speaking,
        )
    else:
        # Default chime (use wake_sound_path override if provided)
        ack = WakeAcknowledgement(
            sound_path=wake_sound_path,
            is_speaking=tts.is_speaking,
        )

    identify_speaker = _build_voice_id_callback()

    # Build post-STT acknowledgment based on acknowledgment_mode config.
    # "sound" → play the chime after STT; "phrase" → speak a filler phrase;
    # "none" → no post-STT acknowledgment.
    ack_mode = getattr(settings, "acknowledgment_mode", "sound")
    post_stt_ack: Callable[[], Awaitable[None]] | None
    if ack_mode == "phrase":
        _phrase = "On it"
        post_stt_ack = lambda: tts.speak(_phrase)  # noqa: E731
    elif ack_mode == "sound":
        post_stt_ack = ack.play
    else:
        post_stt_ack = None

    logger.info(
        "[Pipeline] All stages ready — voice loop active",
        extra={"event": "pipeline_ready"},
    )

    return VoiceLoop(
        assistant,
        wake_listener=wake_listener,
        detection_source=mic.detection_frame,
        record_phrase=mic.record_phrase,
        transcribe=lambda audio: stt.transcribe(audio, sample_rate),
        speak=lambda text: tts.speak(text, speaker_wav=speaker_wav),
        speak_streaming=lambda sentences: tts.speak_streaming(sentences, speaker_wav=speaker_wav),
        warmup=lambda: tts.warmup(speaker_wav=speaker_wav),
        acknowledge=ack.play,
        post_stt_acknowledge=post_stt_ack,
        identify_speaker=identify_speaker,
        sample_rate=sample_rate,
    )


def _resolve_voice_reference() -> str | None:
    """Resolve voice reference for the default user.

    Returns:
        Path to voice sample file, or None if not configured
    """
    try:
        users_map = load_users_map()
        profiles = load_all_profiles()

        # Get default user
        default_user = settings.default_user or settings.user_id or "default"
        user_key = resolve_user_key(default_user, users_map, profiles=profiles)

        if not user_key:
            user_key = default_user

        # Load profile and extract voice reference
        if user_key in profiles:
            return extract_voice_reference(profiles[user_key], user_key=user_key)

        return None
    except Exception as exc:
        logger.warning("Failed to resolve voice reference: %s", exc)
        return None


__all__ = [
    "AsyncMicrophone",
    "WakeAcknowledgement",
    "SpeechToText",
    "SynthesizedAudio",
    "TextToSpeech",
    "VoiceLoop",
    "build_voice_loop",
    "_resolve_voice_reference",
]
