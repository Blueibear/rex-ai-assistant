"""Async voice assistant loop orchestrating wake word, STT, LLM, and TTS.

RELATIONSHIP NOTE — two voice_loop files exist in this repo:
- ``rex/voice_loop.py`` (this file, package): canonical implementation.
  ``rex_loop.py`` imports ``build_voice_loop`` from here and this is the
  authoritative voice loop used when Rex starts.
- ``voice_loop.py`` (repo root): legacy implementation containing
  ``AsyncRexAssistant``. Kept for backward compatibility only. Changes here
  do NOT affect the ``rex_loop.py`` startup path.

Implementation lives in ``rex/voice/`` (one module per concern, see
US-REM-028). This module is the stable facade: every public class, helper,
and module-level object keeps its ``rex.voice_loop.<name>`` import path, and
tests monkeypatch names here (the voice modules resolve patchable names
through this module at call time).
"""

from __future__ import annotations

import asyncio  # noqa: F401  (re-export: tests patch rex.voice_loop.asyncio.*)
import logging
import os  # noqa: F401  (re-export: tests patch rex.voice_loop.os.*)
import warnings
from importlib import import_module  # noqa: F401  (re-export: patched in tests)
from typing import Any

from rex.assistant_errors import (  # noqa: F401  (re-export)
    AudioDeviceError,
    AudioFormatError,
    SpeechToTextError,
    TextToSpeechError,
)
from rex.wake_acknowledgment import ensure_wake_acknowledgment_sound  # noqa: F401

from .config import settings  # noqa: F401  (re-export: patched in tests)

# Suppress torio FFmpeg extension warnings — FFmpeg is not required for audio
# capture/playback (sounddevice handles that).  It is only used internally by
# Coqui XTTS; the XTTS fallback path handles the case where it is absent.
warnings.filterwarnings("ignore", message=".*FFmpeg extension.*")
warnings.filterwarnings("ignore", message=".*libtorio.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torio")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports (US-REM-028). Order matters: the voice modules below resolve
# patchable names (settings, logger, sa, lazy importers, classes) through
# this module at call time, so the bindings above must exist first.
# ---------------------------------------------------------------------------
from rex.voice._types import (  # noqa: E402,F401
    _USE_CONFIG_LANGUAGE,
    AudioArray,
    IdentifySpeakerCallable,
    RecorderCallable,
)
from rex.voice.acknowledgement import WakeAcknowledgement  # noqa: E402,F401
from rex.voice.audio_utils import (  # noqa: E402,F401
    _VOICE_INTERACTION_ID,
    _apply_stt_auto_gain,
    _audio_level,
    _audio_quality_summary,
    _available_input_devices,
    _detect_audio_format,
    _device_name,
    _max_input_channels,
    _prepare_audio_for_stt,
    _to_wav_buffer,
    _validate_input_device_index,
    _voice_log_extra,
)
from rex.voice.builder import (  # noqa: E402,F401
    _build_voice_id_callback,
    _resolve_voice_reference,
    build_voice_loop,
)
from rex.voice.loop import VoiceLoop  # noqa: E402,F401
from rex.voice.microphone import AsyncMicrophone  # noqa: E402,F401
from rex.voice.optional_imports import (  # noqa: E402,F401
    _import_optional,
    _lazy_import_numpy,
    _lazy_import_simpleaudio,
    _lazy_import_soundfile,
    _lazy_import_tts,
    _lazy_import_whisper,
    _load_sounddevice,
    _require_numpy,
    _require_sounddevice,
)
from rex.voice.stt import SpeechToText  # noqa: E402,F401
from rex.voice.transcripts import (  # noqa: E402,F401
    _WARMUP_PHRASE,
    _combine_followup_transcript,
    _extract_completed_sentences,
    _is_low_value_transcript,
    _is_suspicious_voice_transcript,
    _is_weak_transcript_fragment,
    _looks_like_clarification_reply,
    _normalize_transcript_for_guard,
    _protect_abbreviations,
    _sentence_buffer_stream,
    _sentence_stream,
    _split_into_sentences,
    _strip_wake_prefix,
)
from rex.voice.tts import SynthesizedAudio, TextToSpeech  # noqa: E402,F401

# Optional-dependency bindings live on this module (original behavior): tests
# stub ``rex.voice_loop.sa`` / ``rex.voice_loop.sd``, and a fresh import of
# this module with numpy blocked must fall back to ``Any`` for# ``_NDArray``.
np = _lazy_import_numpy()
sa = _lazy_import_simpleaudio()
sd = None

# Backwards-compatible runtime alias used by optional-import tests.
_NDArray = np.ndarray if np is not None else Any

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
