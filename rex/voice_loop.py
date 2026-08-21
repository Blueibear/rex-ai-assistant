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

warnings.filterwarnings("ignore", message=".*FFmpeg extension.*")
warnings.filterwarnings("ignore", message=".*libtorio.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torio")

logger = logging.getLogger(__name__)

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
    build_voice_loop as _build_voice_loop_impl,
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

np = _lazy_import_numpy()
sa = _lazy_import_simpleaudio()
sd = None
_NDArray = np.ndarray if np is not None else Any


def build_voice_loop(assistant, *args: Any, **kwargs: Any) -> VoiceLoop:
    """Build the canonical loop and apply user-scoped spoken-output routing."""
    origin_device_id = kwargs.pop("origin_device_id", None)
    loop = _build_voice_loop_impl(assistant, *args, **kwargs)
    user_id = getattr(assistant, "user_id", None)
    if not isinstance(user_id, str) or not user_id:
        return loop

    original_speak = loop._speak
    original_streaming = loop._speak_streaming

    async def routed_speak(text: str) -> None:
        from rex.output_routing.runtime import get_output_routing_service, user_local_now
        from rex.output_routing.spoken import deliver_spoken_response, send_remote_spoken_text

        try:
            routing = get_output_routing_service()
        except RuntimeError:
            await original_speak(text)
            return
        result = await deliver_spoken_response(
            text,
            routing=routing,
            user_id=user_id,
            origin_device_id=origin_device_id,
            at=user_local_now(user_id),
            remote_sender=send_remote_spoken_text,
            local_speak=original_speak,
        )
        if not result.delivered:
            raise TextToSpeechError(
                f"Spoken response target delivery failed: {result.reason}"
            )

    async def routed_streaming(sentences) -> None:  # noqa: ANN001
        if original_streaming is None:
            parts = [part async for part in sentences]
            if parts:
                await routed_speak(" ".join(parts))
            return
        from rex.output_routing.execution import resolve_spoken_response
        from rex.output_routing.runtime import get_output_routing_service, user_local_now

        try:
            routing = get_output_routing_service()
            route = resolve_spoken_response(
                routing,
                user_id=user_id,
                origin_device_id=origin_device_id,
                at=user_local_now(user_id),
            )
        except RuntimeError:
            await original_streaming(sentences)
            return
        if route.target_id is None or route.suppressed:
            await original_streaming(sentences)
            return
        parts = [part async for part in sentences]
        if parts:
            await routed_speak(" ".join(parts))

    loop._speak = routed_speak
    if original_streaming is not None:
        loop._speak_streaming = routed_streaming
    return loop


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