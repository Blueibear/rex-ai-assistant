"""Async voice assistant loop orchestrating wake word, STT, LLM, and TTS.

This module is a compatibility wrapper that re-exports the optimized implementation.
For the actual implementation, see voice_loop_optimized.py.
"""

from __future__ import annotations

# Re-export all public symbols from the optimized implementation
from .voice_loop_optimized import (
    AsyncMicrophone,
    SpeechToText,
    TextToSpeech,
    VoiceLoop,
    WakeAcknowledgement,
    build_voice_loop,
)

__all__ = [
    "AsyncMicrophone",
    "WakeAcknowledgement",
    "SpeechToText",
    "TextToSpeech",
    "VoiceLoop",
    "build_voice_loop",
]
