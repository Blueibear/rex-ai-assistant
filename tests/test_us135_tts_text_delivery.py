"""US-135: TTS text input delivery from LLM response handler.

Verifies that:
- _speak_response is called with the exact text from language_model.generate
- empty or whitespace-only responses skip TTS
- LLM exceptions cause an early return without calling _speak_response
"""

from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("numpy")

# Ensure project root is importable when running from any working directory.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402


def _make_assistant():
    """Return AsyncRexAssistant with all heavy deps replaced by mocks."""
    from rex.config import AppConfig
    from voice_loop import AsyncRexAssistant

    assistant = AsyncRexAssistant.__new__(AsyncRexAssistant)
    assistant.config = AppConfig()
    assistant.language_model = MagicMock()
    assistant._assistant = None  # no high-level Assistant; use language_model fallback
    assistant._tts = MagicMock()
    assistant._whisper_model = MagicMock()
    assistant._wake_model = MagicMock()
    assistant._wake_keyword = "hey_rex"
    assistant._sample_rate = 16000
    assistant.active_user = "james"
    assistant.user_voice_refs = {"james": None}
    assistant.plugins = {}
    assistant._wake_sound_path = None
    assistant._running = True
    assistant._state = "running"
    return assistant


def _run(assistant, fake_audio, speak_side_effect=None):
    """Helper: patch all I/O and run _process_conversation, return speak call args."""
    speak_calls: list[str] = []

    def capture_speak(text: str) -> None:
        speak_calls.append(text)

    with (
        patch("voice_loop.append_history_entry"),
        patch("voice_loop.export_transcript"),
        patch.object(assistant, "_record_audio", return_value=fake_audio),
        patch.object(assistant, "transcribe", new=AsyncMock(return_value="hello rex")),
        patch.object(assistant, "_play_wake_sound"),
        patch.object(
            assistant,
            "_speak_response",
            side_effect=speak_side_effect or capture_speak,
        ),
    ):
        asyncio.run(assistant._process_conversation())

    return speak_calls


@pytest.mark.unit
def test_speak_response_called_with_llm_text():
    """_speak_response receives the exact text returned by language_model.generate."""
    assistant = _make_assistant()
    assistant.language_model.generate.return_value = "Hello, I am Rex!"
    fake_audio = np.zeros(16000, dtype=np.float32)

    speak_calls: list[str] = []

    def capture(text: str) -> None:
        speak_calls.append(text)

    with (
        patch("voice_loop.append_history_entry"),
        patch("voice_loop.export_transcript"),
        patch.object(assistant, "_record_audio", return_value=fake_audio),
        patch.object(assistant, "transcribe", new=AsyncMock(return_value="hello rex")),
        patch.object(assistant, "_play_wake_sound"),
        patch.object(assistant, "_speak_response", side_effect=capture),
    ):
        asyncio.run(assistant._process_conversation())

    assert speak_calls == ["Hello, I am Rex!"]


@pytest.mark.unit
def test_speak_response_not_called_for_empty_response():
    """_speak_response is skipped when language_model.generate returns an empty string."""
    assistant = _make_assistant()
    assistant.language_model.generate.return_value = ""
    fake_audio = np.zeros(16000, dtype=np.float32)

    speak_calls: list[str] = []

    with (
        patch("voice_loop.append_history_entry"),
        patch("voice_loop.export_transcript"),
        patch.object(assistant, "_record_audio", return_value=fake_audio),
        patch.object(assistant, "transcribe", new=AsyncMock(return_value="hello rex")),
        patch.object(assistant, "_play_wake_sound"),
        patch.object(assistant, "_speak_response", side_effect=lambda t: speak_calls.append(t)),
    ):
        asyncio.run(assistant._process_conversation())

    assert speak_calls == []


@pytest.mark.unit
def test_speak_response_not_called_for_whitespace_response():
    """_speak_response is skipped when language_model.generate returns only whitespace."""
    assistant = _make_assistant()
    assistant.language_model.generate.return_value = "   \n\t  "
    fake_audio = np.zeros(16000, dtype=np.float32)

    speak_calls: list[str] = []

    with (
        patch("voice_loop.append_history_entry"),
        patch("voice_loop.export_transcript"),
        patch.object(assistant, "_record_audio", return_value=fake_audio),
        patch.object(assistant, "transcribe", new=AsyncMock(return_value="hello rex")),
        patch.object(assistant, "_play_wake_sound"),
        patch.object(assistant, "_speak_response", side_effect=lambda t: speak_calls.append(t)),
    ):
        asyncio.run(assistant._process_conversation())

    assert speak_calls == []


@pytest.mark.unit
def test_speak_response_not_called_when_llm_raises():
    """_speak_response is not called when language_model.generate raises an exception."""
    assistant = _make_assistant()
    assistant.language_model.generate.side_effect = RuntimeError("LLM crashed")
    fake_audio = np.zeros(16000, dtype=np.float32)

    speak_calls: list[str] = []

    with (
        patch("voice_loop.append_history_entry"),
        patch("voice_loop.export_transcript"),
        patch.object(assistant, "_record_audio", return_value=fake_audio),
        patch.object(assistant, "transcribe", new=AsyncMock(return_value="hello rex")),
        patch.object(assistant, "_play_wake_sound"),
        patch.object(assistant, "_speak_response", side_effect=lambda t: speak_calls.append(t)),
    ):
        asyncio.run(assistant._process_conversation())

    assert speak_calls == []
