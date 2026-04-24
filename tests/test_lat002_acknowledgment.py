"""Tests for US-LAT-002: Acoustic "thinking" feedback during processing."""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, patch

# ---------------------------------------------------------------------------
# WakeAcknowledgement – chime path (default behaviour)
# ---------------------------------------------------------------------------


def test_ack_plays_chime_when_sound_exists(tmp_path):
    """WakeAcknowledgement.play() invokes asyncio.to_thread when chime file exists."""
    from rex.voice_loop import WakeAcknowledgement

    chime = tmp_path / "ack.wav"
    chime.write_bytes(b"RIFF")  # minimal non-empty file

    ack = WakeAcknowledgement(sound_path=chime)

    played = []

    async def _run():
        with patch("rex.voice_loop.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await ack.play()
            played.append(mock_thread.called)

    asyncio.run(_run())
    assert played[0] is True, "Expected asyncio.to_thread to be called for chime playback"


def test_ack_skips_when_sound_missing_and_generation_fails(tmp_path):
    """WakeAcknowledgement.play() returns silently when the chime file cannot be generated."""
    from rex.voice_loop import WakeAcknowledgement

    missing = tmp_path / "nonexistent.wav"

    # Prevent automatic generation so the file remains absent
    with patch("rex.voice_loop.ensure_wake_acknowledgment_sound", side_effect=OSError("no disk")):
        ack = WakeAcknowledgement(sound_path=missing)

    called = []

    async def _run():
        with patch("rex.voice_loop.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await ack.play()
            called.append(mock_thread.called)

    asyncio.run(_run())
    assert called[0] is False, "asyncio.to_thread should not be called when chime file is missing"


# ---------------------------------------------------------------------------
# WakeAcknowledgement – filler phrase
# ---------------------------------------------------------------------------


def test_ack_uses_filler_phrase_instead_of_chime(tmp_path):
    """WakeAcknowledgement calls filler_speak when a filler phrase is configured."""
    from rex.voice_loop import WakeAcknowledgement

    spoken: list[str] = []

    async def mock_speak(text: str) -> None:
        spoken.append(text)

    ack = WakeAcknowledgement(
        filler_phrase="mm-hmm",
        is_speaking=lambda: False,
        filler_speak=mock_speak,
    )

    asyncio.run(ack.play())

    assert spoken == ["mm-hmm"]


def test_ack_filler_phrase_does_not_play_chime(tmp_path):
    """With filler_phrase set, the chime wav is never played."""
    from rex.voice_loop import WakeAcknowledgement

    chime = tmp_path / "ack.wav"
    chime.write_bytes(b"RIFF")

    spoken: list[str] = []

    async def mock_speak(text: str) -> None:
        spoken.append(text)

    called = []

    async def _run():
        ack = WakeAcknowledgement(
            sound_path=chime,
            filler_phrase="one moment",
            is_speaking=lambda: False,
            filler_speak=mock_speak,
        )
        with patch("rex.voice_loop.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await ack.play()
            called.append(mock_thread.called)

    asyncio.run(_run())
    assert spoken == ["one moment"]
    assert called[0] is False, "Chime should not play when filler_phrase is set"


# ---------------------------------------------------------------------------
# WakeAcknowledgement – is_speaking guard
# ---------------------------------------------------------------------------


def test_ack_skipped_when_tts_is_speaking():
    """WakeAcknowledgement.play() skips all output when is_speaking() returns True."""
    from rex.voice_loop import WakeAcknowledgement

    spoken: list[str] = []

    async def mock_speak(text: str) -> None:
        spoken.append(text)

    called = []

    async def _run():
        ack = WakeAcknowledgement(
            filler_phrase="mm-hmm",
            is_speaking=lambda: True,  # TTS is speaking
            filler_speak=mock_speak,
        )
        with patch("rex.voice_loop.asyncio.to_thread", new_callable=AsyncMock) as mock_thread:
            await ack.play()
            called.append(mock_thread.called)

    asyncio.run(_run())
    assert spoken == [], "Filler speak must not fire when TTS is speaking"
    assert called[0] is False, "Chime must not play when TTS is speaking"


def test_ack_plays_when_tts_not_speaking(tmp_path):
    """WakeAcknowledgement.play() proceeds when is_speaking() returns False."""
    from rex.voice_loop import WakeAcknowledgement

    spoken: list[str] = []

    async def mock_speak(text: str) -> None:
        spoken.append(text)

    ack = WakeAcknowledgement(
        filler_phrase="one moment",
        is_speaking=lambda: False,
        filler_speak=mock_speak,
    )

    asyncio.run(ack.play())
    assert spoken == ["one moment"]


# ---------------------------------------------------------------------------
# TextToSpeech – is_speaking() flag
# ---------------------------------------------------------------------------


def test_tts_is_speaking_set_during_speak():
    """TextToSpeech._speaking event is set while speak() is executing."""
    from rex.voice_loop import TextToSpeech

    tts = TextToSpeech.__new__(TextToSpeech)
    tts._language = "en"
    tts._default_speaker = None
    tts._tts_speed = 1.0
    tts._provider = "none"
    tts._edge_voice = "en-US-AndrewNeural"
    tts._tts = None
    tts._xtts_init_error = None
    tts._speaking = threading.Event()

    captured_state: list[bool] = []

    async def _run():
        async def _mock_speak(text: str, *, speaker_wav=None) -> None:
            # Monkeypatch speak to capture state mid-call
            tts._speaking.set()
            try:
                captured_state.append(tts.is_speaking())
            finally:
                tts._speaking.clear()

        await _mock_speak("hello")

    asyncio.run(_run())
    assert captured_state == [True], "is_speaking() should return True during TTS playback"


def test_tts_is_speaking_clear_after_speak():
    """TextToSpeech._speaking event is cleared after speak() completes."""
    from rex.voice_loop import TextToSpeech

    tts = TextToSpeech.__new__(TextToSpeech)
    tts._language = "en"
    tts._default_speaker = None
    tts._tts_speed = 1.0
    tts._provider = "none"
    tts._edge_voice = "en-US-AndrewNeural"
    tts._tts = None
    tts._xtts_init_error = None
    tts._speaking = threading.Event()

    async def _run():
        # Direct async call to speak with provider="none" (print path)
        with patch("builtins.print"):
            await tts.speak("hello world")

    asyncio.run(_run())
    assert not tts.is_speaking(), "is_speaking() should return False after TTS completes"


def test_tts_strips_tool_request_prefix_when_answer_follows():
    """TTS speaks the natural answer when raw tool syntax is prefixed."""
    from rex.voice_loop import TextToSpeech

    tts = TextToSpeech.__new__(TextToSpeech)
    tts._language = "en"
    tts._default_speaker = None
    tts._tts_speed = 1.0
    tts._provider = "none"
    tts._edge_voice = "en-US-AndrewNeural"
    tts._tts = None
    tts._xtts_init_error = None
    tts._speaking = threading.Event()

    text = (
        'TOOL_REQUEST: {"tool": "time_now", "args": {"location": "Dallas, TX"}}'
        "The current local time in Dallas is 3:33 AM."
    )

    async def _run():
        with patch("builtins.print") as mock_print:
            await tts.speak(text)
            mock_print.assert_called_once_with("Rex: The current local time in Dallas is 3:33 AM.")

    asyncio.run(_run())


def test_tts_clean_text_limits_long_spoken_response(monkeypatch):
    """Long GUI-visible replies are shortened before spoken playback."""
    from rex.voice_loop import TextToSpeech

    monkeypatch.setattr("rex.voice_loop.settings.tts_max_spoken_chars", 80, raising=False)

    tts = TextToSpeech.__new__(TextToSpeech)
    long_text = (
        "I was created as part of the AskRex project to help with voice and desktop tasks. "
        "The longer details can stay visible in the transcript while speech remains concise."
    )

    spoken = tts._clean_text(long_text)

    assert len(spoken) <= 81
    assert spoken.endswith(".")
    assert "transcript while speech remains concise" not in spoken


def test_tts_clean_text_does_not_cut_mid_sentence(monkeypatch):
    """Very long first sentences use an intentional handoff instead of a cut phrase."""
    from rex.voice_loop import TextToSpeech

    monkeypatch.setattr("rex.voice_loop.settings.tts_max_spoken_chars", 80, raising=False)

    tts = TextToSpeech.__new__(TextToSpeech)
    long_text = (
        "This chocolate cake recipe starts with flour sugar cocoa powder eggs oil milk "
        "vanilla and baking powder before moving to oven timing and frosting details."
    )

    spoken = tts._clean_text(long_text)

    assert spoken == "I have a longer answer ready. Please check the transcript for the details."


def test_gui_spoken_reply_compacts_recipe_to_transcript_handoff():
    """GUI voice speaks an intentional recipe handoff while showing the full transcript."""
    from rex_voice_bridge import _compact_spoken_reply_for_gui

    full_reply = (
        "Here is a chocolate cake recipe. Ingredients: flour, sugar, cocoa powder, "
        "eggs, oil, milk, and vanilla. Step 1: mix the dry ingredients. "
        "Step 2: bake until done."
    )

    spoken, compacted = _compact_spoken_reply_for_gui(full_reply, max_chars=120)

    assert compacted is True
    assert "full recipe" in spoken
    assert "transcript" in spoken
    assert "Step 1" not in spoken
    assert spoken.endswith(".")


def test_gui_spoken_reply_compacts_short_time_answer():
    """Short time answers use a tighter spoken form while keeping full transcript text."""
    from rex_voice_bridge import _compact_spoken_reply_for_gui

    spoken, compacted = _compact_spoken_reply_for_gui(
        "It's 6:49 PM in New York City.",
        max_chars=120,
    )

    assert compacted is True
    assert spoken == "6:49 PM in New York City."


def test_tts_fast_short_path_uses_windows_direct(monkeypatch):
    """Short GUI speech can bypass Edge synthesis through the fast local path."""
    from rex.voice_loop import TextToSpeech

    monkeypatch.setattr("rex.voice_loop.os.name", "nt", raising=False)
    monkeypatch.setattr(
        "rex.voice_loop.settings.tts_fast_short_reply_max_chars",
        140,
        raising=False,
    )

    tts = TextToSpeech.__new__(TextToSpeech)
    tts._language = "en"
    tts._default_speaker = None
    tts._tts_speed = 1.0
    tts._provider = "edge"
    tts._edge_voice = "en-US-AndrewNeural"
    tts._tts = None
    tts._xtts_init_error = None
    tts._speaking = threading.Event()
    spoken: list[str] = []

    async def fake_windows_direct(
        text: str,
        *,
        reason: str = "windows_provider",
        request_started_at: float | None = None,
    ) -> dict[str, object]:
        spoken.append(f"{reason}:{text}")
        return {
            "path_used": "windows_sapi",
            "speech_start_delay_s": 0.01,
        }

    async def fail_edge(_: str) -> None:
        raise AssertionError("Edge should not be used for fast short replies")

    monkeypatch.setattr(tts, "_speak_windows_direct", fake_windows_direct)
    monkeypatch.setattr(tts, "_speak_edge", fail_edge)

    result = asyncio.run(tts.speak("hello", prefer_fast=True))

    assert spoken == ["fast_short_reply:hello."]
    assert result["path_used"] == "windows_sapi"
    assert result["fast_short_used"] is True


def test_tts_fast_short_path_falls_back_to_edge(monkeypatch):
    """Short reply fast path records fallback when local SAPI is unavailable."""
    from rex.voice_loop import TextToSpeech

    monkeypatch.setattr("rex.voice_loop.os.name", "nt", raising=False)
    monkeypatch.setattr(
        "rex.voice_loop.settings.tts_fast_short_reply_max_chars",
        140,
        raising=False,
    )

    tts = TextToSpeech.__new__(TextToSpeech)
    tts._language = "en"
    tts._default_speaker = None
    tts._tts_speed = 1.0
    tts._provider = "edge"
    tts._edge_voice = "en-US-AndrewNeural"
    tts._tts = None
    tts._xtts_init_error = None
    tts._speaking = threading.Event()

    async def fail_windows_direct(
        text: str,
        *,
        reason: str = "windows_provider",
        request_started_at: float | None = None,
    ) -> dict[str, object]:
        raise RuntimeError("sapi unavailable")

    async def fake_edge(
        text: str,
        *,
        request_started_at: float,
    ) -> dict[str, object]:
        return {
            "path_used": "edge",
            "speech_start_delay_s": 0.5,
        }

    monkeypatch.setattr(tts, "_speak_windows_direct", fail_windows_direct)
    monkeypatch.setattr(tts, "_speak_edge", fake_edge)

    result = asyncio.run(tts.speak("hello", prefer_fast=True))

    assert result["path_used"] == "edge"
    assert result["fallback_used"] is True
    assert result["fast_short_candidate"] is True
    assert result["fast_short_used"] is False


# ---------------------------------------------------------------------------
# AppConfig – acknowledgment_sound field
# ---------------------------------------------------------------------------


def test_appconfig_has_acknowledgment_sound_field():
    """AppConfig has acknowledgment_sound with default 'chime'."""
    # Test the dataclass default
    import dataclasses

    from rex.config import AppConfig

    defaults = {
        f.name: f.default for f in dataclasses.fields(AppConfig) if f.name == "acknowledgment_sound"
    }
    assert "acknowledgment_sound" in defaults
    assert defaults["acknowledgment_sound"] == "chime"


def test_build_app_config_acknowledgment_sound_default():
    """build_app_config returns 'chime' when acknowledgment.sound is absent."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
        }
    )
    assert cfg.acknowledgment_sound == "chime"


def test_build_app_config_acknowledgment_sound_phrase():
    """build_app_config reads acknowledgment.sound filler phrase from JSON."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
            "acknowledgment": {"sound": "mm-hmm"},
        }
    )
    assert cfg.acknowledgment_sound == "mm-hmm"
