"""Tests for US-169: Pre-warm TTS engine on startup."""

import asyncio
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
VOICE_LOOP_SRC = REPO_ROOT / "rex" / "voice_loop.py"


def _src() -> str:
    return VOICE_LOOP_SRC.read_text(encoding="utf-8")


# ── Source-level checks ────────────────────────────────────────────────────────


class TestSourcePrewarm:
    def test_warmup_phrase_constant_exists(self):
        assert "_WARMUP_PHRASE" in _src()

    def test_warmup_method_on_tts(self):
        assert "async def warmup" in _src()

    def test_warmup_method_accepts_speaker_wav(self):
        src = _src()
        idx = src.index("async def warmup")
        sig_end = src.index(":", idx)
        sig = src[idx:sig_end]
        assert "speaker_wav" in sig

    def test_voice_loop_has_warmup_param(self):
        src = _src()
        idx = src.index("class VoiceLoop")
        init_idx = src.index("def __init__", idx)
        sig_end = src.index(") -> None:", init_idx)
        sig = src[init_idx:sig_end]
        assert "warmup" in sig

    def test_voice_loop_exposes_warmup_coroutine(self):
        src = _src()
        idx = src.index("class VoiceLoop")
        # There should be an async def warmup within VoiceLoop
        warmup_idx = src.index("async def warmup", idx)
        # Confirm it's inside VoiceLoop (before build_voice_loop)
        build_idx = src.index("def build_voice_loop(")
        assert warmup_idx < build_idx

    def test_build_voice_loop_passes_warmup(self):
        src = _src()
        idx = src.index("def build_voice_loop(")
        fn_end = src.index("\n\n\n", idx)
        fn_body = src[idx:fn_end]
        assert "warmup" in fn_body

    def test_warmup_uses_warmup_phrase_constant(self):
        src = _src()
        idx = src.index("async def warmup")
        # Find the body of warmup
        brace_start = src.index(":", idx)
        # Get next ~500 chars
        body = src[brace_start : brace_start + 500]
        assert "_WARMUP_PHRASE" in body

    def test_warmup_logs_completion(self):
        src = _src()
        idx = src.index("async def warmup")
        body = src[idx : idx + 600]
        assert "logger" in body

    def test_warmup_handles_exceptions(self):
        src = _src()
        idx = src.index("async def warmup")
        body = src[idx : idx + 600]
        assert "except" in body


# ── Functional tests ───────────────────────────────────────────────────────────


class TestTTSWarmup:
    @pytest.mark.asyncio
    async def test_warmup_calls_speak(self):
        from rex.voice_loop import TextToSpeech

        tts = TextToSpeech.__new__(TextToSpeech)
        tts._provider = "text"

        spoken = []

        async def _mock_speak(text, *, speaker_wav=None):
            spoken.append(text)

        tts.speak = _mock_speak  # type: ignore[method-assign]

        await tts.warmup()
        assert len(spoken) == 1

    @pytest.mark.asyncio
    async def test_warmup_passes_speaker_wav(self):
        from rex.voice_loop import TextToSpeech

        tts = TextToSpeech.__new__(TextToSpeech)
        tts._provider = "text"

        received_wav = []

        async def _mock_speak(text, *, speaker_wav=None):
            received_wav.append(speaker_wav)

        tts.speak = _mock_speak  # type: ignore[method-assign]

        await tts.warmup(speaker_wav="test.wav")
        assert received_wav == ["test.wav"]

    @pytest.mark.asyncio
    async def test_warmup_does_not_raise_on_speak_failure(self):
        from rex.voice_loop import TextToSpeech

        tts = TextToSpeech.__new__(TextToSpeech)
        tts._provider = "text"

        async def _bad_speak(text, *, speaker_wav=None):
            raise RuntimeError("TTS broken")

        tts.speak = _bad_speak  # type: ignore[method-assign]

        # Should not raise
        await tts.warmup()

    @pytest.mark.asyncio
    async def test_warmup_uses_warmup_phrase(self):
        from rex.voice_loop import _WARMUP_PHRASE, TextToSpeech

        tts = TextToSpeech.__new__(TextToSpeech)
        tts._provider = "text"

        spoken = []

        async def _mock_speak(text, *, speaker_wav=None):
            spoken.append(text)

        tts.speak = _mock_speak  # type: ignore[method-assign]

        await tts.warmup()
        assert spoken[0] == _WARMUP_PHRASE


class TestVoiceLoopWarmup:
    @pytest.mark.asyncio
    async def test_voice_loop_warmup_calls_warmup_callable(self):
        from rex.voice_loop import VoiceLoop

        called = []

        async def _mock_warmup():
            called.append(True)

        loop = VoiceLoop.__new__(VoiceLoop)
        loop._warmup = _mock_warmup

        await loop.warmup()
        assert called == [True]

    @pytest.mark.asyncio
    async def test_voice_loop_warmup_noop_when_not_set(self):
        from rex.voice_loop import VoiceLoop

        loop = VoiceLoop.__new__(VoiceLoop)
        loop._warmup = None

        # Should not raise
        await loop.warmup()

    @pytest.mark.asyncio
    async def test_warmup_can_run_as_background_task(self):
        from rex.voice_loop import VoiceLoop

        done = []

        async def _mock_warmup():
            await asyncio.sleep(0)
            done.append(True)

        loop = VoiceLoop.__new__(VoiceLoop)
        loop._warmup = _mock_warmup

        task = asyncio.create_task(loop.warmup())
        await task
        assert done == [True]
