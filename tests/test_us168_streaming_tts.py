"""Tests for US-168: Streaming TTS — play first audio chunk before full response."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
import pytest

REPO_ROOT = Path(__file__).parent.parent
VOICE_LOOP_SRC = REPO_ROOT / "rex" / "voice_loop.py"


def _src() -> str:
    return VOICE_LOOP_SRC.read_text(encoding="utf-8")


# ── Source-level checks ───────────────────────────────────────────────────────


class TestSourceStreaming:
    def test_split_into_sentences_function_exists(self):
        assert "_split_into_sentences" in _src()

    def test_sentence_stream_async_generator_exists(self):
        assert "_sentence_stream" in _src()

    def test_speak_streaming_method_on_tts(self):
        assert "async def speak_streaming" in _src()

    def test_speak_streaming_accepts_sentences_param(self):
        src = _src()
        idx = src.index("async def speak_streaming")
        sig_end = src.index(":", idx)
        sig = src[idx:sig_end]
        assert "sentences" in sig

    def test_voice_loop_has_speak_streaming_param(self):
        src = _src()
        idx = src.index("class VoiceLoop")
        init_idx = src.index("def __init__", idx)
        sig_end = src.index(") -> None:", init_idx)
        sig = src[init_idx:sig_end]
        assert "speak_streaming" in sig

    def test_voice_loop_run_uses_speak_streaming(self):
        src = _src()
        # Find run method body
        idx = src.index("async def run(self, max_interactions")
        brace_end = len(src)
        depth = 0
        for i, ch in enumerate(src[idx:], start=idx):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    brace_end = i
                    break
        run_body = src[idx:idx + 3000]
        assert "_speak_streaming" in run_body

    def test_sentence_split_in_run(self):
        src = _src()
        assert "_sentence_stream" in src

    def test_build_voice_loop_passes_speak_streaming(self):
        src = _src()
        idx = src.index("def build_voice_loop(")
        fn_end = src.index("\n\n\n", idx)
        fn_body = src[idx:fn_end]
        assert "speak_streaming" in fn_body

    def test_xtts_plays_each_chunk_immediately(self):
        # _synthesize_and_play_chunk should exist for immediate chunk playback
        assert "_synthesize_and_play_chunk" in _src()

    def test_xtts_no_longer_concatenates_all_audio(self):
        src = _src()
        idx = src.index("async def _speak_xtts")
        fn_end = src.index("async def _synthesize_and_play_chunk", idx)
        fn_body = src[idx:fn_end]
        # Should NOT concatenate audio segments (the old approach)
        assert "np.concatenate" not in fn_body


# ── Functional tests ──────────────────────────────────────────────────────────


class TestSplitIntoSentences:
    def test_single_sentence(self):
        from rex.voice_loop import _split_into_sentences
        result = _split_into_sentences("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences(self):
        from rex.voice_loop import _split_into_sentences
        result = _split_into_sentences("Hello world. How are you? I'm fine!")
        assert len(result) == 3

    def test_empty_text(self):
        from rex.voice_loop import _split_into_sentences
        result = _split_into_sentences("")
        assert result == []

    def test_strips_whitespace(self):
        from rex.voice_loop import _split_into_sentences
        result = _split_into_sentences("  Hello.   World.  ")
        assert all(s == s.strip() for s in result)


class TestSentenceStream:
    @pytest.mark.asyncio
    async def test_yields_sentences(self):
        from rex.voice_loop import _sentence_stream
        results = []
        async for sentence in _sentence_stream("Hello. World."):
            results.append(sentence)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_yields_nothing_for_empty(self):
        from rex.voice_loop import _sentence_stream
        results = []
        async for sentence in _sentence_stream(""):
            results.append(sentence)
        assert results == []


class TestSpeakStreamingMethod:
    @pytest.mark.asyncio
    async def test_speak_streaming_calls_speak_for_each_sentence(self):
        from rex.voice_loop import TextToSpeech, _sentence_stream

        tts = TextToSpeech.__new__(TextToSpeech)
        tts._provider = "text"  # fallback that prints

        spoken = []
        original_speak = tts.speak

        async def _mock_speak(text, *, speaker_wav=None):
            spoken.append(text)

        tts.speak = _mock_speak  # type: ignore[method-assign]
        tts._clean_text = lambda t: t  # type: ignore[method-assign]

        async def fake_sentences():
            for s in ["Hello.", "World."]:
                yield s

        await tts.speak_streaming(fake_sentences())
        assert len(spoken) == 2

    @pytest.mark.asyncio
    async def test_speak_streaming_skips_empty_sentences(self):
        from rex.voice_loop import TextToSpeech

        tts = TextToSpeech.__new__(TextToSpeech)
        tts._provider = "text"
        tts._clean_text = lambda t: t  # type: ignore[method-assign]

        spoken = []

        async def _mock_speak(text, *, speaker_wav=None):
            spoken.append(text)

        tts.speak = _mock_speak  # type: ignore[method-assign]

        async def fake_sentences():
            for s in ["", "  ", "Hello."]:
                yield s

        await tts.speak_streaming(fake_sentences())
        assert len(spoken) == 1
        assert spoken[0] == "Hello."
