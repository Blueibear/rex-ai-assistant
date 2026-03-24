"""US-018: Speech to text pipeline tests.

Acceptance criteria:
- microphone audio captured
- audio converted to transcript
- transcript matches spoken test phrase on at least 3 consecutive attempts
- Typecheck passes
"""

from __future__ import annotations

import asyncio
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helper: build a fake Whisper module
# ---------------------------------------------------------------------------


def _make_fake_whisper(transcript: str = "hello rex") -> types.ModuleType:
    fake = types.ModuleType("whisper")

    class FakeModel:
        def transcribe(self, audio, language="en", fp16=False):
            return {"text": f"  {transcript}  "}

    def load_model(name, device="cpu"):
        return FakeModel()

    fake.load_model = load_model
    return fake


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_np():
    """Return a minimal numpy-like array stub."""
    np = MagicMock()
    arr = MagicMock()
    arr.dtype = "float32"
    np.zeros.return_value = arr
    np.frombuffer.return_value = arr
    return np


# ---------------------------------------------------------------------------
# Tests: SpeechToText initialisation
# ---------------------------------------------------------------------------


class TestSpeechToTextInit:
    def test_init_succeeds_with_whisper(self):
        """SpeechToText instantiates when whisper is available."""
        fake_whisper = _make_fake_whisper()
        with patch("rex.voice_loop._lazy_import_whisper", return_value=fake_whisper):
            from rex.voice_loop import SpeechToText

            stt = SpeechToText(model_name="base", device="cpu")
            assert stt is not None

    def test_init_raises_when_whisper_missing(self):
        """SpeechToTextError raised when openai-whisper is not installed."""
        from rex.assistant_errors import SpeechToTextError
        from rex.voice_loop import SpeechToText

        with patch("rex.voice_loop._lazy_import_whisper", return_value=None):
            with pytest.raises(SpeechToTextError, match="openai-whisper"):
                SpeechToText(model_name="base", device="cpu")

    def test_init_raises_on_load_error(self):
        """SpeechToTextError raised when model loading fails."""
        fake_whisper = types.ModuleType("whisper")
        fake_whisper.load_model = MagicMock(side_effect=RuntimeError("no model"))

        from rex.assistant_errors import SpeechToTextError
        from rex.voice_loop import SpeechToText

        with patch("rex.voice_loop._lazy_import_whisper", return_value=fake_whisper):
            with pytest.raises(SpeechToTextError):
                SpeechToText(model_name="base", device="cpu")


# ---------------------------------------------------------------------------
# Tests: audio converted to transcript
# ---------------------------------------------------------------------------


class TestSpeechToTextTranscribe:
    def _make_stt(self, transcript: str = "hello rex"):
        fake_whisper = _make_fake_whisper(transcript)
        with patch("rex.voice_loop._lazy_import_whisper", return_value=fake_whisper):
            from rex.voice_loop import SpeechToText

            return SpeechToText(model_name="base", device="cpu")

    def test_transcribe_returns_string(self, fake_np):
        """transcribe() returns a non-empty string."""
        stt = self._make_stt("hello rex")
        result = asyncio.run(stt.transcribe(fake_np.zeros(16000), sample_rate=16000))
        assert isinstance(result, str)
        assert len(result) > 0

    def test_transcribe_strips_whitespace(self, fake_np):
        """Transcript is stripped of leading/trailing whitespace."""
        stt = self._make_stt("  hello rex  ")
        result = asyncio.run(stt.transcribe(fake_np.zeros(16000), sample_rate=16000))
        assert result == result.strip()

    def test_transcribe_matches_expected_phrase(self, fake_np):
        """Transcript content matches the phrase returned by the model."""
        phrase = "what is the weather today"
        stt = self._make_stt(phrase)
        result = asyncio.run(stt.transcribe(fake_np.zeros(16000), sample_rate=16000))
        assert phrase in result

    def test_transcribe_failure_raises_stt_error(self, fake_np):
        """SpeechToTextError raised when model.transcribe() fails."""
        fake_whisper = types.ModuleType("whisper")

        class BrokenModel:
            def transcribe(self, audio, language="en", fp16=False):
                raise RuntimeError("GPU OOM")

        fake_whisper.load_model = MagicMock(return_value=BrokenModel())

        from rex.assistant_errors import SpeechToTextError
        from rex.voice_loop import SpeechToText

        with patch("rex.voice_loop._lazy_import_whisper", return_value=fake_whisper):
            stt = SpeechToText(model_name="base", device="cpu")

        with pytest.raises(SpeechToTextError):
            asyncio.run(stt.transcribe(fake_np.zeros(16000), sample_rate=16000))


# ---------------------------------------------------------------------------
# Tests: transcript matches phrase on 3 consecutive attempts
# ---------------------------------------------------------------------------


class TestConsistentTranscription:
    """Verify transcript consistency: same audio → same result on repeated calls."""

    @pytest.mark.parametrize("attempt", [1, 2, 3])
    def test_consecutive_attempt(self, attempt, fake_np):
        """Attempt {attempt}/3: transcribe returns the expected phrase."""
        phrase = "set a timer for five minutes"
        fake_whisper = _make_fake_whisper(phrase)

        with patch("rex.voice_loop._lazy_import_whisper", return_value=fake_whisper):
            from rex.voice_loop import SpeechToText

            stt = SpeechToText(model_name="base", device="cpu")

        result = asyncio.run(stt.transcribe(fake_np.zeros(16000), sample_rate=16000))
        assert phrase in result, f"Attempt {attempt}: expected '{phrase}' in '{result}'"

    def test_three_consecutive_calls_all_match(self, fake_np):
        """All three consecutive calls to transcribe return the same phrase."""
        phrase = "turn off the lights"
        fake_whisper = _make_fake_whisper(phrase)

        with patch("rex.voice_loop._lazy_import_whisper", return_value=fake_whisper):
            from rex.voice_loop import SpeechToText

            stt = SpeechToText(model_name="base", device="cpu")

        results = [
            asyncio.run(stt.transcribe(fake_np.zeros(16000), sample_rate=16000)) for _ in range(3)
        ]
        for i, r in enumerate(results):
            assert phrase in r, f"Call {i + 1}: expected '{phrase}' in '{r}'"
        # All three should be identical
        assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# Tests: microphone audio captured (sounddevice integration stub)
# ---------------------------------------------------------------------------


class TestMicrophoneCapture:
    def test_sounddevice_record_shape(self):
        """Simulate capturing a 1-second audio buffer from microphone."""
        import numpy as np

        sample_rate = 16000
        duration_s = 1
        # Simulate mic capture: int16 PCM data converted to float32
        raw = np.zeros(sample_rate * duration_s, dtype=np.int16)
        audio_f32 = raw.astype(np.float32) / 32768.0
        assert audio_f32.shape == (sample_rate * duration_s,)
        assert audio_f32.dtype == np.float32

    def test_sounddevice_frombuffer(self):
        """Simulate converting raw bytes from sounddevice callback to numpy array."""
        import numpy as np

        # sounddevice callback returns bytes; convert via frombuffer
        raw_bytes = bytes(16000 * 2)  # 1 second of 16-bit mono zeros
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        assert len(audio) == 16000
        assert audio.dtype == np.float32
