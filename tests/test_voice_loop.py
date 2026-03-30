import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import rex.voice_loop as _rvl
from rex.assistant_errors import AudioDeviceError, AudioFormatError, SpeechToTextError
from rex.voice_loop import TextToSpeech, VoiceLoop

np = pytest.importorskip("numpy")


class DummyListener:
    def __init__(self):
        self._triggered = False

    async def listen(self, source):
        if self._triggered:
            return
        self._triggered = True
        yield await source()

    def stop(self):
        pass


class DummyAssistant:
    def __init__(self):
        self.calls = []

    async def generate_reply(self, transcript, *, voice_mode: bool = False):
        self.calls.append(transcript)
        return "ok"


async def _constant_frame():
    return np.ones(4, dtype=np.float32)


async def _record_phrase():
    return np.ones(4, dtype=np.float32)


async def _transcribe(_: np.ndarray) -> str:
    return "hello world"


async def _speak(_: str) -> None:
    pass


async def _ack():
    pass


@pytest.mark.unit
def test_voice_loop_processes_interaction():
    assistant = DummyAssistant()
    listener = DummyListener()
    spoken = []

    async def speak(text: str) -> None:
        spoken.append(text)

    loop = VoiceLoop(
        assistant,
        wake_listener=listener,
        detection_source=_constant_frame,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=speak,
        acknowledge=_ack,
    )

    asyncio.run(loop.run(max_interactions=1))

    assert assistant.calls == ["hello world"]
    assert spoken == ["ok."]  # Voice loop adds period for TTS


@pytest.mark.unit
def test_voice_loop_handles_transcription_error():
    assistant = DummyAssistant()
    listener = DummyListener()
    spoken = []

    async def failing_transcribe(_: np.ndarray) -> str:
        raise SpeechToTextError("boom")

    async def speak(text: str) -> None:
        spoken.append(text)

    loop = VoiceLoop(
        assistant,
        wake_listener=listener,
        detection_source=_constant_frame,
        record_phrase=_record_phrase,
        transcribe=failing_transcribe,
        speak=speak,
        acknowledge=None,
    )

    asyncio.run(loop.run(max_interactions=1))

    assert assistant.calls == []
    assert spoken == []


@pytest.mark.unit
def test_voice_loop_handles_audio_format_error(caplog):
    assistant = DummyAssistant()
    listener = DummyListener()

    async def failing_transcribe(_: np.ndarray) -> str:
        raise AudioFormatError("Expected WAV, got ID3")

    loop = VoiceLoop(
        assistant,
        wake_listener=listener,
        detection_source=_constant_frame,
        record_phrase=_record_phrase,
        transcribe=failing_transcribe,
        speak=_speak,
        acknowledge=None,
    )

    with caplog.at_level("ERROR"):
        asyncio.run(loop.run(max_interactions=1))

    assert assistant.calls == []
    assert "STT error: Expected WAV, got ID3" in caplog.text


@pytest.mark.unit
def test_voice_loop_identify_speaker_receives_audio_frame():
    assistant = DummyAssistant()
    listener = DummyListener()
    captured = []

    def identify(audio: np.ndarray) -> str | None:
        captured.append(audio.tolist())
        return "alice"

    loop = VoiceLoop(
        assistant,
        wake_listener=listener,
        detection_source=_constant_frame,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=_speak,
        acknowledge=None,
        identify_speaker=identify,
    )

    asyncio.run(loop.run(max_interactions=1))

    assert captured == [[1.0, 1.0, 1.0, 1.0]]


@pytest.mark.unit
def test_voice_loop_identify_speaker_without_args_still_supported():
    assistant = DummyAssistant()
    listener = DummyListener()
    calls = []

    def identify() -> str | None:
        calls.append("called")
        return "alice"

    loop = VoiceLoop(
        assistant,
        wake_listener=listener,
        detection_source=_constant_frame,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=_speak,
        acknowledge=None,
        identify_speaker=identify,
    )

    asyncio.run(loop.run(max_interactions=1))

    assert calls == ["called"]


def test_voice_loop_propagates_audio_errors():
    assistant = DummyAssistant()
    listener = DummyListener()

    async def broken_source():
        raise AudioDeviceError("no mic")

    loop = VoiceLoop(
        assistant,
        wake_listener=listener,
        detection_source=broken_source,
        record_phrase=_record_phrase,
        transcribe=_transcribe,
        speak=_speak,
        acknowledge=None,
    )

    # The loop handles the error internally and retries until max_interactions reached.
    asyncio.run(loop.run(max_interactions=1))

    assert assistant.calls == []


# ---------------------------------------------------------------------------
# Temp file cleanup tests (US-196)
# ---------------------------------------------------------------------------


def _make_tts(monkeypatch, tmp_path):
    """Build a minimal TextToSpeech with a fake XTTS engine and temp files in tmp_path."""
    _orig_ntf = tempfile.NamedTemporaryFile

    def _patched_ntf(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        return _orig_ntf(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _patched_ntf)
    monkeypatch.setattr(_rvl, "sa", None)  # skip audio playback

    tts = TextToSpeech.__new__(TextToSpeech)
    tts._language = "en"
    tts._default_speaker = None
    tts._tts_speed = 1.0
    tts._provider = "xtts"
    tts._xtts_init_error = None

    fake_engine = MagicMock()

    def _write_dummy_wav(text, speaker_wav, language, file_path, speed):
        Path(file_path).write_bytes(b"\x00" * 44)

    fake_engine.tts_to_file.side_effect = _write_dummy_wav
    tts._tts = fake_engine
    return tts


def test_no_leftover_wav_files_after_synthesize_chunk(monkeypatch, tmp_path):
    """Temp .wav files must be cleaned up after a successful synthesis cycle."""
    tts = _make_tts(monkeypatch, tmp_path)
    sf_mock = MagicMock()

    asyncio.run(tts._synthesize_and_play_chunk("hello world", None, sf_mock))

    leftover = list(tmp_path.glob("*.wav"))
    assert leftover == [], f"Leftover .wav files after success: {leftover}"


def test_temp_wav_cleaned_up_even_on_synthesis_error(monkeypatch, tmp_path):
    """Temp .wav files must be cleaned up even when the TTS engine raises."""
    _orig_ntf = tempfile.NamedTemporaryFile

    def _patched_ntf(*args, **kwargs):
        kwargs["dir"] = str(tmp_path)
        return _orig_ntf(*args, **kwargs)

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _patched_ntf)
    monkeypatch.setattr(_rvl, "sa", None)

    tts = TextToSpeech.__new__(TextToSpeech)
    tts._language = "en"
    tts._default_speaker = None
    tts._tts_speed = 1.0
    tts._provider = "xtts"
    tts._xtts_init_error = None

    fake_engine = MagicMock()
    fake_engine.tts_to_file.side_effect = RuntimeError("synthesis exploded")
    tts._tts = fake_engine

    sf_mock = MagicMock()

    with pytest.raises(RuntimeError):
        asyncio.run(tts._synthesize_and_play_chunk("hello", None, sf_mock))

    leftover = list(tmp_path.glob("*.wav"))
    assert leftover == [], f"Leftover .wav files after error: {leftover}"
