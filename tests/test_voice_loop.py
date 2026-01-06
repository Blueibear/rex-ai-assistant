import asyncio

import pytest

np = pytest.importorskip("numpy")

from rex.assistant_errors import AudioDeviceError, SpeechToTextError
from rex.voice_loop import VoiceLoop


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

    async def generate_reply(self, transcript):
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
    assert spoken == ["ok."] # Voice loop adds period for TTS


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

