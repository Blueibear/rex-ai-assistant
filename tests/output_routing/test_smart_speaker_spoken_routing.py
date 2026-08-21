from __future__ import annotations

from types import SimpleNamespace

import pytest

from rex.audio.speaker_discovery import DiscoveredSpeaker
from rex.voice_loop import _direct_smart_speaker_speak


@pytest.mark.asyncio
async def test_direct_smart_speaker_route_reuses_existing_xtts_wav_path(monkeypatch) -> None:
    source_tts = SimpleNamespace(
        _provider="xtts",
        _language="en",
        _default_speaker="voice.wav",
    )
    routed_devices: list[str | None] = []

    async def original_speak(text: str) -> None:
        _ = (text, source_tts._provider)

    class DedicatedTTS:
        def __init__(self, *, language: str, default_speaker: str | None = None) -> None:
            assert language == "en"
            assert default_speaker == "voice.wav"
            self._provider = "xtts"
            self._tts_output_device: str | None = None

        def _try_smart_speaker(self, _wav_path: str) -> bool:
            routed_devices.append(self._tts_output_device)
            return True

        async def speak(self, _text: str, *, speaker_wav: str | None = None):
            assert speaker_wav == "voice.wav"
            self._try_smart_speaker("reply.wav")
            return {"path_used": "xtts"}

    class Discovery:
        def get_cached_speakers(self):
            return [
                DiscoveredSpeaker(
                    provider="sonos",
                    name="Living Room",
                    ip="192.168.1.50",
                    model="Play:1",
                )
            ]

    monkeypatch.setattr("rex.voice_loop.TextToSpeech", DedicatedTTS)
    monkeypatch.setattr(
        "rex.audio.speaker_discovery.get_speaker_discovery",
        lambda: Discovery(),
    )

    delivered = await _direct_smart_speaker_speak(
        "sonos:192.168.1.50",
        "Hello from Rex",
        original_speak,
    )

    assert delivered is True
    assert routed_devices == ["Living Room"]


@pytest.mark.asyncio
async def test_direct_smart_speaker_route_fails_closed_when_current_tts_cannot_emit_wav() -> None:
    source_tts = SimpleNamespace(
        _provider="edge",
        _language="en",
        _default_speaker=None,
    )
    spoken: list[str] = []

    async def original_speak(text: str) -> None:
        spoken.append(text)
        _ = source_tts._provider

    delivered = await _direct_smart_speaker_speak(
        "sonos:192.168.1.50",
        "Do not duplicate locally",
        original_speak,
    )

    assert delivered is False
    assert spoken == []
