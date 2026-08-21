from __future__ import annotations

import pytest

from rex.audio.speaker_discovery import DiscoveredSpeaker
from rex.voice.tts import TextToSpeech
from rex.voice_loop import _direct_smart_speaker_speak


@pytest.mark.asyncio
async def test_direct_smart_speaker_route_reuses_existing_xtts_wav_path(monkeypatch) -> None:
    tts = TextToSpeech.__new__(TextToSpeech)
    tts._provider = "xtts"
    tts._tts_output_device = None
    observed: list[tuple[str, str | None]] = []

    async def original_speak(text: str) -> None:
        # Referencing tts keeps the canonical engine in the callback closure,
        # matching rex.voice.builder's normal _speak_for_callback shape.
        observed.append((text, tts._tts_output_device))

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
    assert observed == [("Hello from Rex", "Living Room")]
    assert tts._tts_output_device is None


@pytest.mark.asyncio
async def test_direct_smart_speaker_route_fails_closed_when_current_tts_cannot_emit_wav() -> None:
    tts = TextToSpeech.__new__(TextToSpeech)
    tts._provider = "edge"
    tts._tts_output_device = None
    spoken: list[str] = []

    async def original_speak(text: str) -> None:
        spoken.append(text)
        _ = tts._provider

    delivered = await _direct_smart_speaker_speak(
        "sonos:192.168.1.50",
        "Do not duplicate locally",
        original_speak,
    )

    assert delivered is False
    assert spoken == []
