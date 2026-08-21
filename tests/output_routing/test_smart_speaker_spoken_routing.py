from __future__ import annotations

from pathlib import Path

import numpy as np

from rex.voice.tts import TextToSpeech, routed_output_target


def _bare_tts() -> TextToSpeech:
    tts = TextToSpeech.__new__(TextToSpeech)
    tts._tts_output_device = None
    return tts


def test_canonical_sonos_target_routes_existing_wav_by_provider_and_ip(monkeypatch) -> None:
    calls: list[tuple[str, str, str]] = []

    class Output:
        def play_wav(self, wav_path: str, *, provider: str, ip: str) -> bool:
            calls.append((wav_path, provider, ip))
            return True

    monkeypatch.setattr(
        "rex.audio.smart_speaker_output.get_smart_speaker_output",
        lambda: Output(),
    )
    tts = _bare_tts()

    with routed_output_target("sonos:192.168.1.50"):
        assert tts._try_smart_speaker("reply.wav") is True

    assert calls == [("reply.wav", "sonos", "192.168.1.50")]


def test_routed_pcm_is_written_to_wav_and_sent_to_smart_speaker(monkeypatch, tmp_path) -> None:
    observed: list[tuple[str, bytes]] = []
    tts = _bare_tts()

    def route_wav(path: str) -> bool:
        observed.append((Path(path).suffix, Path(path).read_bytes()))
        return True

    monkeypatch.setattr(tts, "_try_smart_speaker", route_wav)
    pcm = np.zeros((160, 1), dtype=np.int16)

    with routed_output_target("bose:192.168.1.60"):
        assert tts._route_pcm_to_smart_speaker(pcm, 16000, 1) is True

    assert observed
    assert observed[0][0] == ".wav"
    assert observed[0][1].startswith(b"RIFF")


def test_output_target_context_is_scoped() -> None:
    tts = _bare_tts()
    assert tts._current_routed_output_target() is None
    with routed_output_target("sonos:192.168.1.50"):
        assert tts._current_routed_output_target() == "sonos:192.168.1.50"
    assert tts._current_routed_output_target() is None
