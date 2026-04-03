from __future__ import annotations

import socket
from types import SimpleNamespace

from rex.audio.speaker_discovery import (
    DiscoveredSpeaker,
    SpeakerDiscoveryService,
    _BOSE_DISCOVERY_PAYLOAD,
    _parse_bose_response,
)


class _FakeSocket:
    def __init__(self, responses: list[tuple[bytes, tuple[str, int]]]) -> None:
        self._responses = list(responses)
        self.sent_packets: list[tuple[bytes, tuple[str, int]]] = []

    def __enter__(self) -> "_FakeSocket":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def setsockopt(self, *args) -> None:
        return None

    def settimeout(self, value: float) -> None:
        return None

    def sendto(self, payload: bytes, address: tuple[str, int]) -> None:
        self.sent_packets.append((payload, address))

    def recvfrom(self, size: int) -> tuple[bytes, tuple[str, int]]:
        if not self._responses:
            raise socket.timeout()
        return self._responses.pop(0)


def test_parse_bose_response_extracts_name_and_model() -> None:
    payload = (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\n\r\n"
        b"<info><name>Kitchen Bose</name><type>SoundTouch 10</type></info>"
    )

    speaker = _parse_bose_response(payload, "192.168.1.20")

    assert speaker == DiscoveredSpeaker(
        provider="bose",
        name="Kitchen Bose",
        ip="192.168.1.20",
        model="SoundTouch 10",
    )


def test_discover_bose_uses_broadcast_and_caches_unique_responses(monkeypatch) -> None:
    responses = [
        (
            b"HTTP/1.1 200 OK\r\n\r\n<info><name>Living Room</name><type>SoundTouch 30</type></info>",
            ("192.168.1.30", 8090),
        ),
        (
            b"HTTP/1.1 200 OK\r\n\r\n<info><name>Living Room</name><type>SoundTouch 30</type></info>",
            ("192.168.1.30", 8090),
        ),
    ]
    fake_socket = _FakeSocket(responses)
    monkeypatch.setattr("rex.audio.speaker_discovery.socket.socket", lambda *args, **kwargs: fake_socket)

    service = SpeakerDiscoveryService()
    speakers = service._discover_bose()

    assert fake_socket.sent_packets == [(_BOSE_DISCOVERY_PAYLOAD, ("255.255.255.255", 8090))]
    assert speakers == [
        DiscoveredSpeaker(
            provider="bose",
            name="Living Room",
            ip="192.168.1.30",
            model="SoundTouch 30",
        )
    ]


def test_discover_sonos_skips_when_soco_missing(monkeypatch) -> None:
    original_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "soco.discovery":
            raise ImportError("missing soco")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)
    service = SpeakerDiscoveryService()

    assert service._discover_sonos() == []


def test_discover_sonos_normalizes_devices(monkeypatch) -> None:
    class _FakeSpeaker:
        player_name = "Office Sonos"
        ip_address = "192.168.1.40"

        def get_speaker_info(self) -> dict[str, str]:
            return {"model_name": "Era 100"}

        def __hash__(self) -> int:
            return hash(self.ip_address)

    speaker = _FakeSpeaker()
    monkeypatch.setitem(__import__("sys").modules, "soco.discovery", SimpleNamespace(discover=lambda timeout: {speaker}))

    service = SpeakerDiscoveryService()
    speakers = service._discover_sonos()

    assert speakers == [
        DiscoveredSpeaker(
            provider="sonos",
            name="Office Sonos",
            ip="192.168.1.40",
            model="Era 100",
        )
    ]


def test_background_discovery_runs_without_blocking(monkeypatch) -> None:
    calls: list[str] = []

    def fake_discover(self: SpeakerDiscoveryService) -> list[DiscoveredSpeaker]:
        calls.append("discover")
        return []

    monkeypatch.setattr(SpeakerDiscoveryService, "_discover_all", fake_discover)
    service = SpeakerDiscoveryService(refresh_interval_seconds=10.0)
    service.start_background_discovery()
    service._thread.join(timeout=0.2)

    assert calls == ["discover"]
    assert service._thread is not None
    assert service._thread.daemon is True
    service.stop_background_discovery()
