"""Smart speaker discovery for Sonos and Bose devices."""

from __future__ import annotations

import logging
import importlib
import socket
import threading
import time
from dataclasses import dataclass
from html import unescape
from typing import Any

logger = logging.getLogger(__name__)

_DISCOVERY_REFRESH_SECONDS = 60.0
_DEFAULT_DISCOVERY_TIMEOUT = 0.5
_BOSE_DISCOVERY_PAYLOAD = (
    b"GET /info HTTP/1.1\r\n"
    b"Host: 255.255.255.255:8090\r\n"
    b"Accept: application/xml\r\n"
    b"Connection: close\r\n\r\n"
)


@dataclass(frozen=True)
class DiscoveredSpeaker:
    """Description of a discovered smart speaker."""

    provider: str
    name: str
    ip: str
    model: str


def _extract_xml_tag(text: str, tag: str) -> str | None:
    start_token = f"<{tag}>"
    end_token = f"</{tag}>"
    start = text.find(start_token)
    end = text.find(end_token)
    if start == -1 or end == -1 or end <= start:
        return None
    return unescape(text[start + len(start_token) : end].strip())


def _parse_bose_response(payload: bytes, ip_address: str) -> DiscoveredSpeaker | None:
    text = payload.decode("utf-8", errors="ignore")
    body = text.split("\r\n\r\n", 1)[1] if "\r\n\r\n" in text else text
    name = _extract_xml_tag(body, "name")
    model = _extract_xml_tag(body, "type") or _extract_xml_tag(body, "model")
    if not name:
        return None
    return DiscoveredSpeaker(
        provider="bose",
        name=name,
        ip=ip_address,
        model=model or "Bose SoundTouch",
    )


class SpeakerDiscoveryService:
    """Caches smart speaker discovery and refreshes it in the background."""

    def __init__(
        self,
        *,
        refresh_interval_seconds: float = _DISCOVERY_REFRESH_SECONDS,
        discovery_timeout_seconds: float = _DEFAULT_DISCOVERY_TIMEOUT,
    ) -> None:
        self._refresh_interval_seconds = refresh_interval_seconds
        self._discovery_timeout_seconds = discovery_timeout_seconds
        self._cached_speakers: list[DiscoveredSpeaker] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._last_refresh_at = 0.0

    def start_background_discovery(self) -> None:
        """Start periodic discovery in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop,
            name="rex-smart-speaker-discovery",
            daemon=True,
        )
        self._thread.start()

    def stop_background_discovery(self) -> None:
        """Stop the background discovery thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_cached_speakers(self) -> list[DiscoveredSpeaker]:
        """Return cached speakers."""
        with self._lock:
            return list(self._cached_speakers)

    def discover_now(self) -> list[DiscoveredSpeaker]:
        """Refresh the cache immediately and return the current speaker list."""
        speakers = self._discover_all()
        with self._lock:
            self._cached_speakers = speakers
            self._last_refresh_at = time.time()
        return list(speakers)

    def _background_loop(self) -> None:
        self.discover_now()
        while not self._stop_event.wait(self._refresh_interval_seconds):
            try:
                self.discover_now()
            except Exception:  # pragma: no cover - defensive
                logger.exception("speaker discovery refresh failed")

    def _discover_all(self) -> list[DiscoveredSpeaker]:
        speakers_by_key: dict[tuple[str, str], DiscoveredSpeaker] = {}
        for speaker in self._discover_sonos():
            speakers_by_key[(speaker.provider, speaker.ip)] = speaker
        for speaker in self._discover_bose():
            speakers_by_key[(speaker.provider, speaker.ip)] = speaker
        return sorted(speakers_by_key.values(), key=lambda item: (item.provider, item.name.lower()))

    def _discover_sonos(self) -> list[DiscoveredSpeaker]:
        try:
            discovery_module = importlib.import_module("soco.discovery")
        except ImportError:
            logger.debug("soco not installed; skipping Sonos discovery")
            return []

        try:
            discover = getattr(discovery_module, "discover")
            devices = discover(timeout=self._discovery_timeout_seconds) or set()
        except Exception:
            logger.exception("Sonos discovery failed")
            return []

        speakers: list[DiscoveredSpeaker] = []
        for device in devices:
            try:
                info: dict[str, Any] = {}
                get_info = getattr(device, "get_speaker_info", None)
                if callable(get_info):
                    info = get_info() or {}
                speakers.append(
                    DiscoveredSpeaker(
                        provider="sonos",
                        name=str(getattr(device, "player_name", info.get("zone_name", "Sonos Speaker"))),
                        ip=str(getattr(device, "ip_address", info.get("ip_address", ""))),
                        model=str(info.get("model_name") or getattr(device, "model_name", "Sonos")),
                    )
                )
            except Exception:
                logger.exception("failed to normalize Sonos speaker")
        return [speaker for speaker in speakers if speaker.ip]

    def _discover_bose(self) -> list[DiscoveredSpeaker]:
        speakers: dict[str, DiscoveredSpeaker] = {}
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            with sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                sock.settimeout(self._discovery_timeout_seconds)
                sock.sendto(_BOSE_DISCOVERY_PAYLOAD, ("255.255.255.255", 8090))
                while True:
                    try:
                        payload, addr = sock.recvfrom(65535)
                    except socket.timeout:
                        break
                    speaker = _parse_bose_response(payload, addr[0])
                    if speaker is not None:
                        speakers[speaker.ip] = speaker
        except OSError:
            logger.exception("Bose discovery failed")
        return list(speakers.values())


_speaker_discovery = SpeakerDiscoveryService()


def get_speaker_discovery() -> SpeakerDiscoveryService:
    """Return the process-wide speaker discovery service."""
    return _speaker_discovery


def start_smart_speaker_discovery() -> None:
    """Kick off background smart-speaker discovery."""
    get_speaker_discovery().start_background_discovery()
