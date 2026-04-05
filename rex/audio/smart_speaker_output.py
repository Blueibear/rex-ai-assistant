"""Smart speaker TTS output routing (US-SP-002).

Supports Sonos (via the ``soco`` library) and Bose SoundTouch (via its REST
API).  Both providers serve the synthesised WAV file over a temporary HTTP
server running on a free local port.  If the speaker is unreachable or the
required library is not installed the caller receives ``False`` and should fall
back to local audio output.
"""

from __future__ import annotations

import importlib
import logging
import socket
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

_BIND_ALL = "0.0.0.0"
_SONOS_PLAY_BUFFER_SECONDS = 1.0
_BOSE_PORT = 8090
_BOSE_SELECT_TIMEOUT = 5
_PORT_RANGE_START = 49152
_PORT_RANGE_END = 65535


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wav_duration_seconds(wav_path: str) -> float:
    """Return the duration of a WAV file in seconds (defaults to 5.0 on error)."""
    try:
        with wave.open(wav_path) as wf:
            return float(wf.getnframes()) / float(wf.getframerate())
    except Exception:
        return 5.0


def _get_local_ip() -> str:
    """Return the primary LAN IP address (not loopback)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        with s:
            s.connect(("8.8.8.8", 80))
            return cast(str, s.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def _find_free_port(start: int = _PORT_RANGE_START, end: int = _PORT_RANGE_END) -> int:
    """Return a free TCP port in [*start*, *end*), or 0 if none found."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return 0


def _make_file_handler(wav_path: str) -> type[BaseHTTPRequestHandler]:
    """Return an HTTP handler class that serves a single WAV file at any path."""

    class _Handler(BaseHTTPRequestHandler):
        _file_path: str = wav_path

        def do_GET(self) -> None:  # noqa: N802
            try:
                data = Path(self._file_path).read_bytes()
            except OSError:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("[smart-speaker-http] " + fmt, *args)

    return _Handler


def _start_http_server(wav_path: str) -> tuple[HTTPServer, str]:
    """Start a temporary HTTP server serving *wav_path*.

    Returns ``(server, uri)`` where *uri* is the URL the smart speaker can use
    to fetch the audio.  The caller is responsible for calling
    ``server.shutdown()`` when done.

    Raises ``RuntimeError`` if no free port is available.
    """
    port = _find_free_port()
    if not port:
        raise RuntimeError("No free TCP port available for smart-speaker HTTP server")

    local_ip = _get_local_ip()
    server = HTTPServer((_BIND_ALL, port), _make_file_handler(wav_path))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    uri = f"http://{local_ip}:{port}/audio.wav"
    return server, uri


# ---------------------------------------------------------------------------
# SmartSpeakerOutput
# ---------------------------------------------------------------------------


class SmartSpeakerOutput:
    """Routes synthesised WAV audio to a smart speaker device.

    Usage::

        ok = SmartSpeakerOutput().play_wav("/tmp/tts.wav", provider="sonos", ip="192.168.1.10")
        if not ok:
            # fall back to local audio
            ...
    """

    def play_wav(self, wav_path: str, *, provider: str, ip: str) -> bool:
        """Play *wav_path* on the smart speaker identified by *provider* and *ip*.

        Returns ``True`` if the audio was successfully sent to the device, or
        ``False`` if the attempt failed (the caller should fall back to local
        audio).
        """
        if provider == "sonos":
            return self._play_sonos(wav_path, ip)
        if provider == "bose":
            return self._play_bose(wav_path, ip)
        logger.warning("[smart-speaker] Unknown provider %r — skipping.", provider)
        return False

    # ------------------------------------------------------------------
    # Sonos (via soco)
    # ------------------------------------------------------------------

    def _play_sonos(self, wav_path: str, ip: str) -> bool:
        try:
            soco_mod = importlib.import_module("soco")
        except ImportError:
            logger.warning("[smart-speaker] soco not installed; cannot play on Sonos.")
            return False

        duration = _wav_duration_seconds(wav_path)
        try:
            server, uri = _start_http_server(wav_path)
        except RuntimeError as exc:
            logger.warning("[smart-speaker] %s", exc)
            return False

        try:
            device = soco_mod.SoCo(ip)
            device.play_uri(uri, title="Rex")
            time.sleep(duration + _SONOS_PLAY_BUFFER_SECONDS)
            try:
                device.stop()
            except Exception:
                pass
            logger.debug("[smart-speaker] Sonos playback complete (%s).", ip)
            return True
        except Exception as exc:
            logger.warning("[smart-speaker] Sonos play failed on %s: %s", ip, exc)
            return False
        finally:
            server.shutdown()

    # ------------------------------------------------------------------
    # Bose SoundTouch (via REST API)
    # ------------------------------------------------------------------

    def _play_bose(self, wav_path: str, ip: str) -> bool:
        try:
            requests = importlib.import_module("requests")
        except ImportError:
            logger.warning("[smart-speaker] requests not installed; cannot play on Bose.")
            return False

        duration = _wav_duration_seconds(wav_path)
        try:
            server, uri = _start_http_server(wav_path)
        except RuntimeError as exc:
            logger.warning("[smart-speaker] %s", exc)
            return False

        try:
            xml_body = (
                f'<ContentItem source="INTERNET_RADIO" location="{uri}">'
                "<itemName>Rex</itemName>"
                "</ContentItem>"
            )
            resp = requests.post(
                f"http://{ip}:{_BOSE_PORT}/select",
                data=xml_body,
                headers={"Content-Type": "application/xml"},
                timeout=_BOSE_SELECT_TIMEOUT,
            )
            resp.raise_for_status()
            time.sleep(duration + _SONOS_PLAY_BUFFER_SECONDS)
            logger.debug("[smart-speaker] Bose playback complete (%s).", ip)
            return True
        except Exception as exc:
            logger.warning("[smart-speaker] Bose play failed on %s: %s", ip, exc)
            return False
        finally:
            server.shutdown()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_smart_speaker_output = SmartSpeakerOutput()


def get_smart_speaker_output() -> SmartSpeakerOutput:
    """Return the process-wide ``SmartSpeakerOutput`` instance."""
    return _smart_speaker_output


__all__ = ["SmartSpeakerOutput", "get_smart_speaker_output"]
