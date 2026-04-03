"""Smart speaker microphone input (US-SP-003).

Captures audio frames from a microphone-enabled smart speaker via its local
HTTP API so the wake word detection pipeline can run without any changes to
the core openWakeWord integration.

Currently supported
-------------------
Sonos
    Local HTTP audio-capture API at ``http://{ip}:1400/api/v1/audio/capture``
    (available on Arc, Beam, Era 300, Era 100, One, Move 2, etc.).
    Returns a raw PCM 16-bit LE mono stream at 16 kHz.

Bose SoundTouch
    No microphone capture API is exposed; always falls back to local mic.

Usage
-----
    mic = SmartSpeakerMic(provider="sonos", ip="192.168.1.10")
    if mic.connect():
        frame = mic.read_frame(duration_seconds=1.0)   # numpy float32 array
        mic.disconnect()
    else:
        # fall back to local sounddevice
        ...
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SONOS_AUDIO_PORT = 1400
_SONOS_CAPTURE_PATH = "/api/v1/audio/capture"
_CONNECT_TIMEOUT = 3.0
_READ_TIMEOUT = 15.0
_CHUNK_BYTES = 1024


class SmartSpeakerMic:
    """Capture audio frames from a microphone-enabled smart speaker.

    Parameters
    ----------
    provider:
        ``"sonos"`` or ``"bose"``.
    ip:
        IP address of the speaker on the local network.
    sample_rate:
        Expected sample rate in Hz (default 16 000 — matches openWakeWord).
    """

    def __init__(
        self,
        *,
        provider: str,
        ip: str,
        sample_rate: int = 16000,
    ) -> None:
        self._provider = provider
        self._ip = ip
        self._sample_rate = sample_rate
        self._connected = False
        self._response: Any = None  # streaming requests.Response

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Establish an audio stream connection to the speaker.

        Returns ``True`` if the connection is ready for :meth:`read_frame`,
        or ``False`` if the device is unavailable (caller should fall back to
        local microphone).
        """
        if self._provider == "sonos":
            return self._connect_sonos()
        logger.debug(
            "[smart-mic] Provider %r has no microphone API; falling back to local mic.",
            self._provider,
        )
        return False

    def read_frame(self, duration_seconds: float) -> Any | None:
        """Read approximately *duration_seconds* of audio from the speaker.

        Returns a float32 numpy array shaped ``(-1,)`` sampled at
        ``self._sample_rate``, or ``None`` if the stream breaks (caller should
        fall back to local mic on the next frame).
        """
        if not self._connected or self._response is None:
            return None
        return self._read_pcm_frame(duration_seconds)

    def disconnect(self) -> None:
        """Close the audio stream."""
        if self._response is not None:
            try:
                self._response.close()
            except Exception:
                pass
            self._response = None
        self._connected = False
        logger.debug("[smart-mic] Disconnected from %s (%s).", self._provider, self._ip)

    # ------------------------------------------------------------------
    # Sonos
    # ------------------------------------------------------------------

    def _connect_sonos(self) -> bool:
        try:
            requests = importlib.import_module("requests")
        except ImportError:
            logger.warning("[smart-mic] requests not installed; cannot stream from Sonos mic.")
            return False

        url = f"http://{self._ip}:{_SONOS_AUDIO_PORT}{_SONOS_CAPTURE_PATH}"
        try:
            resp = requests.get(
                url,
                stream=True,
                timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
            )
            resp.raise_for_status()
            self._response = resp
            self._connected = True
            logger.info("[smart-mic] Connected to Sonos microphone at %s.", url)
            return True
        except Exception as exc:
            logger.warning("[smart-mic] Cannot connect to Sonos mic at %s: %s", url, exc)
            return False

    # ------------------------------------------------------------------
    # PCM frame reader (shared by all providers)
    # ------------------------------------------------------------------

    def _read_pcm_frame(self, duration_seconds: float) -> Any | None:
        """Read raw 16-bit LE PCM bytes and return as a float32 numpy array."""
        try:
            np = importlib.import_module("numpy")
        except ImportError:
            logger.warning("[smart-mic] numpy not installed; cannot decode audio frame.")
            return None

        n_samples = max(1, int(self._sample_rate * duration_seconds))
        n_bytes = n_samples * 2  # 16-bit PCM = 2 bytes per sample

        try:
            buf = bytearray()
            assert self._response is not None
            for chunk in self._response.iter_content(chunk_size=_CHUNK_BYTES):
                buf.extend(chunk)
                if len(buf) >= n_bytes:
                    break

            if len(buf) < n_bytes:
                logger.warning("[smart-mic] Audio stream ended prematurely; disconnecting.")
                self.disconnect()
                return None

            pcm = bytes(buf[:n_bytes])
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            return samples.reshape(-1)

        except Exception as exc:
            logger.warning("[smart-mic] Error reading audio frame: %s", exc)
            self.disconnect()
            return None


__all__ = ["SmartSpeakerMic"]
