"""Blocking wake-word listener with robust error handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import simpleaudio as sa
import sounddevice as sd

from ..config import settings
from ..logging_utils import configure_logger
from .utils import detect_wakeword, load_wakeword_model

LOGGER = configure_logger(__name__)


class WakeWordListener:
    """Listen for the configured wake word and emit notifications."""

    def __init__(self, *, sample_rate: int = 16_000, block_size: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.block_size = block_size or sample_rate
        self._model, self._keyword = load_wakeword_model(keyword=settings.wakeword)

    def listen(self) -> bool:
        """Block until the wake word is detected or an unrecoverable error occurs."""

        LOGGER.info("Listening for wake word '%s'", self._keyword)

        def audio_callback(indata, frames, time, status):  # pragma: no cover - requires audio hardware
            if status:
                LOGGER.warning("Audio stream status: %s", status)
            audio_data = np.squeeze(indata)
            if detect_wakeword(self._model, audio_data):
                LOGGER.info("Wake word '%s' detected", self._keyword)
                raise StopIteration

        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=audio_callback,
            ):
                while True:
                    sd.sleep(100)
        except StopIteration:
            self._play_confirmation_sound()
            return True
        except Exception as exc:  # pragma: no cover - hardware-specific failures
            LOGGER.error("Wakeword listener error: %s", exc)
            return False

    def _play_confirmation_sound(self) -> None:
        path = settings.wake_sound_path
        if not path:
            path = str(Path(__file__).resolve().parents[2] / "assets" / "rex_wake_acknowledgment (1).wav")
        try:
            wave_obj = sa.WaveObject.from_wave_file(path)
        except FileNotFoundError:
            LOGGER.debug("No custom wake sound configured; skipping playback")
            return
        except Exception as exc:  # pragma: no cover - hardware-specific
            LOGGER.error("Could not load wake confirmation sound: %s", exc)
            return

        try:
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as exc:  # pragma: no cover - hardware-specific
            LOGGER.error("Failed to play wake confirmation sound: %s", exc)


__all__ = ["WakeWordListener"]
