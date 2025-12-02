"""Async wake-word listener built around ``detect_wakeword``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from .utils import detect_wakeword

# Optional: TTS
# import pyttsx3
# tts = pyttsx3.init()

logger = logging.getLogger(__name__)


class WakeWordListener:
    """Continuously read audio frames and yield when the wake word fires."""

    def __init__(
        self,
        detector: Callable[[np.ndarray], bool],
        *,
        poll_interval: float = 0.05,
    ) -> None:
        self._detector = detector
        self._poll_interval = poll_interval
        self._running = False

    async def listen(
        self, source: Callable[[], Awaitable[np.ndarray]]
    ) -> AsyncIterator[np.ndarray]:
        self._running = True
        try:
            while self._running:
                frame = await source()
                try:
                    triggered = await asyncio.get_running_loop().run_in_executor(
                        None, self._detector, frame
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("Wake-word detection failed: %s", exc)
                    triggered = False

                if triggered:
                    print("🎤 Wakeword DETECTED! Rex is listening...")

                    # Optional: Speak a response
                    # tts.say("Hello, how can I help?")
                    # tts.runAndWait()

                    yield frame

                await asyncio.sleep(self._poll_interval)
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False


def build_default_detector(model, *, threshold: float = 0.5) -> Callable[[np.ndarray], bool]:
    def _detector(frame: np.ndarray) -> bool:
        return detect_wakeword(model, frame, threshold=threshold)

    return _detector


__all__ = ["WakeWordListener", "build_default_detector"]
