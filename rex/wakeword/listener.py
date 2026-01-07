"""Async wake-word listener built around ``detect_wakeword``."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from ..assistant_errors import WakeWordError
from .utils import detect_wakeword, load_wakeword_model

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


def build_default_detector(
    *,
    sample_rate: int,
    chunk_duration: float,
    threshold: float = 0.5,
    poll_interval: float | None = None,
    keyword: str | None = None,
    model_path: str | None = None,
) -> WakeWordListener:
    """Build a WakeWordListener with the default wake-word model."""
    try:
        model, _ = load_wakeword_model(keyword=keyword, model_path=model_path)
    except Exception as exc:  # pragma: no cover - dependency/setup dependent
        raise WakeWordError(f"Failed to load wake-word model: {exc}") from exc

    def _detector(frame: np.ndarray) -> bool:
        return detect_wakeword(model, frame, threshold=threshold)

    if poll_interval is None:
        poll_interval = min(0.05, max(0.0, chunk_duration / 2))

    return WakeWordListener(_detector, poll_interval=poll_interval)


__all__ = ["WakeWordListener", "build_default_detector"]
