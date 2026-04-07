"""Async wake-word listener built around ``detect_wakeword``."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable

import numpy as np

from ..assistant_errors import WakeWordError
from .utils import detect_wakeword, load_wakeword_model

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
                    detected_at = time.monotonic()
                    logger.info(
                        "Wake word detected; initiating audio capture",
                        extra={"event": "wakeword_detected", "detected_at": detected_at},
                    )
                    capture_at = time.monotonic()
                    logger.debug(
                        "Audio capture started (%.1f ms after detection)",
                        (capture_at - detected_at) * 1000,
                        extra={"event": "audio_capture_start", "capture_at": capture_at},
                    )
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
    embedding_path: str | None = None,
    backend: str | None = None,
    fallback_to_builtin: bool | None = None,
    fallback_keyword: str | None = None,
) -> WakeWordListener:
    """Build a WakeWordListener with the default wake-word model."""
    if keyword is not None and keyword.strip() == "":
        raise WakeWordError(
            "Wake word keyword must not be empty. "
            "Set a valid keyword or leave keyword=None to use the default."
        )
    try:
        model, _ = load_wakeword_model(
            keyword=keyword,
            model_path=model_path,
            embedding_path=embedding_path,
            backend=backend,
            fallback_to_builtin=fallback_to_builtin,
            fallback_keyword=fallback_keyword,
        )
    except Exception as exc:  # pragma: no cover - dependency/setup dependent
        raise WakeWordError(f"Failed to load wake-word model: {exc}") from exc

    def _detector(frame: np.ndarray) -> bool:
        return detect_wakeword(model, frame, threshold=threshold)

    if poll_interval is None:
        poll_interval = min(0.05, max(0.0, chunk_duration / 2))

    return WakeWordListener(_detector, poll_interval=poll_interval)


__all__ = ["WakeWordListener", "build_default_detector"]
