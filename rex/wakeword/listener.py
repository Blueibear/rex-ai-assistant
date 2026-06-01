"""Async wake-word listener built around ``detect_wakeword``."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TypeAlias

import numpy as np

from ..assistant_errors import WakeWordError
from .utils import (
    WakeWordDetectionResult,
    WakeWordModelSelection,
    evaluate_wakeword,
    load_wakeword_model,
    load_wakeword_model_with_metadata,
)

logger = logging.getLogger(__name__)
_DEFAULT_LOAD_WAKEWORD_MODEL = load_wakeword_model

DetectorResult: TypeAlias = bool | WakeWordDetectionResult
DetectorCallable: TypeAlias = Callable[[np.ndarray], DetectorResult]
DetectorFactoryResult: TypeAlias = tuple[DetectorCallable, Callable[[], None] | None, str | None]
DetectorFactory: TypeAlias = Callable[[], DetectorFactoryResult]
WakeEventCallback: TypeAlias = Callable[[dict[str, object]], None]


class WakeWordListener:
    """Continuously read audio frames and yield when the wake word fires."""

    def __init__(
        self,
        detector: DetectorCallable,
        *,
        poll_interval: float = 0.05,
        reset_detector: Callable[[], None] | None = None,
        rebuild_detector: DetectorFactory | None = None,
        threshold: float | None = None,
        keyword: str | None = None,
        backend: str | None = None,
        event_callback: WakeEventCallback | None = None,
    ) -> None:
        self._detector = detector
        self._poll_interval = poll_interval
        self._reset_detector = reset_detector
        self._rebuild_detector = rebuild_detector
        self._threshold = threshold
        self._keyword = keyword
        self._backend = backend
        self._running = False
        self._attempt_count = 0
        self._detector_generation = 1
        self._needs_rebuild = False
        self._event_callback = event_callback
        self._listening_started_at: float | None = None
        self._listening_cycle = 0

    def mark_listening_started(self, *, reason: str = "manual") -> None:
        """Mark a user-visible wake-listening cycle as active."""
        self._attempt_count = 0
        self._listening_cycle += 1
        self._listening_started_at = time.monotonic()
        extra = {
            "event": "wakeword_listening_cycle_started",
            "reason": reason,
            "threshold": self._threshold,
            "keyword": self._keyword,
            "backend": self._backend,
            "detector_generation": self._detector_generation,
            "listening_cycle": self._listening_cycle,
            "listening_started_at": self._listening_started_at,
        }
        logger.info("Wake-word listening cycle started", extra=extra)
        self._emit_event("INFO", "Wake-word listening cycle started", extra)

    def _mark_listening_ended(
        self,
        *,
        reason: str,
        accepted: bool,
        confidence: float | None = None,
        threshold: float | None = None,
    ) -> None:
        ended_at = time.monotonic()
        started_at = self._listening_started_at or ended_at
        extra = {
            "event": "wakeword_listening_cycle_ended",
            "reason": reason,
            "accepted": accepted,
            "threshold": threshold if threshold is not None else self._threshold,
            "confidence": confidence,
            "keyword": self._keyword,
            "backend": self._backend,
            "detector_generation": self._detector_generation,
            "listening_cycle": self._listening_cycle,
            "attempts": self._attempt_count,
            "duration_s": round(ended_at - started_at, 3),
        }
        logger.info("Wake-word listening cycle ended", extra=extra)
        self._emit_event("INFO", "Wake-word listening cycle ended", extra)

    async def listen(
        self, source: Callable[[], Awaitable[np.ndarray]]
    ) -> AsyncIterator[np.ndarray]:
        self._running = True
        if self._listening_started_at is None:
            self.mark_listening_started(reason="listener_loop_entered")
        enter_extra = {
            "event": "wakeword_listener_loop_entered",
            "threshold": self._threshold,
            "keyword": self._keyword,
            "backend": self._backend,
            "detector_generation": self._detector_generation,
            "detector_ready": True,
            "listening_cycle": self._listening_cycle,
            "listening_started_at": self._listening_started_at,
        }
        logger.info(
            "Wake-word listener loop entered",
            extra=enter_extra,
        )
        self._emit_event("INFO", "Wake-word listener loop entered", enter_extra)
        try:
            while self._running:
                frame = await source()
                detector_ready = True
                try:
                    result = await asyncio.get_running_loop().run_in_executor(
                        None, self._detector, frame
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.exception("Wake-word detection failed: %s", exc)
                    detector_ready = False
                    result = False

                self._attempt_count += 1
                (
                    triggered,
                    threshold,
                    confidence,
                    keyword,
                    reason,
                    result_extra,
                ) = self._coerce_result(result)
                accept_reason = "confidence_met_threshold" if triggered else None
                attempt_at = time.monotonic()
                listening_started_at = self._listening_started_at or attempt_at
                time_since_listening_start_s = round(attempt_at - listening_started_at, 3)
                log_level = (
                    logging.INFO
                    if triggered or self._attempt_count <= 8 or self._attempt_count % 4 == 0
                    else logging.DEBUG
                )
                attempt_extra = {
                    "event": "wakeword_attempt",
                    "attempt": self._attempt_count,
                    "detector_ready": detector_ready,
                    "threshold": threshold,
                    "confidence": confidence,
                    "keyword": keyword,
                    "accepted": triggered,
                    "accept_reason": accept_reason,
                    "reject_reason": None if triggered else reason,
                    "detector_generation": self._detector_generation,
                    "listening_cycle": self._listening_cycle,
                    "listening_started_at": listening_started_at,
                    "time_since_wake_listening_start_s": time_since_listening_start_s,
                    "early_listening_attempt": (
                        self._attempt_count <= 8 or time_since_listening_start_s <= 3.0
                    ),
                    **result_extra,
                }
                logger.log(
                    log_level,
                    "Wake-word attempt %d: %s confidence=%.6f threshold=%.3f keyword=%r",
                    self._attempt_count,
                    "accepted" if triggered else "rejected",
                    confidence,
                    threshold,
                    keyword,
                    extra=attempt_extra,
                )
                self._emit_event(
                    logging.getLevelName(log_level),
                    (
                        f"Wake-word attempt {self._attempt_count}: "
                        f"{'accepted' if triggered else 'rejected'}"
                    ),
                    attempt_extra,
                )

                if triggered:
                    detected_at = time.monotonic()
                    self._mark_listening_ended(
                        reason="accepted_wake",
                        accepted=True,
                        confidence=confidence,
                        threshold=threshold,
                    )
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
                    source_owner = getattr(source, "__self__", None)
                    reset_detection_buffer = getattr(source_owner, "reset_detection_buffer", None)
                    if callable(reset_detection_buffer):
                        reset_detection_buffer(reason="accepted_wake")
                    self.reset(reason="accepted_wake")
                    yield frame

                await asyncio.sleep(self._poll_interval)
        finally:
            self._running = False
            exit_extra = {
                "event": "wakeword_listener_loop_exited",
                "threshold": self._threshold,
                "keyword": self._keyword,
                "backend": self._backend,
                "detector_generation": self._detector_generation,
            }
            logger.info(
                "Wake-word listener loop exited",
                extra=exit_extra,
            )
            self._emit_event("INFO", "Wake-word listener loop exited", exit_extra)

    def _emit_event(self, level: str, message: str, extra: dict[str, object]) -> None:
        if self._event_callback is None:
            return
        try:
            self._event_callback({"level": level, "message": message, "extra": dict(extra)})
        except Exception:
            logger.debug("Wake event callback failed", exc_info=True)

    @staticmethod
    def _coerce_result(
        result: DetectorResult,
    ) -> tuple[bool, float, float, str | None, str, dict[str, object]]:
        if isinstance(result, WakeWordDetectionResult):
            return (
                result.triggered,
                result.threshold,
                result.confidence,
                result.keyword,
                result.reason,
                {
                    "audio_rms": result.audio_rms,
                    "audio_peak": result.audio_peak,
                    "effective_peak": result.effective_peak,
                    "applied_gain": result.applied_gain,
                    "gain_limit": result.gain_limit,
                    "target_peak": result.target_peak,
                },
            )
        return bool(result), float("nan"), float("nan"), None, "legacy_bool_detector", {}

    def reset(self, *, reason: str = "manual") -> None:
        """Reset detector state when the backend supports it."""
        self._attempt_count = 0

        if self._rebuild_detector is not None:
            if reason == "accepted_wake":
                self._needs_rebuild = True
                logger.info(
                    "Wake-word detector rebuild deferred until interaction completes",
                    extra={
                        "event": "wakeword_detector_rebuild_deferred",
                        "reason": reason,
                        "detector_generation": self._detector_generation,
                        "threshold": self._threshold,
                        "keyword": self._keyword,
                        "backend": self._backend,
                        "reset_supported": self._reset_detector is not None,
                    },
                )
                return
            if reason == "post_interaction" and self._needs_rebuild:
                self._rebuild(reason=reason)
                return
            if self._reset_detector is None and reason == "post_interaction":
                self._rebuild(reason=reason)
                return

        if self._reset_detector is None:
            logger.info(
                "Wake-word detector reset skipped; backend has no reset hook",
                extra={
                    "event": "wakeword_detector_reset",
                    "reason": reason,
                    "reset": False,
                    "rebuild_available": self._rebuild_detector is not None,
                    "detector_generation": self._detector_generation,
                    "threshold": self._threshold,
                    "keyword": self._keyword,
                    "backend": self._backend,
                },
            )
            return
        try:
            self._reset_detector()
            logger.info(
                "Wake-word detector state reset",
                extra={
                    "event": "wakeword_detector_reset",
                    "reason": reason,
                    "reset": True,
                    "detector_generation": self._detector_generation,
                    "threshold": self._threshold,
                    "keyword": self._keyword,
                    "backend": self._backend,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Wake-word detector reset failed: %s",
                exc,
                extra={"event": "wakeword_detector_reset_failed", "reason": reason},
            )

    def _rebuild(self, *, reason: str) -> None:
        if self._rebuild_detector is None:
            return
        started_at = time.perf_counter()
        try:
            detector, reset_detector, keyword = self._rebuild_detector()
        except Exception as exc:  # pragma: no cover - dependency/setup dependent
            logger.exception(
                "Wake-word detector rebuild failed: %s",
                exc,
                extra={
                    "event": "wakeword_detector_rebuild_failed",
                    "reason": reason,
                    "detector_generation": self._detector_generation,
                    "threshold": self._threshold,
                    "keyword": self._keyword,
                    "backend": self._backend,
                },
            )
            return

        self._detector = detector
        self._reset_detector = reset_detector
        if keyword:
            self._keyword = keyword
        self._detector_generation += 1
        self._needs_rebuild = False
        logger.info(
            "Wake-word detector rebuilt",
            extra={
                "event": "wakeword_detector_rebuilt",
                "reason": reason,
                "detector_generation": self._detector_generation,
                "threshold": self._threshold,
                "keyword": self._keyword,
                "backend": self._backend,
                "reset_supported": self._reset_detector is not None,
                "duration_s": round(time.perf_counter() - started_at, 3),
            },
        )

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
    event_callback: WakeEventCallback | None = None,
) -> WakeWordListener:
    """Build a WakeWordListener with the default wake-word model."""
    threshold = float(threshold)
    if keyword is not None and keyword.strip() == "":
        raise WakeWordError(
            "Wake word keyword must not be empty. "
            "Set a valid keyword or leave keyword=None to use the default."
        )
    if model_path is not None and model_path.strip():
        resolved = Path(model_path)
        if not resolved.is_file():
            raise WakeWordError(
                f"Wake word model file not found: {resolved}. "
                "Check wake_word.model_path in rex_config.json."
            )

    selection: WakeWordModelSelection | None = None

    def _build_detector_instance() -> DetectorFactoryResult:
        nonlocal selection
        if load_wakeword_model is _DEFAULT_LOAD_WAKEWORD_MODEL:
            model, selection = load_wakeword_model_with_metadata(
                keyword=keyword,
                model_path=model_path,
                embedding_path=embedding_path,
                backend=backend,
                fallback_to_builtin=fallback_to_builtin,
                fallback_keyword=fallback_keyword,
            )
            active_label = selection.active_label
        else:
            model, active_label = load_wakeword_model(
                keyword=keyword,
                model_path=model_path,
                embedding_path=embedding_path,
                backend=backend,
                fallback_to_builtin=fallback_to_builtin,
                fallback_keyword=fallback_keyword,
            )
            selection = WakeWordModelSelection(
                requested_backend=backend or "openwakeword",
                active_backend=backend or "openwakeword",
                requested_phrase=keyword,
                active_label=active_label,
                requested_model_path=model_path,
                requested_embedding_path=embedding_path,
                fallback_keyword=fallback_keyword,
            )

        def _detector(frame: np.ndarray) -> WakeWordDetectionResult:
            return evaluate_wakeword(model, frame, threshold=threshold)

        reset = getattr(model, "reset", None)
        reset_detector = reset if callable(reset) else None
        return _detector, reset_detector, active_label

    try:
        detector, reset_detector, active_label = _build_detector_instance()
    except Exception as exc:  # pragma: no cover - dependency/setup dependent
        raise WakeWordError(f"Failed to load wake-word model: {exc}") from exc

    active_backend = (
        selection.active_backend if selection is not None else (backend or "openwakeword")
    )

    if poll_interval is None:
        poll_interval = min(0.05, max(0.0, chunk_duration / 2))

    logger.info(
        "Wake-word listener configured",
        extra={
            "event": "wakeword_listener_configured",
            "threshold": threshold,
            "poll_interval": poll_interval,
            "chunk_duration_s": chunk_duration,
            "keyword": active_label,
            "backend": active_backend,
            "requested_backend": selection.requested_backend if selection else backend,
            "requested_phrase": selection.requested_phrase if selection else keyword,
            "requested_model_path": selection.requested_model_path if selection else model_path,
            "requested_embedding_path": (
                selection.requested_embedding_path if selection else embedding_path
            ),
            "resolved_model_path": selection.resolved_model_path if selection else None,
            "resolved_embedding_path": selection.resolved_embedding_path if selection else None,
            "used_fallback": selection.used_fallback if selection else False,
            "fallback_keyword": selection.fallback_keyword if selection else fallback_keyword,
            "validation_error": selection.validation_error if selection else None,
            "reset_supported": reset_detector is not None,
        },
    )
    logger.info(
        "Wake-word detector instance ready",
        extra={
            "event": "wakeword_detector_instance_ready",
            "threshold": threshold,
            "keyword": active_label,
            "backend": active_backend,
            "requested_backend": selection.requested_backend if selection else backend,
            "requested_phrase": selection.requested_phrase if selection else keyword,
            "requested_model_path": selection.requested_model_path if selection else model_path,
            "requested_embedding_path": (
                selection.requested_embedding_path if selection else embedding_path
            ),
            "resolved_model_path": selection.resolved_model_path if selection else None,
            "resolved_embedding_path": selection.resolved_embedding_path if selection else None,
            "used_fallback": selection.used_fallback if selection else False,
            "fallback_keyword": selection.fallback_keyword if selection else fallback_keyword,
            "validation_error": selection.validation_error if selection else None,
            "detector_generation": 1,
            "reset_supported": reset_detector is not None,
        },
    )

    return WakeWordListener(
        detector,
        poll_interval=poll_interval,
        reset_detector=reset_detector,
        rebuild_detector=_build_detector_instance,
        threshold=threshold,
        keyword=active_label,
        backend=active_backend,
        event_callback=event_callback,
    )


__all__ = ["WakeWordListener", "build_default_detector", "load_wakeword_model"]
