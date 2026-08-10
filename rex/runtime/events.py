"""Ordered, correlated event contracts for assistant turns."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from types import MappingProxyType
from typing import Any

from rex.runtime.turn import TurnContext

logger = logging.getLogger(__name__)


class EventKind(StrEnum):
    TURN_STARTED = "turn_started"
    CONTEXT_PROGRESS = "context_progress"
    ROUTE_PROGRESS = "route_progress"
    CAPABILITY_PROGRESS = "capability_progress"
    ACTION_PROGRESS = "action_progress"
    MODEL_PROGRESS = "model_progress"
    RESPONSE_PROGRESS = "response_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_EVENT_KINDS = frozenset({EventKind.COMPLETED, EventKind.FAILED, EventKind.CANCELLED})


class TerminalStateError(RuntimeError):
    """The turn already emitted a terminal event."""


@dataclass(frozen=True, slots=True)
class TurnEvent:
    """One immutable event correlated to one validated turn."""

    kind: EventKind
    turn_id: str
    user_id: str
    sequence: int
    monotonic_ns: int
    details: Mapping[str, Any]

    @property
    def is_terminal(self) -> bool:
        return self.kind in TERMINAL_EVENT_KINDS


EventObserver = Callable[[TurnEvent], None]


class TurnEventStream:
    """Emit strictly ordered events for one turn and fail closed after terminal."""

    def __init__(
        self,
        context: TurnContext,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        observer: EventObserver | None = None,
    ) -> None:
        self.context = context
        self._clock = clock
        self._observer = observer
        self._sequence = 0
        self._last_monotonic_ns: int | None = None
        self._terminal_event: TurnEvent | None = None
        self._lock = Lock()

    @property
    def is_terminal(self) -> bool:
        return self._terminal_event is not None

    @property
    def terminal_event(self) -> TurnEvent | None:
        return self._terminal_event

    def emit(self, kind: EventKind | str, details: Mapping[str, Any] | None = None) -> TurnEvent:
        resolved = EventKind(kind)
        if resolved in TERMINAL_EVENT_KINDS:
            raise ValueError("terminal events must use finish()")
        return self._record(resolved, details)

    def finish(self, kind: EventKind | str, details: Mapping[str, Any] | None = None) -> TurnEvent:
        resolved = EventKind(kind)
        if resolved not in TERMINAL_EVENT_KINDS:
            raise ValueError("finish() requires a terminal event kind")
        return self._record(resolved, details)

    def _record(self, kind: EventKind, details: Mapping[str, Any] | None) -> TurnEvent:
        with self._lock:
            if self._terminal_event is not None:
                raise TerminalStateError(f"turn {self.context.turn_id} is already terminal")
            timestamp = self._clock()
            if self._last_monotonic_ns is not None and timestamp < self._last_monotonic_ns:
                raise ValueError("turn event clock moved backwards")
            self._sequence += 1
            event = TurnEvent(
                kind=kind,
                turn_id=self.context.turn_id,
                user_id=self.context.user_id,
                sequence=self._sequence,
                monotonic_ns=timestamp,
                details=MappingProxyType(dict(details or {})),
            )
            self._last_monotonic_ns = timestamp
            if event.is_terminal:
                self._terminal_event = event
            observer = self._observer
        if observer is not None:
            try:
                observer(event)
            except Exception:
                logger.exception(
                    "Turn event observer failed for turn %s sequence %s",
                    event.turn_id,
                    event.sequence,
                )
        return event
