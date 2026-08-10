"""Initial interface-agnostic TurnEngine facade."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from rex.runtime.events import EventKind, EventObserver, TurnEventStream
from rex.runtime.turn import TurnContext

T = TypeVar("T")


class TurnEngine:
    """Wrap existing work with canonical events without changing its result contract."""

    def __init__(self, *, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock = clock

    def execute(
        self,
        context: TurnContext,
        operation: Callable[[TurnEventStream], T],
        *,
        on_event: EventObserver | None = None,
    ) -> T:
        stream = TurnEventStream(context, clock=self._clock, observer=on_event)
        stream.emit(EventKind.TURN_STARTED)
        try:
            result = operation(stream)
        except Exception as exc:
            if not stream.is_terminal:
                stream.finish(EventKind.FAILED, {"exception_type": type(exc).__name__})
            raise
        if not stream.is_terminal:
            stream.finish(EventKind.COMPLETED)
        return result

    async def execute_async(
        self,
        context: TurnContext,
        operation: Callable[[TurnEventStream], Awaitable[T]],
        *,
        on_event: EventObserver | None = None,
    ) -> T:
        """Async counterpart to :meth:`execute` with identical lifecycle semantics."""
        stream = TurnEventStream(context, clock=self._clock, observer=on_event)
        stream.emit(EventKind.TURN_STARTED)
        try:
            result = await operation(stream)
        except Exception as exc:
            if not stream.is_terminal:
                stream.finish(EventKind.FAILED, {"exception_type": type(exc).__name__})
            raise
        if not stream.is_terminal:
            stream.finish(EventKind.COMPLETED)
        return result
