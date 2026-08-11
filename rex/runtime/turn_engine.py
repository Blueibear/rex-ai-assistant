"""Initial interface-agnostic TurnEngine facade."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from threading import Lock
from typing import TypeVar

from rex.identity import validate_user_id
from rex.runtime.cancellation import TurnCancelledError, turn_cancellation_scope
from rex.runtime.events import EventKind, EventObserver, TurnEventStream
from rex.runtime.turn import TurnContext

T = TypeVar("T")


class TurnEngine:
    """Wrap existing work with canonical events without changing its result contract."""

    def __init__(self, *, clock: Callable[[], int] = time.monotonic_ns) -> None:
        self._clock = clock
        self._active_turns: dict[str, TurnContext] = {}
        self._active_lock = Lock()

    def _register(self, context: TurnContext) -> None:
        with self._active_lock:
            if context.turn_id in self._active_turns:
                raise RuntimeError(f"turn {context.turn_id} is already active")
            self._active_turns[context.turn_id] = context

    def _unregister(self, context: TurnContext) -> None:
        with self._active_lock:
            if self._active_turns.get(context.turn_id) is context:
                self._active_turns.pop(context.turn_id, None)

    def cancel_turn(self, turn_id: str, *, user_id: str, reason: str = "caller_cancelled") -> bool:
        """Cancel one active turn only when its validated owner matches."""
        try:
            validated_user = validate_user_id(user_id)
        except ValueError:
            return False
        with self._active_lock:
            context = self._active_turns.get(turn_id)
            if context is None or context.user_id != validated_user:
                return False
            return context.cancellation.cancel(reason)

    def execute(
        self,
        context: TurnContext,
        operation: Callable[[TurnEventStream], T],
        *,
        on_event: EventObserver | None = None,
    ) -> T:
        stream = TurnEventStream(context, clock=self._clock, observer=on_event)
        self._register(context)
        try:
            with turn_cancellation_scope(context.cancellation):
                stream.emit(EventKind.TURN_STARTED)
                try:
                    context.cancellation.raise_if_cancelled()
                    result = operation(stream)
                    if not stream.is_terminal:
                        context.cancellation.raise_if_cancelled()
                except TurnCancelledError as exc:
                    if not stream.is_terminal:
                        stream.finish(EventKind.CANCELLED, {"reason": exc.reason})
                    raise
                except Exception as exc:
                    if not stream.is_terminal:
                        stream.finish(EventKind.FAILED, {"exception_type": type(exc).__name__})
                    raise
                if not stream.is_terminal:
                    stream.finish(EventKind.COMPLETED)
                return result
        finally:
            self._unregister(context)

    async def execute_async(
        self,
        context: TurnContext,
        operation: Callable[[TurnEventStream], Awaitable[T]],
        *,
        on_event: EventObserver | None = None,
    ) -> T:
        """Async counterpart to :meth:`execute` with identical lifecycle semantics."""
        stream = TurnEventStream(context, clock=self._clock, observer=on_event)
        self._register(context)
        try:
            with turn_cancellation_scope(context.cancellation):
                stream.emit(EventKind.TURN_STARTED)
                try:
                    context.cancellation.raise_if_cancelled()
                    result = await operation(stream)
                    if not stream.is_terminal:
                        context.cancellation.raise_if_cancelled()
                except asyncio.CancelledError:
                    context.cancellation.cancel("task_cancelled")
                    reason = context.cancellation.reason or "task_cancelled"
                    if not stream.is_terminal:
                        stream.finish(EventKind.CANCELLED, {"reason": reason})
                    raise
                except TurnCancelledError as exc:
                    if not stream.is_terminal:
                        stream.finish(EventKind.CANCELLED, {"reason": exc.reason})
                    raise
                except Exception as exc:
                    if not stream.is_terminal:
                        stream.finish(EventKind.FAILED, {"exception_type": type(exc).__name__})
                    raise
                if not stream.is_terminal:
                    stream.finish(EventKind.COMPLETED)
                return result
        finally:
            self._unregister(context)
