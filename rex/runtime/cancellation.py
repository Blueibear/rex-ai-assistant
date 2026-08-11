"""Turn-scoped cancellation contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event, Lock
from typing import TypeVar

T = TypeVar("T")


class TurnCancelledError(RuntimeError):
    """Raised when work observes cancellation for its owning turn."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Turn cancelled: {reason}")


class TurnCancellation:
    """Idempotent cancellation state owned by exactly one turn."""

    def __init__(self, turn_id: str) -> None:
        if not turn_id.strip():
            raise ValueError("turn_id must not be empty")
        self.turn_id = turn_id
        self._event = Event()
        self._lock = Lock()
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str = "caller_cancelled") -> bool:
        normalized = reason.strip()
        if not normalized:
            raise ValueError("cancellation reason must not be empty")
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = normalized
            self._event.set()
            return True

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise TurnCancelledError(self._reason or "caller_cancelled")

    def wait(self, timeout_seconds: float | None = None) -> bool:
        """Wait for cancellation, returning whether it was observed."""
        return self._event.wait(timeout_seconds)


_CURRENT_CANCELLATION: ContextVar[TurnCancellation | None] = ContextVar(
    "rex_turn_cancellation", default=None
)


def current_turn_cancellation() -> TurnCancellation | None:
    """Return cancellation state for the currently executing turn, if any."""
    return _CURRENT_CANCELLATION.get()


@contextmanager
def turn_cancellation_scope(cancellation: TurnCancellation) -> Iterator[TurnCancellation]:
    """Bind one turn's cancellation state to the current execution context."""
    token = _CURRENT_CANCELLATION.set(cancellation)
    try:
        yield cancellation
    finally:
        _CURRENT_CANCELLATION.reset(token)


async def await_with_cancellation(awaitable: Awaitable[T], *, poll_seconds: float = 0.01) -> T:
    """Await work while allowing the current turn to abandon stale output promptly."""
    cancellation = current_turn_cancellation()
    if cancellation is None:
        return await awaitable
    cancellation.raise_if_cancelled()
    work = asyncio.ensure_future(awaitable)
    try:
        while not work.done():
            if cancellation.cancelled:
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                cancellation.raise_if_cancelled()
            await asyncio.sleep(poll_seconds)
        cancellation.raise_if_cancelled()
        return await work
    finally:
        if cancellation.cancelled and not work.done():
            work.cancel()


__all__ = [
    "TurnCancellation",
    "TurnCancelledError",
    "await_with_cancellation",
    "current_turn_cancellation",
    "turn_cancellation_scope",
]
