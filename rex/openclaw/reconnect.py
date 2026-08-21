"""Fail-closed OpenClaw reconnect coordination.

Connectivity alone never restores remote authority. A recovered gateway must
complete the injected authenticated capability resync before dispatch becomes
ready again.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from rex.openclaw.capability_sync import OpenClawSyncResult
from rex.openclaw.errors import OpenClawStaleAuthorityError

logger = logging.getLogger(__name__)

OPENCLAW_HEALTH_CHANGED_EVENT = "openclaw.health.changed"
_REASON_RE = re.compile(r"^[a-z0-9_.-]{1,64}$")
_T = TypeVar("_T")


class OpenClawReconnectState(StrEnum):
    READY = "ready"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    RESYNCING = "resyncing"
    CLOSED = "closed"


@dataclass(frozen=True)
class _DisconnectReservation:
    generation: int
    worker: threading.Thread | None


class OpenClawReconnectController:
    """Serialize disconnect, bounded reconnect, and authority restoration."""

    def __init__(
        self,
        *,
        health_probe: Callable[[], dict[str, Any]],
        resync: Callable[[], OpenClawSyncResult | None],
        mark_unavailable: Callable[[], Any],
        event_bus: Any = None,
        auto_reconnect: bool = True,
        base_delay_seconds: float = 1.0,
        max_delay_seconds: float = 30.0,
        wait_fn: Callable[[float], bool] | None = None,
    ) -> None:
        self._health_probe = health_probe
        self._resync = resync
        self._mark_unavailable = mark_unavailable
        self._event_bus = event_bus
        self._auto_reconnect = bool(auto_reconnect)
        self._base_delay = max(0.05, float(base_delay_seconds))
        self._max_delay = max(self._base_delay, float(max_delay_seconds))
        self._stop = threading.Event()
        self._wait_fn = wait_fn or self._stop.wait
        self._lock = threading.RLock()
        self._recovery_lock = threading.Lock()
        self._resync_context = threading.local()
        self._state = OpenClawReconnectState.READY
        self._generation = 0
        self._worker: threading.Thread | None = None

    @property
    def state(self) -> OpenClawReconnectState:
        with self._lock:
            return self._state

    @property
    def ready_for_dispatch(self) -> bool:
        return self.state is OpenClawReconnectState.READY

    @property
    def authority_generation(self) -> int:
        with self._lock:
            return self._generation

    def require_ready(self) -> None:
        """Fail closed before dispatch unless fresh remote authority is ready."""
        if self.ready_for_dispatch:
            return
        from rex.openclaw.errors import OpenClawUnavailableError

        raise OpenClawUnavailableError()

    def apply_authority_update(
        self,
        update: Callable[[], _T],
        *,
        expected_generation: int | None = None,
    ) -> _T:
        """Serialize canonical capability publication against remote dispatch."""
        with self._lock:
            if expected_generation is None:
                expected_generation = getattr(self._resync_context, "generation", None)
            if expected_generation is not None and expected_generation != self._generation:
                raise OpenClawStaleAuthorityError("obsolete reconnect generation")
            return update()

    def dispatch_if_ready(
        self,
        dispatch: Callable[[], Any],
        *,
        disconnect_on_error: Callable[[Exception], bool] | None = None,
        reason_code: str = "transport_failure",
    ) -> Any:
        """Run one remote dispatch atomically against disconnect/revocation."""
        from rex.openclaw.errors import OpenClawUnavailableError

        failure: Exception | None = None
        reservation: _DisconnectReservation | None = None
        with self._lock:
            if self._state is not OpenClawReconnectState.READY:
                raise OpenClawUnavailableError()
            try:
                return dispatch()
            except Exception as exc:
                if disconnect_on_error is None or not disconnect_on_error(exc):
                    raise
                failure = exc
                reservation = self._begin_disconnect_locked(reason_code)

        assert reservation is not None
        self._project_unavailable_and_start(reservation)
        assert failure is not None
        raise failure

    def mark_disconnected(self, reason_code: str = "transport_failure") -> None:
        """Disable remote authority and start at most one reconnect worker."""
        with self._lock:
            reservation = self._begin_disconnect_locked(reason_code)
        self._project_unavailable_and_start(reservation)

    def _begin_disconnect_locked(self, reason_code: str) -> _DisconnectReservation:
        if self._state is OpenClawReconnectState.CLOSED:
            return _DisconnectReservation(self._generation, None)
        self._generation += 1
        generation = self._generation
        self._state = OpenClawReconnectState.DISCONNECTED
        self._publish_locked(_safe_reason_code(reason_code), attempt=0, next_delay=None)
        worker: threading.Thread | None = None
        if self._auto_reconnect and self._worker is None:
            worker = threading.Thread(
                target=self._background_reconnect,
                name="openclaw-reconnect",
                daemon=True,
            )
            self._worker = worker
        return _DisconnectReservation(generation, worker)

    def _project_unavailable_and_start(self, reservation: _DisconnectReservation) -> None:
        worker = reservation.worker
        with self._lock:
            if self._state is not OpenClawReconnectState.DISCONNECTED:
                if worker is not None and self._worker is worker:
                    self._worker = None
                return
            if self._generation != reservation.generation:
                # A newer disconnect inherited this not-yet-started reservation.
                # Keep the single reserved worker and start it against current state.
                if worker is None or self._worker is not worker:
                    return
            try:
                self._mark_unavailable()
            except Exception:
                logger.exception("OpenClaw unavailable-state projection failed")
            if worker is not None:
                worker.start()

    def _background_reconnect(self) -> None:
        successor: _DisconnectReservation | None = None
        with self._lock:
            starting_generation = self._generation
        recovered = False
        try:
            recovered = self.run_until_recovered()
        finally:
            current = threading.current_thread()
            with self._lock:
                generation_changed = self._generation != starting_generation
                if self._worker is current:
                    self._worker = None
                    if (
                        self._auto_reconnect
                        and self._state is OpenClawReconnectState.DISCONNECTED
                        and not self._stop.is_set()
                        and (recovered or generation_changed)
                    ):
                        successor = self._begin_disconnect_locked("reconnect_successor")
            if successor is not None:
                self._project_unavailable_and_start(successor)

    def _resync_for_generation(self, generation: int) -> OpenClawSyncResult | None:
        previous = getattr(self._resync_context, "generation", None)
        self._resync_context.generation = generation
        try:
            return self._resync()
        finally:
            if previous is None:
                delattr(self._resync_context, "generation")
            else:
                self._resync_context.generation = previous

    def run_until_recovered(self, *, max_attempts: int | None = None) -> bool:
        """Serialize recovery callers so only one probe/resync loop runs at a time."""
        with self._recovery_lock:
            with self._lock:
                if self._state is OpenClawReconnectState.READY:
                    return True
            return self._run_until_recovered_owned(max_attempts=max_attempts)

    def _run_until_recovered_owned(self, *, max_attempts: int | None = None) -> bool:
        """Probe with capped backoff until authenticated resync restores authority."""
        attempt = 0
        delay = self._base_delay
        while max_attempts is None or attempt < max_attempts:
            with self._lock:
                if self._state is OpenClawReconnectState.CLOSED:
                    return False
                attempt += 1
                attempt_generation = self._generation
                self._state = OpenClawReconnectState.RECONNECTING
                self._publish_locked("reconnect_probe", attempt=attempt, next_delay=None)

            available = False
            try:
                available = self._health_probe().get("available") is True
            except Exception:
                available = False

            with self._lock:
                if self._state is OpenClawReconnectState.CLOSED:
                    return False
                if self._generation != attempt_generation:
                    return False

            if available:
                with self._lock:
                    self._state = OpenClawReconnectState.RESYNCING
                    self._publish_locked("gateway_recovered", attempt=attempt, next_delay=None)
                try:
                    result = self._resync_for_generation(attempt_generation)
                except Exception:
                    result = None
                if result is not None and result.success and not result.stale:
                    with self._lock:
                        if self._state is OpenClawReconnectState.CLOSED:
                            return False
                        if self._generation == attempt_generation:
                            self._state = OpenClawReconnectState.READY
                            self._publish_locked(
                                "resync_verified", attempt=attempt, next_delay=None
                            )
                            return True

            if max_attempts is not None and attempt >= max_attempts:
                with self._lock:
                    if self._state is not OpenClawReconnectState.CLOSED:
                        self._state = OpenClawReconnectState.DISCONNECTED
                        self._publish_locked("reconnect_failed", attempt=attempt, next_delay=None)
                return False

            wait_delay = min(delay, self._max_delay)
            with self._lock:
                if self._state is OpenClawReconnectState.CLOSED:
                    return False
                self._state = OpenClawReconnectState.DISCONNECTED
                self._publish_locked("reconnect_wait", attempt=attempt, next_delay=wait_delay)
            if self._wait_fn(wait_delay):
                return False
            delay = min(wait_delay * 2.0, self._max_delay)
        return False

    def close(self) -> None:
        """Stop reconnect work and permanently close this coordinator."""
        with self._lock:
            if self._state is OpenClawReconnectState.CLOSED:
                return
            self._state = OpenClawReconnectState.CLOSED
            self._stop.set()
            self._publish_locked("closed", attempt=0, next_delay=None)
            worker = self._worker
        if worker is not None and worker is not threading.current_thread() and worker.is_alive():
            worker.join(timeout=1.0)

    def _publish_locked(self, reason_code: str, *, attempt: int, next_delay: float | None) -> None:
        if self._event_bus is None:
            return
        payload = {
            "state": self._state.value,
            "reason_code": _safe_reason_code(reason_code),
            "attempt": max(0, int(attempt)),
            "next_delay_seconds": next_delay,
        }
        try:
            self._event_bus.publish(OPENCLAW_HEALTH_CHANGED_EVENT, payload)
        except Exception:
            logger.exception("OpenClaw health-state event publication failed")


def _safe_reason_code(value: str) -> str:
    candidate = str(value or "").strip().lower()
    return candidate if _REASON_RE.fullmatch(candidate) else "unspecified"


__all__ = [
    "OPENCLAW_HEALTH_CHANGED_EVENT",
    "OpenClawReconnectController",
    "OpenClawReconnectState",
]
