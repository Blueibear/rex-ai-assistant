"""Deadline-driven runtime for AskRex timers and alarms."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from datetime import datetime

from .models import DueEvent, ensure_utc, utc_now
from .service import TimekeepingService

logger = logging.getLogger(__name__)


def _notification_request(event: DueEvent, *, target_id: str | None = None):
    from rex.notification import NotificationRequest

    if event.kind == "timer":
        title = "Timer"
        body = f"{event.name or 'Timer'} finished."
        record_key = "timer_id"
    else:
        title = "Alarm"
        body = f"{event.name or 'Alarm'} is ringing."
        record_key = "alarm_id"
    metadata = {record_key: event.record_id, "user_id": event.user_id}
    if target_id and target_id.startswith("ha:"):
        metadata["ha_entity_id"] = target_id.split(":", 1)[1]
    return NotificationRequest(
        priority="normal",
        title=title,
        body=body,
        channel_preferences=["dashboard", "ha_tts"],
        metadata=metadata,
    )


def _send_due_event_to_target(target_id: str, event: DueEvent) -> bool:
    """Attempt a currently supported targeted due-event delivery."""
    from rex.notification import get_notifier

    # HA targets have a first-class entity selector in the existing TTS client.
    # Other provider-specific target delivery is added by the shared voice/media
    # output adapter; never pretend a generic notification reached that speaker.
    if not target_id.startswith("ha:"):
        return False
    request = _notification_request(event, target_id=target_id)
    get_notifier().send_to_channel("ha_tts", request)
    return True


def deliver_due_event_notification(event: DueEvent) -> None:
    """Resolve current output policy and attempt a due timer/alarm announcement."""
    from rex.notification import get_notifier
    from rex.output_routing.delivery import OutputDeliveryService
    from rex.output_routing.runtime import get_output_routing_service

    try:
        routing = get_output_routing_service()
    except RuntimeError:
        # During very early startup the media registry may not yet be installed.
        # Preserve the legacy dashboard/default-HA notification rather than lose
        # an overdue alarm, but do not label that fallback as targeted delivery.
        get_notifier().send(_notification_request(event))
        return

    result = OutputDeliveryService(
        routing,
        sender=_send_due_event_to_target,
    ).deliver_due_event(event)
    if result.delivered:
        return

    logger.warning(
        "timekeeping due event was not target-delivered kind=%s record_id=%s user_id=%s reason=%s",
        event.kind,
        event.record_id,
        event.user_id,
        result.reason,
    )
    # Keep a non-audio dashboard record when target delivery is unavailable.
    request = _notification_request(event)
    request.channel_preferences = ["dashboard"]
    get_notifier().send(request)


class TimekeepingRuntime:
    """Wait until the nearest persisted deadline and deliver due events."""

    def __init__(
        self,
        service: TimekeepingService,
        *,
        event_handler: Callable[[DueEvent], object],
        now_func: Callable[[], datetime] | None = None,
    ) -> None:
        self._service = service
        self._event_handler = event_handler
        self._now = now_func or utc_now
        self._condition = threading.Condition()
        self._generation = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def _now_utc(self) -> datetime:
        return ensure_utc(self._now())

    @property
    def running(self) -> bool:
        with self._condition:
            return self._running

    def wake(self) -> None:
        with self._condition:
            self._generation += 1
            self._condition.notify_all()

    def process_due_once(self) -> int:
        events = self._service.claim_due_events(self._now_utc())
        for event in events:
            try:
                self._event_handler(event)
            except Exception:
                logger.exception(
                    "timekeeping event delivery failed kind=%s record_id=%s user_id=%s",
                    event.kind,
                    event.record_id,
                    event.user_id,
                )
        return len(events)

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
        self._service.set_change_callback(self.wake)
        self.process_due_once()
        thread = threading.Thread(target=self._run, name="askrex-timekeeping", daemon=True)
        self._thread = thread
        thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        with self._condition:
            if not self._running:
                self._service.set_change_callback(None)
                return
            self._running = False
            self._generation += 1
            self._condition.notify_all()
        self._service.set_change_callback(None)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        self._thread = None

    def _run(self) -> None:
        while True:
            with self._condition:
                if not self._running:
                    return
                observed_generation = self._generation
            self.process_due_once()
            deadline = self._service.next_deadline()
            if deadline is None:
                with self._condition:
                    if not self._running:
                        return
                    if observed_generation == self._generation:
                        self._condition.wait()
                continue
            delay = max(0.0, (deadline - self._now_utc()).total_seconds())
            if delay <= 0:
                continue
            with self._condition:
                if not self._running:
                    return
                if observed_generation != self._generation:
                    continue
                self._condition.wait(timeout=delay)


_global_lock = threading.RLock()
_global_service: TimekeepingService | None = None
_global_runtime: TimekeepingRuntime | None = None


def get_timekeeping_service() -> TimekeepingService:
    global _global_service
    with _global_lock:
        if _global_service is None:
            _global_service = TimekeepingService()
        return _global_service


def ensure_timekeeping_runtime() -> TimekeepingRuntime:
    global _global_runtime
    with _global_lock:
        if _global_runtime is None:
            _global_runtime = TimekeepingRuntime(
                get_timekeeping_service(),
                event_handler=deliver_due_event_notification,
            )
            _global_runtime.start()
        return _global_runtime


def shutdown_timekeeping_runtime() -> None:
    global _global_runtime
    with _global_lock:
        runtime = _global_runtime
        _global_runtime = None
    if runtime is not None:
        runtime.stop()


def set_timekeeping_service(service: TimekeepingService | None) -> None:
    global _global_service
    shutdown_timekeeping_runtime()
    with _global_lock:
        _global_service = service


__all__ = [
    "TimekeepingRuntime",
    "deliver_due_event_notification",
    "ensure_timekeeping_runtime",
    "get_timekeeping_service",
    "set_timekeeping_service",
    "shutdown_timekeeping_runtime",
]
