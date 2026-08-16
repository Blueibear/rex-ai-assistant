from __future__ import annotations

import threading
import time as wall_time
from datetime import UTC, datetime, timedelta

from rex.timekeeping.runtime import TimekeepingRuntime
from rex.timekeeping.service import TimekeepingService


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: float) -> None:
        self.value += timedelta(**kwargs)


def test_process_due_once_reconciles_overdue_restart_once(tmp_path) -> None:
    clock = MutableClock(datetime(2026, 8, 16, 12, 0, tzinfo=UTC))
    path = tmp_path / "timekeeping.json"
    first = TimekeepingService(path, now_func=clock)
    timer = first.create_timer("james", 30, name="tea")
    clock.advance(seconds=45)

    restarted = TimekeepingService(path, now_func=clock)
    delivered = []
    runtime = TimekeepingRuntime(restarted, event_handler=delivered.append, now_func=clock)

    assert runtime.process_due_once() == 1
    assert [event.record_id for event in delivered] == [timer.timer_id]
    assert runtime.process_due_once() == 0


def test_runtime_wakes_at_nearest_deadline_without_polling(tmp_path) -> None:
    service = TimekeepingService(tmp_path / "timekeeping.json")
    fired = threading.Event()
    delivered = []
    runtime = TimekeepingRuntime(
        service,
        event_handler=lambda event: (delivered.append(event), fired.set()),
    )
    runtime.start()
    try:
        started = wall_time.monotonic()
        timer = service.create_timer("james", 0.15, name="quick")
        assert fired.wait(1.0)
        elapsed = wall_time.monotonic() - started
        assert delivered[0].record_id == timer.timer_id
        assert 0.08 <= elapsed < 0.8
    finally:
        runtime.stop()


def test_runtime_reorders_wait_when_earlier_timer_is_added(tmp_path) -> None:
    service = TimekeepingService(tmp_path / "timekeeping.json")
    delivered = []
    both = threading.Event()

    def handle(event) -> None:
        delivered.append(event)
        if len(delivered) == 2:
            both.set()

    runtime = TimekeepingRuntime(service, event_handler=handle)
    runtime.start()
    try:
        later = service.create_timer("james", 0.30, name="later")
        earlier = service.create_timer("james", 0.08, name="earlier")
        assert both.wait(1.5)
        assert [event.record_id for event in delivered] == [earlier.timer_id, later.timer_id]
    finally:
        runtime.stop()


def test_default_due_event_handler_uses_existing_notifier(monkeypatch) -> None:
    from rex.timekeeping.models import DueEvent
    from rex.timekeeping.runtime import deliver_due_event_notification

    sent = []

    class FakeNotifier:
        def send(self, request) -> None:
            sent.append(request)

    monkeypatch.setattr("rex.notification.get_notifier", lambda: FakeNotifier())
    event = DueEvent(
        kind="timer",
        record_id="tmr_test",
        user_id="james",
        name="pasta",
        fired_at=datetime(2026, 8, 16, 12, 0, tzinfo=UTC),
    )

    deliver_due_event_notification(event)

    assert len(sent) == 1
    assert sent[0].title == "Timer"
    assert "pasta" in sent[0].body
    assert sent[0].metadata["timer_id"] == "tmr_test"
    assert sent[0].metadata["user_id"] == "james"
