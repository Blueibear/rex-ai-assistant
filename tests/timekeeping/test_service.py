from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

import pytest

from rex.timekeeping.service import TimekeepingService


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value

    def advance(self, **kwargs: float) -> None:
        self.value += timedelta(**kwargs)


def test_concurrent_timers_are_isolated_by_owner(tmp_path) -> None:
    clock = MutableClock(datetime(2026, 8, 16, 12, 0, tzinfo=UTC))
    service = TimekeepingService(tmp_path / "timekeeping.json", now_func=clock)

    james = service.create_timer("james", 600, name="pasta")
    cole = service.create_timer("cole", 300, name="pasta")
    laundry = service.create_timer("james", 1200, name="laundry")

    assert james.timer_id != cole.timer_id != laundry.timer_id
    assert [t.name for t in service.list_timers("james")] == ["pasta", "laundry"]
    assert [t.name for t in service.list_timers("cole")] == ["pasta"]
    assert service.get_timer(james.timer_id, "cole") is None
    assert service.remaining_seconds(james.timer_id, "james") == pytest.approx(600)


def test_timer_pause_resume_adjust_rename_and_cancel(tmp_path) -> None:
    clock = MutableClock(datetime(2026, 8, 16, 12, 0, tzinfo=UTC))
    service = TimekeepingService(tmp_path / "timekeeping.json", now_func=clock)
    timer = service.create_timer("james", 600, name="pasta")

    clock.advance(seconds=90)
    paused = service.pause_timer(timer.timer_id, "james")
    assert paused is not None and paused.status == "paused"
    assert service.remaining_seconds(timer.timer_id, "james") == pytest.approx(510)

    clock.advance(minutes=5)
    assert service.remaining_seconds(timer.timer_id, "james") == pytest.approx(510)
    resumed = service.resume_timer(timer.timer_id, "james")
    assert resumed is not None and resumed.status == "active"

    adjusted = service.adjust_timer(timer.timer_id, "james", 120)
    assert adjusted is not None
    assert service.remaining_seconds(timer.timer_id, "james") == pytest.approx(630)
    assert service.rename_timer(timer.timer_id, "james", "sauce") is not None
    assert service.cancel_timer(timer.timer_id, "james") is True
    assert service.list_timers("james") == []


def test_timer_persists_and_overdue_timer_reconciles_once(tmp_path) -> None:
    path = tmp_path / "timekeeping.json"
    clock = MutableClock(datetime(2026, 8, 16, 12, 0, tzinfo=UTC))
    first = TimekeepingService(path, now_func=clock)
    timer = first.create_timer("james", 30, name="tea")

    clock.advance(seconds=45)
    restarted = TimekeepingService(path, now_func=clock)
    pending = restarted.get_timer(timer.timer_id, "james")
    assert pending is not None and pending.status == "active"

    due = restarted.claim_due_events()
    assert [(event.kind, event.record_id) for event in due] == [("timer", timer.timer_id)]
    assert restarted.get_timer(timer.timer_id, "james").status == "fired"
    assert restarted.claim_due_events() == []


def test_one_shot_alarm_fires_and_can_be_dismissed(tmp_path) -> None:
    clock = MutableClock(datetime(2026, 8, 16, 12, 0, tzinfo=UTC))
    service = TimekeepingService(tmp_path / "timekeeping.json", now_func=clock)
    alarm = service.create_alarm(
        "james",
        local_time=time(7, 30),
        timezone_name="America/Chicago",
        local_date=date(2026, 8, 17),
        name="morning",
    )

    assert alarm.next_fire_at == datetime(2026, 8, 17, 12, 30, tzinfo=UTC)
    clock.value = alarm.next_fire_at
    due = service.claim_due_events()
    assert [(event.kind, event.record_id) for event in due] == [("alarm", alarm.alarm_id)]
    assert service.get_alarm(alarm.alarm_id, "james").status == "ringing"

    dismissed = service.dismiss_alarm(alarm.alarm_id, "james")
    assert dismissed is not None and dismissed.status == "dismissed"
    assert dismissed.next_fire_at is None


def test_recurring_alarm_recalculates_across_dst(tmp_path) -> None:
    clock = MutableClock(datetime(2026, 10, 30, 15, 0, tzinfo=UTC))  # Friday, CDT
    service = TimekeepingService(tmp_path / "timekeeping.json", now_func=clock)
    alarm = service.create_alarm(
        "james",
        local_time=time(7, 0),
        timezone_name="America/Chicago",
        weekdays=(0, 1, 2, 3, 4),
        name="weekday",
    )

    # DST ends Sunday Nov 1, so Monday 07:00 Chicago is 13:00 UTC.
    assert alarm.next_fire_at == datetime(2026, 11, 2, 13, 0, tzinfo=UTC)
    clock.value = alarm.next_fire_at
    assert len(service.claim_due_events()) == 1
    dismissed = service.dismiss_alarm(alarm.alarm_id, "james")
    assert dismissed is not None and dismissed.status == "active"
    assert dismissed.next_fire_at == datetime(2026, 11, 3, 13, 0, tzinfo=UTC)


def test_alarm_snooze_refires_then_recurring_dismisses_to_next_schedule(tmp_path) -> None:
    clock = MutableClock(datetime(2026, 8, 17, 11, 55, tzinfo=UTC))
    service = TimekeepingService(tmp_path / "timekeeping.json", now_func=clock)
    alarm = service.create_alarm(
        "james",
        local_time=time(7, 0),
        timezone_name="America/Chicago",
        weekdays=(0, 1, 2, 3, 4),
        name="workday",
    )

    clock.value = alarm.next_fire_at
    assert len(service.claim_due_events()) == 1
    snoozed = service.snooze_alarm(alarm.alarm_id, "james", 600)
    assert snoozed is not None and snoozed.status == "active"
    assert snoozed.snooze_count == 1
    assert snoozed.next_fire_at == clock.value + timedelta(minutes=10)

    clock.value = snoozed.next_fire_at
    assert len(service.claim_due_events()) == 1
    dismissed = service.dismiss_alarm(alarm.alarm_id, "james")
    assert dismissed is not None and dismissed.status == "active"
    assert dismissed.next_fire_at > clock.value


def test_alarm_edit_enable_disable_and_owner_isolation(tmp_path) -> None:
    clock = MutableClock(datetime(2026, 8, 16, 12, 0, tzinfo=UTC))
    service = TimekeepingService(tmp_path / "timekeeping.json", now_func=clock)
    james = service.create_alarm(
        "james",
        local_time=time(7, 0),
        timezone_name="America/Chicago",
        local_date=date(2026, 8, 17),
        name="morning",
    )
    cole = service.create_alarm(
        "cole",
        local_time=time(7, 0),
        timezone_name="America/Chicago",
        local_date=date(2026, 8, 17),
        name="morning",
    )

    assert service.get_alarm(james.alarm_id, "cole") is None
    assert [a.alarm_id for a in service.list_alarms("cole")] == [cole.alarm_id]
    assert service.disable_alarm(james.alarm_id, "james") is not None
    assert service.list_alarms("james") == []

    enabled = service.enable_alarm(james.alarm_id, "james")
    assert enabled is not None and enabled.enabled is True
    edited = service.edit_alarm(
        james.alarm_id,
        "james",
        local_time=time(8, 15),
        name="late morning",
    )
    assert edited is not None and edited.name == "late morning"
    assert edited.next_fire_at == datetime(2026, 8, 17, 13, 15, tzinfo=UTC)
