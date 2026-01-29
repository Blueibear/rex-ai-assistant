"""Tests for the follow-up engine, cue store, reminders, and CLI wiring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from rex.calendar_service import CalendarEvent, CalendarService
from rex.cli import cmd_cues, cmd_reminders
from rex.cue_store import CueStore
from rex.followup_engine import FollowupConfig, FollowupEngine
from rex.reminder_service import ReminderService


def test_cue_store_persistence_and_prune(tmp_path: Path) -> None:
    storage_path = tmp_path / "cues.json"
    store = CueStore(storage_path)
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    cue = store.add_cue("Follow up on meeting", source="calendar", source_id="event-1")
    assert store.mark_asked(cue.cue_id, at=now)
    assert store.dismiss(cue.cue_id, at=now + timedelta(hours=1))

    reloaded = CueStore(storage_path)
    cues = reloaded.list_cues()
    assert len(cues) == 1
    assert cues[0].status == "dismissed"

    cues[0].created_at = now - timedelta(hours=200)
    reloaded._save()
    removed = reloaded.prune_expired(expire_hours=168, now=now)
    assert removed == 1


def test_reminder_service_followup_and_status(tmp_path: Path) -> None:
    cue_store = CueStore(tmp_path / "cues.json")
    service = ReminderService(tmp_path / "reminders.json", cue_store=cue_store)
    remind_at = datetime(2024, 2, 1, 9, 0, tzinfo=timezone.utc)

    reminder = service.add_reminder("Call mom", remind_at, follow_up=True)
    assert reminder.follow_up_cue_id is not None
    assert len(cue_store.list_cues()) == 1

    fired = service.fire_due(now=remind_at + timedelta(minutes=1))
    assert fired and fired[0].status == "fired"

    assert service.mark_done(reminder.reminder_id)
    assert service.get(reminder.reminder_id).status == "done"

    reminder_two = service.add_reminder("Submit report", remind_at + timedelta(days=1))
    assert service.cancel(reminder_two.reminder_id)
    assert service.get(reminder_two.reminder_id).status == "canceled"


def test_calendar_followup_cues_dedupe(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    events = [
        CalendarEvent(
            event_id="past-1",
            title="Retro",
            start_time=now - timedelta(hours=5),
            end_time=now - timedelta(hours=4),
        ),
        CalendarEvent(
            event_id="past-2",
            title="Old Event",
            start_time=now - timedelta(days=5, hours=2),
            end_time=now - timedelta(days=5, hours=1),
        ),
        CalendarEvent(
            event_id="future-1",
            title="Planning",
            start_time=now + timedelta(hours=3),
            end_time=now + timedelta(hours=4),
        ),
    ]
    calendar_service = CalendarService(mock_events=events)
    cue_store = CueStore(tmp_path / "cues.json")

    created = calendar_service.generate_followup_cues(cue_store, lookback_hours=24)
    assert created == 1
    assert len(cue_store.list_cues()) == 1

    created_again = calendar_service.generate_followup_cues(cue_store, lookback_hours=24)
    assert created_again == 0


def test_followup_engine_session_limits(tmp_path: Path) -> None:
    now = datetime(2024, 3, 1, 10, 0, tzinfo=timezone.utc)
    store = CueStore(tmp_path / "cues.json")
    store.add_cue("Check in on proposal")
    store.add_cue("Ask about travel plans")

    engine = FollowupEngine(
        cue_store=store,
        config=FollowupConfig(enabled=True, max_per_session=1, lookback_hours=24, expire_hours=72),
        now_fn=lambda: now,
    )

    prompts_first = engine.collect_followups()
    assert len(prompts_first) == 1

    prompts_second = engine.collect_followups()
    assert prompts_second == []


def test_cli_reminders_and_cues(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    cue_store = CueStore(tmp_path / "cues.json")
    reminder_service = ReminderService(tmp_path / "reminders.json", cue_store=cue_store)

    monkeypatch.setattr("rex.cli.get_reminder_service", lambda: reminder_service)
    monkeypatch.setattr("rex.cli.get_cue_store", lambda: cue_store)

    args_add = SimpleNamespace(
        reminders_command="add",
        title="Call mom",
        at="2024-01-02 09:00",
        follow_up=True,
    )
    assert cmd_reminders(args_add) == 0

    args_list = SimpleNamespace(reminders_command="list", status=None)
    assert cmd_reminders(args_list) == 0
    assert "Call mom" in capsys.readouterr().out

    reminder_id = reminder_service.list_reminders()[0].reminder_id
    args_done = SimpleNamespace(reminders_command="done", reminder_id=reminder_id)
    assert cmd_reminders(args_done) == 0

    args_cancel = SimpleNamespace(reminders_command="cancel", reminder_id="missing")
    assert cmd_reminders(args_cancel) == 1

    cue_id = cue_store.list_cues()[0].cue_id
    args_cue_list = SimpleNamespace(cues_command="list", status=None)
    assert cmd_cues(args_cue_list) == 0
    assert cue_id in capsys.readouterr().out

    args_cue_dismiss = SimpleNamespace(cues_command="dismiss", cue_id=cue_id)
    assert cmd_cues(args_cue_dismiss) == 0

    old_cue = cue_store.add_cue("Old cue")
    old_cue.created_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    cue_store._save()

    args_cue_prune = SimpleNamespace(cues_command="prune")
    assert cmd_cues(args_cue_prune) == 0
