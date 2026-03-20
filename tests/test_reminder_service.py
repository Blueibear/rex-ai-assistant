"""Smoke tests for rex.reminder_service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _future(hours: int = 2) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def _past(hours: int = 1) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours)


def test_import():
    """Module imports without error."""
    import rex.reminder_service as rs

    assert rs is not None


def test_reminder_model_defaults():
    """Reminder model sets expected defaults."""
    from rex.reminder_service import Reminder

    r = Reminder(title="Test", remind_at=_future())

    assert r.user_id == "default"
    assert r.status == "pending"
    assert r.reminder_id.startswith("rem_")
    assert r.followup_enabled is False


def test_reminder_is_due_past():
    """is_due returns True for a pending reminder in the past."""
    from rex.reminder_service import Reminder

    r = Reminder(title="Past", remind_at=_past())
    assert r.is_due()


def test_reminder_is_due_future():
    """is_due returns False for a pending reminder in the future."""
    from rex.reminder_service import Reminder

    r = Reminder(title="Future", remind_at=_future())
    assert not r.is_due()


def test_reminder_is_due_already_fired():
    """is_due returns False for a non-pending reminder."""
    from rex.reminder_service import Reminder

    r = Reminder(title="Fired", remind_at=_past(), status="fired")
    assert not r.is_due()


def test_reminder_followup_prompt_default():
    """get_followup_prompt returns default prompt when none set."""
    from rex.reminder_service import Reminder

    r = Reminder(title="Call mom", remind_at=_future())
    assert "Call mom" in r.get_followup_prompt()


def test_reminder_followup_prompt_custom():
    """get_followup_prompt returns custom prompt when set."""
    from rex.reminder_service import Reminder

    r = Reminder(title="Test", remind_at=_future(), followup_prompt="Did you do it?")
    assert r.get_followup_prompt() == "Did you do it?"


def test_create_and_list(tmp_path):
    """create_reminder stores and list_reminders retrieves."""
    from rex.reminder_service import ReminderService

    svc = ReminderService(storage_path=tmp_path / "reminders.json")

    r = svc.create_reminder("alice", "Buy milk", _future())

    assert r.user_id == "alice"
    assert r.title == "Buy milk"

    listed = svc.list_reminders(user_id="alice", status="pending")
    assert len(listed) == 1
    assert listed[0].reminder_id == r.reminder_id


def test_mark_done(tmp_path):
    """mark_done transitions status to done."""
    from rex.reminder_service import ReminderService

    svc = ReminderService(storage_path=tmp_path / "reminders.json")
    r = svc.create_reminder("bob", "Read book", _future())

    result = svc.mark_done(r.reminder_id)

    assert result is True
    updated = svc.get_reminder(r.reminder_id)
    assert updated.status == "done"
    assert updated.done_at is not None


def test_cancel_reminder(tmp_path):
    """cancel_reminder transitions status to canceled."""
    from rex.reminder_service import ReminderService

    svc = ReminderService(storage_path=tmp_path / "reminders.json")
    r = svc.create_reminder("carol", "Meeting", _future())

    result = svc.cancel_reminder(r.reminder_id)

    assert result is True
    updated = svc.get_reminder(r.reminder_id)
    assert updated.status == "canceled"


def test_delete_reminder(tmp_path):
    """delete_reminder removes from storage."""
    from rex.reminder_service import ReminderService

    svc = ReminderService(storage_path=tmp_path / "reminders.json")
    r = svc.create_reminder("dave", "Exercise", _future())

    deleted = svc.delete_reminder(r.reminder_id)

    assert deleted is True
    assert svc.get_reminder(r.reminder_id) is None


def test_fire_due_reminders(tmp_path):
    """fire_due_reminders fires past reminders and marks them as fired."""
    from unittest.mock import patch

    from rex.reminder_service import ReminderService

    svc = ReminderService(storage_path=tmp_path / "reminders.json")
    svc.create_reminder("user", "Past task", _past())
    svc.create_reminder("user", "Future task", _future())

    with patch.object(svc, "_send_notification"), patch.object(svc, "_create_followup_cue"):
        fired = svc.fire_due_reminders()

    assert len(fired) == 1
    assert fired[0].title == "Past task"
    assert fired[0].status == "fired"


def test_stats(tmp_path):
    """stats() returns correct counts."""

    from rex.reminder_service import ReminderService

    svc = ReminderService(storage_path=tmp_path / "reminders.json")
    svc.create_reminder("u", "A", _future())
    svc.create_reminder("u", "B", _future(), followup_enabled=True)

    stats = svc.stats()

    assert stats["total"] == 2
    assert stats["by_status"]["pending"] == 2
    assert stats["with_followup"] == 1


def test_persistence(tmp_path):
    """Reminders persist to disk and reload correctly."""
    from rex.reminder_service import ReminderService

    path = tmp_path / "reminders.json"

    svc1 = ReminderService(storage_path=path)
    r = svc1.create_reminder("user", "Persistent", _future())

    svc2 = ReminderService(storage_path=path)
    loaded = svc2.get_reminder(r.reminder_id)

    assert loaded is not None
    assert loaded.title == "Persistent"


def test_backward_compat_aliases(tmp_path):
    """add_reminder / cancel / fire_due / get / all_reminders aliases work."""

    from rex.reminder_service import ReminderService

    svc = ReminderService(storage_path=tmp_path / "reminders.json")

    r = svc.add_reminder("Alias test", _future())
    assert r.title == "Alias test"

    got = svc.get(r.reminder_id)
    assert got is not None

    all_r = list(svc.all_reminders())
    assert len(all_r) == 1

    svc.cancel(r.reminder_id)
    assert svc.get(r.reminder_id).status == "canceled"


def test_get_reminder_service_singleton(tmp_path):
    """get_reminder_service returns a consistent global instance."""
    from rex.reminder_service import get_reminder_service, set_reminder_service

    set_reminder_service(None)
    svc = get_reminder_service()
    svc2 = get_reminder_service()

    assert svc is svc2

    # cleanup
    set_reminder_service(None)
