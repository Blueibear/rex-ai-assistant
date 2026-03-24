"""Unit tests for rex.notifications.models — Notification and NotificationStore."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rex.notifications.models import Notification, NotificationStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> NotificationStore:
    """Return an in-memory-ish store using a temp file that is auto-cleaned."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return NotificationStore(db_path=Path(tmp.name))


def _make_notification(**kwargs: object) -> Notification:
    defaults: dict[str, object] = {
        "title": "Test notification",
        "body": "Body text",
        "source": "test",
    }
    defaults.update(kwargs)
    return Notification(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Notification model tests
# ---------------------------------------------------------------------------


class TestNotificationModel:
    def test_default_id_is_uuid(self) -> None:
        n = _make_notification()
        assert len(n.id) == 36  # UUID4 string

    def test_default_priority_low(self) -> None:
        n = _make_notification()
        assert n.priority == "low"

    def test_default_channel_desktop(self) -> None:
        n = _make_notification()
        assert n.channel == "desktop"

    def test_default_digest_eligible_false(self) -> None:
        n = _make_notification()
        assert n.digest_eligible is False

    def test_default_quiet_hours_exempt_false(self) -> None:
        n = _make_notification()
        assert n.quiet_hours_exempt is False

    def test_default_created_at_is_utc(self) -> None:
        n = _make_notification()
        assert n.created_at.tzinfo is not None

    def test_optional_fields_default_none(self) -> None:
        n = _make_notification()
        assert n.delivered_at is None
        assert n.read_at is None
        assert n.escalation_due_at is None

    def test_all_priority_levels(self) -> None:
        for p in ("low", "medium", "high", "critical"):
            n = _make_notification(priority=p)
            assert n.priority == p

    def test_all_channel_values(self) -> None:
        for ch in ("desktop", "digest", "sms", "email"):
            n = _make_notification(channel=ch)
            assert n.channel == ch

    def test_model_dump_round_trip(self) -> None:
        dt = datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc)
        n = Notification(
            id="test-id",
            title="Hi",
            body="Body",
            source="email",
            priority="high",
            channel="sms",
            digest_eligible=True,
            quiet_hours_exempt=True,
            created_at=dt,
            escalation_due_at=dt,
        )
        restored = Notification(**n.model_dump())
        assert restored == n


# ---------------------------------------------------------------------------
# NotificationStore CRUD tests
# ---------------------------------------------------------------------------


class TestNotificationStore:
    def test_add_and_get_unread(self) -> None:
        store = _make_store()
        n = _make_notification(title="Unread one")
        store.add(n)
        unread = store.get_unread()
        assert len(unread) == 1
        assert unread[0].id == n.id
        assert unread[0].title == "Unread one"

    def test_get_unread_empty_initially(self) -> None:
        store = _make_store()
        assert store.get_unread() == []

    def test_mark_read_removes_from_unread(self) -> None:
        store = _make_store()
        n = _make_notification()
        store.add(n)
        store.mark_read(n.id)
        assert store.get_unread() == []

    def test_mark_read_nonexistent_id_no_error(self) -> None:
        store = _make_store()
        store.mark_read("nonexistent-id")  # should not raise

    def test_update_changes_fields(self) -> None:
        store = _make_store()
        n = _make_notification(title="Original")
        store.add(n)
        updated = n.model_copy(update={"title": "Updated", "priority": "high"})
        store.update(updated)
        unread = store.get_unread()
        assert unread[0].title == "Updated"
        assert unread[0].priority == "high"

    def test_update_nonexistent_id_no_error(self) -> None:
        store = _make_store()
        n = _make_notification(id="ghost-id")
        store.update(n)  # should not raise

    def test_multiple_notifications_ordered_by_created_at(self) -> None:
        store = _make_store()
        from datetime import timedelta

        base = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
        n1 = _make_notification(title="First", created_at=base)
        n2 = _make_notification(title="Second", created_at=base + timedelta(hours=1))
        store.add(n1)
        store.add(n2)
        unread = store.get_unread()
        assert unread[0].title == "First"
        assert unread[1].title == "Second"

    def test_read_notifications_excluded_from_get_unread(self) -> None:
        store = _make_store()
        n1 = _make_notification(title="Read")
        n2 = _make_notification(title="Unread")
        store.add(n1)
        store.add(n2)
        store.mark_read(n1.id)
        unread = store.get_unread()
        assert len(unread) == 1
        assert unread[0].title == "Unread"

    def test_add_duplicate_id_raises(self) -> None:
        import sqlite3

        store = _make_store()
        n = _make_notification(id="dup-id")
        store.add(n)
        with pytest.raises(sqlite3.IntegrityError):
            store.add(n)

    def test_optional_datetime_fields_persist(self) -> None:
        store = _make_store()
        dt = datetime(2025, 3, 15, 9, 0, tzinfo=timezone.utc)
        n = _make_notification(
            escalation_due_at=dt,
            digest_eligible=True,
            quiet_hours_exempt=True,
        )
        store.add(n)
        unread = store.get_unread()
        assert unread[0].escalation_due_at == dt
        assert unread[0].digest_eligible is True
        assert unread[0].quiet_hours_exempt is True
