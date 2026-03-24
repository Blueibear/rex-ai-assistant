"""Unit tests for rex.notifications.escalation — EscalationEngine."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from rex.notifications.escalation import EscalationEngine
from rex.notifications.models import Notification, NotificationStore
from rex.notifications.router import NotificationRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> NotificationStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return NotificationStore(db_path=Path(tmp.name))


def _make_router(store: NotificationStore) -> NotificationRouter:
    return NotificationRouter(store)


def _past(minutes: int = 10) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


def _future(minutes: int = 10) -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=minutes)


def _make_notification(
    priority: str = "low",
    escalation_due_at: datetime | None = None,
) -> Notification:
    return Notification(
        title="Test",
        body="Body",
        source="test",
        priority=priority,  # type: ignore[arg-type]
        escalation_due_at=escalation_due_at,
    )


# ---------------------------------------------------------------------------
# No escalation when not due
# ---------------------------------------------------------------------------


class TestNoEscalation:
    def test_no_escalation_when_no_due_at(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("low", escalation_due_at=None)
        store.add(n)
        engine = EscalationEngine(store, router, config={})
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            engine.check_escalations()
        mock_desktop.assert_not_called()

    def test_no_escalation_when_due_in_future(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("low", escalation_due_at=_future(60))
        store.add(n)
        engine = EscalationEngine(store, router, config={})
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            engine.check_escalations()
        mock_desktop.assert_not_called()

    def test_no_escalation_when_already_read(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("low", escalation_due_at=_past())
        store.add(n)
        store.mark_read(n.id)
        engine = EscalationEngine(store, router, config={})
        with patch("rex.notifications.router._send_desktop"):
            engine.check_escalations()
        # No unread — nothing to escalate
        assert store.get_unread() == []


# ---------------------------------------------------------------------------
# Priority promotion
# ---------------------------------------------------------------------------


class TestPriorityPromotion:
    def test_low_promoted_to_medium(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("low", escalation_due_at=_past())
        store.add(n)
        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        engine.check_escalations()
        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.priority == "medium"

    def test_medium_promoted_to_high(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("medium", escalation_due_at=_past())
        store.add(n)
        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        engine.check_escalations()
        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.priority == "high"

    def test_high_promoted_to_critical(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("high", escalation_due_at=_past())
        store.add(n)
        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        with patch("rex.notifications.router._send_desktop"):
            engine.check_escalations()
        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.priority == "critical"

    def test_critical_not_escalated_further(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("critical", escalation_due_at=_past())
        store.add(n)
        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        with patch("rex.notifications.router._send_desktop"):
            engine.check_escalations()
        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.priority == "critical"


# ---------------------------------------------------------------------------
# Re-routing and escalation_due_at update
# ---------------------------------------------------------------------------


class TestReroutingAndUpdate:
    def test_re_routes_via_router(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("high", escalation_due_at=_past())
        store.add(n)
        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            engine.check_escalations()
        # critical/high re-routes → desktop called
        mock_desktop.assert_called_once()

    def test_escalation_due_at_updated_to_future(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("low", escalation_due_at=_past())
        store.add(n)
        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        before = datetime.now(timezone.utc)
        engine.check_escalations()
        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.escalation_due_at is not None
        assert updated.escalation_due_at > before

    def test_escalation_delay_applied(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("low", escalation_due_at=_past())
        store.add(n)
        delay = 45
        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": delay}
        )
        before = datetime.now(timezone.utc)
        engine.check_escalations()
        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.escalation_due_at is not None
        # Should be approximately now + delay minutes
        expected_approx = before + timedelta(minutes=delay)
        diff = abs((updated.escalation_due_at - expected_approx).total_seconds())
        assert diff < 5  # within 5 seconds

    def test_original_notification_not_mutated(self) -> None:
        store = _make_store()
        router = _make_router(store)
        original_priority = "low"
        n = _make_notification(original_priority, escalation_due_at=_past())
        store.add(n)
        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        engine.check_escalations()
        assert n.priority == original_priority


# ---------------------------------------------------------------------------
# Default config (30 minutes)
# ---------------------------------------------------------------------------


class TestDefaultDelay:
    def test_default_delay_used_when_config_missing(self) -> None:
        store = _make_store()
        router = _make_router(store)
        n = _make_notification("low", escalation_due_at=_past())
        store.add(n)
        # empty config → should use default 30 min delay
        engine = EscalationEngine(store, router, config={})
        before = datetime.now(timezone.utc)
        engine.check_escalations()
        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.escalation_due_at is not None
        diff = (updated.escalation_due_at - before).total_seconds()
        assert 25 * 60 < diff < 35 * 60  # approximately 30 minutes
