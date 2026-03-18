"""Integrated tests for the notifications pipeline.

Tests the full flow across NotificationRouter, DigestBuilder, QuietHoursGate,
and EscalationEngine working together with a shared store.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from rex.notifications.digest import DigestBackend, DigestBuilder
from rex.notifications.escalation import EscalationEngine
from rex.notifications.models import Notification, NotificationStore
from rex.notifications.quiet_hours import QuietHoursGate
from rex.notifications.router import NotificationRouter

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _store() -> NotificationStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return NotificationStore(db_path=Path(tmp.name))


def _notification(
    priority: str = "low",
    quiet_hours_exempt: bool = False,
    escalation_due_at: datetime | None = None,
) -> Notification:
    return Notification(
        title="Pipeline test",
        body="Integration body",
        source="test",
        priority=priority,  # type: ignore[arg-type]
        quiet_hours_exempt=quiet_hours_exempt,
        escalation_due_at=escalation_due_at,
    )


def _past(minutes: int = 10) -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


# ---------------------------------------------------------------------------
# Router dispatches critical immediately
# ---------------------------------------------------------------------------


class TestRouterCritical:
    def test_critical_dispatched_immediately(self) -> None:
        store = _store()
        router = NotificationRouter(store)
        n = _notification("critical")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once_with(n.title, n.body)

    def test_critical_stored_in_db(self) -> None:
        store = _store()
        router = NotificationRouter(store)
        n = _notification("critical")
        with patch("rex.notifications.router._send_desktop"):
            router.route(n)
        assert any(x.id == n.id for x in store.get_unread())


# ---------------------------------------------------------------------------
# Router queues low as digest
# ---------------------------------------------------------------------------


class TestRouterLowDigest:
    def test_low_no_immediate_desktop(self) -> None:
        store = _store()
        router = NotificationRouter(store)
        n = _notification("low")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_not_called()

    def test_low_stored_as_digest_eligible(self) -> None:
        store = _store()
        router = NotificationRouter(store)
        n = _notification("low")
        router.route(n)
        stored = next(x for x in store.get_unread() if x.id == n.id)
        assert stored.digest_eligible is True


# ---------------------------------------------------------------------------
# Quiet hours suppresses medium
# ---------------------------------------------------------------------------


class TestQuietHoursSuppressMedium:
    def test_medium_suppressed_during_quiet_hours(self) -> None:
        store = _store()
        from datetime import time

        qh = QuietHoursGate(
            config={
                "notifications_quiet_hours_start": "22:00",
                "notifications_quiet_hours_end": "07:00",
            },
            clock=lambda: time(23, 30),
        )
        router = NotificationRouter(store, quiet_hours=qh)
        n = _notification("medium")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_not_called()

    def test_medium_still_stored_when_suppressed(self) -> None:
        store = _store()
        from datetime import time

        qh = QuietHoursGate(
            config={
                "notifications_quiet_hours_start": "22:00",
                "notifications_quiet_hours_end": "07:00",
            },
            clock=lambda: time(23, 30),
        )
        router = NotificationRouter(store, quiet_hours=qh)
        n = _notification("medium")
        with patch("rex.notifications.router._send_desktop"):
            router.route(n)
        assert any(x.id == n.id for x in store.get_unread())
        # delivered_at remains None (queued for post-quiet-hours delivery)
        stored = next(x for x in store.get_unread() if x.id == n.id)
        assert stored.delivered_at is None


# ---------------------------------------------------------------------------
# quiet_hours_exempt critical not suppressed
# ---------------------------------------------------------------------------


class TestQuietHoursExempt:
    def test_exempt_critical_dispatched_even_during_quiet_hours(self) -> None:
        store = _store()
        from datetime import time

        qh = QuietHoursGate(
            config={
                "notifications_quiet_hours_start": "22:00",
                "notifications_quiet_hours_end": "07:00",
            },
            clock=lambda: time(23, 30),
        )
        router = NotificationRouter(store, quiet_hours=qh)
        # critical ignores quiet hours by router design; also exempt flag
        n = _notification("critical", quiet_hours_exempt=True)
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once()

    def test_exempt_medium_dispatched_during_quiet_hours(self) -> None:
        store = _store()
        from datetime import time

        qh = QuietHoursGate(
            config={
                "notifications_quiet_hours_start": "22:00",
                "notifications_quiet_hours_end": "07:00",
            },
            clock=lambda: time(23, 30),
        )
        router = NotificationRouter(store, quiet_hours=qh)
        n = _notification("medium", quiet_hours_exempt=True)
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once()


# ---------------------------------------------------------------------------
# Escalation promotes priority after delay
# ---------------------------------------------------------------------------


class TestEscalationPipeline:
    def test_escalation_promotes_and_redispatches(self) -> None:
        store = _store()
        router = NotificationRouter(store)
        n = _notification("low", escalation_due_at=_past())
        # Low priority — router stores without desktop
        router.route(n)

        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        with patch("rex.notifications.router._send_desktop"):
            engine.check_escalations()

        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.priority == "medium"

    def test_escalation_reschedules_due_at(self) -> None:
        store = _store()
        router = NotificationRouter(store)
        n = _notification("medium", escalation_due_at=_past())
        store.upsert(n)  # add directly with medium priority

        engine = EscalationEngine(
            store, router, config={"notifications_escalation_delay_minutes": 30}
        )
        before = datetime.now(timezone.utc)
        with patch("rex.notifications.router._send_desktop"):
            engine.check_escalations()

        updated = next(x for x in store.get_unread() if x.id == n.id)
        assert updated.escalation_due_at is not None
        assert updated.escalation_due_at > before


# ---------------------------------------------------------------------------
# Digest builder generates summary for N queued items
# ---------------------------------------------------------------------------


class TestDigestPipeline:
    def test_digest_summarises_queued_low_notifications(self) -> None:
        store = _store()
        router = NotificationRouter(store)

        for i in range(3):
            n = Notification(
                title=f"Update {i}",
                body="Body",
                source="test",
                priority="low",
            )
            router.route(n)

        mock_backend: MagicMock = MagicMock(spec=DigestBackend)
        mock_backend.generate.return_value = "You have 3 update(s): ..."

        builder = DigestBuilder(store, backend=mock_backend)
        with patch("rex.notifications.digest._send_desktop") as mock_desktop:
            builder.run_digest()

        mock_desktop.assert_called_once_with("Rex Digest", "You have 3 update(s): ...")
        # All low notifications should now be read
        assert store.get_unread() == []

    def test_digest_includes_all_eligible_in_prompt(self) -> None:
        store = _store()
        router = NotificationRouter(store)

        titles = ["Alpha", "Beta", "Gamma"]
        for title in titles:
            n = Notification(title=title, body="body", source="test", priority="low")
            router.route(n)

        mock_backend: MagicMock = MagicMock(spec=DigestBackend)
        mock_backend.generate.return_value = "Summary"
        builder = DigestBuilder(store, backend=mock_backend)
        builder.build_digest()

        call_messages = mock_backend.generate.call_args[0][0]
        user_content = call_messages[1]["content"]
        for title in titles:
            assert title in user_content

    def test_digest_no_op_when_no_low_items(self) -> None:
        store = _store()
        router = NotificationRouter(store)

        # Only high priority — not digest eligible
        n = _notification("high")
        with patch("rex.notifications.router._send_desktop"):
            router.route(n)

        builder = DigestBuilder(store)
        with patch("rex.notifications.digest._send_desktop") as mock_desktop:
            builder.run_digest()
        mock_desktop.assert_not_called()
