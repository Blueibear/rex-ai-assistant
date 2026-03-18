"""Unit tests for rex.notifications.router — NotificationRouter."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from rex.notifications.models import Notification, NotificationStore
from rex.notifications.router import NotificationRouter, QuietHoursChecker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store() -> NotificationStore:
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return NotificationStore(db_path=Path(tmp.name))


def _make_notification(priority: str = "low", **kwargs: object) -> Notification:
    defaults: dict[str, object] = {
        "title": "Test",
        "body": "Body",
        "source": "test",
        "priority": priority,
    }
    defaults.update(kwargs)
    return Notification(**defaults)  # type: ignore[arg-type]


def _never_suppress() -> QuietHoursChecker:
    mock: MagicMock = MagicMock(spec=QuietHoursChecker)
    mock.should_suppress.return_value = False
    return mock  # type: ignore[return-value]


def _always_suppress() -> QuietHoursChecker:
    mock: MagicMock = MagicMock(spec=QuietHoursChecker)
    mock.should_suppress.return_value = True
    return mock  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Critical priority
# ---------------------------------------------------------------------------


class TestCriticalPriority:
    def test_sends_desktop_notification(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)
        n = _make_notification("critical")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once_with(n.title, n.body)

    def test_stored_to_db(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)
        n = _make_notification("critical")
        with patch("rex.notifications.router._send_desktop"):
            router.route(n)
        assert any(x.id == n.id for x in store.get_unread())

    def test_ignores_quiet_hours(self) -> None:
        store = _make_store()
        router = NotificationRouter(store, quiet_hours=_always_suppress())
        n = _make_notification("critical")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once()


# ---------------------------------------------------------------------------
# High priority
# ---------------------------------------------------------------------------


class TestHighPriority:
    def test_sends_desktop_notification(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)
        n = _make_notification("high")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once_with(n.title, n.body)

    def test_stored_to_db(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)
        n = _make_notification("high")
        with patch("rex.notifications.router._send_desktop"):
            router.route(n)
        assert any(x.id == n.id for x in store.get_unread())

    def test_ignores_quiet_hours(self) -> None:
        store = _make_store()
        router = NotificationRouter(store, quiet_hours=_always_suppress())
        n = _make_notification("high")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once()


# ---------------------------------------------------------------------------
# Medium priority
# ---------------------------------------------------------------------------


class TestMediumPriority:
    def test_sends_desktop_when_not_in_quiet_hours(self) -> None:
        store = _make_store()
        router = NotificationRouter(store, quiet_hours=_never_suppress())
        n = _make_notification("medium")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once_with(n.title, n.body)

    def test_no_desktop_when_in_quiet_hours(self) -> None:
        store = _make_store()
        router = NotificationRouter(store, quiet_hours=_always_suppress())
        n = _make_notification("medium")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_not_called()

    def test_stored_even_when_suppressed(self) -> None:
        store = _make_store()
        router = NotificationRouter(store, quiet_hours=_always_suppress())
        n = _make_notification("medium")
        with patch("rex.notifications.router._send_desktop"):
            router.route(n)
        assert any(x.id == n.id for x in store.get_unread())

    def test_quiet_hours_checker_called(self) -> None:
        store = _make_store()
        qh = _never_suppress()
        router = NotificationRouter(store, quiet_hours=qh)
        n = _make_notification("medium")
        with patch("rex.notifications.router._send_desktop"):
            router.route(n)
        qh.should_suppress.assert_called_once_with(n)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Low priority
# ---------------------------------------------------------------------------


class TestLowPriority:
    def test_no_desktop_notification(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)
        n = _make_notification("low")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_not_called()

    def test_stored_to_db(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)
        n = _make_notification("low")
        router.route(n)
        assert any(x.id == n.id for x in store.get_unread())

    def test_digest_eligible_set_to_true(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)
        n = _make_notification("low", digest_eligible=False)
        router.route(n)
        stored = next(x for x in store.get_unread() if x.id == n.id)
        assert stored.digest_eligible is True

    def test_original_notification_not_mutated(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)
        n = _make_notification("low", digest_eligible=False)
        router.route(n)
        assert n.digest_eligible is False  # original unchanged


# ---------------------------------------------------------------------------
# Default quiet-hours (no-op)
# ---------------------------------------------------------------------------


class TestDefaultNoQuietHours:
    def test_medium_always_dispatched_when_no_quiet_hours_configured(self) -> None:
        store = _make_store()
        router = NotificationRouter(store)  # no quiet_hours arg
        n = _make_notification("medium")
        with patch("rex.notifications.router._send_desktop") as mock_desktop:
            router.route(n)
        mock_desktop.assert_called_once()
