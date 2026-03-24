"""Tests for US-030: Notification routing.

Acceptance criteria:
- notifications generated
- routing rules applied
- delivery attempted
- Typecheck passes
"""

from __future__ import annotations

import pytest

from rex.notification import (
    EscalationManager,
    NotificationRequest,
    Notifier,
    get_notifier,
    set_notifier,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolate_global_notifier(tmp_path):
    """Reset global notifier state before and after each test."""
    set_notifier(None)
    yield
    set_notifier(None)


def make_notifier(tmp_path, **kwargs):
    """Create a Notifier backed by a temp directory."""
    return Notifier(storage_path=tmp_path / "notifications", **kwargs)


# ---------------------------------------------------------------------------
# Notifications generated
# ---------------------------------------------------------------------------


def test_notification_request_has_default_id(tmp_path):
    notif = NotificationRequest(title="Hello", body="World")
    assert notif.id.startswith("notif_")
    assert len(notif.id) > 6


def test_notification_request_defaults(tmp_path):
    notif = NotificationRequest(title="T", body="B")
    assert notif.priority == "normal"
    assert notif.channel_preferences == ["dashboard"]
    assert notif.metadata == {}
    assert notif.idempotency_key is None


def test_notification_request_urgent(tmp_path):
    notif = NotificationRequest(
        title="Alert",
        body="Critical failure",
        priority="urgent",
        channel_preferences=["sms", "email"],
    )
    assert notif.priority == "urgent"
    assert notif.channel_preferences == ["sms", "email"]


def test_notification_request_digest(tmp_path):
    notif = NotificationRequest(title="Weekly", body="Summary", priority="digest")
    assert notif.priority == "digest"


def test_notifier_instantiates(tmp_path):
    notifier = make_notifier(tmp_path)
    assert notifier is not None
    assert notifier.storage_path.exists()


# ---------------------------------------------------------------------------
# Routing rules applied
# ---------------------------------------------------------------------------


def test_urgent_sends_to_all_channels(tmp_path):
    """Urgent notifications are dispatched to every channel_preference."""
    dispatched = []

    notifier = make_notifier(tmp_path)
    notifier._dispatch_to_channel = lambda ch, n: dispatched.append(ch)

    notif = NotificationRequest(
        title="Fire",
        body="Building on fire",
        priority="urgent",
        channel_preferences=["dashboard", "email"],
    )
    notifier.send(notif)
    assert "dashboard" in dispatched
    assert "email" in dispatched


def test_normal_sends_to_first_channel_only(tmp_path):
    """Normal notifications stop after the first successful channel."""
    dispatched = []

    notifier = make_notifier(tmp_path)
    notifier._dispatch_to_channel = lambda ch, n: dispatched.append(ch)

    notif = NotificationRequest(
        title="Reminder",
        body="Check email",
        priority="normal",
        channel_preferences=["dashboard", "email"],
    )
    notifier.send(notif)
    assert dispatched == ["dashboard"]


def test_digest_queues_notification(tmp_path):
    """Digest notifications are queued rather than dispatched immediately."""
    dispatched = []

    notifier = make_notifier(tmp_path)
    notifier._dispatch_to_channel = lambda ch, n: dispatched.append(ch)

    notif = NotificationRequest(
        title="Weekly summary",
        body="Here is the summary",
        priority="digest",
        channel_preferences=["dashboard"],
    )
    notifier.send(notif)

    # Nothing dispatched immediately
    assert dispatched == []
    # Queued for dashboard
    assert "dashboard" in notifier.digest_queues
    assert len(notifier.digest_queues["dashboard"].notifications) == 1


def test_routing_uses_channel_preferences_order(tmp_path):
    """Normal routing uses channel_preferences order and stops on success."""
    dispatched = []

    notifier = make_notifier(tmp_path)
    notifier._dispatch_to_channel = lambda ch, n: dispatched.append(ch)

    notif = NotificationRequest(
        title="Test",
        body="Body",
        priority="normal",
        channel_preferences=["email", "dashboard", "sms"],
    )
    notifier.send(notif)
    assert dispatched == ["email"]


def test_normal_falls_through_to_next_channel_on_failure(tmp_path):
    """Normal routing falls through to next channel if first dispatch fails."""
    dispatched = []

    notifier = make_notifier(tmp_path)

    def fake_dispatch(ch, n):
        if ch == "email":
            raise RuntimeError("email down")
        dispatched.append(ch)

    notifier._dispatch_to_channel = fake_dispatch

    notif = NotificationRequest(
        title="Test",
        body="Body",
        priority="normal",
        channel_preferences=["email", "dashboard"],
    )
    notifier.send(notif)
    assert dispatched == ["dashboard"]


# ---------------------------------------------------------------------------
# Delivery attempted
# ---------------------------------------------------------------------------


def test_send_to_channel_calls_dispatch(tmp_path):
    """send_to_channel() directly calls _dispatch_to_channel."""
    dispatched = []
    notifier = make_notifier(tmp_path)
    notifier._dispatch_to_channel = lambda ch, n: dispatched.append(ch)

    notif = NotificationRequest(title="Direct", body="Send")
    notifier.send_to_channel("dashboard", notif)
    assert dispatched == ["dashboard"]


def test_dashboard_delivery_attempted(tmp_path, monkeypatch):
    """Dashboard channel sends notification to dashboard store."""
    stored = []

    class FakeStore:
        def write(self, **kwargs):
            stored.append(kwargs)

    monkeypatch.setattr(
        "rex.notification.Notifier._send_to_dashboard",
        lambda self, n: stored.append(n.title),
    )

    notifier = make_notifier(tmp_path)
    notif = NotificationRequest(title="Dashboard notif", body="body", priority="normal")
    notifier.send(notif)

    assert len(stored) == 1
    assert stored[0] == "Dashboard notif"


def test_email_delivery_attempted(tmp_path, monkeypatch):
    """Email channel attempts delivery when email is in preferences."""
    attempts = []
    monkeypatch.setattr(
        "rex.notification.Notifier._send_to_email",
        lambda self, n: attempts.append(n.title),
    )
    notifier = make_notifier(tmp_path)
    notif = NotificationRequest(
        title="Email notif",
        body="body",
        priority="normal",
        channel_preferences=["email"],
    )
    notifier.send(notif)
    assert attempts == ["Email notif"]


def test_sms_delivery_attempted(tmp_path, monkeypatch):
    """SMS channel attempts delivery when sms is in preferences."""
    attempts = []
    monkeypatch.setattr(
        "rex.notification.Notifier._send_to_sms",
        lambda self, n: attempts.append(n.title),
    )
    notifier = make_notifier(tmp_path)
    notif = NotificationRequest(
        title="SMS notif",
        body="body",
        priority="normal",
        channel_preferences=["sms"],
    )
    notifier.send(notif)
    assert attempts == ["SMS notif"]


def test_unknown_channel_logs_warning_no_crash(tmp_path, caplog):
    """Unknown channel is logged and does not crash the notifier."""
    import logging

    notifier = make_notifier(tmp_path)
    notif = NotificationRequest(
        title="Test",
        body="Body",
        priority="normal",
        channel_preferences=["carrier_pigeon"],
    )
    with caplog.at_level(logging.WARNING):
        notifier.send(notif)
    assert "Unknown channel" in caplog.text


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_duplicate_notification_skipped(tmp_path):
    """Notifications with the same id are not sent twice."""
    dispatched = []
    notifier = make_notifier(tmp_path)
    notifier._dispatch_to_channel = lambda ch, n: dispatched.append(ch)

    notif = NotificationRequest(title="Once", body="body")
    notifier.send(notif)
    notifier.send(notif)

    assert len(dispatched) == 1


def test_idempotency_key_prevents_duplicate(tmp_path):
    """Notifications sharing the same idempotency_key are only sent once."""
    dispatched = []
    notifier = make_notifier(tmp_path)
    notifier._dispatch_to_channel = lambda ch, n: dispatched.append(ch)

    notif1 = NotificationRequest(title="First", body="body", idempotency_key="key-abc")
    notif2 = NotificationRequest(title="Second", body="body", idempotency_key="key-abc")
    notifier.send(notif1)
    notifier.send(notif2)

    assert len(dispatched) == 1


# ---------------------------------------------------------------------------
# Escalation manager routing
# ---------------------------------------------------------------------------


def test_escalation_manager_quiet_hours_suppress_normal(tmp_path):
    """Normal notifications suppressed during quiet hours."""
    from datetime import datetime, time, timezone

    em = EscalationManager(quiet_hours_start=time(22, 0), quiet_hours_end=time(7, 0))
    # Midnight — within quiet hours
    dt_midnight = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    assert em.is_quiet_hours(dt_midnight)

    notif = NotificationRequest(title="T", body="B", priority="normal")
    # Simulate quiet hours active
    em.is_quiet_hours = lambda dt=None: True  # type: ignore[method-assign]
    assert em.should_suppress(notif)


def test_escalation_manager_urgent_never_suppressed(tmp_path):
    """Urgent notifications are never suppressed by quiet hours."""
    em = EscalationManager()
    em.dnd_enabled = True  # DND on

    notif = NotificationRequest(title="T", body="B", priority="urgent")
    assert not em.should_suppress(notif)


def test_get_notifier_creates_singleton(tmp_path, monkeypatch):
    """get_notifier() creates a Notifier if none is set."""
    monkeypatch.setattr("rex.notification.Notifier.__init__", lambda self, **kw: None)
    monkeypatch.setattr("rex.notification._notifier", None, raising=False)
    # Just test it doesn't crash
    notifier = get_notifier()
    assert notifier is not None
