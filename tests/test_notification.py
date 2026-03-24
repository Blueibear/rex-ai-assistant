"""Tests for notification system with priority routing and digest mode."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.notification import (
    NotificationRequest,
    Notifier,
    get_notifier,
    set_notifier,
)


@pytest.fixture
def temp_notification_path(tmp_path):
    """Create a temporary notification storage path."""
    notif_path = tmp_path / "notifications"
    notif_path.mkdir()
    return notif_path


@pytest.fixture
def notifier(temp_notification_path):
    """Create a Notifier instance with temporary storage."""
    return Notifier(storage_path=temp_notification_path)


def test_notification_request_model():
    """Test NotificationRequest model creation."""
    notif = NotificationRequest(
        priority="urgent",
        title="Test Alert",
        body="This is a test notification",
        channel_preferences=["sms", "email"],
    )

    assert notif.priority == "urgent"
    assert notif.title == "Test Alert"
    assert notif.body == "This is a test notification"
    assert notif.channel_preferences == ["sms", "email"]
    assert notif.id.startswith("notif_")
    assert isinstance(notif.timestamp, datetime)


def test_notification_request_defaults():
    """Test NotificationRequest default values."""
    notif = NotificationRequest(
        title="Test",
        body="Test body",
    )

    assert notif.priority == "normal"
    assert notif.channel_preferences == ["dashboard"]
    assert notif.metadata == {}
    assert notif.acknowledged_at is None


def test_notifier_initialization(temp_notification_path):
    """Test Notifier initialization."""
    notifier = Notifier(
        digest_interval_seconds=7200,
        storage_path=temp_notification_path,
    )

    assert notifier.digest_interval == 7200
    assert notifier.storage_path == temp_notification_path
    assert notifier.digest_queues == {}


def test_notifier_send_urgent(notifier):
    """Test sending urgent notifications to all channels."""
    notif = NotificationRequest(
        priority="urgent",
        title="Critical Alert",
        body="System failure detected",
        channel_preferences=["dashboard", "sms", "email"],
    )

    with patch.object(notifier, "_dispatch_to_channel") as mock_dispatch:
        notifier.send(notif)

        # Urgent should send to ALL channels
        assert mock_dispatch.call_count == 3
        channels_called = [call[0][0] for call in mock_dispatch.call_args_list]
        assert set(channels_called) == {"dashboard", "sms", "email"}


def test_notifier_send_normal(notifier):
    """Test sending normal notifications to first available channel."""
    notif = NotificationRequest(
        priority="normal",
        title="Info Alert",
        body="Regular update",
        channel_preferences=["email", "dashboard"],
    )

    with patch.object(notifier, "_dispatch_to_channel") as mock_dispatch:
        notifier.send(notif)

        # Normal should only send to FIRST channel
        assert mock_dispatch.call_count == 1
        assert mock_dispatch.call_args[0][0] == "email"


def test_notifier_send_normal_fallback(notifier):
    """Test normal notification fallback when first channel fails."""
    notif = NotificationRequest(
        priority="normal",
        title="Info Alert",
        body="Regular update",
        channel_preferences=["email", "dashboard", "sms"],
    )

    call_count = 0

    def mock_dispatch_fail_once(channel, notification):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Channel unavailable")
        # Second call succeeds

    with patch.object(notifier, "_dispatch_to_channel", side_effect=mock_dispatch_fail_once):
        notifier.send(notif)

        # Should have tried email (failed), then dashboard (succeeded)
        assert call_count == 2


def test_notifier_send_digest(notifier):
    """Test sending digest notifications queues them."""
    notif1 = NotificationRequest(
        priority="digest",
        title="Update 1",
        body="First update",
        channel_preferences=["email"],
    )
    notif2 = NotificationRequest(
        priority="digest",
        title="Update 2",
        body="Second update",
        channel_preferences=["email"],
    )

    notifier.send(notif1)
    notifier.send(notif2)

    # Check queue
    assert "email" in notifier.digest_queues
    assert len(notifier.digest_queues["email"].notifications) == 2


def test_notifier_digest_persistence(temp_notification_path):
    """Test digest queues are persisted to disk."""
    notifier = Notifier(storage_path=temp_notification_path)

    notif = NotificationRequest(
        priority="digest",
        title="Test Digest",
        body="Test body",
        channel_preferences=["email"],
    )
    notifier.send(notif)

    # Check file was created
    digest_file = temp_notification_path / "digests.json"
    assert digest_file.exists()

    # Load and verify
    with open(digest_file) as f:
        data = json.load(f)

    assert "email" in data
    assert len(data["email"]["notifications"]) == 1


def test_notifier_flush_digests(notifier):
    """Test flushing digest queues."""
    # Queue some digest notifications
    for i in range(3):
        notif = NotificationRequest(
            priority="digest",
            title=f"Update {i}",
            body=f"Body {i}",
            channel_preferences=["email"],
        )
        notifier.send(notif)

    assert len(notifier.digest_queues["email"].notifications) == 3

    with patch.object(notifier, "_dispatch_to_channel") as mock_dispatch:
        count = notifier.flush_digests()

        # Should flush 1 queue
        assert count == 1
        # Should send 1 digest summary
        assert mock_dispatch.call_count == 1

        # Queue should be empty
        assert len(notifier.digest_queues["email"].notifications) == 0


def test_notifier_flush_specific_channel(notifier):
    """Test flushing a specific channel."""
    # Queue notifications for multiple channels
    for channel in ["email", "dashboard"]:
        notif = NotificationRequest(
            priority="digest",
            title="Test",
            body="Test",
            channel_preferences=[channel],
        )
        notifier.send(notif)

    with patch.object(notifier, "_dispatch_to_channel"):
        count = notifier.flush_digests(channel="email")

        # Should only flush email channel
        assert count == 1
        assert len(notifier.digest_queues["email"].notifications) == 0
        assert len(notifier.digest_queues["dashboard"].notifications) == 1


def test_notifier_list_digests(notifier):
    """Test listing queued digest notifications."""
    notif = NotificationRequest(
        priority="digest",
        title="Test Digest",
        body="Test body",
        channel_preferences=["email"],
    )
    notifier.send(notif)

    digests = notifier.list_digests()

    assert "email" in digests
    assert len(digests["email"]) == 1
    assert digests["email"][0]["title"] == "Test Digest"
    assert digests["email"][0]["body"] == "Test body"


def test_notifier_dispatch_to_dashboard(notifier):
    """Test dashboard notification dispatch."""
    notif = NotificationRequest(
        priority="normal",
        title="Dashboard Test",
        body="Test body",
    )

    # Should not raise an error
    notifier._send_to_dashboard(notif)


def test_notifier_sms_stub_does_not_raise(notifier):
    """_send_to_sms is a logging stub — it must not raise for any input."""
    notif = NotificationRequest(
        priority="urgent",
        title="SMS Test",
        body="Test message",
        metadata={"to_number": "+15551234567"},
    )
    # OPENCLAW-REPLACE: stub; no real SMS delivery until OpenClaw messaging is wired
    notifier._send_to_sms(notif)  # should not raise


def test_notifier_event_subscription():
    """Test event bus subscription setup (via EventBridge)."""
    mock_bridge = MagicMock()
    with patch("rex.openclaw.event_bridge.EventBridge", return_value=mock_bridge):
        notifier = Notifier()
        notifier.setup_event_subscriptions()

        # Should subscribe to email and calendar events via EventBridge
        assert mock_bridge.subscribe.call_count == 2
        calls = mock_bridge.subscribe.call_args_list
        event_types = [call[0][0] for call in calls]
        assert "email.unread" in event_types
        assert "calendar.update" in event_types


def test_notifier_email_unread_handler(notifier):
    """Test handling email.unread events."""
    event = MagicMock()
    event.payload = {
        "emails": [
            {
                "id": "email_1",
                "from_addr": "manager@example.com",
                "subject": "Urgent Request",
                "importance_score": 0.9,
            }
        ]
    }

    with patch.object(notifier, "send") as mock_send:
        notifier._on_email_unread(event)

        # Should create an urgent notification
        mock_send.assert_called_once()
        sent_notif = mock_send.call_args[0][0]
        assert sent_notif.priority == "urgent"
        assert "High Importance Email" in sent_notif.title


def test_notifier_email_unread_handler_low_importance(notifier):
    """Test email handler ignores low importance emails."""
    event = MagicMock()
    event.payload = {
        "emails": [
            {
                "id": "email_2",
                "from_addr": "newsletter@example.com",
                "subject": "Weekly Update",
                "importance_score": 0.3,
            }
        ]
    }

    with patch.object(notifier, "send") as mock_send:
        notifier._on_email_unread(event)

        # Should not send notification for low importance
        mock_send.assert_not_called()


def test_notifier_calendar_update_handler(notifier):
    """Test handling calendar.update events."""
    from datetime import timedelta

    now = datetime.now(timezone.utc)
    soon = now + timedelta(minutes=10)

    event = MagicMock()
    event.payload = {
        "events": [
            {
                "event_id": "evt_1",
                "title": "Team Meeting",
                "start_time": soon.isoformat(),
            }
        ]
    }

    with patch.object(notifier, "send") as mock_send:
        notifier._on_calendar_update(event)

        # Should create a notification for upcoming event
        mock_send.assert_called_once()
        sent_notif = mock_send.call_args[0][0]
        assert sent_notif.priority == "normal"
        assert "Upcoming Calendar Event" in sent_notif.title


def test_global_notifier():
    """Test global notifier getter/setter."""
    notifier1 = get_notifier()
    notifier2 = get_notifier()
    assert notifier1 is notifier2

    # Set custom notifier
    custom_notifier = Notifier(storage_path=Path("/tmp/custom_notifications"))
    set_notifier(custom_notifier)

    notifier3 = get_notifier()
    assert notifier3 is custom_notifier

    # Reset
    set_notifier(None)
