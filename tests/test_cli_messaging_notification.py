"""Tests for CLI messaging and notification commands."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest

from rex.cli import cmd_msg, cmd_notify


@pytest.fixture
def temp_sms_file(tmp_path):
    """Create a temporary SMS mock file."""
    import json

    mock_file = tmp_path / "mock_sms.json"
    mock_file.write_text(json.dumps({"messages": []}, indent=2))
    return mock_file


@pytest.fixture
def temp_notification_path(tmp_path):
    """Create a temporary notification storage path."""
    notif_path = tmp_path / "notifications"
    notif_path.mkdir()
    return notif_path


# --- Messaging CLI Tests ---


def test_cmd_msg_send(capsys):
    """cmd_msg SMS send is a stub — returns 1 with migration message."""
    # OPENCLAW-REPLACE: SMS delivery migrated to OpenClaw; cmd_msg is a stub
    args = argparse.Namespace(
        msg_command="send",
        channel="sms",
        to="+15551234567",
        body="Hello from CLI",
    )
    result = cmd_msg(args)
    assert result == 1
    captured = capsys.readouterr()
    assert "not available" in captured.out


def test_cmd_msg_send_unsupported_channel(capsys):
    """Test sending via unsupported channel."""
    args = argparse.Namespace(
        msg_command="send",
        channel="telegram",
        to="@user",
        body="Test",
    )

    result = cmd_msg(args)

    assert result == 1
    captured = capsys.readouterr()
    assert "Unsupported channel 'telegram'" in captured.out


def test_cmd_msg_receive_empty(capsys):
    """cmd_msg SMS receive is a stub — returns 1 with migration message."""
    # OPENCLAW-REPLACE: SMS receive migrated to OpenClaw; cmd_msg is a stub
    args = argparse.Namespace(
        msg_command="receive",
        channel="sms",
        limit=10,
    )
    result = cmd_msg(args)
    assert result == 1
    captured = capsys.readouterr()
    assert "not available" in captured.out


def test_cmd_msg_receive_with_messages(capsys):
    """cmd_msg SMS receive with messages is a stub — returns 1 with migration message."""
    # OPENCLAW-REPLACE: SMS receive migrated to OpenClaw; cmd_msg is a stub
    args = argparse.Namespace(
        msg_command="receive",
        channel="sms",
        limit=10,
    )
    result = cmd_msg(args)
    assert result == 1
    captured = capsys.readouterr()
    assert "not available" in captured.out


def test_cmd_msg_receive_unsupported_channel(capsys):
    """Test receiving from unsupported channel."""
    args = argparse.Namespace(
        msg_command="receive",
        channel="discord",
        limit=10,
    )

    result = cmd_msg(args)

    assert result == 1
    captured = capsys.readouterr()
    assert "Unsupported channel 'discord'" in captured.out


def test_cmd_msg_unknown_subcommand(capsys):
    """Test unknown messaging subcommand."""
    args = argparse.Namespace(
        msg_command="invalid",
    )

    result = cmd_msg(args)

    assert result == 1
    captured = capsys.readouterr()
    assert "Unknown messaging subcommand" in captured.out


# --- Notification CLI Tests ---


def test_cmd_notify_send(temp_notification_path, capsys):
    """Test sending notification via CLI."""
    args = argparse.Namespace(
        notify_command="send",
        priority="urgent",
        title="Test Alert",
        body="This is a test",
        channels="sms,email",
    )

    with (
        patch("rex.notification.get_notifier") as mock_get_notifier,
        patch("rex.notification.get_escalation_manager") as mock_get_escalation,
    ):
        from rex.notification import EscalationManager, Notifier

        mock_notifier = Notifier(storage_path=temp_notification_path)
        mock_escalation = EscalationManager()

        mock_get_notifier.return_value = mock_notifier
        mock_get_escalation.return_value = mock_escalation

        with patch.object(mock_notifier, "send"):
            result = cmd_notify(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Notification sent successfully" in captured.out
            assert "Priority: urgent" in captured.out
            assert "Channels: sms, email" in captured.out


def test_cmd_notify_send_default_channel(temp_notification_path, capsys):
    """Test sending notification with default channel."""
    args = argparse.Namespace(
        notify_command="send",
        priority="normal",
        title="Test",
        body="Test body",
        channels=None,
    )

    with (
        patch("rex.notification.get_notifier") as mock_get_notifier,
        patch("rex.notification.get_escalation_manager") as mock_get_escalation,
    ):
        from rex.notification import EscalationManager, Notifier

        mock_notifier = Notifier(storage_path=temp_notification_path)
        mock_escalation = EscalationManager()

        mock_get_notifier.return_value = mock_notifier
        mock_get_escalation.return_value = mock_escalation

        with patch.object(mock_notifier, "send"):
            result = cmd_notify(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Channels: dashboard" in captured.out


def test_cmd_notify_list_digests_empty(temp_notification_path, capsys):
    """Test listing digests when none exist."""
    args = argparse.Namespace(
        notify_command="list-digests",
    )

    with patch("rex.notification.get_notifier") as mock_get_notifier:
        from rex.notification import Notifier

        mock_notifier = Notifier(storage_path=temp_notification_path)
        mock_get_notifier.return_value = mock_notifier

        result = cmd_notify(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No queued digest notifications" in captured.out


def test_cmd_notify_list_digests_with_notifications(temp_notification_path, capsys):
    """Test listing queued digest notifications."""
    from rex.notification import NotificationRequest, Notifier

    notifier = Notifier(storage_path=temp_notification_path)

    # Queue some digests
    for i in range(2):
        notif = NotificationRequest(
            priority="digest",
            title=f"Update {i}",
            body=f"Body {i}",
            channel_preferences=["email"],
        )
        notifier.send(notif)

    args = argparse.Namespace(
        notify_command="list-digests",
    )

    with patch("rex.notification.get_notifier", return_value=notifier):
        result = cmd_notify(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Queued Digest Notifications" in captured.out
        assert "Channel: email" in captured.out
        assert "Count: 2" in captured.out
        assert "Update 0" in captured.out
        assert "Update 1" in captured.out


def test_cmd_notify_flush_digests_empty(temp_notification_path, capsys):
    """Test flushing digests when none exist."""
    args = argparse.Namespace(
        notify_command="flush-digests",
        channel=None,
    )

    with patch("rex.notification.get_notifier") as mock_get_notifier:
        from rex.notification import Notifier

        mock_notifier = Notifier(storage_path=temp_notification_path)
        mock_get_notifier.return_value = mock_notifier

        result = cmd_notify(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No digest notifications to flush" in captured.out


def test_cmd_notify_flush_digests_with_notifications(temp_notification_path, capsys):
    """Test flushing digest queues."""
    from rex.notification import NotificationRequest, Notifier

    notifier = Notifier(storage_path=temp_notification_path)

    # Queue some digests
    notif = NotificationRequest(
        priority="digest",
        title="Update",
        body="Body",
        channel_preferences=["email"],
    )
    notifier.send(notif)

    args = argparse.Namespace(
        notify_command="flush-digests",
        channel=None,
    )

    with patch("rex.notification.get_notifier", return_value=notifier):
        with patch.object(notifier, "_dispatch_to_channel"):
            result = cmd_notify(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Flushed 1 digest queue(s)" in captured.out


def test_cmd_notify_flush_digests_specific_channel(temp_notification_path, capsys):
    """Test flushing specific channel."""
    from rex.notification import NotificationRequest, Notifier

    notifier = Notifier(storage_path=temp_notification_path)

    # Queue digests for multiple channels
    for channel in ["email", "dashboard"]:
        notif = NotificationRequest(
            priority="digest",
            title="Update",
            body="Body",
            channel_preferences=[channel],
        )
        notifier.send(notif)

    args = argparse.Namespace(
        notify_command="flush-digests",
        channel="email",
    )

    with patch("rex.notification.get_notifier", return_value=notifier):
        with patch.object(notifier, "_dispatch_to_channel"):
            result = cmd_notify(args)

            assert result == 0
            captured = capsys.readouterr()
            assert "Flushed digest queue for channel: email" in captured.out


def test_cmd_notify_ack(temp_notification_path, capsys):
    """Test acknowledging a notification."""
    args = argparse.Namespace(
        notify_command="ack",
        notification_id="notif_test123",
    )

    with patch("rex.notification.get_escalation_manager") as mock_get_escalation:
        mock_manager = MagicMock()
        mock_manager.acknowledge.return_value = True
        mock_get_escalation.return_value = mock_manager

        result = cmd_notify(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Acknowledged notification: notif_test123" in captured.out


def test_cmd_notify_ack_not_found(temp_notification_path, capsys):
    """Test acknowledging non-existent notification."""
    args = argparse.Namespace(
        notify_command="ack",
        notification_id="notif_invalid",
    )

    with patch("rex.notification.get_escalation_manager") as mock_get_escalation:
        mock_manager = MagicMock()
        mock_manager.acknowledge.return_value = False
        mock_get_escalation.return_value = mock_manager

        result = cmd_notify(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Notification not found or already acknowledged" in captured.out


def test_cmd_notify_unknown_subcommand(capsys):
    """Test unknown notification subcommand."""
    args = argparse.Namespace(
        notify_command="invalid",
    )

    with patch("rex.notification.get_notifier"), patch("rex.notification.get_escalation_manager"):
        result = cmd_notify(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown notification subcommand" in captured.out
