"""Tests for notification escalation manager with quiet hours and escalation rules."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from unittest.mock import patch

import pytest

from rex.notification import (
    EscalationManager,
    NotificationRequest,
    get_escalation_manager,
    set_escalation_manager,
)


@pytest.fixture
def escalation_manager():
    """Create an EscalationManager instance."""
    return EscalationManager(
        quiet_hours_start=time(22, 0),
        quiet_hours_end=time(7, 0),
        escalation_delay_minutes=5,
    )


def test_escalation_manager_initialization():
    """Test EscalationManager initialization."""
    manager = EscalationManager(
        quiet_hours_start=time(23, 0),
        quiet_hours_end=time(6, 0),
        escalation_delay_minutes=10,
    )

    assert manager.quiet_hours_start == time(23, 0)
    assert manager.quiet_hours_end == time(6, 0)
    assert manager.escalation_delay == 10
    assert manager.dnd_enabled is False
    assert len(manager.pending_escalations) == 0


def test_is_quiet_hours_during(escalation_manager):
    """Test quiet hours detection during quiet period."""
    # During quiet hours (e.g., 23:00)
    test_time = datetime(2024, 1, 15, 23, 0, 0, tzinfo=UTC)
    assert escalation_manager.is_quiet_hours(test_time) is True

    # Also during quiet hours (e.g., 3:00 AM)
    test_time = datetime(2024, 1, 15, 3, 0, 0, tzinfo=UTC)
    assert escalation_manager.is_quiet_hours(test_time) is True


def test_is_quiet_hours_outside(escalation_manager):
    """Test quiet hours detection outside quiet period."""
    # Outside quiet hours (e.g., 10:00 AM)
    test_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=UTC)
    assert escalation_manager.is_quiet_hours(test_time) is False

    # Also outside (e.g., 8:00 PM)
    test_time = datetime(2024, 1, 15, 20, 0, 0, tzinfo=UTC)
    assert escalation_manager.is_quiet_hours(test_time) is False


def test_is_quiet_hours_boundaries(escalation_manager):
    """Test quiet hours boundary conditions."""
    # At start of quiet hours (22:00)
    test_time = datetime(2024, 1, 15, 22, 0, 0, tzinfo=UTC)
    assert escalation_manager.is_quiet_hours(test_time) is True

    # Just before end of quiet hours (06:59)
    test_time = datetime(2024, 1, 15, 6, 59, 0, tzinfo=UTC)
    assert escalation_manager.is_quiet_hours(test_time) is True

    # At end of quiet hours (07:00)
    test_time = datetime(2024, 1, 15, 7, 0, 0, tzinfo=UTC)
    assert escalation_manager.is_quiet_hours(test_time) is False


def test_is_quiet_hours_no_midnight_crossing():
    """Test quiet hours that don't cross midnight."""
    manager = EscalationManager(
        quiet_hours_start=time(12, 0),
        quiet_hours_end=time(14, 0),
    )

    # During (13:00)
    assert manager.is_quiet_hours(datetime(2024, 1, 15, 13, 0, tzinfo=UTC)) is True

    # Before (11:00)
    assert manager.is_quiet_hours(datetime(2024, 1, 15, 11, 0, tzinfo=UTC)) is False

    # After (15:00)
    assert manager.is_quiet_hours(datetime(2024, 1, 15, 15, 0, tzinfo=UTC)) is False


def test_should_suppress_urgent_never_suppressed(escalation_manager):
    """Test urgent notifications are never suppressed."""
    notif = NotificationRequest(
        priority="urgent",
        title="Critical Alert",
        body="System failure",
    )

    # Not suppressed even during quiet hours
    with patch.object(escalation_manager, "is_quiet_hours", return_value=True):
        assert escalation_manager.should_suppress(notif) is False

    # Not suppressed even with DND enabled
    escalation_manager.set_dnd(True)
    assert escalation_manager.should_suppress(notif) is False


def test_should_suppress_normal_during_quiet_hours(escalation_manager):
    """Test normal notifications suppressed during quiet hours."""
    notif = NotificationRequest(
        priority="normal",
        title="Info",
        body="Regular update",
    )

    with patch.object(escalation_manager, "is_quiet_hours", return_value=True):
        assert escalation_manager.should_suppress(notif) is True

    with patch.object(escalation_manager, "is_quiet_hours", return_value=False):
        assert escalation_manager.should_suppress(notif) is False


def test_should_suppress_normal_with_dnd(escalation_manager):
    """Test normal notifications suppressed with DND enabled."""
    notif = NotificationRequest(
        priority="normal",
        title="Info",
        body="Regular update",
    )

    escalation_manager.set_dnd(True)
    assert escalation_manager.should_suppress(notif) is True

    escalation_manager.set_dnd(False)
    with patch.object(escalation_manager, "is_quiet_hours", return_value=False):
        assert escalation_manager.should_suppress(notif) is False


def test_should_suppress_digest_during_quiet_hours(escalation_manager):
    """Test digest notifications suppressed during quiet hours."""
    notif = NotificationRequest(
        priority="digest",
        title="Summary",
        body="Daily digest",
    )

    with patch.object(escalation_manager, "is_quiet_hours", return_value=True):
        assert escalation_manager.should_suppress(notif) is True


def test_track_notification(escalation_manager):
    """Test tracking urgent notifications for escalation."""
    notif = NotificationRequest(
        priority="urgent",
        title="Alert",
        body="Test",
    )

    escalation_manager.track_notification(notif, next_channel="email")

    assert notif.id in escalation_manager.pending_escalations
    rule = escalation_manager.pending_escalations[notif.id]
    assert rule.notification_id == notif.id
    assert rule.next_channel == "email"
    assert rule.escalation_delay_minutes == 5
    assert rule.escalated is False


def test_track_notification_ignores_non_urgent(escalation_manager):
    """Test non-urgent notifications are not tracked."""
    notif = NotificationRequest(
        priority="normal",
        title="Info",
        body="Test",
    )

    escalation_manager.track_notification(notif, next_channel="email")

    assert notif.id not in escalation_manager.pending_escalations


def test_acknowledge_notification(escalation_manager):
    """Test acknowledging a tracked notification."""
    notif = NotificationRequest(
        priority="urgent",
        title="Alert",
        body="Test",
    )

    escalation_manager.track_notification(notif)
    assert notif.id in escalation_manager.pending_escalations

    result = escalation_manager.acknowledge(notif.id)

    assert result is True
    assert notif.id not in escalation_manager.pending_escalations


def test_acknowledge_notification_not_found(escalation_manager):
    """Test acknowledging non-existent notification."""
    result = escalation_manager.acknowledge("nonexistent_id")
    assert result is False


def test_check_escalations_none_pending(escalation_manager):
    """Test checking escalations when none are pending."""
    to_escalate = escalation_manager.check_escalations()
    assert to_escalate == []


def test_check_escalations_not_yet_due(escalation_manager):
    """Test checking escalations when none are due yet."""
    notif = NotificationRequest(
        priority="urgent",
        title="Alert",
        body="Test",
    )

    # Track notification (sent just now)
    escalation_manager.track_notification(notif, next_channel="email")

    # Check escalations immediately (not enough time elapsed)
    to_escalate = escalation_manager.check_escalations()
    assert to_escalate == []


def test_check_escalations_due(escalation_manager):
    """Test checking escalations when one is due."""
    notif = NotificationRequest(
        priority="urgent",
        title="Alert",
        body="Test",
    )

    # Track notification with a timestamp 10 minutes ago
    past_time = datetime.now(UTC) - timedelta(minutes=10)
    notif.timestamp = past_time

    escalation_manager.track_notification(notif, next_channel="email")

    # Manually update the sent_at time
    escalation_manager.pending_escalations[notif.id].sent_at = past_time

    # Check escalations (should be due)
    to_escalate = escalation_manager.check_escalations()

    assert len(to_escalate) == 1
    assert to_escalate[0] == (notif.id, "email")

    # Rule should be marked as escalated
    assert escalation_manager.pending_escalations[notif.id].escalated is True


def test_check_escalations_multiple(escalation_manager):
    """Test checking multiple escalations."""
    past_time = datetime.now(UTC) - timedelta(minutes=10)

    for i in range(3):
        notif = NotificationRequest(
            priority="urgent",
            title=f"Alert {i}",
            body="Test",
        )
        notif.timestamp = past_time

        escalation_manager.track_notification(notif, next_channel=f"channel_{i}")
        escalation_manager.pending_escalations[notif.id].sent_at = past_time

    to_escalate = escalation_manager.check_escalations()

    assert len(to_escalate) == 3
    channels = [ch for _, ch in to_escalate]
    assert set(channels) == {"channel_0", "channel_1", "channel_2"}


def test_check_escalations_only_once(escalation_manager):
    """Test escalations only trigger once per notification."""
    notif = NotificationRequest(
        priority="urgent",
        title="Alert",
        body="Test",
    )

    past_time = datetime.now(UTC) - timedelta(minutes=10)
    notif.timestamp = past_time

    escalation_manager.track_notification(notif, next_channel="email")
    escalation_manager.pending_escalations[notif.id].sent_at = past_time

    # First check - should escalate
    to_escalate = escalation_manager.check_escalations()
    assert len(to_escalate) == 1

    # Second check - should NOT escalate again
    to_escalate = escalation_manager.check_escalations()
    assert len(to_escalate) == 0


def test_set_dnd(escalation_manager):
    """Test enabling/disabling do-not-disturb mode."""
    assert escalation_manager.dnd_enabled is False

    escalation_manager.set_dnd(True)
    assert escalation_manager.dnd_enabled is True

    escalation_manager.set_dnd(False)
    assert escalation_manager.dnd_enabled is False


def test_global_escalation_manager():
    """Test global escalation manager getter/setter."""
    manager1 = get_escalation_manager()
    manager2 = get_escalation_manager()
    assert manager1 is manager2

    # Set custom manager
    custom_manager = EscalationManager(escalation_delay_minutes=15)
    set_escalation_manager(custom_manager)

    manager3 = get_escalation_manager()
    assert manager3 is custom_manager

    # Reset
    set_escalation_manager(None)
