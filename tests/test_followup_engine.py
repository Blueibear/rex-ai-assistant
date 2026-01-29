"""Tests for the Follow-up Engine v1.

This test module covers:
1. CueStore add/list/mark/dismiss/prune operations
2. Calendar cue generation
3. Reminder scheduling and follow-up cue creation
4. Conversation integration with cue injection
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


# =============================================================================
# CueStore Tests
# =============================================================================


class TestCueStore:
    """Tests for CueStore operations."""

    def test_add_and_get_cue(self, tmp_path: Path) -> None:
        """Test adding a cue and retrieving it."""
        from rex.cue_store import CueStore

        store = CueStore(storage_path=tmp_path / "cues.json")
        cue = store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-123",
            title="Doctor appointment",
            prompt="How did your doctor appointment go?",
        )

        assert cue.cue_id.startswith("cue_")
        assert cue.user_id == "test_user"
        assert cue.source_type == "calendar"
        assert cue.source_id == "event-123"
        assert cue.status == "pending"

        retrieved = store.get_cue(cue.cue_id)
        assert retrieved is not None
        assert retrieved.cue_id == cue.cue_id

    def test_list_pending_cues(self, tmp_path: Path) -> None:
        """Test listing pending cues with filtering."""
        from rex.cue_store import CueStore

        store = CueStore(storage_path=tmp_path / "cues.json")

        # Add multiple cues for different users
        cue1 = store.add_cue(
            user_id="user1",
            source_type="calendar",
            source_id="event-1",
            title="Event 1",
            prompt="How did event 1 go?",
        )
        store.add_cue(
            user_id="user2",
            source_type="calendar",
            source_id="event-2",
            title="Event 2",
            prompt="How did event 2 go?",
        )

        # List for user1
        pending = store.list_pending_cues("user1")
        assert len(pending) == 1
        assert pending[0].cue_id == cue1.cue_id

    def test_mark_cue_asked(self, tmp_path: Path) -> None:
        """Test marking a cue as asked."""
        from rex.cue_store import CueStore

        store = CueStore(storage_path=tmp_path / "cues.json")
        cue = store.add_cue(
            user_id="test_user",
            source_type="reminder",
            source_id="rem-1",
            title="Task",
            prompt="Did you complete the task?",
        )

        assert store.mark_asked(cue.cue_id)

        updated = store.get_cue(cue.cue_id)
        assert updated is not None
        assert updated.status == "asked"
        assert updated.asked_at is not None

        # Should no longer appear in pending
        pending = store.list_pending_cues("test_user")
        assert len(pending) == 0

    def test_dismiss_cue(self, tmp_path: Path) -> None:
        """Test dismissing a cue."""
        from rex.cue_store import CueStore

        store = CueStore(storage_path=tmp_path / "cues.json")
        cue = store.add_cue(
            user_id="test_user",
            source_type="manual",
            source_id="manual-1",
            title="Check in",
            prompt="How are you doing?",
        )

        assert store.dismiss(cue.cue_id)

        updated = store.get_cue(cue.cue_id)
        assert updated is not None
        assert updated.status == "dismissed"
        assert updated.dismissed_at is not None

    def test_prune_expired_cues(self, tmp_path: Path) -> None:
        """Test pruning expired cues."""
        from rex.cue_store import CueStore

        store = CueStore(storage_path=tmp_path / "cues.json")
        now = _utc_now()

        # Add an expired cue
        store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="old-event",
            title="Old event",
            prompt="How did old event go?",
            expires_at=now - timedelta(hours=1),
        )

        # Add a non-expired cue
        cue2 = store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="new-event",
            title="New event",
            prompt="How did new event go?",
            expires_at=now + timedelta(hours=24),
        )

        assert len(store) == 2

        pruned = store.prune_expired(now)
        assert pruned == 1
        assert len(store) == 1

        # The non-expired cue should still exist
        assert store.get_cue(cue2.cue_id) is not None

    def test_no_duplicate_cues_for_same_source(self, tmp_path: Path) -> None:
        """Test that duplicate cues are not created for the same source."""
        from rex.cue_store import CueStore

        store = CueStore(storage_path=tmp_path / "cues.json")

        cue1 = store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-123",
            title="Event",
            prompt="How did it go?",
        )

        # Try to add a duplicate
        cue2 = store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-123",
            title="Event (duplicate)",
            prompt="How did it go again?",
        )

        # Should return the existing cue
        assert cue2.cue_id == cue1.cue_id
        assert len(store) == 1

    def test_cue_persistence(self, tmp_path: Path) -> None:
        """Test that cues persist across store instances."""
        from rex.cue_store import CueStore

        storage = tmp_path / "cues.json"

        # Create a cue
        store1 = CueStore(storage_path=storage)
        cue = store1.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Event",
            prompt="How did it go?",
        )

        # Load in a new instance
        store2 = CueStore(storage_path=storage)
        loaded = store2.get_cue(cue.cue_id)
        assert loaded is not None
        assert loaded.title == "Event"
        assert loaded.status == "pending"

    def test_list_cues_with_window_hours(self, tmp_path: Path) -> None:
        """Test listing cues with time window filtering."""
        from rex.cue_store import CueStore

        store = CueStore(storage_path=tmp_path / "cues.json")
        now = _utc_now()

        # Add a cue created 100 hours ago
        old_cue = store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="old-event",
            title="Old event",
            prompt="How did it go?",
        )
        # Manually set created_at to simulate old cue
        old_cue.created_at = now - timedelta(hours=100)
        store._save()

        # Add a recent cue
        recent_cue = store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="recent-event",
            title="Recent event",
            prompt="How did it go?",
        )

        # With 72 hour window, only recent cue should appear
        pending = store.list_pending_cues("test_user", now=now, window_hours=72)
        assert len(pending) == 1
        assert pending[0].cue_id == recent_cue.cue_id

        # With larger window, both should appear
        pending_all = store.list_pending_cues("test_user", now=now, window_hours=200)
        assert len(pending_all) == 2


# =============================================================================
# Calendar Cue Generation Tests
# =============================================================================


class TestCalendarCueGeneration:
    """Tests for calendar -> cue generation."""

    def test_generate_cue_for_past_event(self, tmp_path: Path) -> None:
        """Test that cues are created for past events."""
        from rex.calendar_service import CalendarEvent, CalendarService
        from rex.cue_store import CueStore, set_cue_store

        # Set up a cue store
        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        set_cue_store(cue_store)

        # Create a calendar service with a past event
        now = _utc_now()
        past_event = CalendarEvent(
            event_id="past-event-1",
            title="Team meeting",
            start_time=now - timedelta(hours=3),
            end_time=now - timedelta(hours=2),
        )

        service = CalendarService(mock_events=[past_event])
        service.connect()

        # Generate cues
        created = service.generate_followup_cues("test_user", now=now)

        assert created == 1

        # Verify cue was created
        cues = cue_store.list_pending_cues("test_user")
        assert len(cues) == 1
        assert cues[0].source_id == "past-event-1"
        assert "Team meeting" in cues[0].prompt

    def test_no_duplicate_cue_on_second_run(self, tmp_path: Path) -> None:
        """Test that running generate_followup_cues twice doesn't create duplicates."""
        from rex.calendar_service import CalendarEvent, CalendarService
        from rex.cue_store import CueStore, set_cue_store

        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        set_cue_store(cue_store)

        now = _utc_now()
        past_event = CalendarEvent(
            event_id="event-1",
            title="Event",
            start_time=now - timedelta(hours=3),
            end_time=now - timedelta(hours=2),
        )

        service = CalendarService(mock_events=[past_event])
        service.connect()

        # First run
        created1 = service.generate_followup_cues("test_user", now=now)
        assert created1 == 1

        # Second run
        created2 = service.generate_followup_cues("test_user", now=now)
        assert created2 == 0

        # Still only one cue
        assert len(cue_store) == 1

    def test_skip_holiday_events(self, tmp_path: Path) -> None:
        """Test that holiday/all-day events are skipped."""
        from rex.calendar_service import CalendarEvent, CalendarService
        from rex.cue_store import CueStore, set_cue_store

        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        set_cue_store(cue_store)

        now = _utc_now()
        holiday = CalendarEvent(
            event_id="holiday-1",
            title="Company Holiday",
            start_time=now - timedelta(days=1),
            end_time=now - timedelta(hours=1),
            all_day=True,
        )

        service = CalendarService(mock_events=[holiday])
        service.connect()

        created = service.generate_followup_cues("test_user", now=now)

        assert created == 0

    def test_skip_no_followup_events(self, tmp_path: Path) -> None:
        """Test that events marked no-followup are skipped."""
        from rex.calendar_service import CalendarEvent, CalendarService
        from rex.cue_store import CueStore, set_cue_store

        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        set_cue_store(cue_store)

        now = _utc_now()
        event = CalendarEvent(
            event_id="event-1",
            title="Quick sync [no-followup]",
            start_time=now - timedelta(hours=2),
            end_time=now - timedelta(hours=1),
        )

        service = CalendarService(mock_events=[event])
        service.connect()

        created = service.generate_followup_cues("test_user", now=now)

        assert created == 0


# =============================================================================
# Reminder Tests
# =============================================================================


class TestReminderService:
    """Tests for the reminder service."""

    def test_create_reminder(self, tmp_path: Path) -> None:
        """Test creating a reminder."""
        from rex.reminder_service import ReminderService

        service = ReminderService(storage_path=tmp_path / "reminders.json")

        remind_at = _utc_now() + timedelta(hours=2)
        reminder = service.create_reminder(
            user_id="test_user",
            title="Call mom",
            remind_at=remind_at,
        )

        assert reminder.reminder_id.startswith("rem_")
        assert reminder.title == "Call mom"
        assert reminder.status == "pending"
        assert reminder.remind_at == remind_at

    def test_list_reminders(self, tmp_path: Path) -> None:
        """Test listing reminders."""
        from rex.reminder_service import ReminderService

        service = ReminderService(storage_path=tmp_path / "reminders.json")

        now = _utc_now()
        service.create_reminder(
            user_id="test_user",
            title="Reminder 1",
            remind_at=now + timedelta(hours=1),
        )
        service.create_reminder(
            user_id="test_user",
            title="Reminder 2",
            remind_at=now + timedelta(hours=2),
        )

        reminders = service.list_reminders(user_id="test_user")
        assert len(reminders) == 2

    def test_mark_reminder_done(self, tmp_path: Path) -> None:
        """Test marking a reminder as done."""
        from rex.reminder_service import ReminderService

        service = ReminderService(storage_path=tmp_path / "reminders.json")

        reminder = service.create_reminder(
            user_id="test_user",
            title="Task",
            remind_at=_utc_now() + timedelta(hours=1),
        )

        assert service.mark_done(reminder.reminder_id)

        updated = service.get_reminder(reminder.reminder_id)
        assert updated is not None
        assert updated.status == "done"
        assert updated.done_at is not None

    def test_fire_due_reminder_creates_notification(self, tmp_path: Path, monkeypatch) -> None:
        """Test that firing a reminder sends a notification."""
        from rex.reminder_service import ReminderService

        service = ReminderService(storage_path=tmp_path / "reminders.json")
        notification_sent = []

        # Mock the notification
        def mock_send(notification):
            notification_sent.append(notification)

        class MockNotifier:
            def send(self, notification):
                mock_send(notification)

        monkeypatch.setattr("rex.notification.get_notifier", lambda: MockNotifier())

        now = _utc_now()
        service.create_reminder(
            user_id="test_user",
            title="Due reminder",
            remind_at=now - timedelta(minutes=1),  # Already due
        )

        fired = service.fire_due_reminders(now=now)

        assert len(fired) == 1
        assert len(notification_sent) == 1
        assert "Due reminder" in notification_sent[0].body

    def test_fire_reminder_with_followup_creates_cue(self, tmp_path: Path, monkeypatch) -> None:
        """Test that firing a reminder with followup_enabled creates a cue."""
        from rex.cue_store import CueStore, set_cue_store
        from rex.reminder_service import ReminderService

        # Set up cue store
        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        set_cue_store(cue_store)

        service = ReminderService(storage_path=tmp_path / "reminders.json")

        # Mock notification to avoid import issues
        class MockNotifier:
            def send(self, notification):
                pass

        monkeypatch.setattr("rex.notification.get_notifier", lambda: MockNotifier())

        now = _utc_now()
        service.create_reminder(
            user_id="test_user",
            title="Task to follow up on",
            remind_at=now - timedelta(minutes=1),
            followup_enabled=True,
        )

        fired = service.fire_due_reminders(now=now)
        assert len(fired) == 1

        # Check that a cue was created
        cues = cue_store.list_pending_cues("test_user")
        assert len(cues) == 1
        assert cues[0].source_type == "reminder"
        assert "Task to follow up on" in cues[0].prompt


# =============================================================================
# Followup Engine Tests
# =============================================================================


class TestFollowupEngine:
    """Tests for the follow-up engine."""

    def test_get_followup_prompt(self, tmp_path: Path, monkeypatch) -> None:
        """Test getting a follow-up prompt."""
        from rex.cue_store import CueStore, set_cue_store
        from rex.followup_engine import FollowupEngine

        # Set up cue store with a pending cue
        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        cue_store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Meeting",
            prompt="How did the meeting go?",
        )
        set_cue_store(cue_store)

        # Mock config
        def mock_load_config():
            return {
                "conversation": {
                    "followups": {
                        "enabled": True,
                        "max_per_session": 1,
                        "lookback_hours": 72,
                        "expire_hours": 168,
                    }
                }
            }

        monkeypatch.setattr("rex.followup_engine.load_config", mock_load_config)

        engine = FollowupEngine()
        prompt = engine.get_followup_prompt("test_user")

        assert prompt == "How did the meeting go?"

    def test_mark_cue_asked_increments_session_count(self, tmp_path: Path, monkeypatch) -> None:
        """Test that marking a cue as asked increments the session count."""
        from rex.cue_store import CueStore, set_cue_store
        from rex.followup_engine import FollowupEngine

        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        cue_store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Event",
            prompt="How did it go?",
        )
        set_cue_store(cue_store)

        def mock_load_config():
            return {
                "conversation": {
                    "followups": {
                        "enabled": True,
                        "max_per_session": 1,
                        "lookback_hours": 72,
                        "expire_hours": 168,
                    }
                }
            }

        monkeypatch.setattr("rex.followup_engine.load_config", mock_load_config)

        engine = FollowupEngine()
        engine.start_session("test_user")

        # Get the prompt (sets current cue)
        prompt = engine.get_followup_prompt("test_user")
        assert prompt is not None

        assert engine.get_session_cues_asked("test_user") == 0

        # Mark as asked
        engine.mark_current_cue_asked("test_user")

        assert engine.get_session_cues_asked("test_user") == 1

    def test_rate_limit_prevents_multiple_cues(self, tmp_path: Path, monkeypatch) -> None:
        """Test that rate limiting prevents asking multiple cues per session."""
        from rex.cue_store import CueStore, set_cue_store
        from rex.followup_engine import FollowupEngine

        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        cue_store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Event 1",
            prompt="How did event 1 go?",
        )
        cue_store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-2",
            title="Event 2",
            prompt="How did event 2 go?",
        )
        set_cue_store(cue_store)

        def mock_load_config():
            return {
                "conversation": {
                    "followups": {
                        "enabled": True,
                        "max_per_session": 1,
                        "lookback_hours": 72,
                        "expire_hours": 168,
                    }
                }
            }

        monkeypatch.setattr("rex.followup_engine.load_config", mock_load_config)

        engine = FollowupEngine()
        engine.start_session("test_user")

        # Get first prompt
        prompt1 = engine.get_followup_prompt("test_user")
        assert prompt1 is not None
        engine.mark_current_cue_asked("test_user")

        # Try to get second prompt - should be rate limited
        prompt2 = engine.get_followup_prompt("test_user")
        assert prompt2 is None

    def test_disabled_followups(self, tmp_path: Path, monkeypatch) -> None:
        """Test that disabled followups return no prompts."""
        from rex.cue_store import CueStore, set_cue_store
        from rex.followup_engine import FollowupEngine

        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        cue_store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Event",
            prompt="How did it go?",
        )
        set_cue_store(cue_store)

        def mock_load_config():
            return {
                "conversation": {
                    "followups": {
                        "enabled": False,
                    }
                }
            }

        monkeypatch.setattr("rex.followup_engine.load_config", mock_load_config)

        engine = FollowupEngine()
        prompt = engine.get_followup_prompt("test_user")

        assert prompt is None


# =============================================================================
# Conversation Integration Tests
# =============================================================================


class TestConversationIntegration:
    """Tests for conversation integration with followup cues."""

    def test_assistant_prompt_includes_followup(self, tmp_path: Path, monkeypatch) -> None:
        """Test that assistant prompt includes follow-up cue."""
        from rex.cue_store import CueStore, set_cue_store
        from rex.followup_engine import FollowupEngine, set_followup_engine

        # Set up cue store with a pending cue
        cue_store = CueStore(storage_path=tmp_path / "cues.json")
        cue_store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Meeting",
            prompt="How did the meeting go?",
        )
        set_cue_store(cue_store)

        def mock_load_config():
            return {
                "conversation": {
                    "followups": {
                        "enabled": True,
                        "max_per_session": 1,
                        "lookback_hours": 72,
                        "expire_hours": 168,
                    }
                },
                "runtime": {"user_id": "test_user"},
            }

        monkeypatch.setattr("rex.followup_engine.load_config", mock_load_config)

        engine = FollowupEngine()
        set_followup_engine(engine)
        engine.start_session("test_user")

        # Test inject_followup_into_prompt
        base_prompt = "You are a helpful assistant."
        modified = engine.inject_followup_into_prompt("test_user", base_prompt)

        assert "How did the meeting go?" in modified
        assert "natural" in modified.lower() or "small talk" in modified.lower()


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLICommands:
    """Tests for CLI commands."""

    def test_reminders_list_command(self, tmp_path: Path, capsys, monkeypatch) -> None:
        """Test rex reminders list command."""
        import argparse

        from rex.cli import cmd_reminders
        from rex.reminder_service import ReminderService, set_reminder_service

        service = ReminderService(storage_path=tmp_path / "reminders.json")
        set_reminder_service(service)

        service.create_reminder(
            user_id="test_user",
            title="Test reminder",
            remind_at=_utc_now() + timedelta(hours=1),
        )

        args = argparse.Namespace(reminders_command="list", status=None)
        result = cmd_reminders(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Test reminder" in captured.out

    def test_cues_list_command(self, tmp_path: Path, capsys) -> None:
        """Test rex cues list command."""
        import argparse

        from rex.cli import cmd_cues
        from rex.cue_store import CueStore, set_cue_store

        store = CueStore(storage_path=tmp_path / "cues.json")
        set_cue_store(store)

        store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Test event",
            prompt="How did it go?",
        )

        args = argparse.Namespace(cues_command="list", status=None)
        result = cmd_cues(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Test event" in captured.out

    def test_cues_dismiss_command(self, tmp_path: Path, capsys) -> None:
        """Test rex cues dismiss command."""
        import argparse

        from rex.cli import cmd_cues
        from rex.cue_store import CueStore, set_cue_store

        store = CueStore(storage_path=tmp_path / "cues.json")
        set_cue_store(store)

        cue = store.add_cue(
            user_id="test_user",
            source_type="calendar",
            source_id="event-1",
            title="Test event",
            prompt="How did it go?",
        )

        args = argparse.Namespace(cues_command="dismiss", cue_id=cue.cue_id)
        result = cmd_cues(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Dismissed" in captured.out

        # Verify dismissed
        updated = store.get_cue(cue.cue_id)
        assert updated is not None
        assert updated.status == "dismissed"
