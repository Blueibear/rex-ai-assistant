"""Tests for scheduler, email, and calendar CLI commands."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from rex.cli import cmd_calendar, cmd_email, cmd_scheduler


@pytest.fixture
def mock_scheduler():
    """Mock scheduler instance."""
    with patch("rex.cli.get_scheduler") as mock:
        scheduler = MagicMock()
        mock.return_value = scheduler
        yield scheduler


@pytest.fixture
def mock_email_service():
    """Mock email service instance."""
    with patch("rex.cli.get_email_service") as mock:
        service = MagicMock()
        service.connected = False
        mock.return_value = service
        yield service


@pytest.fixture
def mock_calendar_service():
    """Mock calendar service instance."""
    with patch("rex.cli.get_calendar_service") as mock:
        service = MagicMock()
        service.connected = False
        mock.return_value = service
        yield service


class TestSchedulerCLI:
    """Tests for scheduler CLI commands."""

    def test_scheduler_list_empty(self, mock_scheduler, capsys):
        """Test listing scheduler jobs when none exist."""
        mock_scheduler.list_jobs.return_value = []

        args = MagicMock(scheduler_command="list", verbose=False)
        result = cmd_scheduler(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No scheduled jobs" in captured.out

    def test_scheduler_list_with_jobs(self, mock_scheduler, capsys):
        """Test listing scheduler jobs."""
        from rex.scheduler import ScheduledJob

        job1 = ScheduledJob(
            job_id="job1",
            name="Job 1",
            schedule="interval:600",
            enabled=True,
            next_run=datetime.now() + timedelta(minutes=10),
            run_count=5,
        )
        job2 = ScheduledJob(
            job_id="job2",
            name="Job 2",
            schedule="interval:3600",
            enabled=False,
            next_run=datetime.now() + timedelta(hours=1),
            run_count=2,
            max_runs=10,
        )

        mock_scheduler.list_jobs.return_value = [job1, job2]

        args = MagicMock(scheduler_command="list", verbose=False)
        result = cmd_scheduler(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "job1" in captured.out
        assert "Job 1" in captured.out
        assert "job2" in captured.out
        assert "Job 2" in captured.out
        assert "enabled" in captured.out
        assert "disabled" in captured.out

    def test_scheduler_list_verbose(self, mock_scheduler, capsys):
        """Test listing scheduler jobs with verbose output."""
        from rex.scheduler import ScheduledJob

        job = ScheduledJob(
            job_id="job1",
            name="Job 1",
            schedule="interval:600",
            enabled=True,
            next_run=datetime.now(),
            run_count=0,
            callback_name="test_callback",
            workflow_id="workflow-123",
        )

        mock_scheduler.list_jobs.return_value = [job]

        args = MagicMock(scheduler_command="list", verbose=True)
        result = cmd_scheduler(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "test_callback" in captured.out
        assert "workflow-123" in captured.out

    def test_scheduler_run_success(self, mock_scheduler, capsys):
        """Test running a job successfully."""
        mock_scheduler.run_job.return_value = True

        with patch("rex.cli.initialize_scheduler_system"):
            args = MagicMock(scheduler_command="run", job_id="test-job")
            result = cmd_scheduler(args)

        assert result == 0
        mock_scheduler.run_job.assert_called_once_with("test-job", force=True)
        captured = capsys.readouterr()
        assert "executed successfully" in captured.out

    def test_scheduler_run_failure(self, mock_scheduler, capsys):
        """Test running a job that fails."""
        mock_scheduler.run_job.return_value = False

        with patch("rex.cli.initialize_scheduler_system"):
            args = MagicMock(scheduler_command="run", job_id="test-job")
            result = cmd_scheduler(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to run" in captured.out

    def test_scheduler_init(self, capsys):
        """Test initializing scheduler system."""
        with patch("rex.cli.initialize_scheduler_system") as mock_init:
            args = MagicMock(scheduler_command="init")
            result = cmd_scheduler(args)

        assert result == 0
        mock_init.assert_called_once_with(start_scheduler=False)
        captured = capsys.readouterr()
        assert "initialized" in captured.out

    def test_scheduler_unknown_command(self, mock_scheduler, capsys):
        """Test unknown scheduler subcommand."""
        args = MagicMock(scheduler_command="unknown")
        result = cmd_scheduler(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown scheduler subcommand" in captured.out


class TestEmailCLI:
    """Tests for email CLI commands."""

    def test_email_unread_empty(self, mock_email_service, capsys):
        """Test fetching unread emails when none exist."""
        mock_email_service.connect.return_value = True
        mock_email_service.fetch_unread.return_value = []

        args = MagicMock(email_command="unread", limit=10, verbose=False)
        result = cmd_email(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No unread emails" in captured.out

    def test_email_unread_with_emails(self, mock_email_service, capsys):
        """Test fetching unread emails."""
        from rex.email_service import EmailSummary

        email1 = EmailSummary(
            id="email-1",
            from_addr="test@example.com",
            subject="Test Email",
            snippet="This is a test",
            received_at=datetime.now(),
            importance_score=0.8,
        )

        mock_email_service.connect.return_value = True
        mock_email_service.fetch_unread.return_value = [email1]
        mock_email_service.categorize.return_value = "important"

        args = MagicMock(email_command="unread", limit=10, verbose=False)
        result = cmd_email(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "email-1" in captured.out
        assert "Test Email" in captured.out
        assert "test@example.com" in captured.out
        assert "important" in captured.out

    def test_email_unread_verbose(self, mock_email_service, capsys):
        """Test fetching unread emails with verbose output."""
        from rex.email_service import EmailSummary

        email = EmailSummary(
            id="email-1",
            from_addr="test@example.com",
            subject="Test",
            snippet="Test snippet text",
            received_at=datetime.now(),
            importance_score=0.9,
        )

        mock_email_service.connect.return_value = True
        mock_email_service.fetch_unread.return_value = [email]
        mock_email_service.categorize.return_value = "general"

        args = MagicMock(email_command="unread", limit=10, verbose=True)
        result = cmd_email(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "0.90" in captured.out  # Importance score
        assert "Test snippet text" in captured.out

    def test_email_unread_connection_failure(self, mock_email_service, capsys):
        """Test email command when connection fails."""
        mock_email_service.connect.return_value = False

        args = MagicMock(email_command="unread", limit=10, verbose=False)
        result = cmd_email(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to connect" in captured.out

    def test_email_unread_with_limit(self, mock_email_service):
        """Test fetching unread emails with custom limit."""
        mock_email_service.connect.return_value = True
        mock_email_service.fetch_unread.return_value = []

        args = MagicMock(email_command="unread", limit=5, verbose=False)
        cmd_email(args)

        mock_email_service.fetch_unread.assert_called_once_with(limit=5)

    def test_email_unknown_command(self, mock_email_service, capsys):
        """Test unknown email subcommand."""
        args = MagicMock(email_command="unknown")
        result = cmd_email(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown email subcommand" in captured.out

    def test_email_resolves_user_context(self, mock_email_service):
        """Email command resolves active user context (including --user override)."""
        mock_email_service.connect.return_value = True
        mock_email_service.fetch_unread.return_value = []

        with patch("rex.cli._resolve_cli_user", return_value="cole") as mock_resolve:
            args = MagicMock(email_command="unread", limit=10, verbose=False, user="cole")
            result = cmd_email(args)

        assert result == 0
        mock_resolve.assert_called_once_with(args)


class TestCalendarCLI:
    """Tests for calendar CLI commands."""

    def test_calendar_upcoming_empty(self, mock_calendar_service, capsys):
        """Test fetching upcoming events when none exist."""
        mock_calendar_service.connect.return_value = True
        mock_calendar_service.get_upcoming_events.return_value = []

        args = MagicMock(calendar_command="upcoming", days=7, conflicts=False, verbose=False)
        result = cmd_calendar(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "No upcoming events" in captured.out

    def test_calendar_upcoming_with_events(self, mock_calendar_service, capsys):
        """Test fetching upcoming events."""
        from rex.calendar_service import CalendarEvent

        start = datetime.now() + timedelta(days=1)
        end = start + timedelta(hours=1)

        event = CalendarEvent(
            id="event-1",
            title="Team Meeting",
            start_time=start,
            end_time=end,
            attendees=["alice@example.com"],
            location="Office",
            all_day=False,
        )

        mock_calendar_service.connect.return_value = True
        mock_calendar_service.get_upcoming_events.return_value = [event]

        args = MagicMock(calendar_command="upcoming", days=7, conflicts=False, verbose=False)
        result = cmd_calendar(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "event-1" in captured.out
        assert "Team Meeting" in captured.out
        assert "Office" in captured.out
        assert "alice@example.com" in captured.out

    def test_calendar_upcoming_verbose(self, mock_calendar_service, capsys):
        """Test fetching upcoming events with verbose output."""
        from rex.calendar_service import CalendarEvent

        start = datetime.now() + timedelta(days=1)
        end = start + timedelta(hours=1)

        event = CalendarEvent(
            id="event-1",
            title="Meeting",
            start_time=start,
            end_time=end,
            description="Important meeting",
            all_day=False,
        )

        mock_calendar_service.connect.return_value = True
        mock_calendar_service.get_upcoming_events.return_value = [event]

        args = MagicMock(calendar_command="upcoming", days=7, conflicts=False, verbose=True)
        result = cmd_calendar(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Important meeting" in captured.out

    def test_calendar_upcoming_all_day(self, mock_calendar_service, capsys):
        """Test fetching all-day events."""
        from rex.calendar_service import CalendarEvent

        start = datetime.now() + timedelta(days=1)

        event = CalendarEvent(
            id="event-1",
            title="Conference",
            start_time=start,
            end_time=start + timedelta(days=1),
            all_day=True,
        )

        mock_calendar_service.connect.return_value = True
        mock_calendar_service.get_upcoming_events.return_value = [event]

        args = MagicMock(calendar_command="upcoming", days=7, conflicts=False, verbose=False)
        result = cmd_calendar(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "All day" in captured.out

    def test_calendar_upcoming_with_conflicts(self, mock_calendar_service, capsys):
        """Test fetching upcoming events with conflict detection."""
        from rex.calendar_service import CalendarEvent

        start = datetime.now() + timedelta(days=1)

        event1 = CalendarEvent(
            id="event-1",
            title="Meeting 1",
            start_time=start,
            end_time=start + timedelta(hours=2),
            all_day=False,
        )

        event2 = CalendarEvent(
            id="event-2",
            title="Meeting 2",
            start_time=start + timedelta(hours=1),
            end_time=start + timedelta(hours=3),
            all_day=False,
        )

        mock_calendar_service.connect.return_value = True
        mock_calendar_service.get_upcoming_events.return_value = [event1, event2]
        mock_calendar_service.find_conflicts.return_value = [(event1, event2)]

        args = MagicMock(calendar_command="upcoming", days=7, conflicts=True, verbose=False)
        result = cmd_calendar(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "Conflicts Detected" in captured.out
        assert "Meeting 1" in captured.out
        assert "Meeting 2" in captured.out

    def test_calendar_upcoming_custom_days(self, mock_calendar_service):
        """Test fetching upcoming events with custom days."""
        mock_calendar_service.connect.return_value = True
        mock_calendar_service.get_upcoming_events.return_value = []

        args = MagicMock(calendar_command="upcoming", days=14, conflicts=False, verbose=False)
        cmd_calendar(args)

        mock_calendar_service.get_upcoming_events.assert_called_once_with(days=14)

    def test_calendar_upcoming_connection_failure(self, mock_calendar_service, capsys):
        """Test calendar command when connection fails."""
        mock_calendar_service.connect.return_value = False

        args = MagicMock(calendar_command="upcoming", days=7, conflicts=False, verbose=False)
        result = cmd_calendar(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Failed to connect" in captured.out

    def test_calendar_unknown_command(self, mock_calendar_service, capsys):
        """Test unknown calendar subcommand."""
        args = MagicMock(calendar_command="unknown")
        result = cmd_calendar(args)

        assert result == 1
        captured = capsys.readouterr()
        assert "Unknown calendar subcommand" in captured.out

    def test_calendar_resolves_user_context(self, mock_calendar_service):
        """Calendar command resolves active user context (including --user override)."""
        mock_calendar_service.connect.return_value = True
        mock_calendar_service.get_upcoming_events.return_value = []

        with patch("rex.cli._resolve_cli_user", return_value="alex") as mock_resolve:
            args = MagicMock(
                calendar_command="upcoming", days=7, conflicts=False, verbose=False, user="alex"
            )
            result = cmd_calendar(args)

        assert result == 0
        mock_resolve.assert_called_once_with(args)
