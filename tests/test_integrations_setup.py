"""Smoke tests for rex.integrations._setup (scheduler/email/calendar helpers)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_import():
    """Module imports without error."""
    from rex.integrations import _setup

    assert _setup is not None


def _make_mock_job():
    job = MagicMock()
    job.job_id = "test_job"
    return job


def _make_mock_scheduler():
    sched = MagicMock()
    sched.add_job.return_value = _make_mock_job()
    return sched


def test_setup_email_job():
    """setup_email_job registers a callback and returns a ScheduledJob."""
    from rex.integrations._setup import setup_email_job

    mock_scheduler = _make_mock_scheduler()
    mock_event_bus = MagicMock()

    with (
        patch("rex.integrations._setup.get_scheduler", return_value=mock_scheduler),
        patch("rex.integrations._setup.get_event_bus", return_value=mock_event_bus),
    ):
        job = setup_email_job()

    assert job is not None
    mock_scheduler.register_callback.assert_called_once_with(
        "check_email", mock_scheduler.register_callback.call_args[0][1]
    )
    mock_scheduler.add_job.assert_called_once()
    call_kwargs = mock_scheduler.add_job.call_args[1]
    assert call_kwargs["job_id"] == "email_check"


def test_setup_calendar_job():
    """setup_calendar_job registers a callback and returns a ScheduledJob."""
    from rex.integrations._setup import setup_calendar_job

    mock_scheduler = _make_mock_scheduler()
    mock_event_bus = MagicMock()

    with (
        patch("rex.integrations._setup.get_scheduler", return_value=mock_scheduler),
        patch("rex.integrations._setup.get_event_bus", return_value=mock_event_bus),
    ):
        job = setup_calendar_job()

    assert job is not None
    mock_scheduler.add_job.assert_called_once()
    call_kwargs = mock_scheduler.add_job.call_args[1]
    assert call_kwargs["job_id"] == "calendar_sync"


def test_setup_default_event_handlers():
    """setup_default_event_handlers subscribes to email.unread and calendar.update via EventBridge."""
    from rex.integrations._setup import setup_default_event_handlers

    mock_bridge = MagicMock()

    with patch("rex.openclaw.event_bridge.EventBridge", return_value=mock_bridge):
        setup_default_event_handlers()

    assert mock_bridge.subscribe.call_count == 2
    subscribed_events = {call[0][0] for call in mock_bridge.subscribe.call_args_list}
    assert "email.unread" in subscribed_events
    assert "calendar.update" in subscribed_events


def test_initialize_scheduler_system_no_start():
    """initialize_scheduler_system runs without raising even if sub-setups fail."""
    from rex.integrations._setup import initialize_scheduler_system

    mock_scheduler = _make_mock_scheduler()
    mock_event_bus = MagicMock()
    mock_bridge = MagicMock()

    with (
        patch("rex.integrations._setup.get_scheduler", return_value=mock_scheduler),
        patch("rex.integrations._setup.get_event_bus", return_value=mock_event_bus),
        patch("rex.openclaw.event_bridge.EventBridge", return_value=mock_bridge),
        patch("rex.integrations._setup.get_email_service", side_effect=RuntimeError("no email")),
        patch("rex.integrations._setup.get_calendar_service", side_effect=RuntimeError("no cal")),
        patch("rex.integrations._setup._try_register_retention_jobs"),
    ):
        initialize_scheduler_system(start_scheduler=False)

    # Even with failures, event handlers should be registered via EventBridge
    assert mock_bridge.subscribe.call_count >= 2


def test_shutdown_scheduler_system():
    """shutdown_scheduler_system calls scheduler.stop()."""
    from rex.integrations._setup import shutdown_scheduler_system

    mock_scheduler = _make_mock_scheduler()

    with patch("rex.integrations._setup.get_scheduler", return_value=mock_scheduler):
        shutdown_scheduler_system()

    mock_scheduler.stop.assert_called_once()
