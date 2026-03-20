"""Tests for rex.integrations module (scheduler integration setup).

Functions live in rex.integrations._setup and are re-exported from the
rex.integrations package __init__.  Patches must target the _setup module
where the names are bound.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

_SETUP = "rex.integrations._setup"


def test_import():
    """rex.integrations imports without error."""
    import rex.integrations  # noqa: F401


def test_setup_email_job_registers_and_returns_job():
    """setup_email_job registers a callback and adds a job to the scheduler."""
    from rex.integrations import setup_email_job

    mock_scheduler = MagicMock()
    mock_job = MagicMock()
    mock_scheduler.add_job.return_value = mock_job

    with patch(f"{_SETUP}.get_scheduler", return_value=mock_scheduler):
        with patch(f"{_SETUP}.get_event_bus", return_value=MagicMock()):
            result = setup_email_job()

    mock_scheduler.register_callback.assert_called_once()
    assert mock_scheduler.register_callback.call_args[0][0] == "check_email"
    mock_scheduler.add_job.assert_called_once()
    assert result is mock_job


def test_setup_calendar_job_registers_and_returns_job():
    """setup_calendar_job registers a callback and adds a job to the scheduler."""
    from rex.integrations import setup_calendar_job

    mock_scheduler = MagicMock()
    mock_job = MagicMock()
    mock_scheduler.add_job.return_value = mock_job

    with patch(f"{_SETUP}.get_scheduler", return_value=mock_scheduler):
        with patch(f"{_SETUP}.get_event_bus", return_value=MagicMock()):
            result = setup_calendar_job()

    mock_scheduler.register_callback.assert_called_once()
    assert mock_scheduler.register_callback.call_args[0][0] == "sync_calendar"
    mock_scheduler.add_job.assert_called_once()
    assert result is mock_job


def test_setup_default_event_handlers_subscribes_to_events():
    """setup_default_event_handlers subscribes to email.unread and calendar.update."""
    from rex.integrations import setup_default_event_handlers

    mock_event_bus = MagicMock()

    with patch(f"{_SETUP}.get_event_bus", return_value=mock_event_bus):
        setup_default_event_handlers()

    subscribed_events = [c[0][0] for c in mock_event_bus.subscribe.call_args_list]
    assert "email.unread" in subscribed_events
    assert "calendar.update" in subscribed_events


def test_shutdown_scheduler_system_stops_scheduler():
    """shutdown_scheduler_system calls scheduler.stop()."""
    from rex.integrations import shutdown_scheduler_system

    mock_scheduler = MagicMock()

    with patch(f"{_SETUP}.get_scheduler", return_value=mock_scheduler):
        shutdown_scheduler_system()

    mock_scheduler.stop.assert_called_once()


def test_initialize_scheduler_system_no_start():
    """initialize_scheduler_system completes without starting the scheduler."""
    from rex.integrations import initialize_scheduler_system

    mock_scheduler = MagicMock()
    mock_scheduler.add_job.return_value = MagicMock()

    with patch(f"{_SETUP}.get_scheduler", return_value=mock_scheduler):
        with patch(f"{_SETUP}.get_event_bus", return_value=MagicMock()):
            initialize_scheduler_system(start_scheduler=False)

    mock_scheduler.start.assert_not_called()


def test_initialize_scheduler_system_with_start():
    """initialize_scheduler_system starts the scheduler when requested."""
    from rex.integrations import initialize_scheduler_system

    mock_scheduler = MagicMock()
    mock_scheduler.add_job.return_value = MagicMock()

    with patch(f"{_SETUP}.get_scheduler", return_value=mock_scheduler):
        with patch(f"{_SETUP}.get_event_bus", return_value=MagicMock()):
            initialize_scheduler_system(start_scheduler=True)

    mock_scheduler.start.assert_called_once()
