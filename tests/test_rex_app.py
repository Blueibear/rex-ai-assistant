"""Tests for rex.app module (runtime entrypoint with service supervision)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def test_import():
    """rex.app imports without error."""
    import rex.app  # noqa: F401


def test_main_help_exits_zero():
    """main(['--help']) prints help and exits with code 0."""
    from rex.app import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_main_returns_one_on_initialization_error():
    """main() returns 1 when initialize_services raises an exception."""
    from rex.app import main

    with patch("rex.app.initialize_services", side_effect=RuntimeError("boom")):
        with patch("rex.app.configure_logging"):
            with patch("rex.app._setup_signal_handlers"):
                result = main(["--services", "scheduler"])

    assert result == 1


def test_main_keyboard_interrupt_returns_zero():
    """main() returns 0 when interrupted by the user (KeyboardInterrupt)."""
    from rex.app import main

    mock_supervisor = MagicMock()
    mock_services = MagicMock()
    mock_services.event_bus.get_metrics.return_value = {}

    with patch("rex.app.initialize_services", return_value=mock_services):
        with patch("rex.app.ServiceSupervisor", return_value=mock_supervisor):
            with patch("rex.app.configure_logging"):
                with patch("rex.app._setup_signal_handlers"):
                    with patch("rex.app.time.sleep", side_effect=KeyboardInterrupt):
                        result = main(["--services", "event_bus"])

    assert result == 0
    mock_supervisor.stop.assert_called_once()


def test_main_registers_event_bus_service():
    """main() registers the event_bus service with the supervisor."""
    from rex.app import main

    mock_supervisor = MagicMock()
    mock_services = MagicMock()
    mock_services.event_bus.get_metrics.return_value = {}

    with patch("rex.app.initialize_services", return_value=mock_services):
        with patch("rex.app.ServiceSupervisor", return_value=mock_supervisor):
            with patch("rex.app.configure_logging"):
                with patch("rex.app._setup_signal_handlers"):
                    with patch("rex.app.time.sleep", side_effect=KeyboardInterrupt):
                        main(["--services", "event_bus"])

    registered_names = [
        c[1].get("name") or c[0][0] for c in mock_supervisor.register_service.call_args_list
    ]
    assert "event_bus" in registered_names


def test_setup_signal_handlers_registers_sigterm_and_sigint():
    """_setup_signal_handlers registers handlers for SIGTERM and SIGINT."""
    import signal

    from rex.app import _setup_signal_handlers

    mock_supervisor = MagicMock()
    registered = {}

    def fake_signal(signum, handler):
        registered[signum] = handler

    with patch("rex.app.signal.signal", side_effect=fake_signal):
        _setup_signal_handlers(mock_supervisor)

    assert signal.SIGTERM in registered
    assert signal.SIGINT in registered
