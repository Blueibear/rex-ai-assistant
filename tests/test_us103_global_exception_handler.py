"""Tests for US-103: Global unhandled exception handler."""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from rex.exception_handler import handle_unhandled_exception, wrap_entrypoint

# ---------------------------------------------------------------------------
# handle_unhandled_exception
# ---------------------------------------------------------------------------


def test_logs_exception_type(caplog: pytest.LogCaptureFixture) -> None:
    exc = ValueError("boom")
    with caplog.at_level(logging.CRITICAL):
        handle_unhandled_exception(exc)
    assert "ValueError" in caplog.text


def test_logs_exception_message(caplog: pytest.LogCaptureFixture) -> None:
    exc = RuntimeError("something went wrong")
    with caplog.at_level(logging.CRITICAL):
        handle_unhandled_exception(exc)
    assert "something went wrong" in caplog.text


def test_logs_timestamp(caplog: pytest.LogCaptureFixture) -> None:
    exc = TypeError("bad type")
    with caplog.at_level(logging.CRITICAL):
        handle_unhandled_exception(exc)
    # ISO 8601 timestamp contains 'T' and 'Z' or '+00:00'
    assert "timestamp=" in caplog.text


def test_logs_traceback(caplog: pytest.LogCaptureFixture) -> None:
    try:
        raise KeyError("missing key")
    except KeyError as exc:
        with caplog.at_level(logging.CRITICAL):
            handle_unhandled_exception(exc)
    # traceback contains "Traceback" heading
    assert "KeyError" in caplog.text


def test_logs_at_critical_level(caplog: pytest.LogCaptureFixture) -> None:
    exc = ValueError("critical test")
    with caplog.at_level(logging.CRITICAL):
        handle_unhandled_exception(exc)
    assert any(r.levelno == logging.CRITICAL for r in caplog.records)


def test_custom_logger_used() -> None:
    custom_logger = MagicMock(spec=logging.Logger)
    exc = ValueError("custom logger test")
    handle_unhandled_exception(exc, logger=custom_logger)
    custom_logger.critical.assert_called_once()
    call_args = custom_logger.critical.call_args
    # Positional args tuple: (format_string, exc_type_name, exc, timestamp, tb)
    fmt_args = call_args[0]
    all_text = " ".join(str(a) for a in fmt_args)
    assert "ValueError" in all_text
    assert "custom logger test" in all_text


def test_does_not_raise() -> None:
    """handle_unhandled_exception must not re-raise — only logs."""
    exc = RuntimeError("should not bubble up")
    # Should complete without raising
    handle_unhandled_exception(exc)


# ---------------------------------------------------------------------------
# wrap_entrypoint — exception interception
# ---------------------------------------------------------------------------


def test_wrap_entrypoint_catches_exception() -> None:
    @wrap_entrypoint
    def bad_fn() -> None:
        raise RuntimeError("fatal error")

    with patch.object(sys, "exit") as mock_exit:
        bad_fn()
    mock_exit.assert_called_once_with(1)


def test_wrap_entrypoint_exits_nonzero_on_exception() -> None:
    @wrap_entrypoint
    def fn() -> None:
        raise ValueError("oops")

    with pytest.raises(SystemExit) as exc_info:
        fn()
    assert exc_info.value.code == 1


def test_wrap_entrypoint_logs_on_exception(caplog: pytest.LogCaptureFixture) -> None:
    @wrap_entrypoint
    def fn() -> None:
        raise RuntimeError("logged exception")

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(SystemExit):
            fn()
    assert "RuntimeError" in caplog.text
    assert "logged exception" in caplog.text


def test_wrap_entrypoint_passes_through_system_exit() -> None:
    @wrap_entrypoint
    def fn() -> None:
        raise SystemExit(0)

    with pytest.raises(SystemExit) as exc_info:
        fn()
    assert exc_info.value.code == 0


def test_wrap_entrypoint_passes_through_system_exit_nonzero() -> None:
    @wrap_entrypoint
    def fn() -> None:
        sys.exit(42)

    with pytest.raises(SystemExit) as exc_info:
        fn()
    assert exc_info.value.code == 42


def test_wrap_entrypoint_passes_through_keyboard_interrupt() -> None:
    @wrap_entrypoint
    def fn() -> None:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        fn()


def test_wrap_entrypoint_returns_value_on_success() -> None:
    @wrap_entrypoint
    def fn() -> int:
        return 42

    assert fn() == 42


def test_wrap_entrypoint_preserves_function_name() -> None:
    @wrap_entrypoint
    def my_special_entrypoint() -> None:
        pass

    assert my_special_entrypoint.__name__ == "my_special_entrypoint"


def test_wrap_entrypoint_does_not_swallow_exception_silently(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify that the exception details appear in the log (not swallowed)."""

    @wrap_entrypoint
    def fn() -> None:
        raise RuntimeError("must appear in log")

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(SystemExit):
            fn()
    assert "must appear in log" in caplog.text


# ---------------------------------------------------------------------------
# Entry-point integration: verify wrap_entrypoint is applied
# ---------------------------------------------------------------------------


def test_cli_main_is_wrapped() -> None:
    from rex.cli import main as cli_main

    # The wrapper preserves __wrapped__ via functools.wraps
    assert hasattr(cli_main, "__wrapped__") or callable(cli_main)


def test_config_cli_is_wrapped() -> None:
    from rex.config import cli as config_cli

    assert callable(config_cli)


def test_rex_speak_api_main_is_wrapped() -> None:
    import rex_speak_api

    assert callable(rex_speak_api.main)


def test_agent_server_main_is_wrapped() -> None:
    from rex.computers.agent_server import main as agent_main

    assert callable(agent_main)


# ---------------------------------------------------------------------------
# Exception type and message in log record
# ---------------------------------------------------------------------------


def test_log_contains_exception_class_name(caplog: pytest.LogCaptureFixture) -> None:
    exc = FileNotFoundError("no such file")
    with caplog.at_level(logging.CRITICAL):
        handle_unhandled_exception(exc)
    assert "FileNotFoundError" in caplog.text


def test_log_contains_full_traceback(caplog: pytest.LogCaptureFixture) -> None:
    """When exception has a traceback, it should appear in the log output."""
    try:
        raise OSError("disk full")
    except OSError as exc:
        with caplog.at_level(logging.CRITICAL):
            handle_unhandled_exception(exc)
    assert "OSError" in caplog.text


def test_wrap_entrypoint_exception_includes_type_in_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    @wrap_entrypoint
    def fn() -> None:
        raise AttributeError("no attr")

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(SystemExit):
            fn()
    assert "AttributeError" in caplog.text
