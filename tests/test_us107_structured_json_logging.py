"""Tests for US-107: Structured JSON logging.

Verifies:
- JsonFormatter emits valid single-line JSON
- JSON entry contains required fields: timestamp, level, logger, message
- timestamp is ISO 8601 (UTC)
- exception info included when present
- _json_logging_enabled() respects REX_JSON_LOGS env var
- _json_logging_enabled() auto-detects test mode via PYTEST_CURRENT_TEST
- configure_logging() applies JsonFormatter when JSON logging is enabled
- configure_logging() applies plain formatter when JSON logging is disabled
"""

from __future__ import annotations

import json
import logging
import os
from io import StringIO
from unittest.mock import patch

import pytest

from rex.logging_utils import JsonFormatter, _json_logging_enabled, configure_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    msg: str = "hello",
    name: str = "rex.test",
    level: int = logging.INFO,
    exc_info: tuple | None = None,  # type: ignore[type-arg]
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    return record


# ---------------------------------------------------------------------------
# JsonFormatter tests
# ---------------------------------------------------------------------------


class TestJsonFormatter:
    def setup_method(self) -> None:
        self.fmt = JsonFormatter()

    def test_output_is_valid_json(self) -> None:
        record = _make_record("test message")
        output = self.fmt.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_output_is_single_line(self) -> None:
        record = _make_record("single line test")
        output = self.fmt.format(record)
        assert "\n" not in output

    def test_contains_timestamp_field(self) -> None:
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        assert "timestamp" in parsed

    def test_timestamp_is_iso8601_utc(self) -> None:
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        ts = parsed["timestamp"]
        # ISO 8601 with UTC offset (+00:00) or Z
        assert "T" in ts
        assert ts.endswith("+00:00") or ts.endswith("Z")

    def test_contains_level_field(self) -> None:
        record = _make_record(level=logging.WARNING)
        parsed = json.loads(self.fmt.format(record))
        assert parsed["level"] == "WARNING"

    def test_level_info(self) -> None:
        record = _make_record(level=logging.INFO)
        parsed = json.loads(self.fmt.format(record))
        assert parsed["level"] == "INFO"

    def test_level_error(self) -> None:
        record = _make_record(level=logging.ERROR)
        parsed = json.loads(self.fmt.format(record))
        assert parsed["level"] == "ERROR"

    def test_level_debug(self) -> None:
        record = _make_record(level=logging.DEBUG)
        parsed = json.loads(self.fmt.format(record))
        assert parsed["level"] == "DEBUG"

    def test_level_critical(self) -> None:
        record = _make_record(level=logging.CRITICAL)
        parsed = json.loads(self.fmt.format(record))
        assert parsed["level"] == "CRITICAL"

    def test_contains_logger_field(self) -> None:
        record = _make_record(name="rex.some.module")
        parsed = json.loads(self.fmt.format(record))
        assert parsed["logger"] == "rex.some.module"

    def test_contains_message_field(self) -> None:
        record = _make_record(msg="the actual message")
        parsed = json.loads(self.fmt.format(record))
        assert parsed["message"] == "the actual message"

    def test_message_with_format_args(self) -> None:
        record = logging.LogRecord(
            name="rex.test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="value is %d",
            args=(42,),
            exc_info=None,
        )
        parsed = json.loads(self.fmt.format(record))
        assert parsed["message"] == "value is 42"

    def test_no_exception_field_by_default(self) -> None:
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        assert "exception" not in parsed

    def test_exception_field_present_when_exc_info(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = _make_record(exc_info=exc_info)
        parsed = json.loads(self.fmt.format(record))
        assert "exception" in parsed
        assert "ValueError" in parsed["exception"]
        assert "boom" in parsed["exception"]

    def test_minimum_required_fields_present(self) -> None:
        record = _make_record()
        parsed = json.loads(self.fmt.format(record))
        for field in ("timestamp", "level", "logger", "message"):
            assert field in parsed, f"Missing required field: {field}"

    def test_special_characters_in_message(self) -> None:
        record = _make_record(msg='hello "world" & <test>')
        output = self.fmt.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == 'hello "world" & <test>'

    def test_unicode_in_message(self) -> None:
        record = _make_record(msg="こんにちは 🎉")
        parsed = json.loads(self.fmt.format(record))
        assert parsed["message"] == "こんにちは 🎉"


# ---------------------------------------------------------------------------
# _json_logging_enabled tests
# ---------------------------------------------------------------------------


class TestJsonLoggingEnabled:
    def test_returns_true_when_rex_json_logs_1(self) -> None:
        with patch.dict(os.environ, {"REX_JSON_LOGS": "1"}, clear=False):
            assert _json_logging_enabled() is True

    def test_returns_true_when_rex_json_logs_true(self) -> None:
        with patch.dict(os.environ, {"REX_JSON_LOGS": "true"}, clear=False):
            assert _json_logging_enabled() is True

    def test_returns_true_when_rex_json_logs_True_mixed_case(self) -> None:
        with patch.dict(os.environ, {"REX_JSON_LOGS": "True"}, clear=False):
            assert _json_logging_enabled() is True

    def test_returns_false_when_rex_json_logs_0(self) -> None:
        with patch.dict(os.environ, {"REX_JSON_LOGS": "0"}, clear=False):
            assert _json_logging_enabled() is False

    def test_returns_false_when_rex_json_logs_false(self) -> None:
        with patch.dict(os.environ, {"REX_JSON_LOGS": "false"}, clear=False):
            assert _json_logging_enabled() is False

    def test_returns_false_when_rex_json_logs_no(self) -> None:
        with patch.dict(os.environ, {"REX_JSON_LOGS": "no"}, clear=False):
            assert _json_logging_enabled() is False

    def test_auto_off_in_pytest(self) -> None:
        """When PYTEST_CURRENT_TEST is set and REX_JSON_LOGS is unset → off."""
        env = {k: v for k, v in os.environ.items() if k != "REX_JSON_LOGS"}
        env["PYTEST_CURRENT_TEST"] = "some_test"
        with patch.dict(os.environ, env, clear=True):
            assert _json_logging_enabled() is False

    def test_auto_on_outside_pytest(self) -> None:
        """When PYTEST_CURRENT_TEST is absent and REX_JSON_LOGS is unset → on."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("REX_JSON_LOGS", "PYTEST_CURRENT_TEST")
        }
        with patch.dict(os.environ, env, clear=True):
            assert _json_logging_enabled() is True

    def test_explicit_rex_json_logs_overrides_pytest_detection(self) -> None:
        """REX_JSON_LOGS=1 enables JSON even inside pytest."""
        env = {k: v for k, v in os.environ.items() if k != "REX_JSON_LOGS"}
        env["PYTEST_CURRENT_TEST"] = "some_test"
        env["REX_JSON_LOGS"] = "1"
        with patch.dict(os.environ, env, clear=True):
            assert _json_logging_enabled() is True


# ---------------------------------------------------------------------------
# configure_logging integration tests
# ---------------------------------------------------------------------------


class TestConfigureLoggingJsonMode:
    """Test that configure_logging() installs the correct formatter."""

    def _fresh_root_logger(self) -> logging.Logger:
        """Remove all handlers from root logger so configure_logging() runs fresh."""
        root = logging.getLogger()
        for h in root.handlers[:]:
            root.removeHandler(h)
        return root

    def test_json_formatter_applied_when_json_enabled(self) -> None:
        root = self._fresh_root_logger()
        with patch("rex.logging_utils._json_logging_enabled", return_value=True):
            stream = StringIO()
            configure_logging(handlers=[logging.StreamHandler(stream)])
        formatters = [h.formatter for h in root.handlers]
        assert any(isinstance(f, JsonFormatter) for f in formatters)
        # Cleanup
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_plain_formatter_applied_when_json_disabled(self) -> None:
        root = self._fresh_root_logger()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            stream = StringIO()
            configure_logging(handlers=[logging.StreamHandler(stream)])
        formatters = [h.formatter for h in root.handlers]
        assert not any(isinstance(f, JsonFormatter) for f in formatters)
        # Cleanup
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_json_output_parseable_when_enabled(self) -> None:
        root = self._fresh_root_logger()
        stream = StringIO()
        with patch("rex.logging_utils._json_logging_enabled", return_value=True):
            configure_logging(level=logging.DEBUG, handlers=[logging.StreamHandler(stream)])
        logger = logging.getLogger("rex.test.integration")
        logger.info("integration test message")
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["message"] == "integration test message"
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "rex.test.integration"
        # Cleanup
        for h in root.handlers[:]:
            root.removeHandler(h)

    def test_plain_text_output_when_json_disabled(self) -> None:
        root = self._fresh_root_logger()
        stream = StringIO()
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(level=logging.DEBUG, handlers=[logging.StreamHandler(stream)])
        logger = logging.getLogger("rex.test.plain")
        logger.info("plain text message")
        output = stream.getvalue().strip()
        # Should NOT be JSON
        with pytest.raises((json.JSONDecodeError, ValueError)):
            json.loads(output)
        assert "plain text message" in output
        # Cleanup
        for h in root.handlers[:]:
            root.removeHandler(h)
