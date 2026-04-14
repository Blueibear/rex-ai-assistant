"""Tests for US-070: Structured JSON logging system."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler


def test_json_formatter_fields(tmp_path):
    """Each file log line must include timestamp, level, logger, message, extra."""
    from rex.logging_utils import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="rex.test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="hello world",
        args=(),
        exc_info=None,
    )
    line = formatter.format(record)
    entry = json.loads(line)
    assert "timestamp" in entry
    assert entry["level"] == "INFO"
    assert entry["logger"] == "rex.test"
    assert entry["message"] == "hello world"
    assert "extra" in entry


def test_json_formatter_extra_fields(tmp_path):
    """Extra fields attached to a LogRecord are captured in the 'extra' dict."""
    from rex.logging_utils import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="rex.test",
        level=logging.WARNING,
        pathname="",
        lineno=0,
        msg="check extra",
        args=(),
        exc_info=None,
    )
    record.request_id = "abc-123"
    line = formatter.format(record)
    entry = json.loads(line)
    assert entry["extra"].get("request_id") == "abc-123"


def test_json_formatter_timestamp_iso8601(tmp_path):
    """Timestamp must be a valid ISO 8601 string."""
    from datetime import datetime

    from rex.logging_utils import JsonFormatter

    formatter = JsonFormatter()
    record = logging.LogRecord(
        name="rex.ts",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="ts check",
        args=(),
        exc_info=None,
    )
    entry = json.loads(formatter.format(record))
    # datetime.fromisoformat raises ValueError on bad format
    datetime.fromisoformat(entry["timestamp"])


def test_setup_file_logging_creates_file(tmp_path):
    """setup_file_logging must create log file and attach RotatingFileHandler."""
    from rex.logging_config import setup_file_logging

    log_path = tmp_path / "logs" / "rex.log"
    handler = setup_file_logging(log_path)
    assert isinstance(handler, RotatingFileHandler)

    logger = logging.getLogger("rex.setup_test")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.info("structured test message")
    handler.flush()

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1
    entry = json.loads(lines[-1])
    assert entry["message"] == "structured test message"


def test_setup_file_logging_json_format(tmp_path):
    """File output must be valid JSON on every line."""
    from rex.logging_config import setup_file_logging

    log_path = tmp_path / "logs" / "rex.log"
    handler = setup_file_logging(log_path)

    logger = logging.getLogger("rex.json_format_test")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    for msg in ("alpha", "beta", "gamma"):
        logger.warning(msg)
    handler.flush()

    for line in log_path.read_text(encoding="utf-8").strip().splitlines():
        json.loads(line)  # must not raise


def test_setup_file_logging_rotation_config(tmp_path):
    """Handler must be configured with the expected rotation parameters."""
    from rex.logging_config import BACKUP_COUNT, MAX_BYTES, setup_file_logging

    log_path = tmp_path / "rex.log"
    handler = setup_file_logging(log_path)
    assert handler.maxBytes == MAX_BYTES
    assert handler.backupCount == BACKUP_COUNT


def test_max_bytes_is_5mb():
    """MAX_BYTES must be 5 MB per US-323 AC."""
    from rex.logging_config import MAX_BYTES

    assert MAX_BYTES == 5_000_000


def test_setup_file_logging_idempotent(tmp_path):
    """Calling setup_file_logging twice must not add duplicate handlers."""
    from rex.logging_config import setup_file_logging

    log_path = tmp_path / "rex.log"
    root = logging.getLogger()

    setup_file_logging(log_path)
    setup_file_logging(log_path)

    rotating = [
        h
        for h in root.handlers
        if isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_path.resolve())
    ]
    assert len(rotating) == 1


def test_existing_logging_calls_still_work(tmp_path):
    """logging.info/warning/error calls must continue to emit output."""
    from rex.logging_config import setup_file_logging

    log_path = tmp_path / "rex.log"
    handler = setup_file_logging(log_path)

    logger = logging.getLogger("rex.compat")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    handler.flush()

    content = log_path.read_text(encoding="utf-8")
    assert "info message" in content
    assert "warning message" in content
    assert "error message" in content


def test_console_handler_uses_plain_formatter(tmp_path, monkeypatch):
    """configure_logging with JSON disabled must use plain text for console."""
    import io

    from rex.logging_utils import configure_logging

    monkeypatch.setenv("REX_JSON_LOGS", "0")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "")

    buf = io.StringIO()
    stream_handler = logging.StreamHandler(buf)
    # Reset root logger
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers.clear()

    try:
        configure_logging(level=logging.DEBUG, handlers=[stream_handler])
        logging.getLogger("rex.console").info("plain text check")
        stream_handler.flush()
        output = buf.getvalue()
        # Plain formatter: not JSON
        assert not output.strip().startswith("{")
    finally:
        root.handlers.clear()
        root.handlers.extend(old_handlers)


def test_session_marker_written_to_file(tmp_path):
    """setup_file_logging must write a session-start marker to the log file."""
    import logging
    from rex.logging_config import setup_file_logging

    log_path = tmp_path / "rex.log"
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers.clear()
    try:
        setup_file_logging(log_path)
        content = log_path.read_text(encoding="utf-8")
        assert "=== Rex session started at" in content
    finally:
        root.handlers.clear()
        root.handlers.extend(old_handlers)


def test_session_marker_written_by_configure_logging(tmp_path, monkeypatch):
    """configure_logging must write a session-start marker when file logging is enabled."""
    import logging
    from unittest.mock import patch

    from rex.logging_utils import configure_logging

    log_path = tmp_path / "rex.log"
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    root.handlers.clear()
    try:
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
        with patch("rex.logging_utils._json_logging_enabled", return_value=False):
            configure_logging(level=logging.DEBUG, handlers=[fh])
        content = log_path.read_text(encoding="utf-8")
        assert "=== Rex session started at" in content
    finally:
        root.handlers.clear()
        root.handlers.extend(old_handlers)
