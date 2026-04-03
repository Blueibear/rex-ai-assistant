"""Tests for US-LOG-001: Structured logging to rotating file."""

from __future__ import annotations

import json
import logging
from pathlib import Path


def test_setup_file_logging_creates_log_dir(tmp_path: Path) -> None:
    from rex.logging_config import setup_file_logging

    log_file = tmp_path / "custom" / "subdir" / "rex.log"
    setup_file_logging(log_file)
    assert log_file.parent.is_dir()


def test_setup_file_logging_creates_handler(tmp_path: Path) -> None:
    from rex.logging_config import setup_file_logging

    handler = setup_file_logging(tmp_path / "rex.log")
    assert handler is not None


def test_log_line_is_valid_json(tmp_path: Path) -> None:
    from rex.logging_config import setup_file_logging

    log_file = tmp_path / "rex.log"
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        root.setLevel(logging.DEBUG)
        handler = setup_file_logging(log_file)
        logging.getLogger("test.log001").info("hello world")
        handler.flush()
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert lines, "Expected at least one log line"
        parsed = json.loads(lines[-1])
        assert parsed["level"] == "INFO"
        assert parsed["message"] == "hello world"
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_log_line_has_required_fields(tmp_path: Path) -> None:
    from rex.logging_config import setup_file_logging

    log_file = tmp_path / "rex.log"
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        root.setLevel(logging.DEBUG)
        handler = setup_file_logging(log_file)
        logging.getLogger("test.log001.fields").warning("check fields")
        handler.flush()
        line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
        parsed = json.loads(line)
        for field in ("timestamp", "level", "logger", "message", "extra"):
            assert field in parsed, f"Missing field: {field}"
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_log_line_extra_field_is_dict(tmp_path: Path) -> None:
    from rex.logging_config import setup_file_logging

    log_file = tmp_path / "rex.log"
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        root.setLevel(logging.DEBUG)
        handler = setup_file_logging(log_file)
        logging.getLogger("test.log001.extra").info("msg")
        handler.flush()
        line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
        parsed = json.loads(line)
        assert isinstance(parsed["extra"], dict)
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_rotation_config(tmp_path: Path) -> None:
    from logging.handlers import RotatingFileHandler

    from rex.logging_config import BACKUP_COUNT, MAX_BYTES, setup_file_logging

    handler = setup_file_logging(tmp_path / "rex.log")
    assert isinstance(handler, RotatingFileHandler)
    assert handler.maxBytes == MAX_BYTES
    assert handler.backupCount == BACKUP_COUNT


def test_max_bytes_is_5mb() -> None:
    from rex.logging_config import MAX_BYTES

    assert MAX_BYTES == 5_000_000


def test_backup_count_is_5() -> None:
    from rex.logging_config import BACKUP_COUNT

    assert BACKUP_COUNT == 5


def test_setup_file_logging_no_duplicate_handlers(tmp_path: Path) -> None:
    from rex.logging_config import setup_file_logging

    log_file = tmp_path / "rex.log"
    root = logging.getLogger()
    original_handlers = root.handlers[:]

    try:
        setup_file_logging(log_file)
        count_before = len(root.handlers)
        setup_file_logging(log_file)  # second call — should not add duplicate
        assert len(root.handlers) == count_before
    finally:
        root.handlers = original_handlers


def test_json_formatter_timestamp_is_iso(tmp_path: Path) -> None:
    from rex.logging_config import setup_file_logging

    log_file = tmp_path / "rex.log"
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    try:
        root.setLevel(logging.DEBUG)
        handler = setup_file_logging(log_file)
        logging.getLogger("test.log001.ts").info("ts check")
        handler.flush()
        line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
        parsed = json.loads(line)
        ts = parsed["timestamp"]
        # ISO 8601 basic check — starts with year and contains T
        assert "T" in ts
        assert len(ts) >= 20
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_logs_dir_gitignore() -> None:
    """logs/ must be in .gitignore."""
    gitignore = Path(__file__).resolve().parent.parent / ".gitignore"
    if not gitignore.exists():
        return  # pragma: no cover
    content = gitignore.read_text(encoding="utf-8")
    assert "logs/" in content, ".gitignore must contain 'logs/'"
