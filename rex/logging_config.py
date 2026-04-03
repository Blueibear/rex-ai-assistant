"""Structured JSON logging configuration for Rex.

Provides ``setup_file_logging()`` which adds a ``RotatingFileHandler`` writing
valid JSON lines to ``logs/rex.log``.

Each log line format::

    {"timestamp": "<ISO8601>", "level": "INFO", "logger": "rex.foo",
     "message": "...", "extra": {}}

Rotation: 5 MB per file, keep last 5 files.

Usage::

    from rex.logging_config import setup_file_logging
    setup_file_logging()
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .logging_utils import JsonFormatter

LOG_FILE = Path("logs/rex.log")
MAX_BYTES = 5_000_000  # 5 MB
BACKUP_COUNT = 5


def setup_file_logging(
    log_file: str | Path = LOG_FILE,
    *,
    level: int = logging.DEBUG,
    max_bytes: int = MAX_BYTES,
    backup_count: int = BACKUP_COUNT,
) -> RotatingFileHandler:
    """Attach a JSON rotating file handler to the root logger.

    Creates the log directory if it does not exist.

    Args:
        log_file: Path to the primary log file (default ``logs/rex.log``).
        level: Minimum log level to write to the file (default ``DEBUG`` so all
            levels captured by the root logger reach the file).
        max_bytes: Maximum file size before rotation (default 5 MB).
        backup_count: Number of rotated files to keep (default 5).

    Returns:
        The configured ``RotatingFileHandler`` (also attached to root logger).
    """
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    # Avoid duplicate handlers on repeated calls.
    for existing in root.handlers:
        if isinstance(existing, RotatingFileHandler) and existing.baseFilename == str(
            path.resolve()
        ):
            return existing

    root.addHandler(handler)
    return handler


__all__ = ["setup_file_logging", "LOG_FILE", "MAX_BYTES", "BACKUP_COUNT"]
