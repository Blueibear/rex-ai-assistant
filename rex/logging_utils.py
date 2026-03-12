"""Logging helpers used throughout the Rex assistant."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = Path("logs/rex.log")
DEFAULT_ERROR_FILE = Path("logs/error.log")


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object.

    Fields: timestamp (ISO 8601), level, logger, message.
    Optional: exception (if exc_info is set).
    """

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, str] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def _json_logging_enabled() -> bool:
    """Return True if JSON structured logging should be used.

    Resolution order:
    1. ``REX_JSON_LOGS`` env var (``1``/``true`` → on; ``0``/``false`` → off).
    2. Auto-detect: off when running under pytest (``PYTEST_CURRENT_TEST`` is set),
       on otherwise so that production deployments get structured output by default.
    """
    val = os.environ.get("REX_JSON_LOGS")
    if val is not None:
        return val.lower() not in ("0", "false", "no")
    # Default: plain text in tests, JSON in production
    return "PYTEST_CURRENT_TEST" not in os.environ

try:  # pragma: no cover - avoid circular imports during package init
    from .config import settings as settings
except Exception:  # pragma: no cover - fallback when config not initialised
    settings = None  # type: ignore[assignment]


def _resolve_path(candidate: str | os.PathLike[str], default: Path) -> Path:
    path = Path(candidate) if candidate else default
    return path


def configure_logging(
    level: int = logging.INFO, handlers: Iterable[logging.Handler] | None = None
) -> None:
    """Configure application-wide logging with optional file handlers.

    By default, logs to stdout. File logging is enabled only if:
    - runtime.file_logging_enabled=true in rex_config.json (default: false)
    - Or explicitly requested via handlers parameter

    This makes logging container-friendly and follows 12-factor app principles.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    if handlers is None:
        stream_handler = logging.StreamHandler(sys.stdout)

        # Force UTF-8 encoding for console output to handle emoji and Unicode
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass  # Continue without forcing encoding

        handlers_list = [stream_handler]

        # Get file logging setting from config (defaults to False)
        file_logging_enabled = False
        if settings is not None:
            file_logging_enabled = getattr(settings, "file_logging_enabled", False)

        if file_logging_enabled:
            log_path = _resolve_path(
                getattr(settings, "log_path", DEFAULT_LOG_FILE) if settings else DEFAULT_LOG_FILE,
                DEFAULT_LOG_FILE,
            )
            error_path = _resolve_path(
                (
                    getattr(settings, "error_log_path", DEFAULT_ERROR_FILE)
                    if settings
                    else DEFAULT_ERROR_FILE
                ),
                DEFAULT_ERROR_FILE,
            )

            log_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = RotatingFileHandler(
                log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
            )
            error_handler = RotatingFileHandler(
                error_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8"
            )
            error_handler.setLevel(logging.ERROR)

            handlers_list.extend([file_handler, error_handler])  # type: ignore[list-item]

        handlers = tuple(handlers_list)

    if _json_logging_enabled():
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(LOG_FORMAT)

    handler_list = list(handlers)
    for h in handler_list:
        h.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=handler_list)


def get_logger(name: str, *, level: int | None = None) -> logging.Logger:
    """Return a module-level logger backed by the shared configuration."""
    configure_logging(level=level or logging.getLogger().level or logging.INFO)
    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)
        for handler in logger.handlers:
            handler.setLevel(level)

    return logger


def set_global_level(level: int) -> None:
    """Update the configured logging level for all registered handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    for handler in root_logger.handlers:
        handler.setLevel(level)

    manager = logging.Logger.manager
    for logger in manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
