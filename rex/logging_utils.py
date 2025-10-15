"""Logging helpers used throughout the Rex assistant."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable

LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = Path("logs/rex.log")
DEFAULT_ERROR_FILE = Path("logs/error.log")

try:  # pragma: no cover - avoid circular imports during package init
    from .config import settings
except Exception:  # pragma: no cover - fallback when config not initialised
    settings = None  # type: ignore[assignment]


def _resolve_path(candidate: str | os.PathLike[str], default: Path) -> Path:
    path = Path(candidate) if candidate else default
    return path


def configure_logging(level: int = logging.INFO, handlers: Iterable[logging.Handler] | None = None) -> None:
    """Configure application-wide logging with rotating file handlers."""

    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    log_path = _resolve_path(getattr(settings, "log_path", DEFAULT_LOG_FILE), DEFAULT_LOG_FILE)
    error_path = _resolve_path(getattr(settings, "error_log_path", DEFAULT_ERROR_FILE), DEFAULT_ERROR_FILE)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    if handlers is None:
        import sys
        # Force UTF-8 encoding for console output to handle emoji and Unicode
        stream_handler = logging.StreamHandler(sys.stdout)
        if hasattr(sys.stdout, 'reconfigure'):
            try:
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            except (AttributeError, ValueError):
                pass  # Fallback: continue with default encoding
        
        file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5, encoding='utf-8')
        error_handler = RotatingFileHandler(error_path, maxBytes=1_000_000, backupCount=5, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        handlers = (stream_handler, file_handler, error_handler)

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=list(handlers))


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
