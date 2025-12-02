"""Central logging configuration for the Rex assistant."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Create a log directory if not already present
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Default log format
_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_FILE = LOG_DIR / "error.log"

# Max log size before rotation (5MB) and backup count
_MAX_LOG_SIZE = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """
    Return a module-level logger configured for both console and file output.
    Prevents duplicate handlers across reloads.
    """

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_DEFAULT_FORMAT)

    # File handler with log rotation
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=_MAX_LOG_SIZE, backupCount=_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    return logger


def set_global_level(level: int) -> None:
    """
    Update the logging level for all configured handlers across the application.
    """

    logging.getLogger().setLevel(level)

    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)
