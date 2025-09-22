"""Logging helpers centralised for the Rex package."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from .config import settings

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

_DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"


def configure_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured for console and rotating file output."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_DEFAULT_FORMAT)

    file_path = settings.log_file or _LOG_DIR / "assistant.log"
    file_handler: logging.Handler
    if isinstance(file_path, Path):
        file_handler = RotatingFileHandler(file_path, maxBytes=512_000, backupCount=5)
    else:
        file_handler = RotatingFileHandler(str(file_path), maxBytes=512_000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    return logger


def set_global_level(level: int) -> None:
    """Update all configured loggers to a new logging level."""

    logging.getLogger().setLevel(level)
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)


__all__ = ["configure_logger", "set_global_level"]
