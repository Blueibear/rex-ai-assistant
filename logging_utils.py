"""Central logging configuration for the Rex assistant."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, *, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger configured for both console and file output."""

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_DEFAULT_FORMAT)

    file_handler = logging.FileHandler(LOG_DIR / "error.log")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)

    return logger


def set_global_level(level: int) -> None:
    """Update the logging level for all configured handlers."""

    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            for handler in logger.handlers:
                handler.setLevel(level)
    logging.getLogger().setLevel(level)
