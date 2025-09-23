"""Logging helpers for the Rex assistant."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Iterable

LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = os.path.join("logs", "rex.log")


def configure_logging(level: int = logging.INFO, handlers: Iterable[logging.Handler] | None = None) -> None:
    """Configure application-wide logging with sane defaults."""

    logger = logging.getLogger()
    if logger.handlers:
        return

    os.makedirs(os.path.dirname(DEFAULT_LOG_FILE), exist_ok=True)

    if handlers is None:
        stream_handler = logging.StreamHandler()
        file_handler = RotatingFileHandler(DEFAULT_LOG_FILE, maxBytes=1_000_000, backupCount=5)
        handlers = (stream_handler, file_handler)

    logging.basicConfig(level=level, format=LOG_FORMAT, handlers=list(handlers))
