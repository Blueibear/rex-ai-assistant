"""Logging helpers used throughout the Rex assistant."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "[%(asctime)s] %(levelname)s - %(message)s"
DEFAULT_LOG_FILE = Path("logs/rex.log")
DEFAULT_ERROR_FILE = Path("logs/error.log")

# Mapping from string name → logging constant
_LEVEL_NAMES: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


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


def _env_log_level() -> int:
    """Return the log level from the environment.

    Resolution order:
    1. ``LOG_LEVEL`` env var (e.g. ``LOG_LEVEL=DEBUG``).
    2. ``REX_LOG_LEVEL`` env var (legacy name).
    3. Default: ``INFO``.

    Unknown values are silently ignored and the next source is tried.
    """
    for var in ("LOG_LEVEL", "REX_LOG_LEVEL"):
        raw = os.environ.get(var, "").strip().upper()
        if raw in _LEVEL_NAMES:
            return _LEVEL_NAMES[raw]
    return logging.INFO


def _env_module_levels() -> dict[str, int]:
    """Return per-module log-level overrides from the environment.

    Any env var of the form ``LOG_LEVEL_<module>=<level>`` (case-insensitive
    for the level) is treated as a per-module override.  The module name is the
    part after ``LOG_LEVEL_`` with underscores preserved as-is.

    Example::

        LOG_LEVEL_rex.http_errors=DEBUG
        LOG_LEVEL_rex.llm_client=WARNING

    Returns:
        ``dict`` mapping logger name → ``logging`` level constant.
    """
    prefix = "LOG_LEVEL_"
    prefix_upper = prefix.upper()
    result: dict[str, int] = {}
    for key, val in os.environ.items():
        # Compare upper-cased key to handle Windows (which stores keys in upper case).
        if key.upper().startswith(prefix_upper):
            module = key[len(prefix) :].lower()
            level_name = val.strip().upper()
            if module and level_name in _LEVEL_NAMES:
                result[module] = _LEVEL_NAMES[level_name]
    return result


def apply_module_log_levels(module_levels: Mapping[str, int | str]) -> None:
    """Apply per-module log-level overrides.

    Each entry sets the named logger's effective level independently of the
    root logger.  This is additive — previously configured overrides are not
    removed.

    Args:
        module_levels: Mapping of logger name → level (int constant or string
            like ``"DEBUG"``).
    """
    for module, level in module_levels.items():
        if isinstance(level, str):
            resolved: int = _LEVEL_NAMES.get(level.strip().upper(), logging.INFO)
        else:
            resolved = level
        logging.getLogger(module).setLevel(resolved)


try:  # pragma: no cover - avoid circular imports during package init
    from .config import settings as settings
except Exception:  # pragma: no cover - fallback when config not initialised
    settings = None  # noqa: PGH003


def _resolve_path(candidate: str | os.PathLike[str], default: Path) -> Path:
    path = Path(candidate) if candidate else default
    return path


def configure_logging(
    level: int | None = None,
    handlers: Iterable[logging.Handler] | None = None,
    module_levels: Mapping[str, int | str] | None = None,
) -> None:
    """Configure application-wide logging with optional file handlers.

    Log level resolution (when ``level`` is ``None``):
    1. ``LOG_LEVEL`` environment variable.
    2. ``REX_LOG_LEVEL`` environment variable (legacy).
    3. Default: ``INFO``.

    Per-module overrides can be supplied via ``module_levels`` or via env vars
    of the form ``LOG_LEVEL_<module>=<level>`` (e.g.
    ``LOG_LEVEL_rex.http_errors=DEBUG``).

    By default, logs to stdout.  File logging is enabled only if:
    - runtime.file_logging_enabled=true in rex_config.json (default: false)
    - Or explicitly requested via handlers parameter

    This makes logging container-friendly and follows 12-factor app principles.

    Args:
        level: Root log level.  When ``None`` the value is read from the
            ``LOG_LEVEL`` / ``REX_LOG_LEVEL`` environment variable (default
            ``INFO``).
        handlers: Custom log handlers.  When ``None`` a StreamHandler to
            stdout is used (plus file handlers if configured).
        module_levels: Per-module level overrides applied after the root
            logger is configured.
    """
    effective_level = level if level is not None else _env_log_level()

    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured — still apply any new module-level overrides.
        _apply_all_module_levels(module_levels)
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

    logging.basicConfig(level=effective_level, handlers=handler_list)
    _apply_all_module_levels(module_levels)


def _apply_all_module_levels(module_levels: Mapping[str, int | str] | None) -> None:
    """Apply explicit overrides then env-var overrides."""
    env_overrides = _env_module_levels()
    if env_overrides:
        apply_module_log_levels(env_overrides)
    if module_levels:
        apply_module_log_levels(module_levels)


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
