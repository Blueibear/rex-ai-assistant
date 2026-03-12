"""Global unhandled exception handler (US-103).

Wraps application entry points so that any unhandled exception is:
  - logged with full context (type, message, traceback, timestamp)
  - followed by a non-zero exit (sys.exit(1))

:class:`SystemExit` and :class:`KeyboardInterrupt` are passed through
unchanged so that normal ``sys.exit(0)`` calls and Ctrl-C work correctly.

Usage::

    from rex.exception_handler import wrap_entrypoint

    @wrap_entrypoint
    def main() -> int:
        ...
"""
from __future__ import annotations

import logging
import sys
import traceback
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, TypeVar

_LOG = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_unhandled_exception(
    exc: BaseException,
    logger: logging.Logger = _LOG,
) -> None:
    """Log *exc* with full context (type, message, traceback, timestamp).

    This function only logs — it does NOT exit.  Call ``sys.exit(1)`` after
    this if you want to terminate the process.  Keeping log-and-exit separate
    makes the function testable.
    """
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.critical(
        "Unhandled exception — %s: %s | timestamp=%s\n%s",
        type(exc).__name__,
        exc,
        timestamp,
        tb,
    )


def wrap_entrypoint(fn: F) -> F:
    """Decorator: wrap *fn* with a global unhandled-exception handler.

    * Any :class:`Exception` → logged with full context, then ``sys.exit(1)``.
    * :class:`SystemExit` and :class:`KeyboardInterrupt` pass through unchanged.
    """

    @wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            handle_unhandled_exception(exc)
            sys.exit(1)

    return _wrapper  # type: ignore[return-value]
