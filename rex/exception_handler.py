"""Global unhandled exception handler (US-103).

Wraps application entry points so that any unhandled exception is:
  - logged with full context (type, message, traceback, timestamp)
  - shown to the user as a clean "Error: …" message (no raw traceback)
  - followed by a non-zero exit (sys.exit(1))

In debug mode (``REX_DEBUG=1``), the full traceback is also printed to
stderr so developers can see the full call stack.

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
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any, TypeVar

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
    timestamp = datetime.now(UTC).isoformat()
    logger.critical(
        "Unhandled exception — %s: %s | timestamp=%s\n%s",
        type(exc).__name__,
        exc,
        timestamp,
        tb,
    )


def _print_user_error(exc: Exception) -> None:
    """Print a user-friendly error message to stderr.

    In normal mode, only the error message is shown (no traceback).
    In debug mode (``REX_DEBUG=1``), the full traceback is also printed.
    """
    from rex.dep_errors import is_debug_mode

    if is_debug_mode():
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(tb, file=sys.stderr, end="")
    else:
        print(f"\nError: {exc}", file=sys.stderr)
        print("Run with REX_DEBUG=1 for full details.", file=sys.stderr)


def wrap_entrypoint(fn: F) -> F:
    """Decorator: wrap *fn* with a global unhandled-exception handler.

    * Any :class:`Exception` → logged with full context, user-friendly
      message printed to stderr, then ``sys.exit(1)``.
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
            _print_user_error(exc)
            sys.exit(1)

    return _wrapper  # type: ignore[return-value]
