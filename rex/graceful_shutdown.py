"""Graceful shutdown support for Rex servers and processes.

Registers a SIGTERM handler that:
1. Sets a ``shutting_down`` flag so Flask ``before_request`` hooks can reject
   new requests with 503.
2. Waits up to ``drain_timeout`` seconds for in-flight requests to finish.
3. Runs registered cleanup callbacks (database connections, background jobs).
4. Exits with code 0.

Drain timeout is configurable via the ``REX_SHUTDOWN_TIMEOUT`` environment
variable (default: 10 seconds).

Usage (Flask)::

    handler = get_shutdown_handler()
    handler.install()

    @app.before_request
    def _reject_on_shutdown():
        if handler.is_shutting_down:
            return jsonify({"error": "Server is shutting down"}), 503

Usage (background job cleanup)::

    handler.register_cleanup(scheduler.stop)
    handler.register_cleanup(db_conn.close)
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from collections.abc import Callable

logger = logging.getLogger("rex.graceful_shutdown")

_DEFAULT_DRAIN_TIMEOUT: int = 10  # seconds


class GracefulShutdown:
    """SIGTERM-aware shutdown coordinator.

    Args:
        drain_timeout: Seconds to wait for in-flight requests before forcing
            exit.  When ``None``, reads ``REX_SHUTDOWN_TIMEOUT`` from the
            environment, falling back to 10 s.
    """

    def __init__(self, drain_timeout: int | None = None) -> None:
        raw = os.getenv("REX_SHUTDOWN_TIMEOUT", "")
        self._drain_timeout: int = (
            drain_timeout
            if drain_timeout is not None
            else (int(raw) if raw.strip().isdigit() else _DEFAULT_DRAIN_TIMEOUT)
        )
        self._shutting_down = threading.Event()
        self._cleanups: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_shutting_down(self) -> bool:
        """``True`` once SIGTERM has been received."""
        return self._shutting_down.is_set()

    @property
    def drain_timeout(self) -> int:
        """Configured drain timeout in seconds."""
        return self._drain_timeout

    def register_cleanup(self, fn: Callable[[], None]) -> None:
        """Register a cleanup callback invoked during shutdown.

        Callbacks are called in registration order.  Exceptions are caught and
        logged so that subsequent callbacks still run.
        """
        self._cleanups.append(fn)

    def install(self) -> None:
        """Register the SIGTERM signal handler.

        Safe to call multiple times — subsequent calls merely replace the
        handler with an equivalent one.
        """
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        logger.info(
            "Graceful shutdown handler installed (drain_timeout=%ds)",
            self._drain_timeout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _handle_sigterm(self, signum: int, frame: object) -> None:  # noqa: ARG002
        """SIGTERM handler — starts drain-and-exit in a daemon thread."""
        logger.info(
            "SIGTERM received; entering drain mode (drain_timeout=%ds)",
            self._drain_timeout,
        )
        self._shutting_down.set()
        drain_thread = threading.Thread(
            target=self._drain_and_exit,
            daemon=True,
            name="graceful-shutdown-drain",
        )
        drain_thread.start()

    def _drain_and_exit(self) -> None:
        """Wait for in-flight requests, run cleanups, then exit 0."""
        logger.info(
            "Waiting up to %ds for in-flight requests to complete...",
            self._drain_timeout,
        )
        # Sleep for the drain window.  Real in-flight tracking would use a
        # semaphore; for the Flask dev server a timed pause is sufficient.
        threading.Event().wait(timeout=self._drain_timeout)

        logger.info("Drain window elapsed; running %d cleanup(s)", len(self._cleanups))
        for fn in self._cleanups:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                logger.error("Cleanup callback raised an error: %s", exc)

        logger.info("Graceful shutdown complete; exiting with code 0")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Testing helpers
    # ------------------------------------------------------------------

    def trigger_shutdown(self) -> None:
        """Set the shutting_down flag without spawning a drain thread.

        Intended for unit tests that want to verify the rejection behaviour
        without actually calling ``sys.exit``.
        """
        self._shutting_down.set()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_handler: GracefulShutdown | None = None


def get_shutdown_handler(drain_timeout: int | None = None) -> GracefulShutdown:
    """Return the module-level :class:`GracefulShutdown` singleton.

    Creates it on first call.  Subsequent calls ignore ``drain_timeout``
    and return the existing instance.
    """
    global _handler
    if _handler is None:
        _handler = GracefulShutdown(drain_timeout=drain_timeout)
    return _handler


def reset_shutdown_handler() -> None:
    """Reset the module-level singleton.

    For use in tests only — allows each test to get a fresh handler.
    """
    global _handler
    _handler = None


__all__ = [
    "GracefulShutdown",
    "get_shutdown_handler",
    "reset_shutdown_handler",
]
