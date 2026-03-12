"""Database connection pool for SQLite.

Provides a thread-safe connection pool with configurable min/max size,
acquisition timeout, idle connection timeout, and per-query execution
timeout.  All settings are readable from environment variables so they
can be adjusted without code changes.

Environment variables
---------------------
DB_POOL_MIN_SIZE        Minimum connections kept open (default: 1)
DB_POOL_MAX_SIZE        Maximum concurrent connections (default: 5)
DB_POOL_ACQUIRE_TIMEOUT Seconds to wait for an available connection (default: 5.0)
DB_POOL_IDLE_TIMEOUT    Seconds after which an idle connection is replaced (default: 300.0)
DB_QUERY_TIMEOUT        Seconds before a running query is interrupted (default: 10.0).
                        Set to -1 to disable.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Number of SQLite VM instructions between progress-handler calls.
_PROGRESS_HANDLER_INTERVAL = 100


class ConnectionPoolError(Exception):
    """Raised when a connection cannot be acquired from the pool."""


class QueryTimeoutError(Exception):
    """Raised when a query exceeds the configured execution timeout."""


@dataclass
class PoolConfig:
    """Connection pool configuration.

    Attributes:
        min_size: Minimum number of connections pre-created and kept open.
        max_size: Maximum number of concurrent connections.
        acquire_timeout: Seconds to wait for an available connection before raising.
        idle_timeout: Seconds of inactivity after which a connection is replaced.
        query_timeout: Seconds before a running query is interrupted and
            ``QueryTimeoutError`` is raised.  Set to ``-1.0`` to disable.
    """

    min_size: int = 1
    max_size: int = 5
    acquire_timeout: float = 5.0
    idle_timeout: float = 300.0
    query_timeout: float = 10.0

    @classmethod
    def from_env(cls) -> "PoolConfig":
        """Build a ``PoolConfig`` from environment variables.

        Unset or invalid values fall back to the dataclass defaults.
        """

        def _int(key: str, default: int) -> int:
            raw = os.environ.get(key)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError:
                logger.warning("Invalid value for %s=%r; using default %d", key, raw, default)
                return default

        def _float(key: str, default: float) -> float:
            raw = os.environ.get(key)
            if raw is None:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.warning("Invalid value for %s=%r; using default %.1f", key, raw, default)
                return default

        return cls(
            min_size=_int("DB_POOL_MIN_SIZE", 1),
            max_size=_int("DB_POOL_MAX_SIZE", 5),
            acquire_timeout=_float("DB_POOL_ACQUIRE_TIMEOUT", 5.0),
            idle_timeout=_float("DB_POOL_IDLE_TIMEOUT", 300.0),
            query_timeout=_float("DB_QUERY_TIMEOUT", 10.0),
        )


class _PooledConnection:
    """Internal wrapper tracking when a connection was last used."""

    __slots__ = ("conn", "last_used")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn
        self.last_used: float = time.monotonic()


class ConnectionPool:
    """Thread-safe SQLite connection pool.

    Args:
        db_path: Path to the SQLite database file.
        config:  Pool configuration.  Reads from environment variables when
                 not supplied (see module docstring).

    The pool pre-creates ``config.min_size`` connections on construction and
    logs all settings at INFO level so they are visible in production logs.

    Usage::

        pool = ConnectionPool(db_path)
        with pool.connect() as conn:
            conn.execute("SELECT 1")
    """

    def __init__(
        self,
        db_path: Path,
        config: PoolConfig | None = None,
    ) -> None:
        self._db_path = db_path
        self._config = config if config is not None else PoolConfig.from_env()
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._available: list[_PooledConnection] = []
        self._checked_out: int = 0

        # Pre-populate minimum connections.
        for _ in range(self._config.min_size):
            self._available.append(_PooledConnection(self._make_conn()))

        self._log_settings()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_conn(self) -> sqlite3.Connection:
        """Create a new SQLite connection configured for thread-safe reuse."""
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _log_settings(self) -> None:
        cfg = self._config
        logger.info(
            "Database connection pool: min_size=%d max_size=%d "
            "acquire_timeout=%.1fs idle_timeout=%.1fs path=%s",
            cfg.min_size,
            cfg.max_size,
            cfg.acquire_timeout,
            cfg.idle_timeout,
            self._db_path,
        )

    @property
    def _total(self) -> int:
        """Total connections in use (available + checked out). Must hold lock."""
        return len(self._available) + self._checked_out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self) -> sqlite3.Connection:
        """Acquire a connection from the pool.

        Returns:
            A ``sqlite3.Connection`` ready for use.

        Raises:
            ConnectionPoolError: If no connection can be acquired within
                ``config.acquire_timeout`` seconds.
        """
        deadline = time.monotonic() + self._config.acquire_timeout

        with self._condition:
            while True:
                # Try to reuse an available connection.
                if self._available:
                    pooled = self._available.pop()
                    idle_seconds = time.monotonic() - pooled.last_used
                    if idle_seconds > self._config.idle_timeout:
                        logger.debug(
                            "Replacing idle connection (idle=%.1fs > timeout=%.1fs)",
                            idle_seconds,
                            self._config.idle_timeout,
                        )
                        try:
                            pooled.conn.close()
                        except Exception:  # noqa: BLE001
                            pass
                        pooled = _PooledConnection(self._make_conn())
                    else:
                        pooled.last_used = time.monotonic()
                    self._checked_out += 1
                    return pooled.conn

                # Create a new connection if under the max limit.
                if self._total < self._config.max_size:
                    self._checked_out += 1
                    return self._make_conn()

                # Pool exhausted — wait for a release or timeout.
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ConnectionPoolError(
                        f"Could not acquire a database connection within "
                        f"{self._config.acquire_timeout:.1f}s "
                        f"(pool size: {self._config.max_size})"
                    )
                self._condition.wait(timeout=remaining)

                if time.monotonic() >= deadline:
                    raise ConnectionPoolError(
                        f"Could not acquire a database connection within "
                        f"{self._config.acquire_timeout:.1f}s "
                        f"(pool size: {self._config.max_size})"
                    )

    def release(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool.

        The connection is made available for future :meth:`acquire` calls.
        """
        with self._condition:
            self._checked_out -= 1
            pooled = _PooledConnection(conn)
            self._available.append(pooled)
            self._condition.notify()

    @contextmanager
    def connect(
        self, *, query_context: str = ""
    ) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that acquires, yields, and releases a connection.

        Commits on clean exit; rolls back on exception.  If
        ``config.query_timeout`` is positive, installs a SQLite progress
        handler that interrupts any query exceeding the timeout and raises
        :class:`QueryTimeoutError`.

        Args:
            query_context: Optional description of the operation (e.g.
                ``"fetch_notifications"``).  Logged on timeout; must NOT
                contain query parameters or PII.

        Raises:
            ConnectionPoolError: Propagated from :meth:`acquire`.
            QueryTimeoutError: If a query exceeds the configured timeout.
        """
        conn = self.acquire()
        timeout = self._config.query_timeout
        if timeout > 0:
            deadline = time.monotonic() + timeout

            def _progress_handler() -> int:
                return 1 if time.monotonic() > deadline else 0

            conn.set_progress_handler(_progress_handler, _PROGRESS_HANDLER_INTERVAL)
        try:
            yield conn
            conn.commit()
        except sqlite3.OperationalError as exc:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            if "interrupted" in str(exc).lower():
                ctx_msg = f" [context: {query_context}]" if query_context else ""
                logger.warning(
                    "Query timeout after %.1fs%s", timeout, ctx_msg
                )
                raise QueryTimeoutError(
                    f"Query exceeded timeout of {timeout:.1f}s"
                    + (f" ({query_context})" if query_context else "")
                ) from exc
            raise
        except Exception:
            try:
                conn.rollback()
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            if timeout > 0:
                conn.set_progress_handler(None, 0)
            self.release(conn)

    @property
    def size(self) -> int:
        """Current total number of open connections (available + checked out)."""
        with self._lock:
            return self._total

    @property
    def available(self) -> int:
        """Number of connections currently available (not checked out)."""
        with self._lock:
            return len(self._available)

    @property
    def checked_out(self) -> int:
        """Number of connections currently checked out."""
        with self._lock:
            return self._checked_out

    def close_all(self) -> None:
        """Close all available connections.  Should be called on shutdown."""
        with self._condition:
            while self._available:
                pooled = self._available.pop()
                try:
                    pooled.conn.close()
                except Exception:  # noqa: BLE001
                    pass


__all__ = [
    "ConnectionPool",
    "ConnectionPoolError",
    "PoolConfig",
    "QueryTimeoutError",
]
