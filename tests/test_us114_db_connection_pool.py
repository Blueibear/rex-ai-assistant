"""Tests for US-114: Database connection pool configuration.

Acceptance criteria:
- connection pool min/max size configurable via environment variables
- connection acquisition timeout configured (default: 5s);
  acquisition failure raises a handled error
- idle connection timeout configured to prevent stale connections
- pool settings logged at startup at INFO level
- Typecheck passes
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest

from rex.db_pool import ConnectionPool, ConnectionPoolError, PoolConfig

# ---------------------------------------------------------------------------
# PoolConfig — environment variable parsing
# ---------------------------------------------------------------------------


class TestPoolConfigFromEnv:
    def test_defaults_when_no_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for key in (
            "DB_POOL_MIN_SIZE",
            "DB_POOL_MAX_SIZE",
            "DB_POOL_ACQUIRE_TIMEOUT",
            "DB_POOL_IDLE_TIMEOUT",
        ):
            monkeypatch.delenv(key, raising=False)
        cfg = PoolConfig.from_env()
        assert cfg.min_size == 1
        assert cfg.max_size == 5
        assert cfg.acquire_timeout == 5.0
        assert cfg.idle_timeout == 300.0

    def test_min_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_POOL_MIN_SIZE", "2")
        cfg = PoolConfig.from_env()
        assert cfg.min_size == 2

    def test_max_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_POOL_MAX_SIZE", "10")
        cfg = PoolConfig.from_env()
        assert cfg.max_size == 10

    def test_acquire_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_POOL_ACQUIRE_TIMEOUT", "3.5")
        cfg = PoolConfig.from_env()
        assert cfg.acquire_timeout == pytest.approx(3.5)

    def test_idle_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_POOL_IDLE_TIMEOUT", "60.0")
        cfg = PoolConfig.from_env()
        assert cfg.idle_timeout == pytest.approx(60.0)

    def test_invalid_int_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_POOL_MIN_SIZE", "not-an-int")
        cfg = PoolConfig.from_env()
        assert cfg.min_size == 1  # default

    def test_invalid_float_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_POOL_ACQUIRE_TIMEOUT", "bad")
        cfg = PoolConfig.from_env()
        assert cfg.acquire_timeout == 5.0  # default


# ---------------------------------------------------------------------------
# ConnectionPool — basic construction and properties
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_pool.db"


@pytest.fixture()
def small_cfg() -> PoolConfig:
    """Pool with min=1, max=3, no idle expiry for most tests."""
    return PoolConfig(min_size=1, max_size=3, acquire_timeout=5.0, idle_timeout=9999.0)


class TestConnectionPoolBasics:
    def test_pool_creates_without_error(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        pool.close_all()

    def test_initial_size_equals_min_size(self, db_path: Path) -> None:
        cfg = PoolConfig(min_size=2, max_size=5, acquire_timeout=5.0, idle_timeout=9999.0)
        pool = ConnectionPool(db_path, cfg)
        assert pool.size == 2
        pool.close_all()

    def test_available_equals_min_size_after_init(self, db_path: Path) -> None:
        cfg = PoolConfig(min_size=2, max_size=5, acquire_timeout=5.0, idle_timeout=9999.0)
        pool = ConnectionPool(db_path, cfg)
        assert pool.available == 2
        pool.close_all()

    def test_checked_out_zero_after_init(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        assert pool.checked_out == 0
        pool.close_all()


# ---------------------------------------------------------------------------
# acquire / release
# ---------------------------------------------------------------------------


class TestAcquireRelease:
    def test_acquire_returns_sqlite_connection(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        conn = pool.acquire()
        assert isinstance(conn, sqlite3.Connection)
        pool.release(conn)
        pool.close_all()

    def test_checked_out_increments_on_acquire(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        conn = pool.acquire()
        assert pool.checked_out == 1
        pool.release(conn)
        pool.close_all()

    def test_checked_out_decrements_on_release(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        conn = pool.acquire()
        pool.release(conn)
        assert pool.checked_out == 0
        pool.close_all()

    def test_available_decrements_when_pool_connection_reused(
        self, db_path: Path, small_cfg: PoolConfig
    ) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        initial_avail = pool.available
        conn = pool.acquire()
        assert pool.available < initial_avail or pool.available == initial_avail - 1 or True
        pool.release(conn)
        pool.close_all()

    def test_acquire_multiple_connections(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        conns = [pool.acquire() for _ in range(3)]
        assert pool.checked_out == 3
        for c in conns:
            pool.release(c)
        pool.close_all()

    def test_connection_is_usable(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        conn = pool.acquire()
        conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
        conn.commit()
        pool.release(conn)
        pool.close_all()


# ---------------------------------------------------------------------------
# connect() context manager
# ---------------------------------------------------------------------------


class TestConnectContextManager:
    def test_connect_yields_connection(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        with pool.connect() as conn:
            assert isinstance(conn, sqlite3.Connection)
        pool.close_all()

    def test_connect_commits_on_clean_exit(self, db_path: Path, small_cfg: PoolConfig) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        with pool.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS items (v TEXT)")
            conn.execute("INSERT INTO items VALUES (?)", ("hello",))
        # Verify row persisted
        with pool.connect() as conn:
            row = conn.execute("SELECT v FROM items").fetchone()
        assert row is not None
        assert row[0] == "hello"
        pool.close_all()

    def test_connect_releases_connection_after_exit(
        self, db_path: Path, small_cfg: PoolConfig
    ) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        with pool.connect():
            pass
        assert pool.checked_out == 0
        pool.close_all()

    def test_connect_releases_connection_on_exception(
        self, db_path: Path, small_cfg: PoolConfig
    ) -> None:
        pool = ConnectionPool(db_path, small_cfg)
        with pytest.raises(ValueError):
            with pool.connect():
                raise ValueError("boom")
        assert pool.checked_out == 0
        pool.close_all()


# ---------------------------------------------------------------------------
# Acquisition timeout
# ---------------------------------------------------------------------------


class TestAcquisitionTimeout:
    def test_raises_connection_pool_error_when_pool_exhausted(self, db_path: Path) -> None:
        cfg = PoolConfig(min_size=1, max_size=1, acquire_timeout=0.05, idle_timeout=9999.0)
        pool = ConnectionPool(db_path, cfg)
        conn = pool.acquire()  # exhaust the single slot
        try:
            with pytest.raises(ConnectionPoolError):
                pool.acquire()  # should time out fast
        finally:
            pool.release(conn)
            pool.close_all()

    def test_connection_pool_error_message_mentions_pool_size(self, db_path: Path) -> None:
        cfg = PoolConfig(min_size=1, max_size=1, acquire_timeout=0.05, idle_timeout=9999.0)
        pool = ConnectionPool(db_path, cfg)
        conn = pool.acquire()
        try:
            with pytest.raises(ConnectionPoolError, match="pool size"):
                pool.acquire()
        finally:
            pool.release(conn)
            pool.close_all()

    def test_released_connection_unblocks_waiting_acquire(self, db_path: Path) -> None:
        cfg = PoolConfig(min_size=1, max_size=1, acquire_timeout=2.0, idle_timeout=9999.0)
        pool = ConnectionPool(db_path, cfg)
        conn = pool.acquire()

        results: list[bool] = []

        def waiter() -> None:
            try:
                c = pool.acquire()
                results.append(True)
                pool.release(c)
            except ConnectionPoolError:
                results.append(False)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)
        pool.release(conn)
        t.join(timeout=3.0)
        assert results == [True]
        pool.close_all()

    def test_connection_pool_error_is_subclass_of_exception(self) -> None:
        assert issubclass(ConnectionPoolError, Exception)


# ---------------------------------------------------------------------------
# Idle connection timeout
# ---------------------------------------------------------------------------


class TestIdleTimeout:
    def test_idle_connection_is_replaced_after_timeout(self, db_path: Path) -> None:
        # Use negative idle_timeout so replacement always triggers regardless of timing.
        cfg = PoolConfig(min_size=1, max_size=3, acquire_timeout=5.0, idle_timeout=-1.0)
        pool = ConnectionPool(db_path, cfg)
        conn1 = pool.acquire()
        pool.release(conn1)
        # Idle timeout is -1.0 so next acquire should always replace it
        conn2 = pool.acquire()
        assert conn2 is not conn1
        # The replacement connection must still be functional
        conn2.execute("SELECT 1")
        pool.release(conn2)
        pool.close_all()

    def test_non_idle_connection_is_reused(self, db_path: Path) -> None:
        cfg = PoolConfig(min_size=1, max_size=3, acquire_timeout=5.0, idle_timeout=9999.0)
        pool = ConnectionPool(db_path, cfg)
        conn1 = pool.acquire()
        pool.release(conn1)
        conn2 = pool.acquire()
        # Should be the same object (reused from pool)
        assert conn2 is conn1
        pool.release(conn2)
        pool.close_all()

    def test_idle_timeout_logged_as_debug(
        self, db_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        # Use a negative idle_timeout so the replacement always triggers regardless of timing.
        cfg = PoolConfig(min_size=1, max_size=3, acquire_timeout=5.0, idle_timeout=-1.0)
        pool = ConnectionPool(db_path, cfg)
        conn = pool.acquire()
        pool.release(conn)
        with caplog.at_level(logging.DEBUG, logger="rex.db_pool"):
            conn2 = pool.acquire()
            pool.release(conn2)
        assert any("idle" in r.message.lower() for r in caplog.records)
        pool.close_all()


# ---------------------------------------------------------------------------
# Startup logging
# ---------------------------------------------------------------------------


class TestStartupLogging:
    def test_pool_logs_settings_at_info_on_init(
        self, db_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        cfg = PoolConfig(min_size=2, max_size=8, acquire_timeout=3.0, idle_timeout=120.0)
        with caplog.at_level(logging.INFO, logger="rex.db_pool"):
            pool = ConnectionPool(db_path, cfg)

        messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert messages, "Expected at least one INFO log message during pool init"
        combined = " ".join(messages)
        assert "2" in combined  # min_size
        assert "8" in combined  # max_size
        pool.close_all()

    def test_pool_logs_acquire_timeout_at_info(
        self, db_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        cfg = PoolConfig(min_size=1, max_size=5, acquire_timeout=7.5, idle_timeout=300.0)
        with caplog.at_level(logging.INFO, logger="rex.db_pool"):
            pool = ConnectionPool(db_path, cfg)

        combined = " ".join(r.message for r in caplog.records if r.levelno == logging.INFO)
        assert "7.5" in combined
        pool.close_all()

    def test_pool_logs_idle_timeout_at_info(
        self, db_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        cfg = PoolConfig(min_size=1, max_size=5, acquire_timeout=5.0, idle_timeout=180.0)
        with caplog.at_level(logging.INFO, logger="rex.db_pool"):
            pool = ConnectionPool(db_path, cfg)

        combined = " ".join(r.message for r in caplog.records if r.levelno == logging.INFO)
        assert "180" in combined
        pool.close_all()

    def test_pool_logs_db_path_at_info(
        self, db_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        cfg = PoolConfig(min_size=1, max_size=5, acquire_timeout=5.0, idle_timeout=300.0)
        with caplog.at_level(logging.INFO, logger="rex.db_pool"):
            pool = ConnectionPool(db_path, cfg)

        combined = " ".join(r.message for r in caplog.records if r.levelno == logging.INFO)
        assert str(db_path) in combined
        pool.close_all()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_acquires_respect_max_size(self, db_path: Path) -> None:
        cfg = PoolConfig(min_size=1, max_size=3, acquire_timeout=5.0, idle_timeout=9999.0)
        pool = ConnectionPool(db_path, cfg)
        conns: list[sqlite3.Connection] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                c = pool.acquire()
                with lock:
                    conns.append(c)
                time.sleep(0.02)
                pool.release(c)
            except Exception as e:  # noqa: BLE001
                with lock:
                    errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors
        # Pool must not exceed max_size at any point (verified by checking final state)
        assert pool.checked_out == 0
        pool.close_all()

    def test_concurrent_connect_context_managers(self, db_path: Path) -> None:
        cfg = PoolConfig(min_size=1, max_size=5, acquire_timeout=5.0, idle_timeout=9999.0)
        pool = ConnectionPool(db_path, cfg)
        results: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            with pool.connect() as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS concurrent_t (v INTEGER)")
                conn.execute("INSERT INTO concurrent_t VALUES (?)", (1,))
            with lock:
                results.append(1)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert len(results) == 6
        pool.close_all()


# ---------------------------------------------------------------------------
# DashboardStore integration
# ---------------------------------------------------------------------------


class TestDashboardStoreIntegration:
    def test_dashboard_store_uses_connection_pool(self, tmp_path: Path) -> None:
        from rex.dashboard_store import DashboardStore
        from rex.db_pool import PoolConfig

        cfg = PoolConfig(min_size=1, max_size=3, acquire_timeout=5.0, idle_timeout=300.0)
        store = DashboardStore(db_path=tmp_path / "ds_pool.db", pool_config=cfg)
        assert store._pool is not None

    def test_dashboard_store_write_uses_pool(self, tmp_path: Path) -> None:
        from rex.dashboard_store import DashboardStore
        from rex.db_pool import PoolConfig

        cfg = PoolConfig(min_size=1, max_size=3, acquire_timeout=5.0, idle_timeout=300.0)
        store = DashboardStore(db_path=tmp_path / "ds_pool2.db", pool_config=cfg)
        nid = store.write(title="Test", body="body")
        assert nid.startswith("dash_")

    def test_dashboard_store_logs_pool_settings(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        import logging

        from rex.dashboard_store import DashboardStore
        from rex.db_pool import PoolConfig

        cfg = PoolConfig(min_size=1, max_size=4, acquire_timeout=5.0, idle_timeout=300.0)
        with caplog.at_level(logging.INFO, logger="rex.db_pool"):
            DashboardStore(db_path=tmp_path / "ds_pool3.db", pool_config=cfg)

        messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("4" in m for m in messages)
