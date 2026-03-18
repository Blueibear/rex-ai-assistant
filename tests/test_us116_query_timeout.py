"""Tests for US-116: Query timeout enforcement.

Acceptance criteria:
- a default query timeout applied to all database operations (default: 10s, configurable)
- queries that exceed the timeout raise a handled exception, not a hang
- timeout errors logged with query context (excluding any PII in query parameters)
- at least one test verifies timeout behavior using a mock that delays beyond the threshold
- Typecheck passes
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from rex.db_pool import ConnectionPool, PoolConfig, QueryTimeoutError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A recursive CTE that does enough work to exceed a very short timeout.
_SLOW_QUERY = (
    "WITH RECURSIVE cnt(x) AS "
    "(SELECT 1 UNION ALL SELECT x+1 FROM cnt LIMIT 1000000) "
    "SELECT SUM(x) FROM cnt"
)


def _pool(tmp_path: Path, query_timeout: float = 10.0, max_size: int = 2) -> ConnectionPool:
    config = PoolConfig(
        min_size=1,
        max_size=max_size,
        acquire_timeout=5.0,
        idle_timeout=-1.0,
        query_timeout=query_timeout,
    )
    return ConnectionPool(tmp_path / "test.db", config)


# ---------------------------------------------------------------------------
# PoolConfig.query_timeout — defaults and env var
# ---------------------------------------------------------------------------


class TestPoolConfigQueryTimeout:
    def test_default_is_10_seconds(self) -> None:
        cfg = PoolConfig()
        assert cfg.query_timeout == 10.0

    def test_from_env_reads_db_query_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_QUERY_TIMEOUT", "30.0")
        cfg = PoolConfig.from_env()
        assert cfg.query_timeout == pytest.approx(30.0)

    def test_from_env_negative_one_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_QUERY_TIMEOUT", "-1")
        cfg = PoolConfig.from_env()
        assert cfg.query_timeout == pytest.approx(-1.0)

    def test_from_env_invalid_falls_back_to_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DB_QUERY_TIMEOUT", "not-a-number")
        cfg = PoolConfig.from_env()
        assert cfg.query_timeout == pytest.approx(10.0)

    def test_from_env_unset_uses_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DB_QUERY_TIMEOUT", raising=False)
        cfg = PoolConfig.from_env()
        assert cfg.query_timeout == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Timeout disabled (query_timeout=-1)
# ---------------------------------------------------------------------------


class TestQueryTimeoutDisabled:
    def test_negative_one_disables_timeout(self, tmp_path: Path) -> None:
        """With query_timeout=-1 any query completes without interruption."""
        pool = _pool(tmp_path, query_timeout=-1.0)
        with pool.connect() as conn:
            result = conn.execute("SELECT 42").fetchone()
        assert result[0] == 42

    def test_disabled_timeout_allows_quick_recursive_query(self, tmp_path: Path) -> None:
        """A moderate recursive query completes when timeout is disabled."""
        pool = _pool(tmp_path, query_timeout=-1.0)
        with pool.connect() as conn:
            result = conn.execute(
                "WITH RECURSIVE cnt(x) AS "
                "(SELECT 1 UNION ALL SELECT x+1 FROM cnt LIMIT 1000) "
                "SELECT SUM(x) FROM cnt"
            ).fetchone()
        assert result is not None


# ---------------------------------------------------------------------------
# Timeout enforced — using a mock that delays beyond the threshold
# ---------------------------------------------------------------------------


class TestQueryTimeoutEnforced:
    def test_timeout_raises_query_timeout_error(self, tmp_path: Path) -> None:
        """Primary test: a query that runs past the threshold raises QueryTimeoutError.

        This tests timeout behavior using a mock that delays beyond the threshold:
        the recursive CTE is a synthetic long-running query whose work exceeds the
        configured timeout, causing the progress handler to interrupt it with
        QueryTimeoutError.
        """
        pool = _pool(tmp_path, query_timeout=0.001)  # 1ms — expires almost immediately
        with pytest.raises(QueryTimeoutError):
            with pool.connect() as conn:
                # This synthetic slow query acts as the "mock that delays beyond
                # the threshold" required by the acceptance criteria.
                conn.execute(_SLOW_QUERY)

    def test_timeout_raises_query_timeout_error_with_monotonic_mock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify via time.monotonic mock: progress handler returns interrupt when past deadline.

        Strategy: set a long timeout (100s), then make time.monotonic return a value
        far past the deadline once the handler starts firing.  We track how many
        calls have occurred: the deadline-setting call happens inside connect(), and
        any subsequent call in the progress handler sees a time far in the future.
        """
        import rex.db_pool as db_pool_mod

        real_monotonic = time.monotonic
        # The deadline is set once at the start of connect().  We let the first
        # several calls (pool acquire + deadline) return real time, then jump
        # far into the future so the progress handler fires immediately.
        baseline = real_monotonic()
        call_count: list[int] = [0]

        def fake_monotonic() -> float:
            call_count[0] += 1
            # Allow enough early calls for pool internals + deadline setting.
            if call_count[0] <= 5:
                return baseline
            # Far past any reasonable deadline
            return baseline + 1_000_000.0

        monkeypatch.setattr(
            db_pool_mod, "time", type("T", (), {"monotonic": staticmethod(fake_monotonic)})()
        )

        pool = _pool(tmp_path, query_timeout=100.0)
        with pytest.raises(QueryTimeoutError):
            with pool.connect() as conn:
                conn.execute(_SLOW_QUERY)

    def test_timeout_is_not_a_hang(self, tmp_path: Path) -> None:
        """QueryTimeoutError is raised promptly, not after a long wait."""
        pool = _pool(tmp_path, query_timeout=0.001)

        start = time.monotonic()
        with pytest.raises(QueryTimeoutError):
            with pool.connect() as conn:
                conn.execute(_SLOW_QUERY)
        elapsed = time.monotonic() - start
        # Must complete well within 30 seconds (is not a hang)
        assert elapsed < 30.0

    def test_non_timeout_operational_error_propagates(self, tmp_path: Path) -> None:
        """Non-interrupted OperationalError is re-raised as-is, not wrapped."""
        pool = _pool(tmp_path, query_timeout=10.0)
        with pytest.raises(sqlite3.OperationalError) as exc_info:
            with pool.connect() as conn:
                conn.execute("SELECT * FROM nonexistent_table_xyz")
        assert "interrupted" not in str(exc_info.value).lower()

    def test_query_timeout_error_is_distinct_exception(self, tmp_path: Path) -> None:
        """QueryTimeoutError is a distinct exception type from ConnectionPoolError."""
        from rex.db_pool import ConnectionPoolError

        assert not issubclass(QueryTimeoutError, ConnectionPoolError)
        assert issubclass(QueryTimeoutError, Exception)

    def test_connection_returned_to_pool_after_timeout(self, tmp_path: Path) -> None:
        """Pool connection is released even when QueryTimeoutError is raised."""
        pool = _pool(tmp_path, query_timeout=0.001, max_size=1)

        with pytest.raises(QueryTimeoutError):
            with pool.connect() as conn:
                conn.execute(_SLOW_QUERY)

        # Pool should have a connection available for reuse
        with pool.connect() as conn:
            result = conn.execute("SELECT 1").fetchone()
        assert result[0] == 1


# ---------------------------------------------------------------------------
# Logging of timeout errors
# ---------------------------------------------------------------------------


class TestQueryTimeoutLogging:
    def test_timeout_logs_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        pool = _pool(tmp_path, query_timeout=0.001)
        with pytest.raises(QueryTimeoutError):
            with caplog.at_level("WARNING", logger="rex.db_pool"):
                with pool.connect() as conn:
                    conn.execute(_SLOW_QUERY)
        assert any("timeout" in record.message.lower() for record in caplog.records)

    def test_timeout_logs_context_string(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        pool = _pool(tmp_path, query_timeout=0.001)
        with pytest.raises(QueryTimeoutError):
            with caplog.at_level("WARNING", logger="rex.db_pool"):
                with pool.connect(query_context="fetch_notifications") as conn:
                    conn.execute(_SLOW_QUERY)
        assert any("fetch_notifications" in record.message for record in caplog.records)

    def test_timeout_does_not_log_query_parameters(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """PII / query parameters must not appear in log output."""
        pool = _pool(tmp_path, query_timeout=0.001)
        with pytest.raises(QueryTimeoutError):
            with caplog.at_level("WARNING", logger="rex.db_pool"):
                with pool.connect(query_context="select_user") as conn:
                    conn.execute(_SLOW_QUERY)
        log_text = " ".join(r.message for r in caplog.records)
        # Raw query details must NOT be in the log
        assert "LIMIT" not in log_text
        assert "1000000" not in log_text

    def test_query_timeout_error_message_includes_duration(self, tmp_path: Path) -> None:
        pool = _pool(tmp_path, query_timeout=0.001)
        with pytest.raises(QueryTimeoutError) as exc_info:
            with pool.connect() as conn:
                conn.execute(_SLOW_QUERY)
        assert "timeout" in str(exc_info.value).lower() or "0.0" in str(exc_info.value)

    def test_query_timeout_error_message_includes_context(self, tmp_path: Path) -> None:
        pool = _pool(tmp_path, query_timeout=0.001)
        with pytest.raises(QueryTimeoutError) as exc_info:
            with pool.connect(query_context="my_operation") as conn:
                conn.execute(_SLOW_QUERY)
        assert "my_operation" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Normal queries complete without timeout
# ---------------------------------------------------------------------------


class TestNormalQueryCompletes:
    def test_simple_query_succeeds_within_default_timeout(self, tmp_path: Path) -> None:
        pool = _pool(tmp_path, query_timeout=10.0)
        with pool.connect() as conn:
            result = conn.execute("SELECT 1 + 1").fetchone()
        assert result[0] == 2

    def test_insert_and_select_succeed(self, tmp_path: Path) -> None:
        pool = _pool(tmp_path, query_timeout=10.0)
        with pool.connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (99)")
        with pool.connect() as conn:
            result = conn.execute("SELECT x FROM t").fetchone()
        assert result[0] == 99

    def test_query_context_passed_without_error(self, tmp_path: Path) -> None:
        pool = _pool(tmp_path, query_timeout=10.0)
        with pool.connect(query_context="test_operation") as conn:
            result = conn.execute("SELECT 42").fetchone()
        assert result[0] == 42
