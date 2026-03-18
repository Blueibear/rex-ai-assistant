"""Tests for US-127: Service startup sequence and dependency ordering.

Verifies that:
- startup sequence enforces order: config validation → database connectivity
  → migration check → service initialization
- if any step fails, subsequent steps do not run
- each step is logged at INFO level so the log stream shows exactly where
  a failure occurred
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_startup():
    """Import (or reimport) rex.startup so module-level state is fresh."""
    import importlib

    import rex.startup as mod

    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Logging capture fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def captured_logs(
    caplog: pytest.LogCaptureFixture,
) -> Generator[pytest.LogCaptureFixture, None, None]:
    with caplog.at_level(logging.DEBUG, logger="rex.startup"):
        yield caplog


# ---------------------------------------------------------------------------
# Tests: module importability
# ---------------------------------------------------------------------------


class TestStartupModuleImport:
    def test_run_startup_sequence_exported(self) -> None:
        from rex.startup import run_startup_sequence  # noqa: F401

        assert callable(run_startup_sequence)

    def test_log_service_ready_exported(self) -> None:
        from rex.startup import log_service_ready  # noqa: F401

        assert callable(log_service_ready)

    def test_all_exports(self) -> None:
        from rex.startup import __all__

        assert "run_startup_sequence" in __all__
        assert "log_service_ready" in __all__


# ---------------------------------------------------------------------------
# Tests: step logging at INFO level
# ---------------------------------------------------------------------------


class TestStartupLogging:
    def test_config_validation_logged_at_info(
        self, captured_logs: pytest.LogCaptureFixture
    ) -> None:
        from rex.startup import _run_config_validation

        _run_config_validation()

        info_msgs = [r.message for r in captured_logs.records if r.levelno == logging.INFO]
        assert any(
            "step 1" in m.lower() or "config" in m.lower() for m in info_msgs
        ), f"Expected INFO log for config validation step. Got: {info_msgs}"

    def test_database_connectivity_logged_at_info(
        self, captured_logs: pytest.LogCaptureFixture, tmp_path: Path
    ) -> None:
        from rex.startup import _run_database_connectivity

        with patch("rex.startup._check_database_connectivity"):
            _run_database_connectivity()

        info_msgs = [r.message for r in captured_logs.records if r.levelno == logging.INFO]
        assert any(
            "step 2" in m.lower() or "database" in m.lower() for m in info_msgs
        ), f"Expected INFO log for database step. Got: {info_msgs}"

    def test_migration_check_logged_at_info(self, captured_logs: pytest.LogCaptureFixture) -> None:
        from rex.startup import _run_migration_check

        with patch("rex.startup._run_migration_check") as mock_step:
            mock_step.return_value = None
            mock_step()

        # Call the real function with migration check mocked
        with patch("rex.migrations.validate_migration_state"):
            from rex.startup import _run_migration_check as real_fn  # noqa: F401

            _run_migration_check()

        info_msgs = [r.message for r in captured_logs.records if r.levelno == logging.INFO]
        assert any(
            "step 3" in m.lower() or "migration" in m.lower() for m in info_msgs
        ), f"Expected INFO log for migration step. Got: {info_msgs}"

    def test_log_service_ready_logged_at_info(
        self, captured_logs: pytest.LogCaptureFixture
    ) -> None:
        from rex.startup import log_service_ready

        log_service_ready()

        info_msgs = [r.message for r in captured_logs.records if r.levelno == logging.INFO]
        assert any(
            "step 4" in m.lower() or "ready" in m.lower() or "traffic" in m.lower()
            for m in info_msgs
        ), f"Expected INFO log for service-ready step. Got: {info_msgs}"

    def test_each_step_includes_step_number(self, captured_logs: pytest.LogCaptureFixture) -> None:
        """All four steps must appear in the log with step N/4 notation."""
        from rex.startup import log_service_ready, run_startup_sequence

        with (
            patch("rex.startup._check_database_connectivity"),
            patch("rex.migrations.validate_migration_state"),
        ):
            run_startup_sequence()
            log_service_ready()

        info_msgs = [r.message for r in captured_logs.records if r.levelno == logging.INFO]
        combined = " ".join(info_msgs).lower()
        for n in (1, 2, 3, 4):
            assert (
                f"step {n}" in combined or f"{n}/4" in combined
            ), f"Step {n} not logged at INFO. Messages: {info_msgs}"


# ---------------------------------------------------------------------------
# Tests: dependency ordering — successful path
# ---------------------------------------------------------------------------


class TestStartupOrder:
    def test_run_startup_sequence_calls_all_three_steps(self) -> None:
        from rex.startup import run_startup_sequence

        call_order: list[str] = []

        def fake_config() -> None:
            call_order.append("config")

        def fake_db() -> None:
            call_order.append("db")

        def fake_migration() -> None:
            call_order.append("migration")

        with (
            patch("rex.startup._run_config_validation", side_effect=fake_config),
            patch("rex.startup._run_database_connectivity", side_effect=fake_db),
            patch("rex.startup._run_migration_check", side_effect=fake_migration),
        ):
            run_startup_sequence()

        assert call_order == [
            "config",
            "db",
            "migration",
        ], f"Expected config → db → migration, got {call_order}"

    def test_config_runs_before_database(self) -> None:
        from rex.startup import run_startup_sequence

        call_order: list[str] = []

        with (
            patch(
                "rex.startup._run_config_validation",
                side_effect=lambda: call_order.append("config"),
            ),
            patch(
                "rex.startup._run_database_connectivity",
                side_effect=lambda: call_order.append("db"),
            ),
            patch(
                "rex.startup._run_migration_check",
                side_effect=lambda: call_order.append("migration"),
            ),
        ):
            run_startup_sequence()

        assert call_order.index("config") < call_order.index(
            "db"
        ), "Config must run before database"

    def test_database_runs_before_migration(self) -> None:
        from rex.startup import run_startup_sequence

        call_order: list[str] = []

        with (
            patch(
                "rex.startup._run_config_validation",
                side_effect=lambda: call_order.append("config"),
            ),
            patch(
                "rex.startup._run_database_connectivity",
                side_effect=lambda: call_order.append("db"),
            ),
            patch(
                "rex.startup._run_migration_check",
                side_effect=lambda: call_order.append("migration"),
            ),
        ):
            run_startup_sequence()

        assert call_order.index("db") < call_order.index(
            "migration"
        ), "Database must run before migration"


# ---------------------------------------------------------------------------
# Tests: failure halts subsequent steps
# ---------------------------------------------------------------------------


class TestStartupFailureHalts:
    def test_config_failure_prevents_database_step(self) -> None:
        """If config validation exits, database step must not run."""
        from rex.startup import run_startup_sequence

        db_called = []

        with (
            patch("rex.startup._run_config_validation", side_effect=SystemExit(1)),
            patch(
                "rex.startup._run_database_connectivity", side_effect=lambda: db_called.append(True)
            ),
            patch("rex.startup._run_migration_check"),
        ):
            with pytest.raises(SystemExit):
                run_startup_sequence()

        assert not db_called, "Database step must not run when config validation fails"

    def test_config_failure_prevents_migration_step(self) -> None:
        """If config validation exits, migration step must not run."""
        from rex.startup import run_startup_sequence

        migration_called = []

        with (
            patch("rex.startup._run_config_validation", side_effect=SystemExit(1)),
            patch("rex.startup._run_database_connectivity"),
            patch(
                "rex.startup._run_migration_check",
                side_effect=lambda: migration_called.append(True),
            ),
        ):
            with pytest.raises(SystemExit):
                run_startup_sequence()

        assert not migration_called, "Migration step must not run when config validation fails"

    def test_database_failure_prevents_migration_step(self) -> None:
        """If database connectivity exits, migration step must not run."""
        from rex.startup import run_startup_sequence

        migration_called = []

        with (
            patch("rex.startup._run_config_validation"),
            patch("rex.startup._run_database_connectivity", side_effect=SystemExit(1)),
            patch(
                "rex.startup._run_migration_check",
                side_effect=lambda: migration_called.append(True),
            ),
        ):
            with pytest.raises(SystemExit):
                run_startup_sequence()

        assert not migration_called, "Migration step must not run when database connectivity fails"

    def test_database_exception_exits_with_code_1(self) -> None:
        """An unhandled exception in the database step must cause sys.exit(1)."""
        from rex.startup import _run_database_connectivity

        with patch("rex.startup._check_database_connectivity", side_effect=RuntimeError("db down")):
            with pytest.raises(SystemExit) as exc_info:
                _run_database_connectivity()

        assert exc_info.value.code == 1

    def test_config_exception_exits_with_code_1(self) -> None:
        """An unhandled exception in config validation must cause sys.exit(1)."""
        from rex.startup import _run_config_validation

        with patch("rex.startup._validate_env", side_effect=RuntimeError("bad env")):
            with pytest.raises(SystemExit) as exc_info:
                _run_config_validation()

        assert exc_info.value.code == 1

    def test_migration_exception_exits_with_code_1(self) -> None:
        """An unhandled exception in the migration step must cause sys.exit(1)."""
        from rex.startup import _run_migration_check

        with patch(
            "rex.migrations.validate_migration_state", side_effect=RuntimeError("migration fail")
        ):
            with pytest.raises(SystemExit) as exc_info:
                _run_migration_check()

        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Tests: happy-path integration (mocked externals)
# ---------------------------------------------------------------------------


class TestStartupHappyPath:
    def test_run_startup_sequence_succeeds(self) -> None:
        """run_startup_sequence() must return normally when all steps pass."""
        from rex.startup import run_startup_sequence

        with (
            patch("rex.startup._check_database_connectivity"),
            patch("rex.migrations.validate_migration_state"),
        ):
            run_startup_sequence()  # must not raise

    def test_log_service_ready_does_not_raise(self) -> None:
        from rex.startup import log_service_ready

        log_service_ready()  # must not raise


# ---------------------------------------------------------------------------
# Tests: flask_proxy uses startup sequence
# ---------------------------------------------------------------------------


class TestFlaskProxyUsesStartup:
    def test_flask_proxy_imports_startup_functions(self) -> None:
        """flask_proxy.py must import run_startup_sequence and log_service_ready."""
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "flask_proxy.py").read_text()
        tree = ast.parse(source)

        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "rex.startup":
                for alias in node.names:
                    imported_names.add(alias.name)

        assert (
            "run_startup_sequence" in imported_names
        ), "flask_proxy.py must import run_startup_sequence from rex.startup"
        assert (
            "log_service_ready" in imported_names
        ), "flask_proxy.py must import log_service_ready from rex.startup"

    def test_flask_proxy_calls_run_startup_sequence(self) -> None:
        """flask_proxy.py must call run_startup_sequence() at module level."""
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "flask_proxy.py").read_text()
        tree = ast.parse(source)

        call_found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "run_startup_sequence"
            ):
                call_found = True
                break

        assert call_found, "flask_proxy.py must call run_startup_sequence() at module level"

    def test_flask_proxy_calls_log_service_ready(self) -> None:
        """flask_proxy.py must call log_service_ready() at module level."""
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "flask_proxy.py").read_text()
        tree = ast.parse(source)

        call_found = False
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "log_service_ready"
            ):
                call_found = True
                break

        assert call_found, "flask_proxy.py must call log_service_ready() at module level"

    def test_flask_proxy_does_not_call_validate_migration_state_directly(self) -> None:
        """flask_proxy.py must not call validate_migration_state() directly (use startup module)."""
        import ast
        from pathlib import Path

        source = (Path(__file__).resolve().parents[1] / "flask_proxy.py").read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "validate_migration_state"
            ):
                pytest.fail(
                    "flask_proxy.py calls validate_migration_state() directly; "
                    "use run_startup_sequence() instead"
                )
