"""Startup sequence enforcement for Rex Flask applications.

Enforces the required initialization order and logs each step at INFO so that
the log stream shows exactly where a failure occurred:

  Step 1/4 — Config validation      (environment variables are sane)
  Step 2/4 — Database connectivity  (database file is accessible)
  Step 3/4 — Migration state        (no pending migrations)
  Step 4/4 — Service initialization (caller registers blueprints, starts server)

If any step raises an exception or calls ``sys.exit``, subsequent steps do not
run.  The process exits with code 1 when a step fails.

Usage::

    from rex.startup import run_startup_sequence

    run_startup_sequence()          # must be called before Flask app setup
    app = Flask(__name__)           # step 4: caller completes service init
    # … register blueprints …
    logger.info("Startup complete — accepting traffic")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rex.audio.speaker_discovery import start_smart_speaker_discovery

logger = logging.getLogger(__name__)

_TOTAL_STEPS = 4


def _log_step(n: int, description: str) -> None:
    logger.info("Startup [step %d/%d]: %s", n, _TOTAL_STEPS, description)


def run_startup_sequence() -> None:
    """Run the startup sequence in order.

    Logs each step at INFO level.  Exits with code 1 if any step fails so
    that subsequent steps never execute.

    Raises:
        SystemExit: with code 1 if any startup step fails.
    """
    _run_config_validation()
    _run_database_connectivity()
    _run_migration_check()
    start_smart_speaker_discovery()


# ---------------------------------------------------------------------------
# Step 1: Config validation
# ---------------------------------------------------------------------------


def _run_config_validation() -> None:
    _log_step(1, "Config validation — verifying environment")
    try:
        _validate_env()
    except SystemExit:
        raise
    except Exception as exc:
        logger.critical("Config validation failed: %s", exc)
        sys.exit(1)


def _validate_env() -> None:
    """Check that the runtime environment is minimally sane.

    Currently a lightweight check.  Add mandatory variable assertions here as
    requirements grow.  Logs warnings for advisory-only issues.
    """
    import os  # noqa: PLC0415

    proxy_token = os.environ.get("REX_PROXY_TOKEN", "")
    if not proxy_token:
        logger.warning(
            "REX_PROXY_TOKEN is not set — bearer-token auth is disabled. "
            "Set this variable in production."
        )
    logger.debug("Config validation passed")


# ---------------------------------------------------------------------------
# Step 2: Database connectivity
# ---------------------------------------------------------------------------


def _run_database_connectivity() -> None:
    _log_step(2, "Database connectivity — verifying database is accessible")
    try:
        _check_database_connectivity()
    except SystemExit:
        raise
    except Exception as exc:
        logger.critical("Database connectivity check failed: %s", exc)
        sys.exit(1)


def _check_database_connectivity() -> None:
    """Probe the SQLite database for basic read/write access.

    Uses the same default path as the migrations module so that the check is
    consistent with the migration state step.
    """
    import sqlite3  # noqa: PLC0415

    db_path = Path("data/dashboard_notifications.db")

    # Ensure the data directory exists (non-fatal; SQLite will create the file)
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create data directory: %s", exc)

    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        conn.execute("SELECT 1")
        conn.close()
        logger.debug("Database connectivity OK: %s", db_path)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Cannot connect to database {db_path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Step 3: Migration state
# ---------------------------------------------------------------------------


def _run_migration_check() -> None:
    _log_step(3, "Migration state — checking for pending database migrations")
    try:
        from rex.migrations import validate_migration_state  # noqa: PLC0415

        validate_migration_state()
        logger.debug("Migration state check passed")
    except SystemExit:
        raise
    except Exception as exc:
        logger.critical("Migration state check failed: %s", exc)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 4 helper — called by the application entry point
# ---------------------------------------------------------------------------


def log_service_ready() -> None:
    """Log step 4 after Flask setup and blueprint registration are complete."""
    _log_step(4, "Service initialization complete — ready to accept traffic")


__all__ = [
    "run_startup_sequence",
    "log_service_ready",
]
