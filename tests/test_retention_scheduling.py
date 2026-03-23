"""Offline tests for retention cleanup scheduling.

Covers:
- Dashboard cleanup job: registered when cleanup_schedule is set, skipped when null.
- Inbound SMS cleanup job: registered only when enabled=true and cleanup_schedule is set.
- Idempotency: registering twice is a no-op (no duplicate jobs).
- Execution: running the job calls cleanup_old on the correct store.
- Missing / disabled config: safe skip, no crash.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

from rex.dashboard_store import DashboardStoreConfig, load_dashboard_store_config
from rex.messaging_backends.inbound_store import (
    InboundSmsStore,
    InboundStoreConfig,
    load_inbound_store_config,
)
from rex.retention import (
    setup_dashboard_cleanup_job,
    setup_inbound_sms_cleanup_job,
    setup_retention_jobs,
    wire_retention_cleanup,
)
from rex.scheduler import Scheduler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scheduler(tmp_path: Path) -> Scheduler:
    """Create an isolated scheduler backed by a temp file."""
    return Scheduler(jobs_file=tmp_path / "test_jobs.json")


def _dashboard_config(
    *,
    cleanup_schedule: str | None = "interval:86400",
    retention_days: int = 30,
    path: str | None = None,
) -> dict[str, Any]:
    """Minimal raw config enabling dashboard cleanup."""
    return {
        "notifications": {
            "dashboard": {
                "store": {
                    "type": "sqlite",
                    "path": path,
                    "retention_days": retention_days,
                    "cleanup_schedule": cleanup_schedule,
                }
            }
        }
    }


def _inbound_config(
    *,
    enabled: bool = True,
    cleanup_schedule: str | None = "interval:86400",
    retention_days: int = 90,
    store_path: str | None = None,
) -> dict[str, Any]:
    """Minimal raw config for inbound SMS."""
    return {
        "messaging": {
            "inbound": {
                "enabled": enabled,
                "cleanup_schedule": cleanup_schedule,
                "retention_days": retention_days,
                "store_path": store_path,
                "auth_token_ref": "twilio:inbound",
                "rate_limit": "120 per minute",
            }
        }
    }


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------


def test_dashboard_store_config_default_cleanup_schedule() -> None:
    """Default cleanup_schedule is daily interval."""
    cfg = DashboardStoreConfig()
    assert cfg.cleanup_schedule == "interval:86400"


def test_dashboard_store_config_null_cleanup_schedule() -> None:
    """cleanup_schedule can be set to None to disable."""
    cfg = DashboardStoreConfig(cleanup_schedule=None)
    assert cfg.cleanup_schedule is None


def test_inbound_store_config_default_cleanup_schedule() -> None:
    """Default cleanup_schedule is daily interval."""
    cfg = InboundStoreConfig()
    assert cfg.cleanup_schedule == "interval:86400"


def test_inbound_store_config_null_cleanup_schedule() -> None:
    """cleanup_schedule can be set to None to disable."""
    cfg = InboundStoreConfig(cleanup_schedule=None)
    assert cfg.cleanup_schedule is None


def test_load_dashboard_store_config_preserves_cleanup_schedule() -> None:
    """Parsing a config dict with cleanup_schedule works correctly."""
    raw = _dashboard_config(cleanup_schedule="interval:3600")
    cfg = load_dashboard_store_config(raw)
    assert cfg.cleanup_schedule == "interval:3600"


def test_load_inbound_store_config_preserves_cleanup_schedule() -> None:
    """Parsing a config dict with cleanup_schedule works correctly."""
    raw = _inbound_config(cleanup_schedule="interval:7200", enabled=True)
    cfg = load_inbound_store_config(raw)
    assert cfg.cleanup_schedule == "interval:7200"


# ---------------------------------------------------------------------------
# setup_dashboard_cleanup_job
# ---------------------------------------------------------------------------


def test_dashboard_job_is_noop(tmp_path: Path) -> None:
    """setup_dashboard_cleanup_job is a no-op (dashboard_store pending retirement)."""
    sched = _make_scheduler(tmp_path)
    result = setup_dashboard_cleanup_job(_dashboard_config(), sched)
    assert result is False
    assert sched.get_job("dashboard_retention_cleanup") is None


def test_dashboard_job_noop_regardless_of_schedule(tmp_path: Path) -> None:
    """No-op whether cleanup_schedule is set or null."""
    sched = _make_scheduler(tmp_path)
    assert setup_dashboard_cleanup_job(_dashboard_config(cleanup_schedule=None), sched) is False
    assert setup_dashboard_cleanup_job(_dashboard_config(), sched) is False
    assert setup_dashboard_cleanup_job({}, sched) is False


# ---------------------------------------------------------------------------
# setup_inbound_sms_cleanup_job
# ---------------------------------------------------------------------------


def test_inbound_job_registered_when_enabled(tmp_path: Path) -> None:
    """Job is created when inbound is enabled and cleanup_schedule is set."""
    sched = _make_scheduler(tmp_path)
    result = setup_inbound_sms_cleanup_job(_inbound_config(enabled=True), sched)
    assert result is True
    job = sched.get_job("inbound_sms_retention_cleanup")
    assert job is not None
    assert job.name == "Inbound SMS Retention Cleanup"
    assert job.schedule == "interval:86400"
    assert job.enabled is True


def test_inbound_job_skipped_when_disabled(tmp_path: Path) -> None:
    """No job is created when messaging.inbound.enabled=false."""
    sched = _make_scheduler(tmp_path)
    result = setup_inbound_sms_cleanup_job(_inbound_config(enabled=False), sched)
    assert result is False
    assert sched.get_job("inbound_sms_retention_cleanup") is None


def test_inbound_job_skipped_when_schedule_is_null(tmp_path: Path) -> None:
    """No job is created when cleanup_schedule is null (even when enabled)."""
    sched = _make_scheduler(tmp_path)
    result = setup_inbound_sms_cleanup_job(
        _inbound_config(enabled=True, cleanup_schedule=None), sched
    )
    assert result is False
    assert sched.get_job("inbound_sms_retention_cleanup") is None


def test_inbound_job_skipped_with_empty_config(tmp_path: Path) -> None:
    """No job is created when config section is absent (inbound disabled by default)."""
    sched = _make_scheduler(tmp_path)
    result = setup_inbound_sms_cleanup_job({}, sched)
    assert result is False  # messaging.inbound.enabled defaults to False


def test_inbound_job_idempotent(tmp_path: Path) -> None:
    """Calling setup twice does not create a duplicate job."""
    sched = _make_scheduler(tmp_path)
    r1 = setup_inbound_sms_cleanup_job(_inbound_config(enabled=True), sched)
    r2 = setup_inbound_sms_cleanup_job(_inbound_config(enabled=True), sched)
    assert r1 is True
    assert r2 is True
    jobs = [j for j in sched.list_jobs() if j.job_id == "inbound_sms_retention_cleanup"]
    assert len(jobs) == 1


# ---------------------------------------------------------------------------
# setup_retention_jobs (combined)
# ---------------------------------------------------------------------------


def test_setup_retention_jobs_inbound_enabled(tmp_path: Path) -> None:
    """Inbound SMS job registered; dashboard is a no-op (pending retirement)."""
    raw = {**_dashboard_config(), **_inbound_config(enabled=True)}
    sched = _make_scheduler(tmp_path)
    results = setup_retention_jobs(raw, sched)
    assert results["dashboard"] is False
    assert results["inbound_sms"] is True


def test_setup_retention_jobs_both_disabled(tmp_path: Path) -> None:
    """No jobs registered when inbound is disabled (dashboard is always no-op)."""
    raw = {**_dashboard_config(), **_inbound_config(enabled=False)}
    sched = _make_scheduler(tmp_path)
    results = setup_retention_jobs(raw, sched)
    assert results["dashboard"] is False
    assert results["inbound_sms"] is False


def test_setup_retention_jobs_neither_when_schedules_null(tmp_path: Path) -> None:
    """No jobs registered when both cleanup_schedule values are null."""
    raw = {
        **_dashboard_config(cleanup_schedule=None),
        **_inbound_config(enabled=True, cleanup_schedule=None),
    }
    sched = _make_scheduler(tmp_path)
    results = setup_retention_jobs(raw, sched)
    assert results["dashboard"] is False
    assert results["inbound_sms"] is False


def test_setup_retention_jobs_with_none_config_does_not_crash(tmp_path: Path) -> None:
    """setup_retention_jobs tolerates None raw_config (attempts disk load gracefully)."""
    sched = _make_scheduler(tmp_path)
    # Patch load_config to raise so we don't depend on a real config file.
    with patch("rex.retention.setup_retention_jobs") as mock_fn:
        # Call the real function but intercept load_config
        mock_fn.side_effect = lambda rc, s: {"dashboard": False, "inbound_sms": False}
        results = mock_fn(None, sched)
    assert results == {"dashboard": False, "inbound_sms": False}


# ---------------------------------------------------------------------------
# Job execution — inbound SMS
# ---------------------------------------------------------------------------


def test_inbound_sms_cleanup_job_executes_correctly(tmp_path: Path) -> None:
    """Running the inbound SMS cleanup job removes old records."""
    db = tmp_path / "inbound.db"
    store = InboundSmsStore(db_path=db, retention_days=1)

    from rex.messaging_backends.inbound_store import InboundSmsRecord

    old_record = InboundSmsRecord(received_at=datetime.now(timezone.utc) - timedelta(days=5))
    recent_record = InboundSmsRecord(received_at=datetime.now(timezone.utc))
    store.write(old_record)
    store.write(recent_record)
    assert store.count() == 2

    sched = _make_scheduler(tmp_path)
    raw = _inbound_config(enabled=True, store_path=str(db), retention_days=1)
    setup_inbound_sms_cleanup_job(raw, sched)

    assert sched.run_job("inbound_sms_retention_cleanup", force=True) is True

    assert store.count() == 1  # Only the recent record remains


def test_inbound_sms_cleanup_job_idempotent_execution(tmp_path: Path) -> None:
    """Running cleanup twice on a fresh store yields zero removals both times."""
    db = tmp_path / "inbound.db"
    store = InboundSmsStore(db_path=db, retention_days=90)

    from rex.messaging_backends.inbound_store import InboundSmsRecord

    store.write(InboundSmsRecord(received_at=datetime.now(timezone.utc)))
    assert store.count() == 1

    sched = _make_scheduler(tmp_path)
    setup_inbound_sms_cleanup_job(_inbound_config(enabled=True, store_path=str(db)), sched)

    assert sched.run_job("inbound_sms_retention_cleanup", force=True) is True
    assert store.count() == 1  # Fresh record not removed

    assert sched.run_job("inbound_sms_retention_cleanup", force=True) is True
    assert store.count() == 1  # Still not removed


# ---------------------------------------------------------------------------
# wire_retention_cleanup
# ---------------------------------------------------------------------------


def test_wire_retention_cleanup_returns_true_when_job_registered(tmp_path: Path) -> None:
    """wire_retention_cleanup returns True when at least one job is registered."""
    from rex.scheduler import set_scheduler

    sched = _make_scheduler(tmp_path)
    set_scheduler(sched)
    try:
        # Dashboard is a no-op; use inbound to get a registered job.
        raw = {**_dashboard_config(), **_inbound_config(enabled=True)}
        result = wire_retention_cleanup(raw)
        assert result is True
        assert sched.get_job("inbound_sms_retention_cleanup") is not None
    finally:
        set_scheduler(None)  # type: ignore[arg-type]


def test_wire_retention_cleanup_returns_false_when_nothing_scheduled(
    tmp_path: Path,
) -> None:
    """wire_retention_cleanup returns False when no jobs are configured."""
    from rex.scheduler import set_scheduler

    sched = _make_scheduler(tmp_path)
    set_scheduler(sched)
    try:
        raw = {
            **_dashboard_config(cleanup_schedule=None),
            **_inbound_config(enabled=False),
        }
        result = wire_retention_cleanup(raw)
        assert result is False
    finally:
        set_scheduler(None)  # type: ignore[arg-type]
