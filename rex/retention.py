"""Scheduled retention cleanup jobs for dashboard and inbound SMS stores.

Wire-up functions register cleanup jobs in the global scheduler so that
old notifications and inbound SMS records are pruned on a configurable
schedule.

Design rules:
- Dashboard cleanup is scheduled when ``notifications.dashboard.store.cleanup_schedule``
  is non-null (default ``"interval:86400"`` — daily).
- Inbound SMS cleanup is scheduled only when ``messaging.inbound.enabled`` is
  ``true`` AND ``messaging.inbound.cleanup_schedule`` is non-null.
- All failures are caught and logged as warnings so that missing config or
  unavailable stores never crash startup.
- Registering jobs twice is idempotent: a second call is a no-op when the
  job already exists in the scheduler.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def setup_dashboard_cleanup_job(
    raw_config: dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> bool:
    """Register a scheduled job to clean up old dashboard notifications.

    Args:
        raw_config: Full runtime config dict.  Defaults to loading from disk.
        scheduler: Scheduler instance.  Uses the global scheduler if ``None``.

    Returns:
        ``True`` if the job was registered (or already existed), ``False`` if
        skipped due to missing config or disabled schedule.
    """
    from rex.dashboard_store import DashboardStore, load_dashboard_store_config
    from rex.scheduler import get_scheduler

    if scheduler is None:
        scheduler = get_scheduler()

    cfg = load_dashboard_store_config(raw_config or {})
    cleanup_schedule: str | None = cfg.cleanup_schedule

    if not cleanup_schedule:
        logger.debug(
            "Dashboard notification cleanup not scheduled "
            "(notifications.dashboard.store.cleanup_schedule is null/empty)"
        )
        return False

    job_id = "dashboard_retention_cleanup"
    if scheduler.get_job(job_id) is not None:
        logger.debug("Dashboard retention cleanup job already registered, skipping")
        return True

    db_path: Path | None = Path(cfg.path) if cfg.path else None
    retention_days: int = cfg.retention_days

    def _cleanup_dashboard(job: Any) -> None:  # noqa: ANN401
        """Run dashboard store cleanup."""
        try:
            store = DashboardStore(db_path=db_path, retention_days=retention_days)
            removed = store.cleanup_old()
            logger.info("Dashboard cleanup: removed %d old notifications", removed)
        except Exception as exc:
            logger.error("Dashboard cleanup failed: %s", exc, exc_info=True)

    callback_name = "dashboard_retention_cleanup"
    scheduler.register_callback(callback_name, _cleanup_dashboard)
    scheduler.add_job(
        job_id=job_id,
        name="Dashboard Notification Retention Cleanup",
        schedule=cleanup_schedule,
        callback_name=callback_name,
        enabled=True,
    )
    logger.info(
        "Dashboard notification retention cleanup scheduled: %s (retention=%d days)",
        cleanup_schedule,
        retention_days,
    )
    return True


def setup_inbound_sms_cleanup_job(
    raw_config: dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> bool:
    """Register a scheduled job to clean up old inbound SMS records.

    Only schedules the job when ``messaging.inbound.enabled`` is ``true`` AND
    ``messaging.inbound.cleanup_schedule`` is non-null.

    Args:
        raw_config: Full runtime config dict.
        scheduler: Scheduler instance.  Uses the global scheduler if ``None``.

    Returns:
        ``True`` if the job was registered (or already existed), ``False`` if
        skipped because inbound is disabled or schedule is not configured.
    """
    from rex.messaging_backends.inbound_store import InboundSmsStore, load_inbound_store_config
    from rex.scheduler import get_scheduler

    if scheduler is None:
        scheduler = get_scheduler()

    cfg = load_inbound_store_config(raw_config or {})

    if not cfg.enabled:
        logger.debug("Inbound SMS cleanup not scheduled (messaging.inbound.enabled=false)")
        return False

    cleanup_schedule: str | None = cfg.cleanup_schedule
    if not cleanup_schedule:
        logger.debug(
            "Inbound SMS cleanup not scheduled "
            "(messaging.inbound.cleanup_schedule is null/empty)"
        )
        return False

    job_id = "inbound_sms_retention_cleanup"
    if scheduler.get_job(job_id) is not None:
        logger.debug("Inbound SMS retention cleanup job already registered, skipping")
        return True

    db_path: Path | None = Path(cfg.store_path) if cfg.store_path else None
    retention_days: int = cfg.retention_days

    def _cleanup_inbound_sms(job: Any) -> None:  # noqa: ANN401
        """Run inbound SMS store cleanup."""
        try:
            store = InboundSmsStore(db_path=db_path, retention_days=retention_days)
            removed = store.cleanup_old()
            logger.info("Inbound SMS cleanup: removed %d old records", removed)
        except Exception as exc:
            logger.error("Inbound SMS cleanup failed: %s", exc, exc_info=True)

    callback_name = "inbound_sms_retention_cleanup"
    scheduler.register_callback(callback_name, _cleanup_inbound_sms)
    scheduler.add_job(
        job_id=job_id,
        name="Inbound SMS Retention Cleanup",
        schedule=cleanup_schedule,
        callback_name=callback_name,
        enabled=True,
    )
    logger.info(
        "Inbound SMS retention cleanup scheduled: %s (retention=%d days)",
        cleanup_schedule,
        retention_days,
    )
    return True


def setup_retention_jobs(
    raw_config: dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> dict[str, bool]:
    """Register all retention cleanup jobs.

    Calls both :func:`setup_dashboard_cleanup_job` and
    :func:`setup_inbound_sms_cleanup_job`.  Failures in either are caught and
    logged as warnings so that the other job can still be registered.

    Args:
        raw_config: Full runtime config dict.  Pass ``None`` to load from disk.
        scheduler: Scheduler instance.  Uses the global scheduler if ``None``.

    Returns:
        A dict with keys ``"dashboard"`` and ``"inbound_sms"`` whose values are
        ``True`` when the corresponding job was registered and ``False``
        otherwise.
    """
    if raw_config is None:
        try:
            from rex.config_manager import load_config

            raw_config = load_config()
        except Exception as exc:
            logger.warning("Could not load config for retention cleanup setup: %s", exc)
            raw_config = {}

    results: dict[str, bool] = {
        "dashboard": False,
        "inbound_sms": False,
    }

    try:
        results["dashboard"] = setup_dashboard_cleanup_job(raw_config, scheduler)
    except Exception as exc:
        logger.warning("Failed to set up dashboard retention cleanup: %s", exc)

    try:
        results["inbound_sms"] = setup_inbound_sms_cleanup_job(raw_config, scheduler)
    except Exception as exc:
        logger.warning("Failed to set up inbound SMS retention cleanup: %s", exc)

    return results


def wire_retention_cleanup(raw_config: dict[str, Any] | None = None) -> bool:
    """Wire retention cleanup jobs and ensure the scheduler background loop runs.

    Suitable for calling from application entry points such as
    ``flask_proxy.py``.  Loading config from disk when *raw_config* is not
    provided.

    Args:
        raw_config: Full runtime config dict, or ``None`` to load from disk.

    Returns:
        ``True`` if at least one retention job was registered.
    """
    results = setup_retention_jobs(raw_config)

    if any(results.values()):
        # Ensure the scheduler background loop is running so jobs actually execute.
        try:
            from rex.scheduler import get_scheduler

            get_scheduler().start()
        except Exception as exc:
            logger.warning("Could not start scheduler background loop: %s", exc)

    return any(results.values())


__all__ = [
    "setup_dashboard_cleanup_job",
    "setup_inbound_sms_cleanup_job",
    "setup_retention_jobs",
    "wire_retention_cleanup",
]
