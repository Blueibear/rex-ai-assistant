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
from typing import Any

logger = logging.getLogger(__name__)


def setup_dashboard_cleanup_job(
    raw_config: dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> bool:
    """No-op stub — dashboard cleanup is deferred to OpenClaw.

    The legacy dashboard notification store is scheduled for retirement as
    part of the OpenClaw migration.  Dashboard notification retention is no
    longer registered as a scheduled job; OpenClaw will manage its own state
    retention once the legacy store is retired.

    Args:
        raw_config: Ignored (retained for API compatibility).
        scheduler: Ignored (retained for API compatibility).

    Returns:
        Always ``False`` — no job is registered.
    """
    # OPENCLAW-REPLACE: remove this function when the dashboard store is retired.
    logger.debug(
        "Dashboard cleanup job skipped — dashboard store is pending retirement "
        "(OpenClaw migration in progress)"
    )
    return False


def setup_inbound_sms_cleanup_job(
    raw_config: dict[str, Any] | None = None,
    scheduler: Any | None = None,
) -> bool:
    """No-op stub — inbound SMS cleanup is deferred to OpenClaw.

    The legacy ``rex.messaging_backends`` inbound store has been retired as
    part of the OpenClaw migration.  Inbound SMS retention is no longer
    registered as a scheduled job; OpenClaw will manage its own messaging
    state once the migration is complete.

    Args:
        raw_config: Ignored (retained for API compatibility).
        scheduler: Ignored (retained for API compatibility).

    Returns:
        Always ``False`` — no job is registered.
    """
    # OPENCLAW-REPLACE: messaging_backends retired in iter 91.
    logger.debug(
        "Inbound SMS cleanup job skipped — messaging backend retired "
        "(OpenClaw migration in progress)"
    )
    return False


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
