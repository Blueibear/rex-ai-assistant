"""Service initialization for scheduler, email, calendar, messaging, and notifications."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from rex.calendar_service import CalendarService
from rex.email_service import EmailService
from rex.notification import EscalationManager, Notifier
from rex.openclaw.event_bus import EventBus, get_event_bus
from rex.scheduler import Scheduler, set_scheduler


@dataclass
class AppServices:
    event_bus: EventBus
    scheduler: Scheduler
    email: EmailService
    calendar: CalendarService
    # OPENCLAW-REPLACE: sms field retained for API compatibility; None until OpenClaw messaging is wired
    sms: Any
    notifier: Notifier
    escalation_manager: EscalationManager


_SERVICES: AppServices | None = None


def initialize_services(
    *,
    storage_path: Path | str | None = None,
    email_mock_path: Path | str | None = None,
    calendar_mock_path: Path | str | None = None,
    sms_mock_path: Path | str | None = None,
    notifications_path: Path | str | None = None,
    now_func: Callable[[], datetime] | None = None,
) -> AppServices:
    """Initialize core services and register default scheduler jobs."""
    global _SERVICES
    if _SERVICES is not None:
        return _SERVICES

    event_bus = get_event_bus()
    scheduler = Scheduler(storage_path=storage_path, now_func=now_func)  # type: ignore[arg-type]
    set_scheduler(scheduler)
    email = EmailService(event_bus, mock_data_path=email_mock_path)  # type: ignore[arg-type]
    calendar = CalendarService(event_bus, mock_data_path=calendar_mock_path)

    # OPENCLAW-REPLACE: SMS service stub — pending migration to OpenClaw messaging
    sms: Any = None  # sms_mock_path retained in signature for API compatibility

    # Initialize notification system
    if notifications_path:
        notifications_path = (
            Path(notifications_path) if isinstance(notifications_path, str) else notifications_path
        )
    escalation_manager = EscalationManager()
    notifier = Notifier(
        storage_path=notifications_path,  # type: ignore[arg-type]
        escalation_manager=escalation_manager,
    )

    # Register scheduler handlers
    scheduler.register_handler("email_triage", lambda job: email.triage_unread())  # type: ignore[arg-type]
    scheduler.register_handler("calendar_sync", lambda job: calendar.refresh_upcoming())  # type: ignore[arg-type]
    scheduler.register_handler("flush_digests", lambda job: notifier.flush_digests())  # type: ignore[arg-type]
    scheduler.register_handler(
        "check_escalations", lambda job: _check_escalations(notifier, escalation_manager)
    )

    _register_default_jobs(scheduler)

    # Subscribe notifier to event bus events
    notifier.setup_event_subscriptions()

    _SERVICES = AppServices(
        event_bus=event_bus,
        scheduler=scheduler,
        email=email,
        calendar=calendar,
        sms=sms,
        notifier=notifier,
        escalation_manager=escalation_manager,
    )
    return _SERVICES


def reset_services() -> None:
    """Reset cached services (primarily for tests)."""
    global _SERVICES
    _SERVICES = None


def _check_escalations(notifier: Notifier, escalation_manager: EscalationManager) -> None:
    """Check for notifications that need escalation and resend them."""
    to_escalate = escalation_manager.check_escalations()
    for notif_id, next_channel in to_escalate:
        rule = escalation_manager.pending_escalations.get(notif_id)
        if not rule:
            continue
        try:
            notifier.send_to_channel(next_channel, rule.notification)
        except Exception as exc:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "Failed to escalate notification %s to %s: %s",
                notif_id,
                next_channel,
                exc,
            )


def _register_default_jobs(scheduler: Scheduler) -> None:
    defaults = [
        ("Email triage", 900, "email_triage"),
        ("Calendar sync", 1800, "calendar_sync"),
        ("Flush notification digests", 3600, "flush_digests"),
        ("Check notification escalations", 300, "check_escalations"),
    ]
    for name, interval, handler in defaults:
        if scheduler.find_job_by_name(name) is None:
            scheduler.add_job(name, interval, handler)

    # Register retention cleanup jobs (config-driven; skipped safely if config
    # is unavailable or stores are disabled).
    _try_register_retention_jobs(scheduler)


def _try_register_retention_jobs(scheduler: Scheduler) -> None:
    """Attempt to register retention cleanup jobs from config.

    Failures are logged at DEBUG level and do not prevent startup.
    """
    import logging

    _log = logging.getLogger(__name__)
    try:
        from rex.config_manager import load_config
        from rex.retention import setup_retention_jobs

        raw_config = load_config()
        results = setup_retention_jobs(raw_config, scheduler)
        if any(results.values()):
            _log.info("Retention cleanup jobs registered: %s", results)
        else:
            _log.debug("No retention cleanup jobs registered (check config)")
    except Exception as exc:
        _log.debug("Retention cleanup jobs setup skipped: %s", exc)


__all__ = ["AppServices", "initialize_services", "reset_services"]
