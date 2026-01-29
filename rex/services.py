"""Service initialization for scheduler, email, calendar, messaging, and notifications."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Callable

from rex.calendar_service import CalendarService
from rex.email_service import EmailService
from rex.event_bus import EventBus, get_event_bus
from rex.messaging_service import SMSService
from rex.notification import EscalationManager, Notifier
from rex.scheduler import Scheduler, set_scheduler


@dataclass
class AppServices:
    event_bus: EventBus
    scheduler: Scheduler
    email: EmailService
    calendar: CalendarService
    sms: SMSService
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
    scheduler = Scheduler(storage_path=storage_path, now_func=now_func)
    set_scheduler(scheduler)
    email = EmailService(event_bus, mock_data_path=email_mock_path)
    calendar = CalendarService(event_bus, mock_data_path=calendar_mock_path)

    # Initialize messaging service
    if sms_mock_path:
        sms_mock_path = Path(sms_mock_path) if isinstance(sms_mock_path, str) else sms_mock_path
    sms = SMSService(mock_file=sms_mock_path)

    # Initialize notification system
    if notifications_path:
        notifications_path = Path(notifications_path) if isinstance(notifications_path, str) else notifications_path
    escalation_manager = EscalationManager()
    notifier = Notifier(
        storage_path=notifications_path,
        escalation_manager=escalation_manager,
    )

    # Register scheduler handlers
    scheduler.register_handler("email_triage", lambda job: email.triage_unread())
    scheduler.register_handler("calendar_sync", lambda job: calendar.refresh_upcoming())
    scheduler.register_handler("flush_digests", lambda job: notifier.flush_digests())
    scheduler.register_handler("check_escalations", lambda job: _check_escalations(notifier, escalation_manager))

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


__all__ = ["AppServices", "initialize_services", "reset_services"]
