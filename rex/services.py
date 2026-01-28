"""Service initialization for scheduler, email, and calendar integrations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Callable

from rex.calendar_service import CalendarService
from rex.email_service import EmailService
from rex.event_bus import EventBus, get_event_bus
from rex.scheduler import Scheduler


@dataclass
class AppServices:
    event_bus: EventBus
    scheduler: Scheduler
    email: EmailService
    calendar: CalendarService


_SERVICES: AppServices | None = None


def initialize_services(
    *,
    storage_path: Path | str | None = None,
    email_mock_path: Path | str | None = None,
    calendar_mock_path: Path | str | None = None,
    now_func: Callable[[], datetime] | None = None,
) -> AppServices:
    """Initialize core services and register default scheduler jobs."""
    global _SERVICES
    if _SERVICES is not None:
        return _SERVICES

    event_bus = get_event_bus()
    scheduler = Scheduler(storage_path=storage_path, now_func=now_func)
    email = EmailService(event_bus, mock_data_path=email_mock_path)
    calendar = CalendarService(event_bus, mock_data_path=calendar_mock_path)

    scheduler.register_handler("email_triage", lambda job: email.triage_unread())
    scheduler.register_handler("calendar_sync", lambda job: calendar.refresh_upcoming())

    _register_default_jobs(scheduler)

    _SERVICES = AppServices(
        event_bus=event_bus,
        scheduler=scheduler,
        email=email,
        calendar=calendar,
    )
    return _SERVICES


def reset_services() -> None:
    """Reset cached services (primarily for tests)."""
    global _SERVICES
    _SERVICES = None


def _register_default_jobs(scheduler: Scheduler) -> None:
    defaults = [
        ("Email triage", 900, "email_triage"),
        ("Calendar sync", 1800, "calendar_sync"),
    ]
    for name, interval, handler in defaults:
        if scheduler.find_job_by_name(name) is None:
            scheduler.add_job(name, interval, handler)


__all__ = ["AppServices", "initialize_services", "reset_services"]
