"""Rex integrations package — email, calendar, SMS, and notifications."""

from rex.integrations._setup import (
    initialize_scheduler_system,
    setup_calendar_job,
    setup_default_event_handlers,
    setup_email_job,
    shutdown_scheduler_system,
)

__all__ = [
    "initialize_scheduler_system",
    "setup_calendar_job",
    "setup_default_event_handlers",
    "setup_email_job",
    "shutdown_scheduler_system",
]
