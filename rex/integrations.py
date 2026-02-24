"""
Integration module for Rex AI Assistant.

Sets up default scheduler jobs and event bus subscriptions for email,
calendar, and other automated tasks.
"""

import logging

from rex.calendar_service import get_calendar_service
from rex.email_service import get_email_service
from rex.event_bus import Event, get_event_bus
from rex.scheduler import ScheduledJob, get_scheduler

logger = logging.getLogger(__name__)


def setup_email_job() -> ScheduledJob:
    """
    Set up the default email checking job.

    Returns:
        Created ScheduledJob
    """
    scheduler = get_scheduler()
    event_bus = get_event_bus()

    # Register email check callback
    def check_email(job: ScheduledJob) -> None:
        """Check for unread emails and publish event."""
        logger.info("Running scheduled email check")
        email_service = get_email_service()

        try:
            if not email_service.connected:
                email_service.connect()

            # Fetch unread emails
            unread_emails = email_service.fetch_unread(limit=10)

            if unread_emails:
                # Categorize emails
                for email in unread_emails:
                    email.category = email_service.categorize(email)

                # Publish event
                event = Event(
                    event_type="email.unread",
                    payload={
                        "count": len(unread_emails),
                        "emails": [email.model_dump() for email in unread_emails],
                    },
                )
                event_bus.publish(event)

                logger.info(f"Published email.unread event with {len(unread_emails)} emails")
            else:
                logger.debug("No unread emails found")

        except Exception as e:
            logger.error(f"Error checking emails: {e}", exc_info=True)

    scheduler.register_callback("check_email", check_email)

    # Add job (runs every 10 minutes = 600 seconds)
    job = scheduler.add_job(
        job_id="email_check",
        name="Check Email",
        schedule="interval:600",
        callback_name="check_email",
        enabled=True,
    )

    logger.info("Email check job registered")
    return job


def setup_calendar_job() -> ScheduledJob:
    """
    Set up the default calendar sync job.

    Returns:
        Created ScheduledJob
    """
    scheduler = get_scheduler()
    event_bus = get_event_bus()

    # Register calendar sync callback
    def sync_calendar(job: ScheduledJob) -> None:
        """Sync calendar and publish event."""
        logger.info("Running scheduled calendar sync")
        calendar_service = get_calendar_service()

        try:
            if not calendar_service.connected:
                calendar_service.connect()

            # Get upcoming events (next 7 days)
            events = calendar_service.get_upcoming_events(days=7)

            # Publish event
            event = Event(
                event_type="calendar.update",
                payload={"count": len(events), "events": [e.model_dump() for e in events]},
            )
            event_bus.publish(event)

            logger.info(f"Published calendar.update event with {len(events)} events")

        except Exception as e:
            logger.error(f"Error syncing calendar: {e}", exc_info=True)

    scheduler.register_callback("sync_calendar", sync_calendar)

    # Add job (runs every hour = 3600 seconds)
    job = scheduler.add_job(
        job_id="calendar_sync",
        name="Sync Calendar",
        schedule="interval:3600",
        callback_name="sync_calendar",
        enabled=True,
    )

    logger.info("Calendar sync job registered")
    return job


def setup_default_event_handlers() -> None:
    """Set up default event bus handlers for logging."""
    event_bus = get_event_bus()

    def log_email_event(event: Event) -> None:
        """Log email events."""
        count = event.payload.get("count", 0)
        logger.info(f"Email event received: {count} unread email(s)")

    def log_calendar_event(event: Event) -> None:
        """Log calendar events."""
        count = event.payload.get("count", 0)
        logger.info(f"Calendar event received: {count} upcoming event(s)")

    # Subscribe handlers
    event_bus.subscribe("email.unread", log_email_event)
    event_bus.subscribe("calendar.update", log_calendar_event)

    logger.info("Default event handlers registered")


def initialize_scheduler_system(start_scheduler: bool = False) -> None:
    """
    Initialize the scheduler system with default jobs and event handlers.

    Args:
        start_scheduler: If True, start the scheduler background thread
    """
    logger.info("Initializing scheduler system")

    # Set up default jobs
    setup_email_job()
    setup_calendar_job()
    _try_register_retention_jobs()

    # Set up event handlers
    setup_default_event_handlers()

    # Start scheduler if requested
    if start_scheduler:
        scheduler = get_scheduler()
        scheduler.start()
        logger.info("Scheduler started")

    logger.info("Scheduler system initialized")


def _try_register_retention_jobs() -> None:
    """Register retention cleanup jobs from config when available.

    Failures are logged at DEBUG and do not prevent scheduler initialization.
    """
    try:
        from rex.config_manager import load_config
        from rex.retention import setup_retention_jobs

        raw_config = load_config()
        results = setup_retention_jobs(raw_config)
        if any(results.values()):
            logger.info("Retention cleanup jobs registered: %s", results)
        else:
            logger.debug("No retention cleanup jobs registered (check config)")
    except Exception as exc:
        logger.debug("Retention cleanup jobs setup skipped: %s", exc)


def shutdown_scheduler_system() -> None:
    """Shutdown the scheduler system."""
    logger.info("Shutting down scheduler system")
    scheduler = get_scheduler()
    scheduler.stop()
    logger.info("Scheduler stopped")
