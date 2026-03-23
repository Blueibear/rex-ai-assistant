"""Core package for the Rex voice assistant.

Importing :mod:`rex` used to configure logging immediately, which proved
problematic for applications that need to install their own handlers before
any log messages are emitted. The package now exposes the helper functions
without triggering side effects so callers can opt in to
:func:`rex.logging_utils.configure_logging` when appropriate.
"""

from __future__ import annotations

from .logging_utils import configure_logging

try:
    from .config import reload_settings, settings
except Exception as exc:  # pragma: no cover - defensive guard for import-time config failures
    _config_import_error = exc

    def reload_settings(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Rex configuration failed to load.") from _config_import_error

    settings = None  # type: ignore[assignment]

# Credential management
# Calendar service
from .calendar_service import (
    CalendarEvent,
    CalendarService,
    get_calendar_service,
    set_calendar_service,
)
from .credentials import (
    Credential,
    CredentialManager,
    CredentialRefreshError,
    get_credential_manager,
    mask_token,
    set_credential_manager,
)

# Email service
from .email_service import (
    EmailService,
    EmailSummary,
    get_email_service,
    set_email_service,
)

# Event bus
from .event_bus import (
    Event,
    EventBus,
    get_event_bus,
    set_event_bus,
)

# Notification system
from .notification import (
    EscalationManager,
    NotificationRequest,
    Notifier,
    get_escalation_manager,
    get_notifier,
    set_escalation_manager,
    set_notifier,
)

# Scheduler
from .scheduler import (
    ScheduledJob,
    Scheduler,
    get_scheduler,
    set_scheduler,
)

# Tool registry
from .tool_registry import (
    MissingCredentialError,
    ToolMeta,
    ToolNotFoundError,
    ToolRegistry,
    get_tool_registry,
    register_tool,
    set_tool_registry,
)

__all__ = [
    # Configuration
    "settings",
    "reload_settings",
    "configure_logging",
    # Credential management
    "Credential",
    "CredentialManager",
    "CredentialRefreshError",
    "get_credential_manager",
    "set_credential_manager",
    "mask_token",
    # Tool registry
    "ToolMeta",
    "ToolRegistry",
    "ToolNotFoundError",
    "MissingCredentialError",
    "get_tool_registry",
    "set_tool_registry",
    "register_tool",
    # Scheduler
    "ScheduledJob",
    "Scheduler",
    "get_scheduler",
    "set_scheduler",
    # Event bus
    "Event",
    "EventBus",
    "get_event_bus",
    "set_event_bus",
    # Email service
    "EmailSummary",
    "EmailService",
    "get_email_service",
    "set_email_service",
    # Calendar service
    "CalendarEvent",
    "CalendarService",
    "get_calendar_service",
    "set_calendar_service",
    # Notification system
    "NotificationRequest",
    "Notifier",
    "EscalationManager",
    "get_notifier",
    "set_notifier",
    "get_escalation_manager",
    "set_escalation_manager",
]
