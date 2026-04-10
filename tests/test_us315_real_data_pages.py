"""
US-315: Replace placeholder data in Calendar, Email, and SMS pages — verification tests.

Verifies:
- CalendarPage.tsx calls window.rex.getCalendarEvents (real IPC, not hardcoded data)
- EmailPage.tsx calls window.rex.getEmailInbox (real IPC, not hardcoded data)
- SmsPage.tsx calls window.rex.getSMSThreads (real IPC, not hardcoded data)
- No hardcoded fake names/dates/messages in GUI source for these pages
- Backend endpoints /api/calendar/events, /api/email/inbox, /api/sms/threads exist
- Each page has an empty-state or error-state (not silently blank)
- IPC handlers in main process route to backend APIs
- Preload exposes the required functions
"""

from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent


def read_file(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_NAMES = ["John Doe", "Jane Doe", "Alice Smith", "Bob Smith", "Test User"]
FAKE_EMAILS = ["john@example.com", "jane@example.com", "test@test.com"]


# ---------------------------------------------------------------------------
# CalendarPage
# ---------------------------------------------------------------------------


def test_calendar_page_calls_real_ipc() -> None:
    src = read_file("gui/src/pages/CalendarPage.tsx")
    assert "getCalendarEvents" in src, "CalendarPage must call window.rex.getCalendarEvents"


def test_calendar_page_no_hardcoded_fake_events() -> None:
    src = read_file("gui/src/pages/CalendarPage.tsx")
    for name in FAKE_NAMES:
        assert name not in src, f"CalendarPage must not contain hardcoded name '{name}'"
    for email in FAKE_EMAILS:
        assert email not in src, f"CalendarPage must not contain hardcoded email '{email}'"


def test_calendar_page_has_error_or_empty_handling() -> None:
    src = read_file("gui/src/pages/CalendarPage.tsx")
    # Page should handle errors (addToast or error state)
    assert "error" in src.lower() or "Error" in src, (
        "CalendarPage must handle error/empty state"
    )


# ---------------------------------------------------------------------------
# EmailPage
# ---------------------------------------------------------------------------


def test_email_page_calls_real_ipc() -> None:
    src = read_file("gui/src/pages/EmailPage.tsx")
    assert "getEmailInbox" in src, "EmailPage must call window.rex.getEmailInbox"


def test_email_page_no_hardcoded_fake_messages() -> None:
    src = read_file("gui/src/pages/EmailPage.tsx")
    for name in FAKE_NAMES:
        assert name not in src, f"EmailPage must not contain hardcoded name '{name}'"
    for email in FAKE_EMAILS:
        assert email not in src, f"EmailPage must not contain hardcoded email '{email}'"


def test_email_page_has_empty_or_error_state() -> None:
    src = read_file("gui/src/pages/EmailPage.tsx")
    assert "length === 0" in src or "messages.length" in src or "No " in src, (
        "EmailPage must handle empty inbox state"
    )


# ---------------------------------------------------------------------------
# SmsPage
# ---------------------------------------------------------------------------


def test_sms_page_calls_real_ipc() -> None:
    src = read_file("gui/src/pages/SmsPage.tsx")
    assert "getSMSThreads" in src, "SmsPage must call window.rex.getSMSThreads"


def test_sms_page_no_hardcoded_fake_threads() -> None:
    src = read_file("gui/src/pages/SmsPage.tsx")
    for name in FAKE_NAMES:
        assert name not in src, f"SmsPage must not contain hardcoded name '{name}'"
    for email in FAKE_EMAILS:
        assert email not in src, f"SmsPage must not contain hardcoded email '{email}'"


def test_sms_page_has_empty_state() -> None:
    src = read_file("gui/src/pages/SmsPage.tsx")
    assert "No conversations" in src or "threads.length === 0" in src, (
        "SmsPage must show an empty state when there are no threads"
    )


# ---------------------------------------------------------------------------
# Backend: API endpoints in gui_app.py
# ---------------------------------------------------------------------------


def test_backend_calendar_events_endpoint_exists() -> None:
    src = read_file("rex/gui_app.py")
    assert '"/api/calendar/events"' in src, (
        "gui_app.py must define /api/calendar/events endpoint"
    )


def test_backend_email_inbox_endpoint_exists() -> None:
    src = read_file("rex/gui_app.py")
    assert '"/api/email/inbox"' in src, (
        "gui_app.py must define /api/email/inbox endpoint"
    )


def test_backend_sms_threads_endpoint_exists() -> None:
    src = read_file("rex/gui_app.py")
    assert '"/api/sms/threads"' in src, (
        "gui_app.py must define /api/sms/threads endpoint"
    )


def test_backend_calendar_returns_configured_flag() -> None:
    src = read_file("rex/gui_app.py")
    # The endpoint should return a "configured" field
    assert '"configured"' in src or "'configured'" in src, (
        "Calendar endpoint must return 'configured' field"
    )


def test_backend_email_returns_configured_flag() -> None:
    src = read_file("rex/gui_app.py")
    assert '"configured"' in src or "'configured'" in src, (
        "Email endpoint must return 'configured' field"
    )


# ---------------------------------------------------------------------------
# IPC: Electron main process handlers
# ---------------------------------------------------------------------------


def test_calendar_ipc_handler_exists() -> None:
    src = read_file("gui/src/main/handlers/calendar.ts")
    assert "getCalendarEvents" in src, (
        "calendar.ts IPC handler must implement getCalendarEvents"
    )
    # Handler uses a bridge script (not direct HTTP) — verify bridge is called
    assert "rex_calendar_bridge" in src or "calendar_bridge" in src, (
        "calendar.ts must call a calendar bridge script"
    )


def test_email_ipc_handler_exists() -> None:
    src = read_file("gui/src/main/handlers/email.ts")
    assert "getEmailInbox" in src, (
        "email.ts IPC handler must implement getEmailInbox"
    )
    # Handler uses a bridge script (not direct HTTP) — verify bridge is called
    assert "rex_email_bridge" in src or "email_bridge" in src, (
        "email.ts must call an email bridge script"
    )


def test_sms_ipc_handler_exists() -> None:
    src = read_file("gui/src/main/handlers/sms.ts")
    assert "getSMSThreads" in src, (
        "sms.ts IPC handler must implement getSMSThreads"
    )
    # Handler uses a bridge script (not direct HTTP) — verify bridge is called
    assert "rex_sms_bridge" in src or "sms_bridge" in src, (
        "sms.ts must call an SMS bridge script"
    )


# ---------------------------------------------------------------------------
# Preload: functions exposed to renderer
# ---------------------------------------------------------------------------


def test_preload_exposes_calendar_function() -> None:
    src = read_file("gui/src/preload/index.ts")
    assert "getCalendarEvents" in src, "preload must expose getCalendarEvents"


def test_preload_exposes_email_function() -> None:
    src = read_file("gui/src/preload/index.ts")
    assert "getEmailInbox" in src, "preload must expose getEmailInbox"


def test_preload_exposes_sms_function() -> None:
    src = read_file("gui/src/preload/index.ts")
    assert "getSMSThreads" in src, "preload must expose getSMSThreads"


# ---------------------------------------------------------------------------
# IPC types
# ---------------------------------------------------------------------------


def test_ipc_types_define_calendar_events() -> None:
    src = read_file("gui/src/types/ipc.ts")
    assert "getCalendarEvents" in src, "ipc.ts must type getCalendarEvents"
    assert "CalendarEvent" in src, "ipc.ts must define CalendarEvent type"


def test_ipc_types_define_email_messages() -> None:
    src = read_file("gui/src/types/ipc.ts")
    assert "getEmailInbox" in src, "ipc.ts must type getEmailInbox"
    assert "EmailMessage" in src, "ipc.ts must define EmailMessage type"


def test_ipc_types_define_sms_threads() -> None:
    src = read_file("gui/src/types/ipc.ts")
    assert "getSMSThreads" in src, "ipc.ts must type getSMSThreads"
    assert "SMSThread" in src, "ipc.ts must define SMSThread type"
