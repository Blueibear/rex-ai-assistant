"""Lazy service getters for Rex CLI command modules.

Extracted verbatim from ``rex/cli.py`` (US-REM-027). All getters import
their backing service lazily so optional integrations degrade gracefully.

These are re-exported by ``rex.cli``; tests monkeypatch them as
``rex.cli.get_*`` and command modules resolve them through ``rex.cli``
at call time.
"""

from __future__ import annotations


def get_browser_service():
    from rex.openclaw.browser_bridge import BrowserBridge

    return BrowserBridge()


def get_os_service():
    from rex.os_automation import get_os_service as _get_os_service

    return _get_os_service()


def get_github_service():
    from rex.github_service import get_github_service as _get_github_service

    return _get_github_service()


def get_vscode_service():
    from rex.vscode_service import get_vscode_service as _get_vscode_service

    return _get_vscode_service()


def get_scheduler():
    from rex.scheduler import get_scheduler as _get_scheduler

    return _get_scheduler()


def get_email_service():
    from rex.email_service import get_email_service as _get_email_service

    return _get_email_service()


def get_calendar_service():
    from rex.calendar_service import get_calendar_service as _get_calendar_service

    return _get_calendar_service()


def get_reminder_service():
    from rex.reminder_service import get_reminder_service as _get_reminder_service

    return _get_reminder_service()


def get_cue_store():
    from rex.cue_store import get_cue_store as _get_cue_store

    return _get_cue_store()


def initialize_scheduler_system(*args, **kwargs):
    from rex.integrations import initialize_scheduler_system as _initialize_scheduler_system

    return _initialize_scheduler_system(*args, **kwargs)


def get_computer_service():
    from rex.computers.service import get_computer_service as _get_computer_service

    return _get_computer_service()
