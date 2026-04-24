from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import rex.config
import rex_calendar_bridge
import rex_email_bridge
from rex.config import build_app_config

REPO = Path(__file__).parent.parent


def test_config_loads_email_and_calendar_provider_selection() -> None:
    cfg = build_app_config(
        {
            "email": {"provider": "outlook"},
            "calendar": {"provider": "gmail"},
        }
    )

    assert cfg.email_provider == "outlook"
    assert cfg.calendar_provider == "google"


def test_email_bridge_reports_outlook_as_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(
        rex.config,
        "load_config",
        lambda: SimpleNamespace(email_provider="outlook"),
    )

    result = rex_email_bridge._handle_list(10)

    assert result["ok"] is False
    assert result["configured"] is True
    assert "Outlook email sync is not implemented yet" in result["error"]


def test_calendar_bridge_reports_outlook_list_as_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(
        rex.config,
        "load_config",
        lambda: SimpleNamespace(calendar_provider="outlook"),
    )

    result = rex_calendar_bridge._handle_list("", "")

    assert result["ok"] is False
    assert result["configured"] is True
    assert "Outlook calendar sync is not implemented yet" in result["error"]


def test_calendar_bridge_reports_outlook_create_as_unsupported(monkeypatch) -> None:
    monkeypatch.setattr(
        rex.config,
        "load_config",
        lambda: SimpleNamespace(calendar_provider="outlook"),
    )

    result = rex_calendar_bridge._handle_create(
        {
            "title": "Doctor appointment",
            "start": "2026-04-22T10:00:00+00:00",
            "end": "2026-04-22T11:00:00+00:00",
        }
    )

    assert result["ok"] is False
    assert result["configured"] is True
    assert "Outlook calendar sync is not implemented yet" in result["error"]


def test_electron_outlook_status_is_not_marked_connected_by_credentials_only() -> None:
    src = (REPO / "gui" / "src" / "main" / "index.ts").read_text(encoding="utf-8")

    assert "OUTLOOK_EMAIL_UNSUPPORTED" in src
    assert "OUTLOOK_CALENDAR_UNSUPPORTED" in src
    assert "hasConfiguredOutlookEmail(integrations)" in src
    assert "hasConfiguredOutlookCalendar(integrations)" in src


def test_email_and_calendar_handlers_surface_bridge_errors() -> None:
    email_src = (REPO / "gui" / "src" / "main" / "handlers" / "email.ts").read_text(
        encoding="utf-8"
    )
    calendar_src = (
        REPO / "gui" / "src" / "main" / "handlers" / "calendar.ts"
    ).read_text(encoding="utf-8")

    assert "throw new Error(result.error" in email_src
    assert "throw new Error(result.error" in calendar_src
    assert "callCalendarBridge('create'" in calendar_src
