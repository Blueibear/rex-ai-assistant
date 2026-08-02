from __future__ import annotations

from pathlib import Path

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
    import rex.config_manager

    monkeypatch.setattr(
        rex.config_manager,
        "load_config",
        lambda *a, **k: {"email": {"provider": "outlook"}},
    )

    result = rex_email_bridge._handle_list("default", 10)

    assert result["ok"] is False
    assert result["configured"] is True
    assert "Outlook email sync is not implemented yet" in result["error"]


def test_calendar_bridge_reports_outlook_list_as_unsupported(monkeypatch) -> None:
    import rex.config_manager

    monkeypatch.setattr(
        rex.config_manager,
        "load_config",
        lambda *a, **k: {"calendar": {"provider": "outlook"}},
    )

    # Legacy global provider config serves only the explicit default profile
    # (issue #303); the honest "unsupported" answer is preserved for it.
    result = rex_calendar_bridge._handle_list("default", "", "")

    assert result["ok"] is False
    assert result["configured"] is True
    assert "Outlook calendar sync is not implemented yet" in result["error"]


def test_calendar_bridge_reports_outlook_create_as_unsupported(monkeypatch) -> None:
    import rex.config_manager

    monkeypatch.setattr(
        rex.config_manager,
        "load_config",
        lambda *a, **k: {"calendar": {"provider": "outlook"}},
    )

    result = rex_calendar_bridge._handle_create(
        "default",
        {
            "title": "Doctor appointment",
            "start": "2026-04-22T10:00:00+00:00",
            "end": "2026-04-22T11:00:00+00:00",
        },
    )

    assert result["ok"] is False
    assert result["configured"] is True
    assert "Outlook calendar sync is not implemented yet" in result["error"]


def test_electron_outlook_status_is_not_marked_connected_by_credentials_only() -> None:
    src = (REPO / "gui" / "src" / "main" / "integrationStatus.ts").read_text(encoding="utf-8")

    assert "OUTLOOK_EMAIL_UNSUPPORTED" in src
    assert "OUTLOOK_CALENDAR_UNSUPPORTED" in src
    assert "unsupportedOutlookStatus(type, integrations)" in src
    assert "state: 'unavailable'" in src
    assert "if (unsupported) return unsupported" in src


def test_email_and_calendar_handlers_surface_bridge_errors() -> None:
    email_src = (REPO / "gui" / "src" / "main" / "handlers" / "email.ts").read_text(
        encoding="utf-8"
    )
    calendar_src = (REPO / "gui" / "src" / "main" / "handlers" / "calendar.ts").read_text(
        encoding="utf-8"
    )

    assert "throw new Error" in email_src and "result.error" in email_src
    assert "throw new Error" in calendar_src and "result.error" in calendar_src
    assert "callCalendarBridge(session, 'create'" in calendar_src
