"""Calendar surface isolation tests: GUI route, bridge, OpenClaw tool,
local tool executor, and scheduler (issue #303).

Each surface must resolve one validated user, fail closed without identity,
and never route through another user's account, provider, or credentials.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from rex.calendar_accounts import CalendarAccountResolver

ALICE = "alice"
BOB = "bob"


# ---------------------------------------------------------------------------
# GUI route: /api/calendar/events
# ---------------------------------------------------------------------------


@pytest.fixture()
def flask_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("REX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REX_JWT_SECRET", "test-cal-303-secret")
    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestCalendarRoute:
    def test_rejects_missing_identity(self, flask_client, monkeypatch):
        """Regression: the base implementation served the global provider
        calendar to any caller with no identity at all."""
        monkeypatch.setattr("rex.identity.resolve_active_user", lambda *a, **k: None)
        resp = flask_client.get("/api/calendar/events")
        assert resp.status_code == 403
        data = resp.get_json()
        assert data["ok"] is False
        assert data["events"] == []

    def test_rejects_malformed_explicit_user(self, flask_client):
        resp = flask_client.get("/api/calendar/events?user=..%2F..%2Fevil")
        assert resp.status_code == 403
        assert resp.get_json()["ok"] is False

    def test_named_user_without_assignment_gets_not_configured(self, flask_client, monkeypatch):
        """A named user never inherits the legacy global provider/token."""
        monkeypatch.setenv("GOOGLE_CALENDAR_ACCESS_TOKEN", "global-token")
        monkeypatch.setattr(
            "rex.config_manager.load_config",
            lambda *a, **k: {"calendar": {"provider": "google"}},
        )
        resp = flask_client.get("/api/calendar/events?user=james")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["configured"] is False
        assert data["events"] == []

    def test_resolved_user_is_passed_to_service_factory(self, flask_client, monkeypatch):
        captured: dict[str, Any] = {}

        def fake_factory(user_id, raw_config=None):
            captured["user_id"] = user_id
            return None, "none"

        monkeypatch.setattr(
            "rex.integrations.calendar_service.create_calendar_service_for_user",
            fake_factory,
        )
        resp = flask_client.get("/api/calendar/events?user=alice")
        assert resp.status_code == 200
        assert captured["user_id"] == "alice"


# ---------------------------------------------------------------------------
# Electron bridge: bridge/rex_calendar_bridge.py
# ---------------------------------------------------------------------------


class TestCalendarBridge:
    def test_missing_user_fails_closed(self, monkeypatch):
        import bridge.rex_calendar_bridge as bridge_mod

        monkeypatch.setattr(bridge_mod, "_resolve_user", lambda payload: None)
        called = {"factory": False}

        def fake_factory(user_id):
            called["factory"] = True
            return None, "none"

        monkeypatch.setattr(bridge_mod, "_service_for_user", fake_factory)

        import io
        import sys as _sys

        monkeypatch.setattr(_sys, "stdin", io.StringIO(json.dumps({"command": "list"})))
        out = io.StringIO()
        monkeypatch.setattr(_sys, "stdout", out)
        bridge_mod.main()
        result = json.loads(out.getvalue())
        assert result["ok"] is False
        assert "No active user" in result["error"]
        assert called["factory"] is False

    def test_list_passes_resolved_user_to_factory(self, monkeypatch):
        import bridge.rex_calendar_bridge as bridge_mod

        captured: dict[str, Any] = {}

        def fake_factory(user_id):
            captured["user_id"] = user_id
            return None, "none"

        monkeypatch.setattr(bridge_mod, "_service_for_user", fake_factory)
        result = bridge_mod._handle_list(ALICE, "", "")
        assert result == {"ok": True, "events": [], "configured": False}
        assert captured["user_id"] == ALICE

    def test_create_without_account_fails_closed(self, monkeypatch):
        import bridge.rex_calendar_bridge as bridge_mod

        monkeypatch.setattr(bridge_mod, "_service_for_user", lambda user_id: (None, "none"))
        result = bridge_mod._handle_create(ALICE, {"title": "x"})
        assert result["ok"] is False
        assert result["configured"] is False

    def test_malformed_explicit_user_resolves_to_none(self):
        import bridge.rex_calendar_bridge as bridge_mod

        assert bridge_mod._resolve_user({"user": "../evil"}) is None


# ---------------------------------------------------------------------------
# OpenClaw tool: calendar_create
# ---------------------------------------------------------------------------


class TestCalendarCreateTool:
    def test_requires_user_identity(self):
        """Regression: the base implementation created events with no user
        identity at all."""
        from rex.openclaw.tools.calendar_tool import calendar_create

        result = calendar_create(
            title="Standup",
            start_time="2027-03-23T09:00:00",
            end_time="2027-03-23T09:30:00",
        )
        assert result["ok"] is False
        assert "user identity" in result["error"]

    @pytest.mark.parametrize("bad", ["", "  ", "../evil", ".."])
    def test_invalid_user_identity_fails_closed(self, bad):
        from rex.openclaw.tools.calendar_tool import calendar_create

        result = calendar_create(
            title="Standup",
            start_time="2027-03-23T09:00:00",
            end_time="2027-03-23T09:30:00",
            _user_id=bad,
        )
        assert result["ok"] is False

    def test_creates_in_dispatching_users_store(self, monkeypatch):
        from rex.calendar_service import CalendarService
        from rex.openclaw.tools import calendar_tool

        svc = CalendarService(mock_events=[], owner_user_id=ALICE)
        monkeypatch.setattr(calendar_tool, "_get_calendar_service", lambda: svc)

        result = calendar_tool.calendar_create(
            title="Standup",
            start_time="2027-03-23T09:00:00",
            end_time="2027-03-23T09:30:00",
            _user_id=ALICE,
            transcript="create a standup meeting",
        )
        assert result["ok"] is True
        assert [e.title for e in svc.get_all_events(user_id=ALICE)] == ["Standup"]
        assert svc.get_all_events(user_id=BOB) == []

    def test_foreign_account_id_gets_generic_error(self, monkeypatch, tmp_path):
        from rex.calendar_service import CalendarService
        from rex.openclaw.tools import calendar_tool

        resolver = CalendarAccountResolver.from_raw_config(
            {
                "calendar": {
                    "accounts": [
                        {"id": "bob-cal", "provider": "stub"},
                    ]
                },
                "users": {BOB: {"calendar_accounts": [{"account_id": "bob-cal"}]}},
            }
        )
        svc = CalendarService(account_resolver=resolver)
        monkeypatch.setattr(calendar_tool, "_get_calendar_service", lambda: svc)

        result = calendar_tool.calendar_create(
            title="Standup",
            start_time="2027-03-23T09:00:00",
            end_time="2027-03-23T09:30:00",
            _user_id=ALICE,
            account_id="bob-cal",
        )
        assert result["ok"] is False
        assert "not available for user" in result["error"]

    def test_arbitrary_credential_reference_kwarg_is_ignored(self, monkeypatch):
        """Callers cannot inject credential references through tool kwargs."""
        from rex.calendar_service import CalendarService
        from rex.openclaw.tools import calendar_tool

        svc = CalendarService(mock_events=[], owner_user_id=ALICE)
        monkeypatch.setattr(calendar_tool, "_get_calendar_service", lambda: svc)

        result = calendar_tool.calendar_create(
            title="Standup",
            start_time="2027-03-23T09:00:00",
            end_time="2027-03-23T09:30:00",
            _user_id=ALICE,
            credential_ref="EVIL_TOKEN_ENV",
            credentials_key="EVIL_TOKEN_ENV",
        )
        assert result["ok"] is True  # absorbed and ignored, not honored


# ---------------------------------------------------------------------------
# Local tool executor: calendar_create_event
# ---------------------------------------------------------------------------


class TestLocalToolExecutor:
    @pytest.fixture(autouse=True)
    def _isolated(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "appdata"))
        empty = CalendarAccountResolver.from_raw_config({})
        monkeypatch.setattr(CalendarAccountResolver, "load", classmethod(lambda cls: empty))

    def test_missing_user_fails_closed(self):
        """Regression: the base implementation created events without any
        user identity."""
        from rex.local_tool_executor import execute_tool

        result = execute_tool("calendar_create_event", {"title": "Standup"})
        assert "a valid user identity is required" in result

    def test_invalid_user_fails_closed(self):
        from rex.local_tool_executor import execute_tool

        result = execute_tool("calendar_create_event", {"title": "Standup", "_user_id": "../evil"})
        assert "a valid user identity is required" in result

    def test_creates_in_requesting_users_store_only(self):
        from rex.calendar_service import CalendarService
        from rex.local_tool_executor import execute_tool

        result = execute_tool(
            "calendar_create_event",
            {
                "title": "Alice tool event",
                "start": "2027-03-23T09:00:00",
                "end": "2027-03-23T09:30:00",
                "_user_id": ALICE,
            },
        )
        assert "Calendar event created" in result

        svc = CalendarService()
        alice_titles = [e.title for e in svc.get_all_events(user_id=ALICE)]
        bob_titles = [e.title for e in svc.get_all_events(user_id=BOB)]
        assert "Alice tool event" in alice_titles
        assert "Alice tool event" not in bob_titles


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------


class TestSchedulerIsolation:
    def test_run_calendar_sync_runs_per_stored_owner(self, monkeypatch):
        from rex import services

        resolver = CalendarAccountResolver.from_raw_config(
            {
                "calendar": {
                    "accounts": [
                        {"id": "alice-cal", "provider": "stub"},
                        {"id": "bob-cal", "provider": "stub"},
                    ]
                },
                "users": {
                    ALICE: {"calendar_accounts": [{"account_id": "alice-cal"}]},
                    BOB: {"calendar_accounts": [{"account_id": "bob-cal"}]},
                },
            }
        )
        monkeypatch.setattr(CalendarAccountResolver, "load", classmethod(lambda cls: resolver))

        calendar = MagicMock()
        services._run_calendar_sync(calendar)
        users = [c.kwargs["user_id"] for c in calendar.refresh_upcoming.call_args_list]
        assert users == [ALICE, BOB]

    def test_one_owner_failure_does_not_fall_through(self, monkeypatch):
        from rex import services

        resolver = CalendarAccountResolver.from_raw_config(
            {
                "calendar": {
                    "accounts": [
                        {"id": "alice-cal", "provider": "stub"},
                        {"id": "bob-cal", "provider": "stub"},
                    ]
                },
                "users": {
                    ALICE: {"calendar_accounts": [{"account_id": "alice-cal"}]},
                    BOB: {"calendar_accounts": [{"account_id": "bob-cal"}]},
                },
            }
        )
        monkeypatch.setattr(CalendarAccountResolver, "load", classmethod(lambda cls: resolver))

        calendar = MagicMock()

        def refresh(*, user_id):
            if user_id == ALICE:
                raise RuntimeError("alice backend down")
            return []

        calendar.refresh_upcoming.side_effect = refresh
        services._run_calendar_sync(calendar)  # must not raise
        users = [c.kwargs["user_id"] for c in calendar.refresh_upcoming.call_args_list]
        assert users == [ALICE, BOB]

    def test_ownerless_setup_never_touches_named_users(self, monkeypatch):
        """With nothing configured, the legacy sync runs only as the
        explicit default profile — never as a named user."""
        from rex import services

        empty = CalendarAccountResolver.from_raw_config({})
        monkeypatch.setattr(CalendarAccountResolver, "load", classmethod(lambda cls: empty))

        calendar = MagicMock()
        services._run_calendar_sync(calendar)
        users = [c.kwargs["user_id"] for c in calendar.refresh_upcoming.call_args_list]
        assert users == ["default"]

    def test_scheduled_job_publishes_safe_envelope_per_owner(self, monkeypatch):
        from rex.integrations import _setup

        resolver = CalendarAccountResolver.from_raw_config(
            {
                "calendar": {"accounts": [{"id": "alice-cal", "provider": "stub"}]},
                "users": {ALICE: {"calendar_accounts": [{"account_id": "alice-cal"}]}},
            }
        )
        monkeypatch.setattr(CalendarAccountResolver, "load", classmethod(lambda cls: resolver))

        callbacks: dict[str, Any] = {}
        scheduler = MagicMock()
        scheduler.register_callback.side_effect = lambda name, fn: callbacks.__setitem__(name, fn)
        published: list[Any] = []
        bus = MagicMock()
        bus.publish.side_effect = published.append
        monkeypatch.setattr(_setup, "get_scheduler", lambda: scheduler)
        monkeypatch.setattr(_setup, "get_event_bus", lambda: bus)

        from rex.calendar_service import CalendarEvent

        start = datetime.now(UTC) + timedelta(hours=1)
        event = CalendarEvent(
            title="Alice secret sync event",
            start_time=start,
            end_time=start + timedelta(hours=1),
            attendees=["private@example.com"],
            location="Secret site",
        )
        calendar = MagicMock()
        calendar.get_upcoming_events.return_value = [event]
        monkeypatch.setattr(_setup, "get_calendar_service", lambda: calendar)

        _setup.setup_calendar_job()
        callbacks["sync_calendar"](MagicMock())

        assert calendar.get_upcoming_events.call_args.kwargs["user_id"] == ALICE

        by_topic = {e.event_type: e.payload for e in published}
        assert f"calendar.update.user.{ALICE}" in by_topic
        assert "calendar.update" in by_topic
        shared = by_topic["calendar.update"]
        assert set(shared) == {"count", "user_id"}
        assert "secret" not in str(shared).lower()
        private = by_topic[f"calendar.update.user.{ALICE}"]
        assert private["events"][0]["title"] == "Alice secret sync event"

    def test_scheduled_job_skips_when_no_owners(self, monkeypatch):
        from rex.integrations import _setup

        empty = CalendarAccountResolver.from_raw_config({})
        monkeypatch.setattr(CalendarAccountResolver, "load", classmethod(lambda cls: empty))

        callbacks: dict[str, Any] = {}
        scheduler = MagicMock()
        scheduler.register_callback.side_effect = lambda name, fn: callbacks.__setitem__(name, fn)
        bus = MagicMock()
        service_touched = {"value": False}

        def get_service():
            service_touched["value"] = True
            return MagicMock()

        monkeypatch.setattr(_setup, "get_scheduler", lambda: scheduler)
        monkeypatch.setattr(_setup, "get_event_bus", lambda: bus)
        monkeypatch.setattr(_setup, "get_calendar_service", get_service)

        _setup.setup_calendar_job()
        callbacks["sync_calendar"](MagicMock())

        assert service_touched["value"] is False
        bus.publish.assert_not_called()
