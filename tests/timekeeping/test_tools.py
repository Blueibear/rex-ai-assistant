from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest

from rex.capabilities.registry import CapabilityRegistry
from rex.timekeeping.runtime import set_timekeeping_service, shutdown_timekeeping_runtime
from rex.timekeeping.service import TimekeepingService
from rex.tools.dispatcher import ToolDispatcher
from rex.tools.registry import _build_default_registry


@pytest.fixture
def service(tmp_path, monkeypatch):
    instance = TimekeepingService(tmp_path / "timekeeping.json")
    set_timekeeping_service(instance)
    monkeypatch.setattr(
        "rex.timekeeping.tools.resolve_user_timezone", lambda user_id: "America/Chicago"
    )
    yield instance
    shutdown_timekeeping_runtime()
    set_timekeeping_service(None)


def test_timekeeping_tools_have_canonical_security_metadata() -> None:
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    read = registry.get("timekeeping_read")
    manage = registry.get("timekeeping_manage")

    assert read is not None and read.operation == "read" and read.requires_identity is True
    assert manage is not None and manage.operation == "mutation"
    assert manage.requires_identity is True
    assert manage.verifier is not None

    read_card = registry.capability_registry.get("timekeeping_read")
    manage_card = registry.capability_registry.get("timekeeping_manage")
    assert read_card is not None and read_card.verification_supported is False
    assert manage_card is not None and manage_card.verification_supported is True


def test_manage_tool_creates_verified_timer_and_read_tool_reports_remaining(service) -> None:
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    created = dispatcher.dispatch(
        "timekeeping_manage",
        {"transcript": "set a 10-minute pasta timer"},
        {"user_id": "james", "request_id": "create-pasta"},
    )
    assert created.success is True
    assert created.status == "verified"
    assert created.output["record_type"] == "timer"
    assert created.output["name"] == "pasta"

    queried = dispatcher.dispatch(
        "timekeeping_read",
        {"transcript": "how much time is left on the pasta timer"},
        {"user_id": "james"},
    )
    assert queried.success is True
    assert queried.output["timer"]["name"] == "pasta"
    assert 590 <= queried.output["timer"]["remaining_seconds"] <= 600


def test_read_tool_does_not_leak_other_users_timer(service) -> None:
    service.create_timer("james", 600, name="private")
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "timekeeping_read",
        {"transcript": "how much time is left on the private timer"},
        {"user_id": "cole"},
    )

    assert result.success is True
    assert result.output["found"] is False
    assert "private" not in result.output.get("details", "")


def test_read_tool_surfaces_ambiguous_same_name_without_guessing(service) -> None:
    service.create_timer("james", 300, name="pasta")
    service.create_timer("james", 600, name="pasta")
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "timekeeping_read",
        {"transcript": "how much time is left on the pasta timer"},
        {"user_id": "james"},
    )

    assert result.success is True
    assert result.output["ambiguous"] is True
    assert len(result.output["matches"]) == 2


def test_manage_tool_requires_identity(service) -> None:
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "timekeeping_manage",
        {"transcript": "set a 5-minute timer"},
        {},
    )

    assert result.success is False
    assert result.status == "denied"


def test_structured_alarm_create_is_verified(service) -> None:
    chicago = ZoneInfo("America/Chicago")
    alarm_date = (datetime.now(chicago) + timedelta(days=2)).date()
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "timekeeping_manage",
        {
            "action": "create_alarm",
            "alarm_time": "07:00",
            "alarm_date": alarm_date.isoformat(),
            "timezone_name": "America/Chicago",
            "reference": "morning",
        },
        {"user_id": "james", "request_id": "alarm-create"},
    )

    assert result.success is True
    assert result.status == "verified"
    alarm = service.get_alarm(result.output["record_id"], "james")
    assert alarm is not None
    assert alarm.next_fire_at == datetime.combine(alarm_date, time(7), tzinfo=chicago).astimezone(
        UTC
    )


def test_mobile_timekeeping_uses_explicit_task_scopes(service) -> None:
    from rex.mobile_api.action_context import MobileActionDeniedError, mobile_action_context

    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    with mobile_action_context(frozenset({"tasks.write"})):
        created = dispatcher.dispatch(
            "timekeeping_manage",
            {"action": "create_timer", "duration_seconds": 60, "reference": "mobile"},
            {"user_id": "james", "request_id": "mobile-timer"},
        )
    assert created.success is True

    with mobile_action_context(frozenset({"chat.send"})):
        with pytest.raises(MobileActionDeniedError):
            dispatcher.dispatch(
                "timekeeping_manage",
                {"action": "create_timer", "duration_seconds": 60},
                {"user_id": "james", "request_id": "mobile-timer-denied"},
            )

    with mobile_action_context(frozenset({"tasks.read"})):
        listed = dispatcher.dispatch(
            "timekeeping_read",
            {"action": "list_timers"},
            {"user_id": "james"},
        )
    assert listed.success is True


def test_structured_alarm_edit_can_change_recurrence_and_timezone(service) -> None:
    alarm_date = (datetime.now(ZoneInfo("America/Chicago")) + timedelta(days=2)).date()
    alarm = service.create_alarm(
        "james",
        local_time=datetime.strptime("07:00", "%H:%M").time(),
        timezone_name="America/Chicago",
        local_date=alarm_date,
        name="work",
    )
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "timekeeping_manage",
        {
            "action": "edit_alarm",
            "reference": alarm.alarm_id,
            "alarm_time": "08:30",
            "weekdays": [0, 2, 4],
            "timezone_name": "America/New_York",
        },
        {"user_id": "james", "request_id": "edit-work-alarm"},
    )

    assert result.success is True
    assert result.status == "verified"
    edited = service.get_alarm(alarm.alarm_id, "james")
    assert edited is not None
    assert edited.local_time == "08:30:00"
    assert edited.weekdays == (0, 2, 4)
    assert edited.local_date is None
    assert edited.timezone_name == "America/New_York"
