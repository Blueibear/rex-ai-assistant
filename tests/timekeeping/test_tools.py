from __future__ import annotations

from datetime import UTC, datetime

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
    registry = _build_default_registry(capability_registry=CapabilityRegistry())
    dispatcher = ToolDispatcher(registry)

    result = dispatcher.dispatch(
        "timekeeping_manage",
        {
            "action": "create_alarm",
            "alarm_time": "07:00",
            "alarm_date": "2026-08-17",
            "timezone_name": "America/Chicago",
            "reference": "morning",
        },
        {"user_id": "james", "request_id": "alarm-create"},
    )

    assert result.success is True
    assert result.status == "verified"
    alarm = service.get_alarm(result.output["record_id"], "james")
    assert alarm is not None
    assert alarm.next_fire_at == datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
