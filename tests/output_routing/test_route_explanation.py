from __future__ import annotations

from datetime import time

from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.models import UserOutputPolicy
from rex.output_routing.service import OutputRoutingService
from rex.timekeeping.parser import parse_timekeeping_command
from rex.timekeeping.runtime import set_timekeeping_service, shutdown_timekeeping_runtime
from rex.timekeeping.service import TimekeepingService
from rex.timekeeping.tools import timekeeping_read


def _target() -> AudioTarget:
    return AudioTarget(
        id="test:bedroom",
        native_id="bedroom",
        provider="test",
        kind=TargetKind.SPEAKER,
        display_name="Bedroom Speaker",
        aliases=(),
        room="bedroom",
        capabilities=frozenset({MediaCapability.PLAY}),
        online=True,
        health="healthy",
    )


def test_parser_recognizes_named_alarm_route_question() -> None:
    command = parse_timekeeping_command(
        "where will my morning alarm play?",
        user_timezone="America/Chicago",
    )

    assert command is not None
    assert command.action == "query_alarm_route"
    assert command.reference == "morning"


def test_timekeeping_read_explains_resolved_alarm_route(monkeypatch, tmp_path) -> None:
    target = _target()
    registry = AudioTargetRegistry(
        (target,),
        authorized_target_ids={"james": {target.id}},
    )
    routing = OutputRoutingService(registry, root=tmp_path / "routing")
    routing.save_policy("james", UserOutputPolicy(alarm_target_id=target.id))

    timekeeping = TimekeepingService(tmp_path / "timekeeping.json")
    set_timekeeping_service(timekeeping)
    try:
        alarm = timekeeping.create_alarm(
            "james",
            local_time=time(7, 0),
            timezone_name="America/Chicago",
            weekdays=(0, 1, 2, 3, 4),
            name="morning",
        )
        monkeypatch.setattr(
            "rex.timekeeping.tools.get_output_routing_service",
            lambda: routing,
        )

        result = timekeeping_read(
            transcript="where will my morning alarm play?",
            _user_id="james",
            timezone_name="America/Chicago",
        )
    finally:
        shutdown_timekeeping_runtime()
        set_timekeeping_service(None)

    assert result["alarm"]["alarm_id"] == alarm.alarm_id
    assert result["route"] == {
        "target_id": target.id,
        "reason": "configured_default",
        "fallback_mode": None,
        "fallback_from": None,
        "rule_index": None,
        "target_volume": None,
        "suppressed": False,
        "requires_confirmation": False,
    }
