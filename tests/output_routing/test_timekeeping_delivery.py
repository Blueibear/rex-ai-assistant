from __future__ import annotations

from datetime import UTC, datetime

from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.delivery import OutputDeliveryService
from rex.output_routing.models import FallbackMode, UserOutputPolicy
from rex.output_routing.service import OutputRoutingService
from rex.timekeeping.models import DueEvent

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def _target(target_id: str, name: str, *, online: bool = True) -> AudioTarget:
    return AudioTarget(
        id=target_id,
        native_id=target_id,
        provider="test",
        kind=TargetKind.SPEAKER,
        display_name=name,
        aliases=(),
        room=name.replace(" Speaker", ""),
        capabilities=frozenset({MediaCapability.PLAY}),
        online=online,
        health="healthy" if online else "offline",
    )


def test_due_event_uses_current_named_fallback_when_default_is_offline(tmp_path) -> None:
    offline = _target("test:bedroom", "Bedroom Speaker", online=False)
    kitchen = _target("test:kitchen", "Kitchen Speaker")
    registry = AudioTargetRegistry(
        (offline, kitchen),
        authorized_target_ids={"james": {offline.id, kitchen.id}},
    )
    routing = OutputRoutingService(registry, root=tmp_path)
    routing.save_policy(
        "james",
        UserOutputPolicy(
            alarm_target_id=offline.id,
            alarm_fallback=FallbackMode.NAMED,
            alarm_fallback_target_id=kitchen.id,
        ),
    )
    delivered: list[tuple[str, DueEvent]] = []
    service = OutputDeliveryService(
        routing,
        sender=lambda target_id, event: delivered.append((target_id, event)) or True,
        now_func=lambda: NOW,
    )
    event = DueEvent(
        kind="alarm",
        record_id="alm_1",
        user_id="james",
        name="Morning",
        fired_at=NOW,
    )

    result = service.deliver_due_event(event)

    assert result.delivered is True
    assert result.target_id == kitchen.id
    assert result.reason == "named_fallback"
    assert delivered == [(kitchen.id, event)]


def test_due_event_explicit_target_beats_current_default(tmp_path) -> None:
    bedroom = _target("test:bedroom", "Bedroom Speaker")
    kitchen = _target("test:kitchen", "Kitchen Speaker")
    registry = AudioTargetRegistry(
        (bedroom, kitchen),
        authorized_target_ids={"james": {bedroom.id, kitchen.id}},
    )
    routing = OutputRoutingService(registry, root=tmp_path)
    routing.save_policy("james", UserOutputPolicy(timer_target_id=kitchen.id))
    delivered: list[str] = []
    service = OutputDeliveryService(
        routing,
        sender=lambda target_id, _event: delivered.append(target_id) or True,
        now_func=lambda: NOW,
    )
    event = DueEvent(
        kind="timer",
        record_id="tmr_1",
        user_id="james",
        name="Pasta",
        fired_at=NOW,
        output_target_id=bedroom.id,
    )

    result = service.deliver_due_event(event)

    assert result.delivered is True
    assert result.target_id == bedroom.id
    assert result.reason == "explicit_target"
    assert delivered == [bedroom.id]


def test_due_event_reports_unavailable_route_without_false_success(tmp_path) -> None:
    registry = AudioTargetRegistry((), authorized_target_ids={"james": set()})
    routing = OutputRoutingService(registry, root=tmp_path)
    service = OutputDeliveryService(
        routing,
        sender=lambda _target_id, _event: True,
        now_func=lambda: NOW,
    )
    event = DueEvent(
        kind="timer",
        record_id="tmr_1",
        user_id="james",
        name=None,
        fired_at=NOW,
    )

    result = service.deliver_due_event(event)

    assert result.delivered is False
    assert result.target_id is None
    assert result.reason == "target_required"
