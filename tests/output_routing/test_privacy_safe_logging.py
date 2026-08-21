from __future__ import annotations

import logging
from datetime import UTC, datetime

from rex.media.models import AudioTarget, MediaCapability, TargetKind
from rex.media.registry import AudioTargetRegistry
from rex.output_routing.models import OutputKind
from rex.output_routing.service import OutputRoutingService


def test_route_decision_log_is_structured_and_content_free(caplog, tmp_path) -> None:
    target = AudioTarget(
        id="test:private-bedroom",
        native_id="private-bedroom",
        provider="test",
        kind=TargetKind.SPEAKER,
        display_name="Private Bedroom Speaker",
        aliases=(),
        room="private bedroom",
        capabilities=frozenset({MediaCapability.PLAY}),
        online=True,
        health="healthy",
    )
    registry = AudioTargetRegistry(
        (target,),
        authorized_target_ids={"james": {target.id}},
    )
    routing = OutputRoutingService(registry, root=tmp_path)

    with caplog.at_level(logging.INFO, logger="rex.output_routing.service"):
        route = routing.resolve(
            user_id="james",
            output_kind=OutputKind.MEDIA,
            explicit_target_text="private bedroom",
            origin_device_id=None,
            at=datetime(2026, 8, 21, 12, 0, tzinfo=UTC),
        )

    assert route.target_id == target.id
    records = [
        record
        for record in caplog.records
        if getattr(record, "event", None) == "output_routing_decision"
    ]
    assert len(records) == 1
    record = records[0]
    assert record.output_kind == "media"
    assert record.reason == "explicit_target"
    assert record.has_target is True
    assert record.fallback_mode is None
    assert record.rule_index is None
    assert record.suppressed is False
    assert record.requires_confirmation is False
    assert record.volume_configured is False

    rendered = record.getMessage().casefold()
    assert "james" not in rendered
    assert "private bedroom" not in rendered
    assert target.id.casefold() not in rendered
    assert not hasattr(record, "user_id")
    assert not hasattr(record, "target_id")
    assert not hasattr(record, "explicit_target_text")
