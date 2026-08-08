from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rex.ha.mutation_service import (
    HAMutation,
    HAMutationService,
    HAOutcome,
    HARisk,
    classify_ha_risk,
)


class RecordingHAClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def call_service(self, domain: str, service: str, data: dict[str, Any]) -> None:
        self.calls.append((domain, service, data))

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        return None


def _service(tmp_path: Path, client: RecordingHAClient) -> HAMutationService:
    return HAMutationService(
        client,
        confirmation_secret=b"us047-confirmation-secret",
        state_db=tmp_path / "ha.db",
        audit_path=tmp_path / "audit.jsonl",
        verification_interval_seconds=0,
        sleep=lambda _seconds: None,
    )


@pytest.mark.parametrize(
    ("domain", "entity_id", "service_name"),
    [
        ("lock", "lock.front_door", "unlock"),
        ("cover", "cover.garage", "open_cover"),
        ("alarm_control_panel", "alarm_control_panel.home", "alarm_disarm"),
        ("script", "script.cleanup", "turn_on"),
        ("scene", "scene.evening", "turn_on"),
    ],
)
def test_risky_domains_require_confirmation_before_dispatch_and_confirmed_retry_proceeds(
    tmp_path: Path, domain: str, entity_id: str, service_name: str
) -> None:
    client = RecordingHAClient()
    svc = _service(tmp_path, client)
    command = HAMutation(
        user_id="james",
        entity_id=entity_id,
        domain=domain,
        service=service_name,
        parameters={},
        request_id=f"us047-{domain}",
    )

    assert classify_ha_risk(domain, service_name) == HARisk.SENSITIVE

    pending = svc.execute(command)

    assert pending.status == HAOutcome.CONFIRMATION_REQUIRED
    assert pending.confirmation_token
    assert client.calls == []

    confirmed = svc.execute(replace(command, confirmation_token=pending.confirmation_token))

    assert confirmed.status in {HAOutcome.VERIFIED, HAOutcome.ATTEMPTED_UNVERIFIED}
    assert client.calls == [(domain, service_name, {"entity_id": entity_id})]
