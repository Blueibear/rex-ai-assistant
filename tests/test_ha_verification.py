from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from rex.ha.mutation_service import HAMutation, HAMutationService, HAOutcome


class FakeHAClient:
    def __init__(self, states: list[dict[str, Any] | None]) -> None:
        self.states = list(states)
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def call_service(self, domain: str, service: str, data: dict[str, Any]) -> None:
        self.calls.append((domain, service, data))

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        return self.states.pop(0) if self.states else None


def _service(tmp_path: Path, client: FakeHAClient) -> HAMutationService:
    return HAMutationService(
        client,
        confirmation_secret=b"us048-confirmation-secret",
        state_db=tmp_path / "ha.db",
        audit_path=tmp_path / "audit.jsonl",
        verification_interval_seconds=0,
        sleep=lambda _seconds: None,
    )


def _execute(
    tmp_path: Path,
    *,
    domain: str,
    service: str,
    entity_id: str,
    observed_state: str,
) -> tuple[dict[str, Any], FakeHAClient]:
    client = FakeHAClient([{"state": observed_state, "attributes": {}}])
    mutation = HAMutation(
        user_id="james",
        entity_id=entity_id,
        domain=domain,
        service=service,
        parameters={},
        request_id=f"us048-{domain}-{service}",
    )
    svc = _service(tmp_path, client)
    result = svc.execute(mutation)
    if result.status == HAOutcome.CONFIRMATION_REQUIRED:
        result = svc.execute(replace(mutation, confirmation_token=result.confirmation_token))
    return result.to_dict(), client


@pytest.mark.parametrize(
    ("domain", "service", "entity_id", "observed_state", "expected_state"),
    [
        ("switch", "turn_on", "switch.fan", "on", "on"),
        ("light", "turn_off", "light.kitchen", "off", "off"),
        ("lock", "lock", "lock.front", "locked", "locked"),
        ("cover", "open_cover", "cover.garage", "open", "open"),
    ],
)
def test_switchable_domains_return_verified_state_evidence(
    tmp_path: Path,
    domain: str,
    service: str,
    entity_id: str,
    observed_state: str,
    expected_state: str,
) -> None:
    result, client = _execute(
        tmp_path,
        domain=domain,
        service=service,
        entity_id=entity_id,
        observed_state=observed_state,
    )

    assert result["status"] == "verified"
    assert result["expected"] == {"state": expected_state, "attributes": {}}
    assert result["actual"] == {"state": observed_state, "attributes": {}}
    assert isinstance(result["latency_ms"], float)
    assert result["latency_ms"] >= 0
    assert len(client.calls) == 1


def test_state_did_not_change_returns_attempted_with_evidence(tmp_path: Path) -> None:
    client = FakeHAClient([{"state": "off", "attributes": {}}] * 4)
    svc = _service(tmp_path, client)
    result = svc.execute(
        HAMutation(
            user_id="james",
            entity_id="switch.fan",
            domain="switch",
            service="turn_on",
            parameters={},
            request_id="us048-stale-switch",
        )
    ).to_dict()

    assert result["status"] == "attempted_unverified"
    assert result["expected"] == {"state": "on", "attributes": {}}
    assert result["actual"] == {"state": "off", "attributes": {}}
    assert result["latency_ms"] >= 0


def test_electron_device_command_contract_preserves_verification_evidence() -> None:
    home_assistant = Path("gui/src/main/homeAssistant.ts").read_text(encoding="utf-8")
    ipc_types = Path("gui/src/types/ipc.ts").read_text(encoding="utf-8")

    for source in (home_assistant, ipc_types):
        assert "expected?:" in source
        assert "actual?:" in source
        assert "latencyMs?: number" in source

    assert "expected: result.expected" in home_assistant
    assert "actual: result.actual" in home_assistant
    assert "latencyMs: result.latency_ms" in home_assistant
