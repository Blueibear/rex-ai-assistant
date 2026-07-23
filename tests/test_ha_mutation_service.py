from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from rex.ha.mutation_service import (
    HAMutation,
    HAMutationService,
    HAOutcome,
    HARisk,
    classify_ha_risk,
)


class FakeHAClient:
    def __init__(self, states: list[dict[str, Any] | None] | None = None) -> None:
        self.states = list(states or [])
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.call_error: Exception | None = None
        self.state_error: Exception | None = None

    def call_service(self, domain: str, service: str, data: dict[str, Any]) -> None:
        self.calls.append((domain, service, data))
        if self.call_error:
            raise self.call_error

    def get_state(self, entity_id: str) -> dict[str, Any] | None:
        if self.state_error:
            raise self.state_error
        return self.states.pop(0) if self.states else None


def service(tmp_path: Path, client: FakeHAClient, **kwargs) -> HAMutationService:
    return HAMutationService(
        client,
        confirmation_secret=b"test-confirmation-secret",
        state_db=tmp_path / "ha.db",
        audit_path=tmp_path / "audit.jsonl",
        verification_interval_seconds=0,
        sleep=lambda _seconds: None,
        **kwargs,
    )


def mutation(
    *,
    user: str = "james",
    entity: str = "light.kitchen",
    domain: str = "light",
    action: str = "turn_on",
    parameters: dict[str, Any] | None = None,
    request_id: str = "request-1",
) -> HAMutation:
    return HAMutation(user, entity, domain, action, parameters or {}, request_id)


def test_risk_classification_is_fail_closed() -> None:
    assert classify_ha_risk("light", "turn_on") == HARisk.SAFE
    assert classify_ha_risk("lock", "unlock") == HARisk.SENSITIVE
    assert classify_ha_risk("alarm_control_panel", "alarm_disarm") == HARisk.SENSITIVE
    assert classify_ha_risk("cover", "open_cover") == HARisk.SENSITIVE
    assert classify_ha_risk("script", "turn_on") == HARisk.PROHIBITED
    assert classify_ha_risk("unknown", "do_thing") == HARisk.PROHIBITED


def test_light_requires_observed_state_not_http_success(tmp_path: Path) -> None:
    client = FakeHAClient([{"state": "off", "attributes": {}}] * 4)
    result = service(tmp_path, client).execute(mutation())
    assert result.status == HAOutcome.ATTEMPTED_UNVERIFIED
    assert len(client.calls) == 1
    assert result.expected == {"state": "on", "attributes": {}}
    assert result.actual == {"state": "off", "attributes": {}}
    assert result.latency_ms >= 0


def test_light_verified_after_stale_state(tmp_path: Path) -> None:
    client = FakeHAClient(
        [
            {"state": "off", "attributes": {}},
            {"state": "on", "attributes": {"brightness": 128}},
        ]
    )
    command = mutation(parameters={"brightness_pct": 50})
    result = service(tmp_path, client).execute(command)
    assert result.status == HAOutcome.VERIFIED


def test_sensitive_confirmation_binds_user_action_and_expires(tmp_path: Path) -> None:
    now = [1000.0]
    client = FakeHAClient([{"state": "unlocked", "attributes": {}}])
    svc = service(tmp_path, client, now=lambda: now[0], confirmation_ttl_seconds=10)
    command = mutation(entity="lock.front", domain="lock", action="unlock")
    pending = svc.execute(command)
    assert pending.status == HAOutcome.CONFIRMATION_REQUIRED
    assert client.calls == []

    mismatched = svc.execute(
        replace(
            command,
            user_id="cole",
            request_id="request-cross-user",
            confirmation_token=pending.confirmation_token,
        )
    )
    assert mismatched.status == HAOutcome.DENIED
    now[0] = 1011.0
    expired = svc.execute(
        replace(
            command,
            request_id="request-expired",
            confirmation_token=pending.confirmation_token,
        )
    )
    assert expired.status == HAOutcome.DENIED
    assert "expired" in expired.detail.lower()


def test_confirmation_is_single_use_and_lock_state_is_verified(tmp_path: Path) -> None:
    client = FakeHAClient([{"state": "locked", "attributes": {}}])
    svc = service(tmp_path, client)
    first = mutation(entity="lock.front", domain="lock", action="lock")
    pending = svc.execute(first)
    confirmed = svc.execute(replace(first, confirmation_token=pending.confirmation_token))
    assert confirmed.status == HAOutcome.VERIFIED

    replay = svc.execute(
        replace(first, request_id="request-replay", confirmation_token=pending.confirmation_token)
    )
    assert replay.status == HAOutcome.DENIED
    assert "already been used" in replay.detail


def test_duplicate_request_returns_prior_result_without_second_write(tmp_path: Path) -> None:
    client = FakeHAClient([{"state": "on", "attributes": {}}])
    svc = service(tmp_path, client)
    command = mutation()
    assert svc.execute(command).status == HAOutcome.VERIFIED
    assert svc.execute(command).status == HAOutcome.VERIFIED
    assert len(client.calls) == 1


def test_cross_user_request_id_is_denied(tmp_path: Path) -> None:
    client = FakeHAClient([{"state": "on", "attributes": {}}])
    svc = service(tmp_path, client)
    assert svc.execute(mutation()).status == HAOutcome.VERIFIED
    assert svc.execute(mutation(user="cole")).status == HAOutcome.DENIED


def test_timeout_after_possible_write_is_unverified_not_failed(tmp_path: Path) -> None:
    client = FakeHAClient()
    client.call_error = TimeoutError("network timeout")
    result = service(tmp_path, client).execute(mutation())
    assert result.status == HAOutcome.ATTEMPTED_UNVERIFIED


def test_transport_failure_is_failed(tmp_path: Path) -> None:
    client = FakeHAClient()
    client.call_error = ConnectionError("offline")
    result = service(tmp_path, client).execute(mutation())
    assert result.status == HAOutcome.FAILED


def test_state_query_failure_after_write_is_unverified(tmp_path: Path) -> None:
    client = FakeHAClient()
    client.state_error = ConnectionError("state endpoint unavailable")
    result = service(tmp_path, client).execute(mutation())
    assert result.status == HAOutcome.ATTEMPTED_UNVERIFIED


def test_invalid_or_prohibited_commands_never_dispatch(tmp_path: Path) -> None:
    client = FakeHAClient()
    svc = service(tmp_path, client)
    mismatch = mutation(entity="lock.front", domain="light")
    prohibited = mutation(entity="script.cleanup", domain="script")
    assert svc.execute(mismatch).status == HAOutcome.DENIED
    assert svc.execute(prohibited).status == HAOutcome.DENIED
    assert client.calls == []


def test_audit_evidence_is_redacted(tmp_path: Path) -> None:
    client = FakeHAClient([{"state": "on", "attributes": {}}])
    svc = service(tmp_path, client)
    svc.execute(mutation(parameters={"secret_code": "do-not-log"}))  # pragma: allowlist secret
    audit = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "do-not-log" not in audit
    assert "confirmation" not in audit
    assert '"status": "verified"' in audit
