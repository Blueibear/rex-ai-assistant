"""Policy-controlled Home Assistant mutations with independent verification."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from rex.identity import validate_user_id


class HARisk(StrEnum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    PROHIBITED = "prohibited"


class HAOutcome(StrEnum):
    VERIFIED = "verified"
    ATTEMPTED_UNVERIFIED = "attempted_unverified"
    CONFIRMATION_REQUIRED = "confirmation_required"
    DENIED = "denied"
    FAILED = "failed"


@dataclass(frozen=True)
class HAMutation:
    user_id: str
    entity_id: str
    domain: str
    service: str
    parameters: dict[str, Any]
    request_id: str
    confirmation_token: str | None = None


@dataclass
class HAMutationResult:
    status: HAOutcome
    detail: str
    entity_id: str
    domain: str
    service: str
    request_id: str
    risk: HARisk
    confirmation_token: str | None = None
    expected: dict[str, Any] | None = None
    actual: dict[str, Any] | None = None
    latency_ms: float = 0.0

    @property
    def success(self) -> bool:
        return self.status == HAOutcome.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["status"] = self.status.value
        value["risk"] = self.risk.value
        value["success"] = self.success
        return value


class HAClient(Protocol):
    def call_service(self, domain: str, service: str, data: dict[str, Any]) -> None: ...

    def get_state(self, entity_id: str) -> dict[str, Any] | None: ...


_SAFE_DOMAINS = {"light", "switch", "fan", "climate", "media_player"}
_SENSITIVE_DOMAINS = {"lock", "alarm_control_panel", "cover"}
_PROHIBITED_DOMAINS = {
    "automation",
    "hassio",
    "homeassistant",
    "python_script",
    "script",
    "shell_command",
    "update",
}


def _runtime_data_dir() -> Path:
    configured = os.environ.get("REX_DATA_DIR")
    if configured:
        return Path(configured)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base / "rex-ai"


def classify_ha_risk(domain: str, service: str) -> HARisk:
    domain = domain.strip().lower()
    service = service.strip().lower()
    if domain in _PROHIBITED_DOMAINS:
        return HARisk.PROHIBITED
    if domain in _SENSITIVE_DOMAINS:
        return HARisk.SENSITIVE
    if domain in _SAFE_DOMAINS and service:
        return HARisk.SAFE
    return HARisk.PROHIBITED


class HAMutationService:
    def __init__(
        self,
        client: HAClient,
        *,
        confirmation_secret: bytes,
        state_db: Path | None = None,
        audit_path: Path | None = None,
        confirmation_ttl_seconds: int = 120,
        verification_attempts: int = 4,
        verification_interval_seconds: float = 0.25,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], Any] = time.sleep,
        authorized_users: set[str] | None = None,
    ) -> None:
        if not confirmation_secret:
            raise ValueError("confirmation_secret is required")
        self._client = client
        self._secret = confirmation_secret
        runtime_dir = _runtime_data_dir()
        self._state_db = state_db or runtime_dir / "ha_mutations.db"
        self._audit_path = audit_path or runtime_dir / "logs" / "ha_mutations.jsonl"
        self._ttl = confirmation_ttl_seconds
        self._verification_attempts = verification_attempts
        self._verification_interval = verification_interval_seconds
        self._now = now
        self._sleep = sleep
        self._authorized_users = authorized_users
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        self._state_db.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self._state_db)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ha_requests ("
                "request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, result_json TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ha_confirmations ("
                "token_hash TEXT PRIMARY KEY, consumed_at REAL NOT NULL)"
            )

    @staticmethod
    def _canonical_action(mutation: HAMutation) -> dict[str, Any]:
        return {
            "user_id": mutation.user_id,
            "entity_id": mutation.entity_id,
            "domain": mutation.domain,
            "service": mutation.service,
            "parameters": mutation.parameters,
        }

    def issue_confirmation(self, mutation: HAMutation) -> str:
        payload = {
            **self._canonical_action(mutation),
            "expires_at": int(self._now()) + self._ttl,
            "nonce": uuid.uuid4().hex,
        }
        encoded = (
            base64.urlsafe_b64encode(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            )
            .decode()
            .rstrip("=")
        )
        signature = hmac.new(self._secret, encoded.encode(), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def _consume_confirmation(self, mutation: HAMutation, token: str) -> tuple[bool, str]:
        try:
            encoded, provided_signature = token.rsplit(".", 1)
            expected_signature = hmac.new(
                self._secret, encoded.encode(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(provided_signature, expected_signature):
                return False, "Confirmation signature is invalid."
            padded = encoded + "=" * (-len(encoded) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        except (ValueError, json.JSONDecodeError):
            return False, "Confirmation token is invalid."

        if payload.get("expires_at", 0) < int(self._now()):
            return False, "Confirmation has expired."
        expected = self._canonical_action(mutation)
        if any(payload.get(key) != value for key, value in expected.items()):
            return False, "Confirmation does not match this user and action."

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO ha_confirmations (token_hash, consumed_at) VALUES (?, ?)",
                    (token_hash, self._now()),
                )
        except sqlite3.IntegrityError:
            return False, "Confirmation has already been used."
        return True, "confirmed"

    def execute(self, mutation: HAMutation) -> HAMutationResult:
        started = time.perf_counter()
        try:
            user_id = validate_user_id(mutation.user_id)
            domain = mutation.domain.strip().lower()
            service = mutation.service.strip().lower()
            entity_id = mutation.entity_id.strip()
            if not mutation.request_id.strip():
                raise ValueError("request_id is required")
            if entity_id != f"{domain}.{entity_id.split('.', 1)[-1]}" or "." not in entity_id:
                raise ValueError("entity_id must match the requested domain")
            if mutation.parameters.get("entity_id", entity_id) != entity_id:
                raise ValueError("parameters.entity_id must match entity_id")
        except ValueError as exc:
            return self._result(mutation, HAOutcome.DENIED, HARisk.PROHIBITED, str(exc))

        try:
            prior = self._prior_result(mutation.request_id, user_id)
        except PermissionError as exc:
            return self._result(mutation, HAOutcome.DENIED, HARisk.PROHIBITED, str(exc))
        if prior is not None:
            return prior

        risk = classify_ha_risk(domain, service)
        if self._authorized_users is not None and user_id not in self._authorized_users:
            result = self._result(mutation, HAOutcome.DENIED, risk, "User is not permitted.")
            return self._record(result, user_id)
        if risk == HARisk.PROHIBITED:
            result = self._result(
                mutation, HAOutcome.DENIED, risk, "This Home Assistant action is prohibited."
            )
            return self._record(result, user_id)
        if risk == HARisk.SENSITIVE:
            if not mutation.confirmation_token:
                token = self.issue_confirmation(mutation)
                return self._result(
                    mutation,
                    HAOutcome.CONFIRMATION_REQUIRED,
                    risk,
                    f"Confirm {domain}.{service} for {entity_id} within {self._ttl} seconds.",
                    token,
                )
            valid, reason = self._consume_confirmation(mutation, mutation.confirmation_token)
            if not valid:
                result = self._result(mutation, HAOutcome.DENIED, risk, reason)
                return self._record(result, user_id)

        data = {**mutation.parameters, "entity_id": entity_id}
        try:
            self._client.call_service(domain, service, data)
        except Exception as exc:
            uncertain = isinstance(exc, TimeoutError) or exc.__class__.__name__.lower().endswith(
                "timeout"
            )
            status = HAOutcome.ATTEMPTED_UNVERIFIED if uncertain else HAOutcome.FAILED
            prefix = "Dispatch timed out after a possible write" if uncertain else "Dispatch failed"
            result = self._result(mutation, status, risk, f"{prefix}: {exc}")
            result.latency_ms = (time.perf_counter() - started) * 1000
            return self._record(result, user_id)

        expected = self._expected_state(domain, service, data)
        if expected is None:
            result = self._result(
                mutation,
                HAOutcome.ATTEMPTED_UNVERIFIED,
                risk,
                "Home Assistant accepted the request, but no independent state proof is defined.",
            )
            result.latency_ms = (time.perf_counter() - started) * 1000
            return self._record(result, user_id)

        last_state: dict[str, Any] | None = None
        for attempt in range(self._verification_attempts):
            try:
                state = self._client.get_state(entity_id)
            except Exception:
                state = None
            last_state = state
            if self._state_matches(state, expected):
                result = self._result(
                    mutation, HAOutcome.VERIFIED, risk, f"Verified {entity_id} state."
                )
                result.expected = {"state": expected[0], "attributes": expected[1]}
                result.actual = state
                result.latency_ms = (time.perf_counter() - started) * 1000
                return self._record(result, user_id)
            if attempt + 1 < self._verification_attempts:
                self._sleep(self._verification_interval)

        result = self._result(
            mutation,
            HAOutcome.ATTEMPTED_UNVERIFIED,
            risk,
            "The service call was attempted, but the requested state was not observed.",
        )
        result.expected = {"state": expected[0], "attributes": expected[1]}
        result.actual = last_state
        result.latency_ms = (time.perf_counter() - started) * 1000
        return self._record(result, user_id)

    @staticmethod
    def _expected_state(
        domain: str, service: str, data: dict[str, Any]
    ) -> tuple[str, dict[str, Any]] | None:
        state_by_action = {
            ("lock", "lock"): "locked",
            ("lock", "unlock"): "unlocked",
            ("cover", "open_cover"): "open",
            ("cover", "close_cover"): "closed",
            ("alarm_control_panel", "alarm_disarm"): "disarmed",
            ("alarm_control_panel", "alarm_arm_home"): "armed_home",
            ("alarm_control_panel", "alarm_arm_away"): "armed_away",
        }
        if service == "turn_on":
            attrs = {key: data[key] for key in ("brightness", "brightness_pct") if key in data}
            return "on", attrs
        if service == "turn_off":
            return "off", {}
        state = state_by_action.get((domain, service))
        return (state, {}) if state else None

    @staticmethod
    def _state_matches(state: dict[str, Any] | None, expected: tuple[str, dict[str, Any]]) -> bool:
        if not state or state.get("state") != expected[0]:
            return False
        raw_attributes = state.get("attributes")
        attributes: dict[str, Any] = raw_attributes if isinstance(raw_attributes, dict) else {}
        for key, value in expected[1].items():
            if key == "brightness_pct":
                observed = attributes.get("brightness")
                target = round(float(value) * 255 / 100)
                if not isinstance(observed, (int, float)) or abs(observed - target) > 3:
                    return False
            elif attributes.get(key) != value:
                return False
        return True

    def _prior_result(self, request_id: str, user_id: str) -> HAMutationResult | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id, result_json FROM ha_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
        if row is None:
            return None
        if row[0] != user_id:
            raise PermissionError("request_id belongs to another user")
        payload = json.loads(row[1])
        payload["status"] = HAOutcome(payload["status"])
        payload["risk"] = HARisk(payload["risk"])
        payload.pop("success", None)
        return HAMutationResult(**payload)

    def _record(self, result: HAMutationResult, user_id: str) -> HAMutationResult:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO ha_requests (request_id, user_id, result_json) VALUES (?, ?, ?)",
                (result.request_id, user_id, json.dumps(result.to_dict(), sort_keys=True)),
            )
        self._audit(result, user_id)
        return result

    def _audit(self, result: HAMutationResult, user_id: str) -> None:
        self._audit_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = {
            "timestamp": datetime.now(UTC).isoformat(),
            "user_id": user_id,
            "request_id_hash": hashlib.sha256(result.request_id.encode()).hexdigest()[:16],
            "entity_id": result.entity_id,
            "domain": result.domain,
            "service": result.service,
            "risk": result.risk.value,
            "status": result.status.value,
        }
        with self._audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(evidence, sort_keys=True) + "\n")

    @staticmethod
    def _result(
        mutation: HAMutation,
        status: HAOutcome,
        risk: HARisk,
        detail: str,
        confirmation_token: str | None = None,
    ) -> HAMutationResult:
        return HAMutationResult(
            status=status,
            detail=detail,
            entity_id=mutation.entity_id,
            domain=mutation.domain,
            service=mutation.service,
            request_id=mutation.request_id,
            risk=risk,
            confirmation_token=confirmation_token,
        )


__all__ = [
    "HAClient",
    "HAMutation",
    "HAMutationResult",
    "HAMutationService",
    "HAOutcome",
    "HARisk",
    "classify_ha_risk",
]
