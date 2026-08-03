"""One-time, action-bound strong authentication for privileged mobile actions (S8).

The server issues a short-lived challenge for one exact canonical action.  The
paired device signs a domain-separated transcript with its enrolled P-256 key.
Successful verification creates a second short-lived approval identifier that
is consumed atomically by the exact action.  Neither a recent biometric
 timestamp nor a reusable session flag can authorize another action.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import secrets
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from rex.ha.mutation_service import HARisk, classify_ha_risk
from rex.mobile_api.authorization import require_scope
from rex.mobile_api.db import connect
from rex.mobile_api.device_proof import ProofError, verify_proof

logger = logging.getLogger(__name__)

STRONG_AUTH_DOMAIN = b"AskRex-Strong-Auth-v1"
STRONG_AUTH_TRANSCRIPT_TYPE = "askrex-strong-auth-proof"
STRONG_AUTH_TRANSCRIPT_VERSION = 1
CHALLENGE_TTL_SECONDS = 90
APPROVAL_TTL_SECONDS = 45
_MAX_ACTION_BYTES = 32 * 1024
_MAX_DEPTH = 10
_MAX_COLLECTION_ITEMS = 256
_MAX_STRING_CHARS = 8_000
_SERVER_ACTION_FIELDS = {"_user_id", "_request_id", "context", "strong_auth_approval_id"}
_HA_ACTION_FIELDS = {"domain", "service", "entity_id", "data"}


class MobileActionRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class ActionPolicy:
    action_name: str
    required_scope: str
    risk_level: MobileActionRisk

    @property
    def requires_strong_auth(self) -> bool:
        return self.risk_level in {MobileActionRisk.HIGH, MobileActionRisk.CRITICAL}


@dataclass(frozen=True)
class StrongAuthChallenge:
    challenge_id: str
    action_name: str
    action_hash: str
    risk_level: str
    required_scope: str
    nonce_b64: str
    desktop_id: str
    session_id: str
    user_id: str
    device_id: str
    grant_id: str
    grant_version: int
    expires_at: str


@dataclass(frozen=True)
class StrongAuthApproval:
    approval_id: str
    action_name: str
    action_hash: str
    risk_level: str
    expires_at: str


class PrincipalBinding(Protocol):
    user_id: str
    session_id: str
    paired_device_id: str | None
    grant_id: str | None
    desktop_id: str | None
    grant_version: int | None
    scopes: frozenset[str]
    permissions: frozenset[str]


class StrongAuthError(PermissionError):
    """Stable, secret-free S8 denial with a machine-readable reason."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)


def _normalize_action_name(value: object) -> str:
    if not isinstance(value, str):
        raise StrongAuthError("invalid_action", "The requested action is invalid.")
    name = value.strip().lower()
    if (
        not name
        or len(name) > 128
        or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789_.:-" for ch in name)
    ):
        raise StrongAuthError("invalid_action", "The requested action is invalid.")
    return name


def _canonical_json_value(value: Any, *, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        raise StrongAuthError("invalid_payload", "The action payload is too deeply nested.")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and len(value) > _MAX_STRING_CHARS:
            raise StrongAuthError(
                "invalid_payload", "The action payload contains an oversized string."
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StrongAuthError(
                "invalid_payload", "The action payload contains an invalid number."
            )
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise StrongAuthError("invalid_payload", "The action payload contains too many fields.")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise StrongAuthError(
                    "invalid_payload", "The action payload contains an invalid field name."
                )
            result[key] = _canonical_json_value(item, depth=depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > _MAX_COLLECTION_ITEMS:
            raise StrongAuthError("invalid_payload", "The action payload contains too many items.")
        return [_canonical_json_value(item, depth=depth + 1) for item in value]
    raise StrongAuthError("invalid_payload", "The action payload contains an unsupported value.")


def _canonical_ha_action(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise StrongAuthError("invalid_payload", "The action payload must be an object.")
    supplied = set(payload) - _SERVER_ACTION_FIELDS
    missing = {"domain", "service", "entity_id"} - supplied
    unknown = supplied - _HA_ACTION_FIELDS
    if missing or unknown:
        raise StrongAuthError(
            "invalid_payload",
            "The Home Assistant action fields are invalid.",
        )

    def required_text(name: str, *, lowercase: bool = False) -> str:
        value = payload.get(name)
        if not isinstance(value, str) or not value.strip() or len(value) > 128:
            raise StrongAuthError(
                "invalid_payload",
                "The Home Assistant action fields are invalid.",
            )
        normalized = value.strip()
        return normalized.lower() if lowercase else normalized

    domain = required_text("domain", lowercase=True)
    entity_id = required_text("entity_id")
    if "." not in entity_id or entity_id.split(".", 1)[0].lower() != domain:
        raise StrongAuthError(
            "invalid_payload",
            "The Home Assistant entity does not match the requested domain.",
        )
    data = payload.get("data", {})
    if not isinstance(data, Mapping):
        raise StrongAuthError("invalid_payload", "The Home Assistant action data is invalid.")
    return {
        "domain": domain,
        "service": required_text("service", lowercase=True),
        "entity_id": entity_id,
        "data": _canonical_json_value(data),
    }


def canonical_action(action_name: object, payload: object) -> tuple[str, bytes, str]:
    """Return normalized name, canonical action bytes, and SHA-256 hash."""
    name = _normalize_action_name(action_name)
    if not isinstance(payload, Mapping):
        raise StrongAuthError("invalid_payload", "The action payload must be an object.")
    canonical_payload = (
        _canonical_ha_action(payload)
        if name == "home_assistant_call_service"
        else _canonical_json_value(
            {key: value for key, value in payload.items() if key not in _SERVER_ACTION_FIELDS}
        )
    )
    body = json.dumps(
        {"action_name": name, "payload": canonical_payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    if len(body) > _MAX_ACTION_BYTES:
        raise StrongAuthError("invalid_payload", "The action payload is too large.")
    return name, body, hashlib.sha256(body).hexdigest()


def policy_for_action(action_name: object, payload: object) -> ActionPolicy | None:
    """Return the server-owned mobile risk policy for an allowlisted action."""
    name, _, _ = canonical_action(action_name, payload)
    if name == "home_assistant_call_service":
        values = _canonical_ha_action(payload)
        domain = values["domain"]
        service = values["service"]
        ha_risk = classify_ha_risk(domain, service)
        if ha_risk is HARisk.PROHIBITED:
            return None
        return ActionPolicy(
            action_name=name,
            required_scope="home.control",
            risk_level=(
                MobileActionRisk.CRITICAL if ha_risk is HARisk.SENSITIVE else MobileActionRisk.HIGH
            ),
        )
    if name in {"music_play", "music_pause", "music_resume", "music_skip", "music_volume"}:
        return ActionPolicy(name, "home.control", MobileActionRisk.MEDIUM)
    return None


def public_challenge_payload(challenge: StrongAuthChallenge) -> dict[str, Any]:
    """Return the complete public wire contract for a device proof challenge."""
    return {
        "challenge_id": challenge.challenge_id,
        "action_name": challenge.action_name,
        "action_hash": challenge.action_hash,
        "risk_level": challenge.risk_level,
        "required_scope": challenge.required_scope,
        "nonce": challenge.nonce_b64,
        "desktop_id": challenge.desktop_id,
        "session_id": challenge.session_id,
        "user_id": challenge.user_id,
        "device_id": challenge.device_id,
        "grant_id": challenge.grant_id,
        "grant_version": challenge.grant_version,
        "expires_at": challenge.expires_at,
        "transcript_type": STRONG_AUTH_TRANSCRIPT_TYPE,
        "transcript_version": STRONG_AUTH_TRANSCRIPT_VERSION,
    }


def canonical_strong_auth_transcript(challenge: StrongAuthChallenge) -> bytes:
    payload = {
        "typ": STRONG_AUTH_TRANSCRIPT_TYPE,
        "v": STRONG_AUTH_TRANSCRIPT_VERSION,
        "challenge_id": challenge.challenge_id,
        "nonce": challenge.nonce_b64,
        "action_name": challenge.action_name,
        "action_hash": challenge.action_hash,
        "risk_level": challenge.risk_level,
        "required_scope": challenge.required_scope,
        "desktop_id": challenge.desktop_id,
        "session_id": challenge.session_id,
        "user_id": challenge.user_id,
        "device_id": challenge.device_id,
        "grant_id": challenge.grant_id,
        "grant_version": challenge.grant_version,
        "expires_at": challenge.expires_at,
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    return STRONG_AUTH_DOMAIN + b"\n" + body


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class StrongAuthAuthority:
    @staticmethod
    def _rollback(conn: Any, operation: str) -> None:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            logger.warning(
                "Strong-auth transaction rollback failed after %s.",
                operation,
                exc_info=True,
            )

    def __init__(
        self,
        db_path: Path | str,
        *,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
        nonce_generator: Callable[[], bytes] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.now = clock or (lambda: datetime.now(UTC))
        self._id_generator = id_generator or (lambda: str(uuid.uuid4()))
        self._nonce_generator = nonce_generator or (lambda: secrets.token_bytes(32))

    def _insert_audit(
        self,
        conn: Any,
        event_type: str,
        *,
        principal: PrincipalBinding | None = None,
        challenge_id: str | None = None,
        approval_id: str | None = None,
        action_name: str | None = None,
        action_hash: str | None = None,
        risk_level: str | None = None,
        reason: str | None = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO mobile_strong_auth_audit (
                event_id, event_type, challenge_id, approval_id, session_id,
                user_id, device_id, grant_id, action_name, action_hash,
                risk_level, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                event_type,
                challenge_id,
                approval_id,
                principal.session_id if principal is not None else None,
                principal.user_id if principal is not None else None,
                principal.paired_device_id if principal is not None else None,
                principal.grant_id if principal is not None else None,
                action_name,
                action_hash,
                risk_level,
                reason,
                self.now().isoformat(),
            ),
        )

    def _audit_denial(self, event_type: str, **values: Any) -> None:
        conn = connect(self.db_path)
        try:
            self._insert_audit(conn, event_type, **values)
        finally:
            conn.close()

    @staticmethod
    def _require_paired(principal: PrincipalBinding) -> tuple[str, str, str, int]:
        if (
            principal.paired_device_id is None
            or principal.grant_id is None
            or principal.desktop_id is None
            or principal.grant_version is None
        ):
            raise StrongAuthError(
                "paired_session_required",
                "A paired device session is required for this action.",
            )
        return (
            principal.paired_device_id,
            principal.grant_id,
            principal.desktop_id,
            principal.grant_version,
        )

    @staticmethod
    def _load_active_binding(conn: Any, principal: PrincipalBinding, now: datetime) -> Any:
        device_id, grant_id, desktop_id, grant_version = StrongAuthAuthority._require_paired(
            principal
        )
        row = conn.execute(
            """
            SELECT d.public_key_b64, d.revoked_at AS device_revoked_at,
                   g.revoked_at AS grant_revoked_at, g.expires_at, g.scopes_json,
                   g.version, g.desktop_id, g.user_id, g.device_id
            FROM mobile_paired_devices d
            JOIN mobile_device_grants g ON g.device_id = d.device_id
            WHERE d.device_id = ? AND d.desktop_id = ? AND d.user_id = ?
              AND g.grant_id = ? AND g.desktop_id = ? AND g.user_id = ?
            """,
            (device_id, desktop_id, principal.user_id, grant_id, desktop_id, principal.user_id),
        ).fetchone()
        if (
            row is None
            or row["device_revoked_at"] is not None
            or row["grant_revoked_at"] is not None
        ):
            raise StrongAuthError("binding_revoked", "The paired device grant is no longer valid.")
        if int(row["version"]) != grant_version or str(row["device_id"]) != device_id:
            raise StrongAuthError("binding_changed", "The paired device grant has changed.")
        try:
            if now >= _parse_time(str(row["expires_at"])):
                raise StrongAuthError("binding_expired", "The paired device grant has expired.")
            scopes = frozenset(json.loads(str(row["scopes_json"])))
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise StrongAuthError("binding_invalid", "The paired device grant is invalid.") from exc
        if scopes != principal.scopes:
            raise StrongAuthError("binding_changed", "The paired device grant has changed.")
        return row

    def create_challenge(
        self,
        principal: PrincipalBinding,
        *,
        action_name: object,
        payload: object,
    ) -> StrongAuthChallenge:
        name, _, action_hash = canonical_action(action_name, payload)
        policy = policy_for_action(name, payload)
        try:
            if policy is None or not policy.requires_strong_auth:
                raise StrongAuthError(
                    "strong_auth_not_applicable",
                    "This action does not use the strong-authentication protocol.",
                )
            try:
                require_scope(
                    principal.scopes,
                    policy.required_scope,
                    permissions=principal.permissions,
                )
            except ValueError as exc:
                raise StrongAuthError(
                    "scope_denied",
                    "The user and paired device are not authorized for this action.",
                ) from exc
            device_id, grant_id, desktop_id, grant_version = self._require_paired(principal)
        except StrongAuthError as exc:
            self._audit_denial(
                "challenge_denied",
                principal=principal,
                action_name=name,
                action_hash=action_hash,
                risk_level=policy.risk_level.value if policy is not None else None,
                reason=exc.reason,
            )
            raise
        now = self.now()
        expires = now + timedelta(seconds=CHALLENGE_TTL_SECONDS)
        challenge = StrongAuthChallenge(
            challenge_id=self._id_generator(),
            action_name=name,
            action_hash=action_hash,
            risk_level=policy.risk_level.value,
            required_scope=policy.required_scope,
            nonce_b64=base64.b64encode(self._nonce_generator()).decode("ascii"),
            desktop_id=desktop_id,
            session_id=principal.session_id,
            user_id=principal.user_id,
            device_id=device_id,
            grant_id=grant_id,
            grant_version=grant_version,
            expires_at=expires.isoformat(),
        )
        conn = connect(self.db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            self._load_active_binding(conn, principal, now)
            conn.execute(
                """
                INSERT INTO mobile_strong_auth_challenges (
                    challenge_id, session_id, user_id, device_id, grant_id, grant_version,
                    desktop_id, action_name, action_hash, risk_level, required_scope,
                    nonce_b64, created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    challenge.challenge_id,
                    challenge.session_id,
                    challenge.user_id,
                    challenge.device_id,
                    challenge.grant_id,
                    challenge.grant_version,
                    challenge.desktop_id,
                    challenge.action_name,
                    challenge.action_hash,
                    challenge.risk_level,
                    challenge.required_scope,
                    challenge.nonce_b64,
                    now.isoformat(),
                    challenge.expires_at,
                ),
            )
            self._insert_audit(
                conn,
                "challenge_issued",
                principal=principal,
                challenge_id=challenge.challenge_id,
                action_name=challenge.action_name,
                action_hash=challenge.action_hash,
                risk_level=challenge.risk_level,
            )
            conn.execute("COMMIT")
            return challenge
        except StrongAuthError as exc:
            self._rollback(conn, "challenge denial")
            self._audit_denial(
                "challenge_denied",
                principal=principal,
                challenge_id=challenge.challenge_id,
                action_name=challenge.action_name,
                action_hash=challenge.action_hash,
                risk_level=challenge.risk_level,
                reason=exc.reason,
            )
            raise
        except BaseException:
            self._rollback(conn, "challenge failure")
            raise
        finally:
            conn.close()

    @staticmethod
    def _challenge_from_row(row: Any) -> StrongAuthChallenge:
        return StrongAuthChallenge(
            challenge_id=str(row["challenge_id"]),
            action_name=str(row["action_name"]),
            action_hash=str(row["action_hash"]),
            risk_level=str(row["risk_level"]),
            required_scope=str(row["required_scope"]),
            nonce_b64=str(row["nonce_b64"]),
            desktop_id=str(row["desktop_id"]),
            session_id=str(row["session_id"]),
            user_id=str(row["user_id"]),
            device_id=str(row["device_id"]),
            grant_id=str(row["grant_id"]),
            grant_version=int(row["grant_version"]),
            expires_at=str(row["expires_at"]),
        )

    @staticmethod
    def _require_same_principal(row: Any, principal: PrincipalBinding) -> None:
        device_id, grant_id, desktop_id, grant_version = StrongAuthAuthority._require_paired(
            principal
        )
        expected = (
            principal.session_id,
            principal.user_id,
            device_id,
            grant_id,
            grant_version,
            desktop_id,
        )
        actual = (
            str(row["session_id"]),
            str(row["user_id"]),
            str(row["device_id"]),
            str(row["grant_id"]),
            int(row["grant_version"]),
            str(row["desktop_id"]),
        )
        if actual != expected:
            raise StrongAuthError(
                "binding_mismatch", "The approval does not belong to this session."
            )

    def verify_challenge(
        self,
        principal: PrincipalBinding,
        *,
        challenge_id: str,
        signature_b64: str,
    ) -> StrongAuthApproval:
        if not isinstance(challenge_id, str) or not challenge_id or len(challenge_id) > 128:
            raise StrongAuthError(
                "invalid_challenge", "The strong-authentication challenge is invalid."
            )
        now = self.now()
        conn = connect(self.db_path)
        row: Any | None = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM mobile_strong_auth_challenges WHERE challenge_id = ?",
                (challenge_id,),
            ).fetchone()
            if row is None:
                raise StrongAuthError(
                    "invalid_challenge", "The strong-authentication challenge is invalid."
                )
            self._require_same_principal(row, principal)
            if row["verified_at"] is not None or row["approval_id"] is not None:
                raise StrongAuthError(
                    "challenge_replayed", "The strong-authentication challenge was already used."
                )
            if now >= _parse_time(str(row["expires_at"])):
                raise StrongAuthError(
                    "challenge_expired", "The strong-authentication challenge has expired."
                )
            binding = self._load_active_binding(conn, principal, now)
            challenge = self._challenge_from_row(row)
            try:
                verify_proof(
                    public_key_b64=str(binding["public_key_b64"]),
                    signature_b64=signature_b64,
                    transcript=canonical_strong_auth_transcript(challenge),
                )
            except ProofError as exc:
                raise StrongAuthError("invalid_signature", "The device proof is invalid.") from exc
            approval_id = self._id_generator()
            approval_expires = now + timedelta(seconds=APPROVAL_TTL_SECONDS)
            updated = conn.execute(
                """
                UPDATE mobile_strong_auth_challenges
                SET approval_id = ?, verified_at = ?, approval_expires_at = ?
                WHERE challenge_id = ? AND verified_at IS NULL AND approval_id IS NULL
                """,
                (approval_id, now.isoformat(), approval_expires.isoformat(), challenge_id),
            ).rowcount
            if updated != 1:
                raise StrongAuthError(
                    "challenge_replayed", "The strong-authentication challenge was already used."
                )
            conn.execute(
                "UPDATE mobile_sessions SET strong_auth_at = ? WHERE session_id = ?",
                (now.isoformat(), principal.session_id),
            )
            conn.execute(
                "UPDATE mobile_device_grants SET last_strong_auth_at = ? WHERE grant_id = ?",
                (now.isoformat(), principal.grant_id),
            )
            self._insert_audit(
                conn,
                "proof_verified",
                principal=principal,
                challenge_id=challenge.challenge_id,
                approval_id=approval_id,
                action_name=challenge.action_name,
                action_hash=challenge.action_hash,
                risk_level=challenge.risk_level,
            )
            conn.execute("COMMIT")
            return StrongAuthApproval(
                approval_id=approval_id,
                action_name=challenge.action_name,
                action_hash=challenge.action_hash,
                risk_level=challenge.risk_level,
                expires_at=approval_expires.isoformat(),
            )
        except StrongAuthError as exc:
            self._rollback(conn, "proof denial")
            self._audit_denial(
                "proof_denied",
                principal=principal,
                challenge_id=challenge_id,
                approval_id=(
                    str(row["approval_id"])
                    if row is not None and row["approval_id"] is not None
                    else None
                ),
                action_name=str(row["action_name"]) if row is not None else None,
                action_hash=str(row["action_hash"]) if row is not None else None,
                risk_level=str(row["risk_level"]) if row is not None else None,
                reason=exc.reason,
            )
            raise
        except BaseException:
            self._rollback(conn, "proof failure")
            raise
        finally:
            conn.close()

    def consume_approval(
        self,
        principal: PrincipalBinding,
        *,
        approval_id: str,
        action_name: object,
        payload: object,
    ) -> StrongAuthApproval:
        if not isinstance(approval_id, str) or not approval_id or len(approval_id) > 128:
            raise StrongAuthError(
                "approval_required", "Strong authentication is required for this action."
            )
        name, _, action_hash = canonical_action(action_name, payload)
        now = self.now()
        conn = connect(self.db_path)
        row: Any | None = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM mobile_strong_auth_challenges WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None or row["verified_at"] is None or row["approval_expires_at"] is None:
                raise StrongAuthError(
                    "approval_invalid", "The strong-authentication approval is invalid."
                )
            self._require_same_principal(row, principal)
            if row["consumed_at"] is not None:
                raise StrongAuthError(
                    "approval_replayed", "The strong-authentication approval was already used."
                )
            if now >= _parse_time(str(row["approval_expires_at"])):
                raise StrongAuthError(
                    "approval_expired", "The strong-authentication approval has expired."
                )
            if str(row["action_name"]) != name or str(row["action_hash"]) != action_hash:
                raise StrongAuthError(
                    "action_changed", "The approved action does not match this request."
                )
            self._load_active_binding(conn, principal, now)
            updated = conn.execute(
                """
                UPDATE mobile_strong_auth_challenges SET consumed_at = ?
                WHERE approval_id = ? AND consumed_at IS NULL
                """,
                (now.isoformat(), approval_id),
            ).rowcount
            if updated != 1:
                raise StrongAuthError(
                    "approval_replayed", "The strong-authentication approval was already used."
                )
            self._insert_audit(
                conn,
                "approval_consumed",
                principal=principal,
                challenge_id=str(row["challenge_id"]),
                approval_id=approval_id,
                action_name=name,
                action_hash=action_hash,
                risk_level=str(row["risk_level"]),
            )
            conn.execute("COMMIT")
            return StrongAuthApproval(
                approval_id=approval_id,
                action_name=name,
                action_hash=action_hash,
                risk_level=str(row["risk_level"]),
                expires_at=str(row["approval_expires_at"]),
            )
        except StrongAuthError as exc:
            self._rollback(conn, "approval denial")
            self._audit_denial(
                "approval_denied",
                principal=principal,
                challenge_id=str(row["challenge_id"]) if row is not None else None,
                approval_id=approval_id,
                action_name=name,
                action_hash=action_hash,
                risk_level=str(row["risk_level"]) if row is not None else None,
                reason=exc.reason,
            )
            raise
        except BaseException:
            self._rollback(conn, "approval failure")
            raise
        finally:
            conn.close()


__all__ = [
    "APPROVAL_TTL_SECONDS",
    "CHALLENGE_TTL_SECONDS",
    "STRONG_AUTH_DOMAIN",
    "STRONG_AUTH_TRANSCRIPT_TYPE",
    "STRONG_AUTH_TRANSCRIPT_VERSION",
    "ActionPolicy",
    "MobileActionRisk",
    "StrongAuthApproval",
    "StrongAuthAuthority",
    "StrongAuthChallenge",
    "StrongAuthError",
    "canonical_action",
    "canonical_strong_auth_transcript",
    "policy_for_action",
    "public_challenge_payload",
]
