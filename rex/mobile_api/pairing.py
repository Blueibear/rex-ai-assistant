"""Desktop-owned mobile device pairing authority (S5).

The mobile HTTP surface may submit a cryptographic proof and poll its status.
Only trusted desktop-local callers may create challenges, approve/deny
requests, or revoke devices. Password sessions never gain those methods.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rex.identity import validate_user_id
from rex.mobile_api.db import connect
from rex.mobile_api.device_proof import (
    ProofError,
    canonical_public_key_spki_b64,
    canonical_transcript,
    load_p256_public_key,
    public_key_thumbprint,
    verify_proof,
)
from rex.mobile_api.grants import Grant, ScopeError, canonicalize_scopes

CHALLENGE_TTL_SECONDS = 120
GRANT_TTL_DAYS = 365
PAIRING_PENDING = "pending"
PAIRING_APPROVED = "approved"
PAIRING_DENIED = "denied"
PAIRING_EXPIRED = "expired"


class PairingError(ValueError):
    """Expected pairing failure with a stable, secret-free message."""


@dataclass(frozen=True)
class PairingChallenge:
    challenge_id: str
    desktop_id: str
    nonce_b64: str
    code: str
    user_id: str
    scopes: tuple[str, ...]
    created_at: str
    expires_at: str

    def qr_payload(self) -> dict[str, Any]:
        return {
            "type": "askrex-pairing",
            "version": 1,
            "desktop_id": self.desktop_id,
            "challenge_id": self.challenge_id,
            "nonce": self.nonce_b64,
            "code": self.code,
            "user_id": self.user_id,
            "scopes": list(self.scopes),
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class PairingSubmission:
    request_id: str
    poll_token: str
    status: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid.uuid4())


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _code_hash(challenge_id: str, nonce_b64: str, code: str) -> str:
    material = f"{challenge_id}\n{nonce_b64}\n{code}".encode()
    return hashlib.sha256(material).hexdigest()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PairingAuthority:
    """SQLite-backed desktop pairing authority with injectable time/IDs."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        clock: Callable[[], datetime] | None = None,
        id_generator: Callable[[], str] | None = None,
        code_generator: Callable[[], str] | None = None,
        token_generator: Callable[[], str] | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._clock = clock or _utc_now
        self._id = id_generator or _new_id
        self._code = code_generator or (lambda: f"{secrets.randbelow(100_000_000):08d}")
        self._token = token_generator or (lambda: secrets.token_urlsafe(32))

    def now(self) -> datetime:
        return self._clock()

    def desktop_id(self) -> str:
        conn = connect(self._db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT desktop_id FROM mobile_pairing_authority WHERE singleton = 1"
            ).fetchone()
            if row is None:
                desktop_id = self._id()
                conn.execute(
                    "INSERT INTO mobile_pairing_authority(singleton, desktop_id, created_at) VALUES (1, ?, ?)",
                    (desktop_id, self.now().isoformat()),
                )
            else:
                desktop_id = str(row["desktop_id"])
            conn.execute("COMMIT")
            return desktop_id
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def create_challenge(self, *, user_id: str, scopes: object) -> PairingChallenge:
        user_id = validate_user_id(user_id)
        try:
            canonical_scopes = canonicalize_scopes(scopes)
        except ScopeError as exc:
            raise PairingError(str(exc)) from exc
        now = self.now()
        challenge_id = self._id()
        desktop_id = self.desktop_id()
        nonce_b64 = secrets.token_urlsafe(32)
        code = self._code()
        if len(code) != 8 or not code.isdigit():
            raise PairingError("Pairing code generator returned an invalid code.")
        expires = now + timedelta(seconds=CHALLENGE_TTL_SECONDS)
        conn = connect(self._db_path)
        try:
            conn.execute(
                """INSERT INTO mobile_pairing_challenges(
                    challenge_id, desktop_id, user_id, nonce_b64, code_hash,
                    scopes_json, created_at, expires_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    challenge_id,
                    desktop_id,
                    user_id,
                    nonce_b64,
                    _code_hash(challenge_id, nonce_b64, code),
                    json.dumps(canonical_scopes),
                    now.isoformat(),
                    expires.isoformat(),
                    PAIRING_PENDING,
                ),
            )
            self._audit(conn, "challenge_created", desktop_id=desktop_id, user_id=user_id)
        finally:
            conn.close()
        return PairingChallenge(
            challenge_id,
            desktop_id,
            nonce_b64,
            code,
            user_id,
            canonical_scopes,
            now.isoformat(),
            expires.isoformat(),
        )

    def submit_proof(self, payload: dict[str, Any]) -> PairingSubmission:
        required = {
            "desktop_id",
            "challenge_id",
            "nonce",
            "code",
            "user_id",
            "scopes",
            "public_key",
            "signature",
            "device_name",
            "platform",
        }
        if set(payload) != required:
            raise PairingError("Pairing request fields are invalid.")
        values = {k: payload[k] for k in required}
        for name in required - {"scopes"}:
            if not isinstance(values[name], str) or not values[name]:
                raise PairingError("Pairing request fields are invalid.")
        try:
            user_id = validate_user_id(values["user_id"])
            scopes = canonicalize_scopes(values["scopes"])
        except (ValueError, ScopeError) as exc:
            raise PairingError("Pairing request is invalid.") from exc

        conn = connect(self._db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM mobile_pairing_challenges WHERE challenge_id = ?",
                (values["challenge_id"],),
            ).fetchone()
            if row is None:
                raise PairingError("Pairing challenge is invalid or expired.")
            if row["status"] != PAIRING_PENDING or row["used_at"] is not None:
                raise PairingError("Pairing challenge has already been used.")
            if self.now() >= _parse_time(row["expires_at"]):
                conn.execute(
                    "UPDATE mobile_pairing_challenges SET status = ? WHERE challenge_id = ?",
                    (PAIRING_EXPIRED, row["challenge_id"]),
                )
                raise PairingError("Pairing challenge is invalid or expired.")
            stored_scopes = tuple(json.loads(row["scopes_json"]))
            if (
                values["desktop_id"] != row["desktop_id"]
                or values["nonce"] != row["nonce_b64"]
                or user_id != row["user_id"]
                or scopes != stored_scopes
            ):
                raise PairingError("Pairing request does not match the challenge.")
            expected_hash = _code_hash(row["challenge_id"], row["nonce_b64"], values["code"])
            if not hmac.compare_digest(expected_hash, row["code_hash"]):
                raise PairingError("Pairing code is invalid.")
            try:
                key = load_p256_public_key(values["public_key"])
                canonical_key = canonical_public_key_spki_b64(key)
                transcript = canonical_transcript(
                    desktop_id=row["desktop_id"],
                    challenge_id=row["challenge_id"],
                    nonce_b64=row["nonce_b64"],
                    mobile_public_key_b64=canonical_key,
                    user_id=row["user_id"],
                    scopes=stored_scopes,
                    code=values["code"],
                )
                verify_proof(
                    public_key_b64=canonical_key,
                    signature_b64=values["signature"],
                    transcript=transcript,
                )
            except ProofError as exc:
                raise PairingError("Device proof could not be verified.") from exc

            request_id = self._id()
            poll_token = self._token()
            submitted_at = self.now().isoformat()
            conn.execute(
                """INSERT INTO mobile_pairing_requests(
                    request_id, challenge_id, desktop_id, user_id, public_key_b64,
                    key_thumbprint, device_name, platform, scopes_json,
                    poll_token_hash, submitted_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    request_id,
                    row["challenge_id"],
                    row["desktop_id"],
                    row["user_id"],
                    canonical_key,
                    public_key_thumbprint(canonical_key),
                    values["device_name"][:128],
                    values["platform"][:64],
                    json.dumps(stored_scopes),
                    _token_hash(poll_token),
                    submitted_at,
                    PAIRING_PENDING,
                ),
            )
            conn.execute(
                "UPDATE mobile_pairing_challenges SET used_at = ?, status = ? WHERE challenge_id = ?",
                (submitted_at, "submitted", row["challenge_id"]),
            )
            self._audit(
                conn,
                "proof_submitted",
                request_id=request_id,
                desktop_id=row["desktop_id"],
                user_id=row["user_id"],
            )
            conn.execute("COMMIT")
            return PairingSubmission(request_id, poll_token, PAIRING_PENDING)
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def list_pending(self) -> list[dict[str, Any]]:
        return self._list_requests(PAIRING_PENDING)

    def _list_requests(self, status: str) -> list[dict[str, Any]]:
        conn = connect(self._db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM mobile_pairing_requests WHERE status = ? ORDER BY submitted_at",
                (status,),
            ).fetchall()
            return [self._request_projection(row) for row in rows]
        finally:
            conn.close()

    def approve(
        self, request_id: str, *, approved_by: str, expires_days: int = GRANT_TTL_DAYS
    ) -> Grant:
        if not approved_by.strip():
            raise PairingError("Desktop approver identity is required.")
        now = self.now()
        conn = connect(self._db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM mobile_pairing_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None or row["status"] != PAIRING_PENDING:
                raise PairingError("Pairing request is not pending approval.")
            existing = conn.execute(
                "SELECT device_id FROM mobile_paired_devices WHERE desktop_id = ? AND key_thumbprint = ?",
                (row["desktop_id"], row["key_thumbprint"]),
            ).fetchone()
            device_id = str(existing["device_id"]) if existing else self._id()
            if existing is None:
                conn.execute(
                    """INSERT INTO mobile_paired_devices(
                        device_id, desktop_id, user_id, public_key_b64, key_thumbprint,
                        device_name, platform, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        device_id,
                        row["desktop_id"],
                        row["user_id"],
                        row["public_key_b64"],
                        row["key_thumbprint"],
                        row["device_name"],
                        row["platform"],
                        now.isoformat(),
                    ),
                )
            elif (
                conn.execute(
                    "SELECT revoked_at FROM mobile_paired_devices WHERE device_id = ?", (device_id,)
                ).fetchone()["revoked_at"]
                is not None
            ):
                raise PairingError("Paired device has been revoked.")
            version_row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) AS v FROM mobile_device_grants WHERE device_id = ?",
                (device_id,),
            ).fetchone()
            version = int(version_row["v"]) + 1
            grant_id = self._id()
            expires = now + timedelta(days=max(1, min(expires_days, GRANT_TTL_DAYS)))
            scopes = tuple(json.loads(row["scopes_json"]))
            conn.execute(
                """INSERT INTO mobile_device_grants(
                    grant_id, device_id, desktop_id, user_id, version, scopes_json,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    grant_id,
                    device_id,
                    row["desktop_id"],
                    row["user_id"],
                    version,
                    json.dumps(scopes),
                    now.isoformat(),
                    expires.isoformat(),
                ),
            )
            conn.execute(
                """UPDATE mobile_pairing_requests SET status = ?, decision_at = ?,
                   decision_by = ?, device_id = ?, grant_id = ? WHERE request_id = ?""",
                (
                    PAIRING_APPROVED,
                    now.isoformat(),
                    approved_by[:128],
                    device_id,
                    grant_id,
                    request_id,
                ),
            )
            self._audit(
                conn,
                "request_approved",
                request_id=request_id,
                device_id=device_id,
                grant_id=grant_id,
                desktop_id=row["desktop_id"],
                user_id=row["user_id"],
            )
            conn.execute("COMMIT")
            return Grant(
                grant_id,
                device_id,
                row["desktop_id"],
                row["user_id"],
                version,
                scopes,
                now.isoformat(),
                expires.isoformat(),
            )
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def deny(self, request_id: str, *, denied_by: str, reason: str = "denied_by_owner") -> None:
        conn = connect(self._db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT desktop_id, user_id FROM mobile_pairing_requests WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            cursor = conn.execute(
                """UPDATE mobile_pairing_requests SET status = ?, decision_at = ?,
                   decision_by = ?, denial_reason = ?
                   WHERE request_id = ? AND status = ?""",
                (
                    PAIRING_DENIED,
                    self.now().isoformat(),
                    denied_by[:128],
                    reason[:128],
                    request_id,
                    PAIRING_PENDING,
                ),
            )
            if cursor.rowcount != 1 or row is None:
                raise PairingError("Pairing request is not pending approval.")
            self._audit(
                conn,
                "request_denied",
                request_id=request_id,
                desktop_id=row["desktop_id"],
                user_id=row["user_id"],
                detail=reason[:128],
            )
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def poll_status(self, request_id: str, poll_token: str) -> dict[str, Any]:
        conn = connect(self._db_path)
        try:
            row = conn.execute(
                "SELECT * FROM mobile_pairing_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None or not hmac.compare_digest(
                row["poll_token_hash"], _token_hash(poll_token)
            ):
                raise PairingError("Pairing status credentials are invalid.")
            result = {"request_id": request_id, "status": row["status"]}
            if row["status"] == PAIRING_APPROVED:
                grant = conn.execute(
                    "SELECT * FROM mobile_device_grants WHERE grant_id = ?", (row["grant_id"],)
                ).fetchone()
                result.update(
                    {
                        "device_id": row["device_id"],
                        "grant_id": row["grant_id"],
                        "scopes": json.loads(grant["scopes_json"]),
                        "expires_at": grant["expires_at"],
                    }
                )
            return result
        finally:
            conn.close()

    def list_devices(self) -> list[dict[str, Any]]:
        conn = connect(self._db_path)
        try:
            rows = conn.execute("""SELECT d.*, g.grant_id, g.version, g.scopes_json, g.expires_at,
                   g.revoked_at AS grant_revoked_at
                   FROM mobile_paired_devices d
                   LEFT JOIN mobile_device_grants g ON g.grant_id = (
                     SELECT grant_id FROM mobile_device_grants x
                     WHERE x.device_id = d.device_id ORDER BY version DESC LIMIT 1
                   ) ORDER BY d.created_at DESC""").fetchall()
            return [
                {
                    "device_id": row["device_id"],
                    "desktop_id": row["desktop_id"],
                    "user_id": row["user_id"],
                    "device_name": row["device_name"],
                    "platform": row["platform"],
                    "key_thumbprint": row["key_thumbprint"],
                    "created_at": row["created_at"],
                    "revoked_at": row["revoked_at"],
                    "grant_id": row["grant_id"],
                    "grant_version": row["version"],
                    "scopes": json.loads(row["scopes_json"]) if row["scopes_json"] else [],
                    "grant_expires_at": row["expires_at"],
                    "grant_revoked_at": row["grant_revoked_at"],
                }
                for row in rows
            ]
        finally:
            conn.close()

    def revoke_device(
        self, device_id: str, *, revoked_by: str, reason: str = "revoked_by_owner"
    ) -> bool:
        now = self.now().isoformat()
        conn = connect(self._db_path)
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "UPDATE mobile_paired_devices SET revoked_at = ?, revoke_reason = ? WHERE device_id = ? AND revoked_at IS NULL",
                (now, reason[:128], device_id),
            )
            conn.execute(
                "UPDATE mobile_device_grants SET revoked_at = ?, revoke_reason = ? WHERE device_id = ? AND revoked_at IS NULL",
                (now, reason[:128], device_id),
            )
            if cursor.rowcount:
                self._audit(conn, "device_revoked", device_id=device_id, detail=revoked_by[:128])
            conn.execute("COMMIT")
            return cursor.rowcount == 1
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    @staticmethod
    def _request_projection(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "request_id": row["request_id"],
            "desktop_id": row["desktop_id"],
            "user_id": row["user_id"],
            "device_name": row["device_name"],
            "platform": row["platform"],
            "key_thumbprint": row["key_thumbprint"],
            "scopes": json.loads(row["scopes_json"]),
            "submitted_at": row["submitted_at"],
            "status": row["status"],
        }

    def _audit(self, conn: sqlite3.Connection, event_type: str, **values: Any) -> None:
        conn.execute(
            """INSERT INTO mobile_pairing_audit(
                event_id, event_type, request_id, device_id, grant_id,
                desktop_id, user_id, created_at, detail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self._id(),
                event_type,
                values.get("request_id"),
                values.get("device_id"),
                values.get("grant_id"),
                values.get("desktop_id"),
                values.get("user_id"),
                self.now().isoformat(),
                values.get("detail"),
            ),
        )


__all__ = [
    "CHALLENGE_TTL_SECONDS",
    "GRANT_TTL_DAYS",
    "PAIRING_APPROVED",
    "PAIRING_DENIED",
    "PAIRING_EXPIRED",
    "PAIRING_PENDING",
    "PairingAuthority",
    "PairingChallenge",
    "PairingError",
    "PairingSubmission",
]
