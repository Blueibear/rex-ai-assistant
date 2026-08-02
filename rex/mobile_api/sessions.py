"""Mobile session and rotating refresh-token lifecycle (issue #323).

Security properties enforced here:

- Refresh tokens are high-entropy opaque values; only SHA-256 hashes are
  stored, never raw values (the source tokens carry >= 256 bits of entropy,
  so an unsalted fast hash is sufficient and allows primary-key lookup).
- Rotation happens inside one ``BEGIN IMMEDIATE`` SQLite transaction, and the
  consume step is an ``UPDATE ... WHERE consumed_at IS NULL`` so concurrent
  use of one refresh token yields exactly one success.
- Reuse of a consumed token revokes the whole token family and its session,
  and records an audit log event containing safe IDs only.
- Sessions are per-device; logout revokes one session, logout-all revokes
  every session belonging to one user and no one else's.

Clock, token, and ID generators are injectable for deterministic tests.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rex.identity import validate_user_id
from rex.mobile_api.db import connect

logger = logging.getLogger(__name__)

# Rotation outcome statuses.
ROTATED = "rotated"
INVALID = "invalid"
EXPIRED = "expired"
REUSED = "reused"
SESSION_REVOKED = "session_revoked"
USER_INACTIVE = "user_inactive"

_MAX_DEVICE_FIELD_LENGTH = 128
DEVICE_SESSION_CHALLENGE_TTL_SECONDS = 120


def hash_refresh_token(raw_token: str) -> str:
    """Return the hex SHA-256 hash under which a refresh token is stored."""
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _default_token_generator() -> str:
    # 48 bytes -> 384 bits of entropy, URL-safe base64 encoded.
    return secrets.token_urlsafe(48)


def _default_id_generator() -> str:
    return str(uuid.uuid4())


@dataclass(frozen=True)
class DeviceInfo:
    """Validated, presentation-only device metadata attached to a session."""

    device_id: str
    name: str = ""
    platform: str = ""
    app_version: str = ""


@dataclass(frozen=True)
class CreatedSession:
    """Result of creating a session: the raw refresh token appears only here."""

    session_id: str
    user_id: str
    refresh_token: str
    refresh_expires_at: datetime
    family_id: str


@dataclass(frozen=True)
class RotationResult:
    """Outcome of a refresh-token rotation attempt."""

    status: str
    session_id: str | None = None
    user_id: str | None = None
    refresh_token: str | None = None
    refresh_expires_at: datetime | None = None


@dataclass(frozen=True)
class DeviceSessionChallenge:
    """Short-lived proof challenge for replacing a bootstrap session."""

    challenge_id: str
    bootstrap_session_id: str
    user_id: str
    device_id: str
    grant_id: str
    grant_version: int
    desktop_id: str
    nonce_b64: str
    expires_at: datetime


class DeviceSessionError(ValueError):
    """Stable, secret-free device-session activation failure."""


class MobileSessionStore:
    """SQLite-backed store for mobile sessions and refresh-token families."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        refresh_ttl_seconds: int,
        clock: Callable[[], datetime] | None = None,
        token_generator: Callable[[], str] | None = None,
        id_generator: Callable[[], str] | None = None,
        audit_logger: object | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._refresh_ttl_seconds = int(refresh_ttl_seconds)
        self._clock = clock or _default_clock
        self._token_generator = token_generator or _default_token_generator
        self._id_generator = id_generator or _default_id_generator
        # Anything with a ``log(LogEntry)`` method; defaults to the canonical
        # rex.audit logger, resolved lazily so importing this module stays
        # side-effect free. Tests inject a recorder so no files are written.
        self._audit_logger = audit_logger

    def _resolve_audit_logger(self) -> object:
        if self._audit_logger is None:
            from rex.audit import get_audit_logger  # noqa: PLC0415

            self._audit_logger = get_audit_logger()
        return self._audit_logger

    def _emit_reuse_audit_event(self, *, session_id: str, family_id: str, user_id: str) -> None:
        """Persist a structured security-audit event for refresh-token reuse.

        The event carries safe identifiers and status only — never raw
        tokens, token hashes, access tokens, passwords, or request bodies.
        Audit failures are logged but never break the security response
        (the revocation has already been committed).
        """
        try:
            from rex.audit import LogEntry  # noqa: PLC0415

            entry = LogEntry(
                action_id=self._id_generator(),
                tool="mobile_auth",
                tool_call_args={
                    "event_type": "mobile_refresh_token_reuse",
                    "session_id": session_id,
                    "family_id": family_id,
                    "user_id": user_id,
                    "revocation_result": "family_and_session_revoked",
                },
                policy_decision="denied",
                requested_by=user_id,
                timestamp=self.now(),
            )
            self._resolve_audit_logger().log(entry)  # type: ignore[attr-defined]
        except Exception:
            logger.exception(
                "Failed to write mobile refresh-reuse audit event: session=%s",
                session_id,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        """Return the injected current UTC time."""
        return self._clock()

    def _connect(self) -> sqlite3.Connection:
        return connect(self._db_path)

    @staticmethod
    def _parse_ts(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

    def session_is_active(self, row: sqlite3.Row | None, now: datetime) -> bool:
        """Return True when a session row exists, is unrevoked, and unexpired."""
        if row is None:
            return False
        if row["revoked_at"] is not None:
            return False
        expires_at = self._parse_ts(row["expires_at"])
        return expires_at is not None and now < expires_at

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, user_id: str, device: DeviceInfo) -> CreatedSession:
        """Create a per-device session and its first refresh token."""
        user_id = validate_user_id(user_id)
        now = self.now()
        expires_at = now + timedelta(seconds=self._refresh_ttl_seconds)
        session_id = self._id_generator()
        family_id = self._id_generator()
        raw_token = self._token_generator()

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                INSERT INTO mobile_sessions (
                    session_id, user_id, device_id, device_name, platform,
                    app_version, created_at, last_seen_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_id,
                    device.device_id[:_MAX_DEVICE_FIELD_LENGTH],
                    device.name[:_MAX_DEVICE_FIELD_LENGTH],
                    device.platform[:_MAX_DEVICE_FIELD_LENGTH],
                    device.app_version[:_MAX_DEVICE_FIELD_LENGTH],
                    now.isoformat(),
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
            )
            conn.execute(
                """
                INSERT INTO mobile_refresh_tokens (
                    token_hash, family_id, session_id, user_id,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    hash_refresh_token(raw_token),
                    family_id,
                    session_id,
                    user_id,
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
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

        logger.info("Mobile session created: session=%s user=%s", session_id, user_id)
        return CreatedSession(
            session_id=session_id,
            user_id=user_id,
            refresh_token=raw_token,
            refresh_expires_at=expires_at,
            family_id=family_id,
        )

    def create_device_session_challenge(
        self,
        *,
        bootstrap_session_id: str,
        user_id: str,
        device_id: str,
        grant_id: str,
    ) -> DeviceSessionChallenge:
        """Create a single-use challenge bound to one bootstrap session/grant."""
        from rex.mobile_api.authorization import (  # noqa: PLC0415
            GrantAuthorizationError,
            load_active_grant,
        )

        user_id = validate_user_id(user_id)
        if not bootstrap_session_id or len(bootstrap_session_id) > 128:
            raise DeviceSessionError("Bootstrap session is invalid.")
        if not device_id or len(device_id) > 128 or not grant_id or len(grant_id) > 128:
            raise DeviceSessionError("Device grant is invalid.")
        now = self.now()
        expires_at = now + timedelta(seconds=DEVICE_SESSION_CHALLENGE_TTL_SECONDS)
        challenge_id = self._id_generator()
        nonce_b64 = secrets.token_urlsafe(32)
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            session = conn.execute(
                "SELECT * FROM mobile_sessions WHERE session_id = ?",
                (bootstrap_session_id,),
            ).fetchone()
            if (
                session is None
                or session["user_id"] != user_id
                or not self.session_is_active(session, now)
                or any(
                    session[name] is not None
                    for name in ("paired_device_id", "grant_id", "grant_version", "desktop_id")
                )
            ):
                raise DeviceSessionError("Bootstrap session is not eligible for activation.")
            try:
                grant = load_active_grant(
                    conn,
                    device_id=device_id,
                    grant_id=grant_id,
                    expected_user_id=user_id,
                    now=now,
                )
            except GrantAuthorizationError as exc:
                raise DeviceSessionError("Device grant is not active.") from exc
            conn.execute(
                """INSERT INTO mobile_device_session_challenges(
                       challenge_id, bootstrap_session_id, user_id, device_id,
                       grant_id, grant_version, desktop_id, nonce_b64,
                       created_at, expires_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    challenge_id,
                    bootstrap_session_id,
                    user_id,
                    grant.device_id,
                    grant.grant_id,
                    grant.version,
                    grant.desktop_id,
                    nonce_b64,
                    now.isoformat(),
                    expires_at.isoformat(),
                ),
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
        return DeviceSessionChallenge(
            challenge_id=challenge_id,
            bootstrap_session_id=bootstrap_session_id,
            user_id=user_id,
            device_id=grant.device_id,
            grant_id=grant.grant_id,
            grant_version=grant.version,
            desktop_id=grant.desktop_id,
            nonce_b64=nonce_b64,
            expires_at=expires_at,
        )

    def activate_device_session(
        self,
        *,
        bootstrap_session_id: str,
        user_id: str,
        challenge_id: str,
        signature_b64: str,
    ) -> CreatedSession:
        """Verify device proof and atomically replace the bootstrap session."""
        from rex.mobile_api.authorization import (  # noqa: PLC0415
            GrantAuthorizationError,
            load_active_grant,
        )
        from rex.mobile_api.device_proof import (  # noqa: PLC0415
            ProofError,
            canonical_session_transcript,
            verify_proof,
        )

        user_id = validate_user_id(user_id)
        if not challenge_id or len(challenge_id) > 128:
            raise DeviceSessionError("Device session challenge is invalid or expired.")
        if not isinstance(signature_b64, str) or not signature_b64:
            raise DeviceSessionError("Device proof could not be verified.")
        now = self.now()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            challenge = conn.execute(
                "SELECT * FROM mobile_device_session_challenges WHERE challenge_id = ?",
                (challenge_id,),
            ).fetchone()
            expires_at = self._parse_ts(challenge["expires_at"]) if challenge is not None else None
            if (
                challenge is None
                or challenge["bootstrap_session_id"] != bootstrap_session_id
                or challenge["user_id"] != user_id
                or challenge["used_at"] is not None
                or expires_at is None
                or now >= expires_at
            ):
                raise DeviceSessionError("Device session challenge is invalid or expired.")
            bootstrap = conn.execute(
                "SELECT * FROM mobile_sessions WHERE session_id = ?",
                (bootstrap_session_id,),
            ).fetchone()
            if (
                bootstrap is None
                or bootstrap["user_id"] != user_id
                or not self.session_is_active(bootstrap, now)
                or any(
                    bootstrap[name] is not None
                    for name in ("paired_device_id", "grant_id", "grant_version", "desktop_id")
                )
            ):
                raise DeviceSessionError("Bootstrap session is not eligible for activation.")
            try:
                grant = load_active_grant(
                    conn,
                    device_id=str(challenge["device_id"]),
                    grant_id=str(challenge["grant_id"]),
                    expected_user_id=user_id,
                    expected_desktop_id=str(challenge["desktop_id"]),
                    expected_version=int(challenge["grant_version"]),
                    now=now,
                )
                transcript = canonical_session_transcript(
                    desktop_id=grant.desktop_id,
                    bootstrap_session_id=bootstrap_session_id,
                    challenge_id=challenge_id,
                    nonce_b64=str(challenge["nonce_b64"]),
                    device_id=grant.device_id,
                    grant_id=grant.grant_id,
                    grant_version=grant.version,
                    user_id=user_id,
                )
                verify_proof(
                    public_key_b64=grant.public_key_b64,
                    signature_b64=signature_b64,
                    transcript=transcript,
                )
            except (GrantAuthorizationError, ProofError) as exc:
                raise DeviceSessionError("Device proof could not be verified.") from exc

            new_session_id = self._id_generator()
            family_id = self._id_generator()
            raw_token = self._token_generator()
            refresh_expires_at = now + timedelta(seconds=self._refresh_ttl_seconds)
            conn.execute(
                """INSERT INTO mobile_sessions(
                       session_id, user_id, device_id, device_name, platform,
                       app_version, paired_device_id, grant_id, grant_version,
                       desktop_id, strong_auth_at, created_at, last_seen_at, expires_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_session_id,
                    user_id,
                    bootstrap["device_id"],
                    bootstrap["device_name"],
                    bootstrap["platform"],
                    bootstrap["app_version"],
                    grant.device_id,
                    grant.grant_id,
                    grant.version,
                    grant.desktop_id,
                    None,
                    now.isoformat(),
                    now.isoformat(),
                    refresh_expires_at.isoformat(),
                ),
            )
            conn.execute(
                """INSERT INTO mobile_refresh_tokens(
                       token_hash, family_id, session_id, user_id, created_at, expires_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    hash_refresh_token(raw_token),
                    family_id,
                    new_session_id,
                    user_id,
                    now.isoformat(),
                    refresh_expires_at.isoformat(),
                ),
            )
            conn.execute(
                """UPDATE mobile_device_session_challenges
                   SET used_at = ?, replacement_session_id = ?
                   WHERE challenge_id = ? AND used_at IS NULL""",
                (now.isoformat(), new_session_id, challenge_id),
            )
            conn.execute(
                """UPDATE mobile_sessions SET revoked_at = ?, revoke_reason = ?
                   WHERE session_id = ? AND revoked_at IS NULL""",
                (now.isoformat(), "device_session_activated", bootstrap_session_id),
            )
            conn.execute(
                """UPDATE mobile_refresh_tokens SET revoked_at = ?
                   WHERE session_id = ? AND revoked_at IS NULL""",
                (now.isoformat(), bootstrap_session_id),
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
        logger.info(
            "Mobile paired session activated: session=%s user=%s device=%s grant=%s",
            new_session_id,
            user_id,
            grant.device_id,
            grant.grant_id,
        )
        return CreatedSession(
            session_id=new_session_id,
            user_id=user_id,
            refresh_token=raw_token,
            refresh_expires_at=refresh_expires_at,
            family_id=family_id,
        )

    def get_session(self, session_id: str) -> sqlite3.Row | None:
        """Return the session row for *session_id*, or None."""
        conn = self._connect()
        try:
            row: sqlite3.Row | None = conn.execute(
                "SELECT * FROM mobile_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row
        finally:
            conn.close()

    def touch_session(self, session_id: str) -> None:
        """Update the session's last-seen timestamp."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE mobile_sessions SET last_seen_at = ? WHERE session_id = ?",
                (self.now().isoformat(), session_id),
            )
        finally:
            conn.close()

    def revoke_session(self, session_id: str, reason: str) -> bool:
        """Revoke one session and all of its refresh tokens (idempotent)."""
        now_iso = self.now().isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE mobile_sessions
                SET revoked_at = ?, revoke_reason = ?
                WHERE session_id = ? AND revoked_at IS NULL
                """,
                (now_iso, reason, session_id),
            )
            conn.execute(
                """
                UPDATE mobile_refresh_tokens
                SET revoked_at = ?
                WHERE session_id = ? AND revoked_at IS NULL
                """,
                (now_iso, session_id),
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
        revoked = cursor.rowcount > 0
        if revoked:
            logger.info("Mobile session revoked: session=%s reason=%s", session_id, reason)
        return revoked

    def revoke_all_sessions_for_user(self, user_id: str, reason: str) -> int:
        """Revoke every active session for one validated user only."""
        user_id = validate_user_id(user_id)
        now_iso = self.now().isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                """
                UPDATE mobile_sessions
                SET revoked_at = ?, revoke_reason = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (now_iso, reason, user_id),
            )
            conn.execute(
                """
                UPDATE mobile_refresh_tokens
                SET revoked_at = ?
                WHERE user_id = ? AND revoked_at IS NULL
                """,
                (now_iso, user_id),
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
        count = cursor.rowcount
        logger.info(
            "Mobile logout-all: user=%s revoked_sessions=%d reason=%s",
            user_id,
            count,
            reason,
        )
        return count

    # ------------------------------------------------------------------
    # Refresh rotation
    # ------------------------------------------------------------------

    def rotate_refresh_token(self, raw_token: str) -> RotationResult:
        """Atomically rotate a refresh token.

        Exactly one concurrent caller can succeed for a given token: the
        consume step updates ``consumed_at`` only where it is still NULL.
        Reuse of a consumed token revokes the family and session.
        """
        if not raw_token or not isinstance(raw_token, str):
            return RotationResult(status=INVALID)
        token_hash = hash_refresh_token(raw_token)

        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM mobile_refresh_tokens WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return RotationResult(status=INVALID)

            now = self.now()

            if row["revoked_at"] is not None:
                conn.execute("COMMIT")
                return RotationResult(status=INVALID)

            if row["consumed_at"] is not None:
                self._revoke_family_locked(conn, row["family_id"], row["session_id"], now)
                conn.execute("COMMIT")
                logger.warning(
                    "Mobile refresh token reuse detected: session=%s family=%s "
                    "user=%s — family and session revoked",
                    row["session_id"],
                    row["family_id"],
                    row["user_id"],
                )
                self._emit_reuse_audit_event(
                    session_id=row["session_id"],
                    family_id=row["family_id"],
                    user_id=row["user_id"],
                )
                return RotationResult(
                    status=REUSED,
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                )

            token_expires = self._parse_ts(row["expires_at"])
            if token_expires is None or now >= token_expires:
                conn.execute("COMMIT")
                return RotationResult(status=EXPIRED)

            session = conn.execute(
                "SELECT * FROM mobile_sessions WHERE session_id = ?",
                (row["session_id"],),
            ).fetchone()
            if not self.session_is_active(session, now):
                conn.execute("COMMIT")
                return RotationResult(status=SESSION_REVOKED)
            try:
                from rex.mobile_api.authorization import (  # noqa: PLC0415
                    GrantAuthorizationError,
                    resolve_session_grant,
                )

                resolve_session_grant(conn, session, now=now)
            except GrantAuthorizationError:
                self._revoke_family_locked(
                    conn,
                    row["family_id"],
                    row["session_id"],
                    now,
                    "device_grant_invalid",
                )
                conn.execute("COMMIT")
                return RotationResult(status=SESSION_REVOKED)

            user = conn.execute("SELECT * FROM users WHERE id = ?", (row["user_id"],)).fetchone()
            user_disabled = user is None or user["disabled_at"] is not None
            if user_disabled:
                self._revoke_family_locked(
                    conn,
                    row["family_id"],
                    row["session_id"],
                    now,
                    "user_inactive",
                )
                conn.execute("COMMIT")
                logger.info(
                    "Mobile refresh rejected for inactive user: session=%s",
                    row["session_id"],
                )
                return RotationResult(status=USER_INACTIVE)

            # Consume — this is the concurrency arbitration point.
            consumed = conn.execute(
                """
                UPDATE mobile_refresh_tokens
                SET consumed_at = ?
                WHERE token_hash = ? AND consumed_at IS NULL
                """,
                (now.isoformat(), token_hash),
            )
            if consumed.rowcount != 1:
                # A concurrent rotation won the race: treat as reuse.
                self._revoke_family_locked(conn, row["family_id"], row["session_id"], now)
                conn.execute("COMMIT")
                logger.warning(
                    "Mobile refresh concurrency loser treated as reuse: session=%s family=%s",
                    row["session_id"],
                    row["family_id"],
                )
                self._emit_reuse_audit_event(
                    session_id=row["session_id"],
                    family_id=row["family_id"],
                    user_id=row["user_id"],
                )
                return RotationResult(
                    status=REUSED,
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                )

            new_raw = self._token_generator()
            new_hash = hash_refresh_token(new_raw)
            new_expires = now + timedelta(seconds=self._refresh_ttl_seconds)
            conn.execute(
                """
                INSERT INTO mobile_refresh_tokens (
                    token_hash, family_id, session_id, user_id,
                    created_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    new_hash,
                    row["family_id"],
                    row["session_id"],
                    row["user_id"],
                    now.isoformat(),
                    new_expires.isoformat(),
                ),
            )
            conn.execute(
                """
                UPDATE mobile_refresh_tokens
                SET replacement_hash = ?
                WHERE token_hash = ?
                """,
                (new_hash, token_hash),
            )
            conn.execute(
                """
                UPDATE mobile_sessions
                SET last_seen_at = ?, expires_at = ?
                WHERE session_id = ?
                """,
                (now.isoformat(), new_expires.isoformat(), row["session_id"]),
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

        return RotationResult(
            status=ROTATED,
            session_id=row["session_id"],
            user_id=row["user_id"],
            refresh_token=new_raw,
            refresh_expires_at=new_expires,
        )

    @staticmethod
    def _revoke_family_locked(
        conn: sqlite3.Connection,
        family_id: str,
        session_id: str,
        now: datetime,
        reason: str = "refresh_token_reuse",
    ) -> None:
        """Revoke a token family and its session inside the caller's transaction."""
        now_iso = now.isoformat()
        conn.execute(
            """
            UPDATE mobile_refresh_tokens
            SET revoked_at = ?
            WHERE family_id = ? AND revoked_at IS NULL
            """,
            (now_iso, family_id),
        )
        conn.execute(
            """
            UPDATE mobile_sessions
            SET revoked_at = ?, revoke_reason = ?
            WHERE session_id = ? AND revoked_at IS NULL
            """,
            (now_iso, reason[:128], session_id),
        )


__all__ = [
    "EXPIRED",
    "INVALID",
    "REUSED",
    "ROTATED",
    "SESSION_REVOKED",
    "USER_INACTIVE",
    "CreatedSession",
    "DEVICE_SESSION_CHALLENGE_TTL_SECONDS",
    "DeviceSessionChallenge",
    "DeviceSessionError",
    "DeviceInfo",
    "MobileSessionStore",
    "RotationResult",
    "hash_refresh_token",
]
