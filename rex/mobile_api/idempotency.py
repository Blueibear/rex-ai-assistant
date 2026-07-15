"""Cross-transport chat idempotency (issue #323, Session 2).

HTTP, SSE, and WebSocket chat share one durable reservation store keyed by
``(user_id, message_id)`` in the canonical users database
(``mobile_message_requests``).  The reservation is inserted *before* any
acknowledgement or Assistant/tool execution, so a retried or replayed
message ID can never execute twice — regardless of which transport carried
the retry.

Semantics:

- A new ``(user_id, message_id)`` is reserved atomically (``BEGIN
  IMMEDIATE`` + ``INSERT``); exactly one concurrent caller wins.
- An exact duplicate (same deterministic request hash) of a *completed*
  request replays the stored terminal result without executing anything.
- An exact duplicate of a *failed* request replays the stored error code —
  clients retry a genuinely new attempt with a new message ID.
- An exact duplicate of a request that is *still processing* is reported as
  in progress and never starts a second execution.
- The same message ID with a *different* request hash is a conflict and
  never executes.
- The same message ID from a different user is an independent request.

The request hash covers only semantic execution fields (message text, IDs,
mode, timestamp, client context) — never transport artifacts, tokens, or
headers.  Stored rows contain the terminal response body (which the client
already received) and never tokens, passwords, or credentials.

Retention: rows older than ``mobile_api.idempotency_retention_hours``
(default 48h) are pruned opportunistically on reservation; the window is
deliberately longer than any realistic reconnect/retry cycle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rex.identity import validate_user_id
from rex.mobile_api.db import connect

logger = logging.getLogger(__name__)

# Reservation outcomes.
RESERVED = "reserved"
DUPLICATE_PROCESSING = "duplicate_processing"
DUPLICATE_COMPLETED = "duplicate_completed"
DUPLICATE_FAILED = "duplicate_failed"
CONFLICT = "conflict"

# Row statuses.
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Error code recorded when a streaming client disconnected before the
# terminal event.  Stored (not a wire code) so a replay of the same message
# ID reports the terminal failure instead of executing tools twice.
CLIENT_DISCONNECTED = "CLIENT_DISCONNECTED"

# Semantic execution fields included in the deterministic request hash.
_HASH_FIELDS = (
    "message_id",
    "conversation_id",
    "sent_at",
    "message",
    "mode",
    "client_context",
)


def compute_request_hash(payload: Mapping[str, Any]) -> str:
    """Return the deterministic hash of a chat request's semantic fields.

    Only fields that influence execution are included; transport-only
    artifacts (headers, frame types, tokens) never are.  A replayed message
    must carry the exact original semantic payload to be treated as the
    same request.
    """
    canonical = {name: payload.get(name) for name in _HASH_FIELDS}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Reservation:
    """Outcome of a reservation attempt for ``(user_id, message_id)``."""

    outcome: str
    status: str | None = None
    response_json: str | None = None
    error_code: str | None = None

    @property
    def is_new(self) -> bool:
        return self.outcome == RESERVED


def _default_clock() -> datetime:
    return datetime.now(UTC)


class MobileMessageStore:
    """SQLite-backed shared idempotency store for mobile chat requests."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        retention_hours: int = 48,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._retention = timedelta(hours=retention_hours)
        self._clock = clock or _default_clock

    def now(self) -> datetime:
        return self._clock()

    def _connect(self) -> sqlite3.Connection:
        return connect(self._db_path)

    # ── Reservation ────────────────────────────────────────────────────

    def reserve(
        self,
        user_id: str,
        message_id: str,
        conversation_id: str,
        request_hash: str,
    ) -> Reservation:
        """Durably reserve ``(user_id, message_id)`` before any execution.

        Exactly one concurrent caller receives ``RESERVED``; every other
        caller observes the existing row's state without executing.
        """
        user_id = validate_user_id(user_id)
        now = self.now().isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT request_hash, status, response_json, error_code "
                "FROM mobile_message_requests WHERE user_id = ? AND message_id = ?",
                (user_id, message_id),
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO mobile_message_requests "
                    "(user_id, message_id, conversation_id, request_hash, status, "
                    " created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        user_id,
                        message_id,
                        conversation_id,
                        request_hash,
                        STATUS_PROCESSING,
                        now,
                        now,
                    ),
                )
                conn.execute("COMMIT")
                self._prune_expired_safely()
                return Reservation(outcome=RESERVED, status=STATUS_PROCESSING)
            conn.execute("COMMIT")
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()

        if row["request_hash"] != request_hash:
            return Reservation(outcome=CONFLICT, status=str(row["status"]))
        status = str(row["status"])
        if status == STATUS_COMPLETED:
            return Reservation(
                outcome=DUPLICATE_COMPLETED,
                status=status,
                response_json=row["response_json"],
            )
        if status == STATUS_FAILED:
            return Reservation(
                outcome=DUPLICATE_FAILED,
                status=status,
                error_code=row["error_code"],
            )
        return Reservation(outcome=DUPLICATE_PROCESSING, status=status)

    # ── Terminal state ─────────────────────────────────────────────────

    def complete(self, user_id: str, message_id: str, response_json: str) -> None:
        """Store the terminal successful result for a reserved request."""
        self._finish(user_id, message_id, STATUS_COMPLETED, response_json, None)

    def fail(self, user_id: str, message_id: str, error_code: str) -> None:
        """Store the terminal failure code for a reserved request."""
        self._finish(user_id, message_id, STATUS_FAILED, None, error_code)

    def _finish(
        self,
        user_id: str,
        message_id: str,
        status: str,
        response_json: str | None,
        error_code: str | None,
    ) -> None:
        user_id = validate_user_id(user_id)
        now = self.now().isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "UPDATE mobile_message_requests "
                "SET status = ?, response_json = ?, error_code = ?, updated_at = ? "
                "WHERE user_id = ? AND message_id = ? AND status = ?",
                (status, response_json, error_code, now, user_id, message_id, STATUS_PROCESSING),
            )
            conn.execute("COMMIT")
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def get(self, user_id: str, message_id: str) -> sqlite3.Row | None:
        """Return the stored request row (or None)."""
        user_id = validate_user_id(user_id)
        conn = self._connect()
        try:
            row: sqlite3.Row | None = conn.execute(
                "SELECT * FROM mobile_message_requests WHERE user_id = ? AND message_id = ?",
                (user_id, message_id),
            ).fetchone()
            return row
        finally:
            conn.close()

    # ── Retention ──────────────────────────────────────────────────────

    def prune_expired(self) -> int:
        """Delete rows older than the retention window; return count removed.

        Terminal and stale ``processing`` rows (e.g. after a crash) are both
        removed once older than retention.  Active rows are never touched.
        """
        cutoff = (self.now() - self._retention).isoformat()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.execute(
                "DELETE FROM mobile_message_requests WHERE created_at < ?",
                (cutoff,),
            )
            conn.execute("COMMIT")
            return cursor.rowcount
        except BaseException:
            self._rollback(conn)
            raise
        finally:
            conn.close()

    def _prune_expired_safely(self) -> None:
        try:
            removed = self.prune_expired()
            if removed:
                logger.info("Pruned %d expired mobile message request(s)", removed)
        except sqlite3.Error:  # pragma: no cover - opportunistic housekeeping
            logger.warning("Mobile idempotency pruning failed", exc_info=True)

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - already rolled back
            pass


__all__ = [
    "CLIENT_DISCONNECTED",
    "CONFLICT",
    "DUPLICATE_COMPLETED",
    "DUPLICATE_FAILED",
    "DUPLICATE_PROCESSING",
    "RESERVED",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_PROCESSING",
    "MobileMessageStore",
    "Reservation",
    "compute_request_hash",
]
