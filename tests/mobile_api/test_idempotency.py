"""Cross-transport idempotency store tests.

Matrix rows: IDP-001, IDP-002, IDP-007, IDP-008, IDP-009, IDP-011, IDP-012.
Transport-level duplicate rows (IDP-003..IDP-006, IDP-010) are covered in
the HTTP/SSE/WebSocket test modules.
"""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

import pytest

from rex.mobile_api import idempotency as idem
from tests.mobile_api.conftest import FakeClock


@pytest.fixture()
def store(tmp_path: Path, clock: FakeClock) -> idem.MobileMessageStore:
    from rex.mobile_api.db import migrate_users_db

    db_path = tmp_path / "users.db"
    migrate_users_db(db_path)
    return idem.MobileMessageStore(db_path, retention_hours=48, clock=clock)


def _fields(message: str = "hello", **overrides) -> dict:
    fields = {
        "message_id": "5f8f1f2a-1111-4111-8111-111111111111",
        "conversation_id": "5f8f1f2a-2222-4222-8222-222222222222",
        "sent_at": "2026-07-15T12:00:00+00:00",
        "message": message,
        "mode": "mobile_text",
        "client_context": {"device": "iphone"},
    }
    fields.update(overrides)
    return fields


USER_A = "3d6f2c1e-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
USER_B = "3d6f2c1e-bbbb-4bbb-8bbb-bbbbbbbbbbbb"


class TestRequestHash:
    def test_hash_is_deterministic_and_semantic(self) -> None:
        """IDP-012: semantic fields drive the hash; key order does not."""
        fields = _fields()
        reordered = dict(reversed(list(fields.items())))
        assert idem.compute_request_hash(fields) == idem.compute_request_hash(reordered)

    def test_hash_changes_with_message(self) -> None:
        assert idem.compute_request_hash(_fields("a")) != idem.compute_request_hash(_fields("b"))

    def test_hash_ignores_transport_fields(self) -> None:
        """IDP-012: transport-only fields are excluded."""
        with_transport = _fields()
        with_transport["type"] = "chat"
        with_transport["authorization"] = "Bearer nope"
        assert idem.compute_request_hash(with_transport) == idem.compute_request_hash(_fields())


class TestReservation:
    def test_first_reservation_is_new(self, store) -> None:
        """IDP-001: the first reservation wins and is durable."""
        fields = _fields()
        result = store.reserve(
            USER_A,
            fields["message_id"],
            fields["conversation_id"],
            idem.compute_request_hash(fields),
        )
        assert result.outcome == idem.RESERVED
        row = store.get(USER_A, fields["message_id"])
        assert row is not None and row["status"] == idem.STATUS_PROCESSING

    def test_exact_duplicate_of_processing(self, store) -> None:
        fields = _fields()
        request_hash = idem.compute_request_hash(fields)
        store.reserve(USER_A, fields["message_id"], fields["conversation_id"], request_hash)
        dup = store.reserve(USER_A, fields["message_id"], fields["conversation_id"], request_hash)
        assert dup.outcome == idem.DUPLICATE_PROCESSING

    def test_exact_duplicate_of_completed_replays_result(self, store) -> None:
        """IDP-002: a completed duplicate returns the stored result."""
        fields = _fields()
        request_hash = idem.compute_request_hash(fields)
        store.reserve(USER_A, fields["message_id"], fields["conversation_id"], request_hash)
        store.complete(USER_A, fields["message_id"], '{"response": "done"}')
        dup = store.reserve(USER_A, fields["message_id"], fields["conversation_id"], request_hash)
        assert dup.outcome == idem.DUPLICATE_COMPLETED
        assert dup.response_json == '{"response": "done"}'

    def test_failed_duplicate_replays_error_code(self, store) -> None:
        """IDP-010: a failed request's terminal code is replayed, not re-run."""
        fields = _fields()
        request_hash = idem.compute_request_hash(fields)
        store.reserve(USER_A, fields["message_id"], fields["conversation_id"], request_hash)
        store.fail(USER_A, fields["message_id"], "BACKEND_UNAVAILABLE")
        dup = store.reserve(USER_A, fields["message_id"], fields["conversation_id"], request_hash)
        assert dup.outcome == idem.DUPLICATE_FAILED
        assert dup.error_code == "BACKEND_UNAVAILABLE"

    def test_same_id_different_payload_conflicts(self, store) -> None:
        """IDP-007: a reused ID with different semantics never executes."""
        fields = _fields()
        store.reserve(
            USER_A,
            fields["message_id"],
            fields["conversation_id"],
            idem.compute_request_hash(fields),
        )
        other = idem.compute_request_hash(_fields("something else"))
        result = store.reserve(USER_A, fields["message_id"], fields["conversation_id"], other)
        assert result.outcome == idem.CONFLICT

    def test_same_id_different_user_is_independent(self, store) -> None:
        """IDP-008: no cross-user reservation sharing or result leak."""
        fields = _fields()
        request_hash = idem.compute_request_hash(fields)
        first = store.reserve(USER_A, fields["message_id"], fields["conversation_id"], request_hash)
        second = store.reserve(
            USER_B, fields["message_id"], fields["conversation_id"], request_hash
        )
        assert first.outcome == idem.RESERVED
        assert second.outcome == idem.RESERVED
        store.complete(USER_A, fields["message_id"], '{"response": "user-a-only"}')
        dup_b = store.reserve(USER_B, fields["message_id"], fields["conversation_id"], request_hash)
        assert dup_b.outcome == idem.DUPLICATE_PROCESSING
        assert dup_b.response_json is None

    def test_invalid_user_id_fails_before_db(self, store) -> None:
        with pytest.raises(ValueError):
            store.reserve("../evil", "m", "c", "h")

    def test_concurrent_duplicates_yield_one_winner(self, tmp_path) -> None:
        """IDP-006: exactly one concurrent reservation wins."""
        from rex.mobile_api.db import migrate_users_db

        db_path = tmp_path / "concurrent.db"
        migrate_users_db(db_path)
        real_store = idem.MobileMessageStore(db_path)
        fields = _fields()
        request_hash = idem.compute_request_hash(fields)
        outcomes: list[str] = []
        lock = threading.Lock()

        def attempt() -> None:
            result = real_store.reserve(
                USER_A, fields["message_id"], fields["conversation_id"], request_hash
            )
            with lock:
                outcomes.append(result.outcome)

        threads = [threading.Thread(target=attempt) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outcomes.count(idem.RESERVED) == 1
        assert len(outcomes) == 8


class TestRetention:
    def test_completed_result_survives_new_store_instance(self, tmp_path, clock) -> None:
        """IDP-009: terminal results persist across service restarts."""
        from rex.mobile_api.db import migrate_users_db

        db_path = tmp_path / "users.db"
        migrate_users_db(db_path)
        first = idem.MobileMessageStore(db_path, clock=clock)
        fields = _fields()
        request_hash = idem.compute_request_hash(fields)
        first.reserve(USER_A, fields["message_id"], fields["conversation_id"], request_hash)
        first.complete(USER_A, fields["message_id"], '{"response": "persisted"}')

        restarted = idem.MobileMessageStore(db_path, clock=clock)
        dup = restarted.reserve(
            USER_A, fields["message_id"], fields["conversation_id"], request_hash
        )
        assert dup.outcome == idem.DUPLICATE_COMPLETED
        assert dup.response_json == '{"response": "persisted"}'

    def test_prune_removes_only_expired_rows(self, store, clock) -> None:
        """IDP-011: expired rows go; active rows stay."""
        old = _fields(message_id=str(uuid.uuid4()))
        store.reserve(USER_A, old["message_id"], old["conversation_id"], "hash-old")
        store.complete(USER_A, old["message_id"], "{}")
        clock.advance(days=3)
        removed = store.prune_expired()
        assert removed == 1
        assert store.get(USER_A, old["message_id"]) is None
        fresh = _fields(message_id=str(uuid.uuid4()))
        store.reserve(USER_A, fresh["message_id"], fresh["conversation_id"], "hash-new")
        assert store.prune_expired() == 0
        assert store.get(USER_A, fresh["message_id"]) is not None
