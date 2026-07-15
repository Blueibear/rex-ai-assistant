"""Refresh rotation, reuse detection, concurrency, and revocation tests.

Matrix rows: REF-001..REF-010, REF-016, REF-018.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from tests.mobile_api.conftest import (
    create_user,
    disable_user,
    login_tokens,
    sequential_token_generator,
)

_REFRESH_URL = "/mobile/auth/refresh"


def _refresh(client, token):
    return client.post(_REFRESH_URL, json={"refresh_token": token})


def _db_dump(db_path: Path) -> str:
    conn = sqlite3.connect(str(db_path))
    try:
        return "\n".join(conn.iterdump())
    finally:
        conn.close()


class TestRefreshRotation:
    def test_valid_refresh_returns_new_pair_same_session(self, client) -> None:
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        response = _refresh(client, tokens["refresh_token"])
        assert response.status_code == 200
        body = response.get_json()
        assert body["session_id"] == tokens["session_id"]
        assert body["refresh_token"] != tokens["refresh_token"]
        assert body["access_token"]
        assert body["user"]["id"]

    def test_raw_refresh_tokens_never_stored(self, client, services) -> None:
        """REF-002: only hashes are stored; raw values are absent from the DB."""
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        rotated = _refresh(client, tokens["refresh_token"]).get_json()
        dump = _db_dump(services.db_path)
        assert tokens["refresh_token"] not in dump
        assert rotated["refresh_token"] not in dump

    def test_reuse_of_consumed_token_revokes_family_and_session(self, client) -> None:
        """REF-003: replaying a rotated token kills the whole family."""
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        rotated = _refresh(client, tokens["refresh_token"]).get_json()

        reuse = _refresh(client, tokens["refresh_token"])
        assert reuse.status_code == 401
        assert reuse.get_json()["error"]["code"] == "AUTH_REFRESH_REUSED"

        # The still-fresh replacement token is now revoked too.
        follow_up = _refresh(client, rotated["refresh_token"])
        assert follow_up.status_code == 401

        # And the session no longer authenticates the rotated access token.
        session = client.get(
            "/mobile/auth/session",
            headers={"Authorization": f"Bearer {rotated['access_token']}"},
        )
        assert session.status_code == 401

    def test_concurrent_refresh_yields_exactly_one_success(self, mobile_env: Path, clock) -> None:
        """REF-004: two concurrent rotations of one token → one winner."""
        from rex.mobile_api.db import migrate_users_db
        from rex.mobile_api.sessions import (
            ROTATED,
            DeviceInfo,
            MobileSessionStore,
        )
        from tests.mobile_api.conftest import RecordingAuditLogger

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)
        user_id = create_user("james", "pw-123456")
        store = MobileSessionStore(
            db_path,
            refresh_ttl_seconds=30 * 86400,
            clock=clock,
            token_generator=sequential_token_generator(),
            audit_logger=RecordingAuditLogger(),
        )
        created = store.create_session(user_id, DeviceInfo(device_id="device-1"))

        barrier = threading.Barrier(2)
        results: list = [None, None]

        def _rotate(slot: int) -> None:
            barrier.wait()
            results[slot] = store.rotate_refresh_token(created.refresh_token)

        threads = [threading.Thread(target=_rotate, args=(slot,)) for slot in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        statuses = sorted(result.status for result in results)
        assert statuses.count(ROTATED) == 1
        issued = [r.refresh_token for r in results if r.refresh_token]
        assert len(issued) == 1


class TestRefreshRejection:
    def test_expired_refresh_token_rejected(self, client, clock) -> None:
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        clock.advance(days=31)
        response = _refresh(client, tokens["refresh_token"])
        assert response.status_code == 401
        assert response.get_json()["error"]["code"] == "AUTH_TOKEN_EXPIRED"

    def test_revoked_refresh_token_rejected(self, client) -> None:
        """REF-006: logout revokes the refresh token."""
        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        client.post(
            "/mobile/auth/logout",
            headers={"Authorization": f"Bearer {tokens['access_token']}"},
        )
        response = _refresh(client, tokens["refresh_token"])
        assert response.status_code == 401

    def test_disabled_user_refresh_rejected_and_session_revoked(self, client, services) -> None:
        """REF-008 / USR-005: a disabled user cannot refresh."""
        user_id = create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        disable_user(services.db_path, user_id)
        response = _refresh(client, tokens["refresh_token"])
        assert response.status_code == 401
        session = services.session_store.get_session(tokens["session_id"])
        assert session["revoked_at"] is not None

    def test_random_token_rejected_without_enumeration(self, client) -> None:
        response = _refresh(client, "r" * 64)
        assert response.status_code == 401
        assert response.get_json()["error"]["code"] == "AUTH_TOKEN_INVALID"

    def test_malformed_refresh_payloads_rejected(self, client) -> None:
        assert client.post(_REFRESH_URL, json={}).status_code == 400
        assert client.post(_REFRESH_URL, json={"refresh_token": 12345}).status_code == 400
        assert client.post(_REFRESH_URL, json={"refresh_token": ""}).status_code == 400
        assert client.post(_REFRESH_URL, json={"refresh_token": "x" * 4096}).status_code == 400

    def test_refresh_rate_limit(self, mobile_env: Path, clock) -> None:
        """REF-016: the refresh route has its own limit and leaks no tokens."""
        from rex.config import MobileApiConfig
        from rex.mobile_api.app import create_mobile_app
        from rex.mobile_api.db import migrate_users_db
        from rex.mobile_api.services import MobileApiServices

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)
        config = MobileApiConfig(
            rate_limit_refresh="2 per minute", rate_limit_default="100 per minute"
        )
        services = MobileApiServices.build(config, db_path=db_path, clock=clock)
        app = create_mobile_app(services=services)
        app.config["TESTING"] = True
        with app.test_client() as client:
            for _ in range(2):
                _refresh(client, "does-not-matter")
            limited = _refresh(client, "does-not-matter")
        assert limited.status_code == 429
        assert "does-not-matter" not in limited.get_data(as_text=True)


class TestReuseAudit:
    def test_reuse_event_logged_with_safe_ids_only(self, client, caplog) -> None:
        """REF-018: the audit event has session/family IDs but no raw token."""
        import logging

        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        _refresh(client, tokens["refresh_token"])
        with caplog.at_level(logging.WARNING):
            _refresh(client, tokens["refresh_token"])
        reuse_records = [
            record for record in caplog.records if "reuse" in record.getMessage().lower()
        ]
        assert reuse_records
        log_text = " ".join(record.getMessage() for record in reuse_records)
        assert tokens["session_id"] in log_text
        assert tokens["refresh_token"] not in log_text

    def test_reuse_creates_exactly_one_persistent_audit_event(
        self, client, services, audit_recorder
    ) -> None:
        """A structured security-audit event is persisted on reuse."""
        user_id = create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")

        _refresh(client, tokens["refresh_token"])  # normal rotation
        assert audit_recorder.entries == []  # no reuse event on rotation

        reuse = _refresh(client, tokens["refresh_token"])
        assert reuse.status_code == 401
        assert len(audit_recorder.entries) == 1

        entry = audit_recorder.entries[0]
        args = entry.tool_call_args
        assert args["event_type"] == "mobile_refresh_token_reuse"
        assert args["session_id"] == tokens["session_id"]
        assert args["family_id"]
        assert args["user_id"] == user_id
        assert args["revocation_result"] == "family_and_session_revoked"
        assert entry.action_id
        assert entry.timestamp is not None

        # The revocation actually happened.
        session = services.session_store.get_session(tokens["session_id"])
        assert session["revoked_at"] is not None

    def test_audit_event_contains_no_token_material(self, client, audit_recorder) -> None:
        from rex.mobile_api.sessions import hash_refresh_token

        create_user("james", "pw-123456")
        tokens = login_tokens(client, "james", "pw-123456")
        rotated = _refresh(client, tokens["refresh_token"]).get_json()
        _refresh(client, tokens["refresh_token"])  # reuse

        assert len(audit_recorder.entries) == 1
        serialized = audit_recorder.entries[0].model_dump_json()
        for forbidden in (
            tokens["refresh_token"],
            rotated["refresh_token"],
            hash_refresh_token(tokens["refresh_token"]),
            hash_refresh_token(rotated["refresh_token"]),
            tokens["access_token"],
            rotated["access_token"],
            "pw-123456",
        ):
            assert forbidden not in serialized

    def test_two_users_reuse_events_stay_separate(self, client, audit_recorder) -> None:
        james_id = create_user("james", "pw-123456")
        sarah_id = create_user("sarah", "pw-abcdef")
        james = login_tokens(client, "james", "pw-123456")
        sarah = login_tokens(client, "sarah", "pw-abcdef")

        _refresh(client, james["refresh_token"])
        _refresh(client, james["refresh_token"])  # james reuse
        _refresh(client, sarah["refresh_token"])
        _refresh(client, sarah["refresh_token"])  # sarah reuse

        assert len(audit_recorder.entries) == 2
        by_user = {e.tool_call_args["user_id"]: e.tool_call_args for e in audit_recorder.entries}
        assert by_user[james_id]["session_id"] == james["session_id"]
        assert by_user[sarah_id]["session_id"] == sarah["session_id"]

    def test_default_audit_logger_writes_persistent_event(
        self, mobile_env, clock, tmp_path
    ) -> None:
        """Without injection, the canonical rex.audit logger persists the event."""
        from rex.audit import AuditLogger
        from rex.mobile_api.db import migrate_users_db
        from rex.mobile_api.sessions import DeviceInfo, MobileSessionStore

        db_path = mobile_env / "users.db"
        migrate_users_db(db_path)
        user_id = create_user("james", "pw-123456")
        audit = AuditLogger(log_path=tmp_path / "logs" / "audit.log")
        store = MobileSessionStore(
            db_path,
            refresh_ttl_seconds=30 * 86400,
            clock=clock,
            audit_logger=audit,
        )
        created = store.create_session(user_id, DeviceInfo(device_id="dev-1"))
        store.rotate_refresh_token(created.refresh_token)
        store.rotate_refresh_token(created.refresh_token)  # reuse

        entries = audit.read()
        reuse_entries = [
            e for e in entries if e.tool_call_args.get("event_type") == "mobile_refresh_token_reuse"
        ]
        assert len(reuse_entries) == 1
        assert reuse_entries[0].tool_call_args["user_id"] == user_id
        assert created.refresh_token not in (tmp_path / "logs" / "audit.log").read_text(
            encoding="utf-8"
        )
