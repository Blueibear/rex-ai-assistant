"""Tests for the inbound SMS SQLite store.

All tests use ``tmp_path`` — no tracked repo files are written.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rex.messaging_backends.inbound_store import (
    InboundSmsRecord,
    InboundSmsStore,
    InboundStoreConfig,
    load_inbound_store_config,
)


@pytest.fixture()
def store(tmp_path):
    """Create a fresh InboundSmsStore backed by a temp directory."""
    db_path = tmp_path / "test_inbound.db"
    return InboundSmsStore(db_path=db_path, retention_days=30)


class TestInboundSmsStore:
    """Tests for InboundSmsStore write/query/cleanup."""

    def test_write_and_query(self, store: InboundSmsStore) -> None:
        """Write a record and retrieve it."""
        record = InboundSmsRecord(
            sid="SM123",
            from_number="+15551111111",
            to_number="+15552222222",
            body="Hello test",
            account_id="primary",
            user_id="alice",
            routed=True,
        )
        record_id = store.write(record)
        assert record_id == record.id

        results = store.query_recent(limit=10)
        assert len(results) == 1
        assert results[0].id == record.id
        assert results[0].sid == "SM123"
        assert results[0].from_number == "+15551111111"
        assert results[0].to_number == "+15552222222"
        assert results[0].body == "Hello test"
        assert results[0].account_id == "primary"
        assert results[0].user_id == "alice"
        assert results[0].routed is True

    def test_query_ordered_newest_first(self, store: InboundSmsStore) -> None:
        """Records are returned newest first."""
        now = datetime.now(timezone.utc)
        for i in range(5):
            rec = InboundSmsRecord(
                sid=f"SM{i}",
                from_number="+1555",
                to_number="+1666",
                body=f"msg {i}",
                received_at=now - timedelta(minutes=5 - i),
            )
            store.write(rec)

        results = store.query_recent(limit=10)
        assert len(results) == 5
        # Most recent first
        assert results[0].sid == "SM4"
        assert results[-1].sid == "SM0"

    def test_query_limit(self, store: InboundSmsStore) -> None:
        """Limit parameter restricts result count."""
        for i in range(10):
            store.write(InboundSmsRecord(sid=f"SM{i}", from_number="+1", to_number="+2", body="x"))
        results = store.query_recent(limit=3)
        assert len(results) == 3

    def test_query_filter_by_user_id(self, store: InboundSmsStore) -> None:
        """Filter results by user_id."""
        store.write(
            InboundSmsRecord(sid="SM1", from_number="+1", to_number="+2", body="a", user_id="alice")
        )
        store.write(
            InboundSmsRecord(sid="SM2", from_number="+1", to_number="+2", body="b", user_id="bob")
        )

        alice_msgs = store.query_recent(user_id="alice")
        assert len(alice_msgs) == 1
        assert alice_msgs[0].user_id == "alice"

    def test_query_filter_by_account_id(self, store: InboundSmsStore) -> None:
        """Filter results by account_id."""
        store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1",
                to_number="+2",
                body="a",
                account_id="primary",
            )
        )
        store.write(
            InboundSmsRecord(
                sid="SM2",
                from_number="+1",
                to_number="+2",
                body="b",
                account_id="secondary",
            )
        )

        primary_msgs = store.query_recent(account_id="primary")
        assert len(primary_msgs) == 1
        assert primary_msgs[0].account_id == "primary"

    def test_count(self, store: InboundSmsStore) -> None:
        """Count returns correct total."""
        assert store.count() == 0
        store.write(InboundSmsRecord(sid="SM1", from_number="+1", to_number="+2", body="a"))
        store.write(
            InboundSmsRecord(sid="SM2", from_number="+1", to_number="+2", body="b", user_id="alice")
        )
        assert store.count() == 2
        assert store.count(user_id="alice") == 1

    def test_cleanup_old(self, tmp_path) -> None:
        """Cleanup removes records older than retention_days."""
        store = InboundSmsStore(db_path=tmp_path / "cleanup.db", retention_days=7)
        old_time = datetime.now(timezone.utc) - timedelta(days=10)
        recent_time = datetime.now(timezone.utc) - timedelta(hours=1)

        store.write(
            InboundSmsRecord(
                sid="OLD",
                from_number="+1",
                to_number="+2",
                body="old",
                received_at=old_time,
            )
        )
        store.write(
            InboundSmsRecord(
                sid="RECENT",
                from_number="+1",
                to_number="+2",
                body="recent",
                received_at=recent_time,
            )
        )

        deleted = store.cleanup_old()
        assert deleted == 1
        remaining = store.query_recent()
        assert len(remaining) == 1
        assert remaining[0].sid == "RECENT"

    def test_record_to_dict(self) -> None:
        """InboundSmsRecord.to_dict() produces expected keys."""
        rec = InboundSmsRecord(
            sid="SM1",
            from_number="+1",
            to_number="+2",
            body="hello",
            account_id="primary",
            user_id="alice",
            routed=True,
        )
        d = rec.to_dict()
        assert d["sid"] == "SM1"
        assert d["from_number"] == "+1"
        assert d["to_number"] == "+2"
        assert d["body"] == "hello"
        assert d["account_id"] == "primary"
        assert d["user_id"] == "alice"
        assert d["routed"] is True
        assert "id" in d
        assert "received_at" in d

    def test_unrouted_message(self, store: InboundSmsStore) -> None:
        """Unrouted messages are stored with routed=False."""
        store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1",
                to_number="+2",
                body="unrouted",
                routed=False,
            )
        )
        results = store.query_recent()
        assert len(results) == 1
        assert results[0].routed is False
        assert results[0].account_id is None


class TestInboundStoreConfig:
    """Tests for config parsing."""

    def test_defaults(self) -> None:
        """Default config has inbound disabled."""
        config = InboundStoreConfig()
        assert config.enabled is False
        assert config.retention_days == 90
        assert config.auth_token_ref == "twilio:inbound"

    def test_load_from_raw_config(self) -> None:
        """Parse inbound config from a full runtime config dict."""
        raw = {
            "messaging": {
                "inbound": {
                    "enabled": True,
                    "auth_token_ref": "twilio:primary",
                    "store_path": "/tmp/test.db",
                    "retention_days": 60,
                }
            }
        }
        config = load_inbound_store_config(raw)
        assert config.enabled is True
        assert config.auth_token_ref == "twilio:primary"
        assert config.store_path == "/tmp/test.db"
        assert config.retention_days == 60

    def test_load_empty_config(self) -> None:
        """Missing messaging section returns defaults."""
        config = load_inbound_store_config({})
        assert config.enabled is False

    def test_load_missing_inbound_section(self) -> None:
        """Missing inbound subsection returns defaults."""
        config = load_inbound_store_config({"messaging": {"backend": "stub"}})
        assert config.enabled is False
