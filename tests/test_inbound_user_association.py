"""Tests for inbound SMS user association (Cycle 4.3).

Covers:
- Webhook persists ``user_id`` when account has ``owner_user_id``
- Webhook leaves ``user_id`` as None when account has no ``owner_user_id``
- Inbound store migration adds ``user_id`` column to legacy tables
- ``SMSService.receive(user_id=...)`` filters correctly
- CLI ``rex msg receive --user <id>`` resolves identity and passes through

All tests are offline and deterministic — uses ``tmp_path``.
"""

from __future__ import annotations

import argparse
import sqlite3
from unittest.mock import patch

import pytest
from flask import Flask

from rex.messaging_backends.inbound_store import InboundSmsRecord, InboundSmsStore
from rex.messaging_backends.inbound_webhook import create_inbound_sms_blueprint

_AUTH_TOKEN = "test_auth_token_user_assoc"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def inbound_store(tmp_path):
    """Create a temp-backed inbound store."""
    return InboundSmsStore(db_path=tmp_path / "inbound_user_test.db")


@pytest.fixture()
def raw_config_with_owner():
    """Config where the personal account has an owner_user_id."""
    return {
        "messaging": {
            "backend": "twilio",
            "default_account_id": "personal",
            "accounts": [
                {
                    "id": "personal",
                    "from_number": "+15551111111",
                    "credential_ref": "twilio:personal",
                    "owner_user_id": "alice",
                },
                {
                    "id": "business",
                    "label": "Business Line",
                    "from_number": "+15552222222",
                    "credential_ref": "twilio:business",
                },
            ],
        }
    }


@pytest.fixture()
def raw_config_no_owner():
    """Config where no accounts have owner_user_id."""
    return {
        "messaging": {
            "backend": "twilio",
            "accounts": [
                {
                    "id": "personal",
                    "from_number": "+15551111111",
                    "credential_ref": "twilio:personal",
                },
            ],
        }
    }


@pytest.fixture()
def app_with_owner(inbound_store, raw_config_with_owner):
    """Flask app with owner_user_id configured on one account."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    bp = create_inbound_sms_blueprint(
        auth_token=_AUTH_TOKEN,
        inbound_store=inbound_store,
        raw_config=raw_config_with_owner,
        signature_verification=False,
    )
    flask_app.register_blueprint(bp)
    return flask_app


@pytest.fixture()
def client_with_owner(app_with_owner):
    return app_with_owner.test_client()


@pytest.fixture()
def app_no_owner(inbound_store, raw_config_no_owner):
    """Flask app without owner_user_id."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    bp = create_inbound_sms_blueprint(
        auth_token=_AUTH_TOKEN,
        inbound_store=inbound_store,
        raw_config=raw_config_no_owner,
        signature_verification=False,
    )
    flask_app.register_blueprint(bp)
    return flask_app


@pytest.fixture()
def client_no_owner(app_no_owner):
    return app_no_owner.test_client()


def _make_twilio_params(
    *,
    message_sid: str = "SM0001",
    from_number: str = "+15559999999",
    to_number: str = "+15551111111",
    body: str = "Hello from test",
) -> dict[str, str]:
    return {
        "MessageSid": message_sid,
        "From": from_number,
        "To": to_number,
        "Body": body,
    }


# ---------------------------------------------------------------------------
# Webhook user_id persistence
# ---------------------------------------------------------------------------


class TestWebhookUserAssociation:
    """Webhook persists user_id when account has owner_user_id."""

    def test_user_id_set_when_owner_configured(self, client_with_owner, inbound_store) -> None:
        """Inbound to owned account stores owner_user_id as user_id."""
        params = _make_twilio_params(to_number="+15551111111")
        resp = client_with_owner.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        assert records[0].account_id == "personal"
        assert records[0].user_id == "alice"
        assert records[0].routed is True

    def test_user_id_none_when_no_owner(self, client_with_owner, inbound_store) -> None:
        """Inbound to account without owner_user_id stores user_id=None."""
        params = _make_twilio_params(to_number="+15552222222")
        resp = client_with_owner.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        assert records[0].account_id == "business"
        assert records[0].user_id is None
        assert records[0].routed is True

    def test_user_id_none_when_unrouted(self, client_with_owner, inbound_store) -> None:
        """Inbound to unknown number has no user_id."""
        params = _make_twilio_params(to_number="+15559999999")
        resp = client_with_owner.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        assert records[0].account_id is None
        assert records[0].user_id is None
        assert records[0].routed is False

    def test_user_id_none_when_config_has_no_owner(self, client_no_owner, inbound_store) -> None:
        """Accounts without owner_user_id produce user_id=None."""
        params = _make_twilio_params(to_number="+15551111111")
        resp = client_no_owner.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        assert records[0].account_id == "personal"
        assert records[0].user_id is None


# ---------------------------------------------------------------------------
# Inbound store migration
# ---------------------------------------------------------------------------


class TestInboundStoreMigration:
    """Migration adds user_id column to legacy databases."""

    def test_migration_adds_user_id_column(self, tmp_path) -> None:
        """Opening a store on a legacy DB (no user_id column) adds it."""
        db_path = tmp_path / "legacy.db"

        # Create a legacy table WITHOUT user_id
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE inbound_sms (
                id            TEXT PRIMARY KEY,
                sid           TEXT NOT NULL DEFAULT '',
                from_number   TEXT NOT NULL DEFAULT '',
                to_number     TEXT NOT NULL DEFAULT '',
                body          TEXT NOT NULL DEFAULT '',
                received_at   TEXT NOT NULL,
                account_id    TEXT,
                routed        INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO inbound_sms (id, sid, from_number, to_number, body, received_at, routed)
            VALUES ('old1', 'SM_OLD', '+1', '+2', 'legacy msg', '2025-01-01T00:00:00+00:00', 1)
            """
        )
        conn.commit()
        conn.close()

        # Open the store — migration should add user_id
        store = InboundSmsStore(db_path=db_path)

        # Verify old record is intact and user_id is NULL
        records = store.query_recent()
        assert len(records) == 1
        assert records[0].id == "old1"
        assert records[0].sid == "SM_OLD"
        assert records[0].user_id is None

        # Verify new records can be written with user_id
        store.write(
            InboundSmsRecord(
                sid="SM_NEW",
                from_number="+1",
                to_number="+2",
                body="new msg",
                user_id="alice",
            )
        )
        alice_records = store.query_recent(user_id="alice")
        assert len(alice_records) == 1
        assert alice_records[0].user_id == "alice"

    def test_migration_idempotent(self, tmp_path) -> None:
        """Opening the store multiple times does not error."""
        db_path = tmp_path / "idempotent.db"
        store1 = InboundSmsStore(db_path=db_path)
        store1.write(InboundSmsRecord(sid="SM1", from_number="+1", to_number="+2", body="x"))

        # Open again — should not fail
        store2 = InboundSmsStore(db_path=db_path)
        assert store2.count() == 1


# ---------------------------------------------------------------------------
# SMSService.receive() user_id filtering
# ---------------------------------------------------------------------------


class TestSMSServiceUserFiltering:
    """SMSService.receive(user_id=...) filters inbound store results."""

    def test_receive_user_id_only(self, tmp_path) -> None:
        """Filtering by user_id returns only that user's messages."""
        from rex.messaging_backends.stub import StubSmsBackend
        from rex.messaging_service import SMSService

        store = InboundSmsStore(db_path=tmp_path / "svc_user.db")
        backend = StubSmsBackend(fixture_path=tmp_path / "mock_sms.json")
        svc = SMSService(backend=backend, inbound_store=store)

        store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1555",
                to_number="+1666",
                body="alice msg",
                user_id="alice",
                account_id="primary",
            )
        )
        store.write(
            InboundSmsRecord(
                sid="SM2",
                from_number="+1555",
                to_number="+1666",
                body="bob msg",
                user_id="bob",
                account_id="primary",
            )
        )
        store.write(
            InboundSmsRecord(
                sid="SM3",
                from_number="+1555",
                to_number="+1666",
                body="no user msg",
                account_id="primary",
            )
        )

        # Filter for alice
        alice_msgs = svc.receive(limit=10, user_id="alice")
        assert len(alice_msgs) == 1
        assert alice_msgs[0].body == "alice msg"

        # Filter for bob
        bob_msgs = svc.receive(limit=10, user_id="bob")
        assert len(bob_msgs) == 1
        assert bob_msgs[0].body == "bob msg"

    def test_receive_user_id_and_account_id(self, tmp_path) -> None:
        """Combining user_id and account_id narrows results."""
        from rex.messaging_backends.stub import StubSmsBackend
        from rex.messaging_service import SMSService

        store = InboundSmsStore(db_path=tmp_path / "svc_combo.db")
        backend = StubSmsBackend(fixture_path=tmp_path / "mock_sms.json")
        svc = SMSService(backend=backend, inbound_store=store)

        store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1555",
                to_number="+1666",
                body="alice primary",
                user_id="alice",
                account_id="primary",
            )
        )
        store.write(
            InboundSmsRecord(
                sid="SM2",
                from_number="+1555",
                to_number="+1777",
                body="alice secondary",
                user_id="alice",
                account_id="secondary",
            )
        )

        # Filter for alice + primary
        msgs = svc.receive(limit=10, user_id="alice", account_id="primary")
        assert len(msgs) == 1
        assert msgs[0].body == "alice primary"

    def test_receive_no_user_returns_all(self, tmp_path) -> None:
        """Without user_id filter, all messages are returned."""
        from rex.messaging_backends.stub import StubSmsBackend
        from rex.messaging_service import SMSService

        store = InboundSmsStore(db_path=tmp_path / "svc_all.db")
        backend = StubSmsBackend(fixture_path=tmp_path / "mock_sms.json")
        svc = SMSService(backend=backend, inbound_store=store)

        store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1555",
                to_number="+1666",
                body="alice msg",
                user_id="alice",
            )
        )
        store.write(
            InboundSmsRecord(
                sid="SM2",
                from_number="+1555",
                to_number="+1666",
                body="unowned msg",
            )
        )

        msgs = svc.receive(limit=10)
        assert len(msgs) == 2

    def test_receive_nonexistent_user_returns_empty(self, tmp_path) -> None:
        """Filtering for a user with no messages returns empty list."""
        from rex.messaging_backends.stub import StubSmsBackend
        from rex.messaging_service import SMSService

        store = InboundSmsStore(db_path=tmp_path / "svc_empty.db")
        backend = StubSmsBackend(fixture_path=tmp_path / "mock_sms.json")
        svc = SMSService(backend=backend, inbound_store=store)

        store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1555",
                to_number="+1666",
                body="alice msg",
                user_id="alice",
            )
        )

        msgs = svc.receive(limit=10, user_id="unknown")
        assert msgs == []


# ---------------------------------------------------------------------------
# CLI identity resolution passthrough
# ---------------------------------------------------------------------------


class TestCLIUserPassthrough:
    """CLI ``rex msg receive --user <id>`` resolves identity and passes through."""

    def test_cli_receive_passes_user_id(self, tmp_path, capsys) -> None:
        """``--user alice`` is resolved and passed to SMSService.receive()."""
        from rex.messaging_backends.stub import StubSmsBackend
        from rex.messaging_service import SMSService

        store = InboundSmsStore(db_path=tmp_path / "cli_user.db")
        backend = StubSmsBackend(fixture_path=tmp_path / "mock_sms.json")
        svc = SMSService(backend=backend, inbound_store=store)

        store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1555",
                to_number="+1666",
                body="alice inbound",
                user_id="alice",
            )
        )
        store.write(
            InboundSmsRecord(
                sid="SM2",
                from_number="+1555",
                to_number="+1666",
                body="bob inbound",
                user_id="bob",
            )
        )

        args = argparse.Namespace(
            msg_command="receive",
            channel="sms",
            limit=10,
            user="alice",
        )

        with (
            patch("rex.messaging_service.get_sms_service", return_value=svc),
            patch("rex.cli._resolve_cli_user", return_value="alice"),
        ):
            from rex.cli import cmd_msg

            result = cmd_msg(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "alice inbound" in captured.out
        assert "bob inbound" not in captured.out

    def test_cli_receive_no_user_shows_all(self, tmp_path, capsys) -> None:
        """Without --user, all inbound messages are shown."""
        from rex.messaging_backends.stub import StubSmsBackend
        from rex.messaging_service import SMSService

        store = InboundSmsStore(db_path=tmp_path / "cli_all.db")
        backend = StubSmsBackend(fixture_path=tmp_path / "mock_sms.json")
        svc = SMSService(backend=backend, inbound_store=store)

        store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1555",
                to_number="+1666",
                body="alice inbound",
                user_id="alice",
            )
        )
        store.write(
            InboundSmsRecord(
                sid="SM2",
                from_number="+1555",
                to_number="+1666",
                body="bob inbound",
                user_id="bob",
            )
        )

        args = argparse.Namespace(
            msg_command="receive",
            channel="sms",
            limit=10,
        )

        with (
            patch("rex.messaging_service.get_sms_service", return_value=svc),
            patch("rex.cli._resolve_cli_user", return_value=None),
        ):
            from rex.cli import cmd_msg

            result = cmd_msg(args)

        assert result == 0
        captured = capsys.readouterr()
        assert "alice inbound" in captured.out
        assert "bob inbound" in captured.out
        assert "Total: 2 messages" in captured.out


# ---------------------------------------------------------------------------
# MessagingAccountConfig owner_user_id
# ---------------------------------------------------------------------------


class TestMessagingAccountConfigOwner:
    """owner_user_id field on MessagingAccountConfig."""

    def test_owner_user_id_accepted(self) -> None:
        """Config with owner_user_id parses successfully."""
        from rex.messaging_backends.account_config import load_messaging_config

        raw = {
            "messaging": {
                "backend": "stub",
                "accounts": [
                    {
                        "id": "primary",
                        "from_number": "+15551234567",
                        "credential_ref": "twilio:primary",
                        "owner_user_id": "alice",
                    }
                ],
            }
        }
        config = load_messaging_config(raw)
        assert len(config.accounts) == 1
        assert config.accounts[0].owner_user_id == "alice"

    def test_owner_user_id_defaults_none(self) -> None:
        """Config without owner_user_id defaults to None."""
        from rex.messaging_backends.account_config import load_messaging_config

        raw = {
            "messaging": {
                "backend": "stub",
                "accounts": [
                    {
                        "id": "primary",
                        "from_number": "+15551234567",
                        "credential_ref": "twilio:primary",
                    }
                ],
            }
        }
        config = load_messaging_config(raw)
        assert config.accounts[0].owner_user_id is None

    def test_owner_user_id_null_in_json(self) -> None:
        """Explicit null for owner_user_id parses as None."""
        from rex.messaging_backends.account_config import load_messaging_config

        raw = {
            "messaging": {
                "backend": "stub",
                "accounts": [
                    {
                        "id": "primary",
                        "from_number": "+15551234567",
                        "credential_ref": "twilio:primary",
                        "owner_user_id": None,
                    }
                ],
            }
        }
        config = load_messaging_config(raw)
        assert config.accounts[0].owner_user_id is None
