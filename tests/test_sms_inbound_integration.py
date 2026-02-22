"""Integration tests for SMSService with inbound store.

Verifies that ``rex msg receive`` path (via SMSService.receive()) correctly
merges messages from both the backend and the inbound webhook store.

All tests are offline and deterministic — uses tmp_path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rex.messaging_backends.inbound_store import InboundSmsRecord, InboundSmsStore
from rex.messaging_backends.stub import StubSmsBackend
from rex.messaging_service import SMSService


@pytest.fixture()
def inbound_store(tmp_path):
    """Create a temp-backed inbound store."""
    return InboundSmsStore(db_path=tmp_path / "inbound_test.db")


@pytest.fixture()
def stub_backend(tmp_path):
    """Create a stub SMS backend in a temp directory."""
    return StubSmsBackend(fixture_path=tmp_path / "mock_sms.json")


class TestSMSServiceWithInboundStore:
    """Tests for SMSService.receive() with inbound store integration."""

    def test_receive_from_inbound_store_only(self, stub_backend, inbound_store) -> None:
        """When backend has no messages, inbound store messages are returned."""
        svc = SMSService(
            backend=stub_backend,
            inbound_store=inbound_store,
        )

        # Add a message to the inbound store
        inbound_store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+15559999999",
                to_number="+15551111111",
                body="Webhook message",
            )
        )

        messages = svc.receive(limit=10)
        assert len(messages) == 1
        assert messages[0].body == "Webhook message"
        assert messages[0].from_ == "+15559999999"

    def test_receive_merges_backend_and_store(self, stub_backend, inbound_store) -> None:
        """Messages from both backend and inbound store are merged."""
        svc = SMSService(
            backend=stub_backend,
            inbound_store=inbound_store,
        )

        # Add inbound via backend (stub)
        stub_backend.inject_inbound(
            from_number="+15550001111",
            body="Backend message",
        )

        # Add inbound via webhook store
        inbound_store.write(
            InboundSmsRecord(
                sid="SM_WH",
                from_number="+15550002222",
                to_number="+15551111111",
                body="Webhook message",
            )
        )

        messages = svc.receive(limit=10)
        assert len(messages) == 2
        bodies = {m.body for m in messages}
        assert "Backend message" in bodies
        assert "Webhook message" in bodies

    def test_receive_respects_limit(self, stub_backend, inbound_store) -> None:
        """Limit is applied to merged results."""
        svc = SMSService(
            backend=stub_backend,
            inbound_store=inbound_store,
        )

        for i in range(5):
            inbound_store.write(
                InboundSmsRecord(
                    sid=f"SM{i}",
                    from_number="+1555",
                    to_number="+1666",
                    body=f"msg {i}",
                )
            )

        messages = svc.receive(limit=3)
        assert len(messages) == 3

    def test_receive_newest_first(self, stub_backend, inbound_store) -> None:
        """Merged results are sorted newest first."""
        svc = SMSService(
            backend=stub_backend,
            inbound_store=inbound_store,
        )
        now = datetime.now(timezone.utc)

        inbound_store.write(
            InboundSmsRecord(
                sid="OLD",
                from_number="+1555",
                to_number="+1666",
                body="old",
                received_at=now - timedelta(hours=2),
            )
        )
        inbound_store.write(
            InboundSmsRecord(
                sid="NEW",
                from_number="+1555",
                to_number="+1666",
                body="new",
                received_at=now,
            )
        )

        messages = svc.receive(limit=10)
        assert messages[0].body == "new"
        assert messages[1].body == "old"

    def test_receive_filters_by_user_id(self, stub_backend, inbound_store) -> None:
        """User ID filter is passed to the inbound store."""
        svc = SMSService(
            backend=stub_backend,
            inbound_store=inbound_store,
        )

        inbound_store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1555",
                to_number="+1666",
                body="alice msg",
                user_id="alice",
            )
        )
        inbound_store.write(
            InboundSmsRecord(
                sid="SM2",
                from_number="+1555",
                to_number="+1666",
                body="bob msg",
                user_id="bob",
            )
        )

        messages = svc.receive(limit=10, user_id="alice")
        # Only alice's webhook message; backend messages are unfiltered
        webhook_bodies = [m.body for m in messages]
        assert "alice msg" in webhook_bodies
        # bob's message should not be in the inbound store results
        assert "bob msg" not in webhook_bodies

    def test_receive_filters_by_account_id(self, stub_backend, inbound_store) -> None:
        """Account ID filter is passed to the inbound store."""
        svc = SMSService(
            backend=stub_backend,
            inbound_store=inbound_store,
        )

        inbound_store.write(
            InboundSmsRecord(
                sid="SM1",
                from_number="+1555",
                to_number="+1666",
                body="primary msg",
                account_id="primary",
            )
        )
        inbound_store.write(
            InboundSmsRecord(
                sid="SM2",
                from_number="+1555",
                to_number="+1666",
                body="secondary msg",
                account_id="secondary",
            )
        )

        messages = svc.receive(limit=10, account_id="primary")
        webhook_bodies = [m.body for m in messages]
        assert "primary msg" in webhook_bodies
        assert "secondary msg" not in webhook_bodies

    def test_receive_without_inbound_store(self, stub_backend) -> None:
        """When no inbound store, receive works as before (backend only)."""
        svc = SMSService(backend=stub_backend)

        stub_backend.inject_inbound(
            from_number="+15550001111",
            body="Backend only message",
        )

        messages = svc.receive(limit=10)
        assert len(messages) == 1
        assert messages[0].body == "Backend only message"

    def test_receive_with_mock_file_and_inbound_store(self, tmp_path, inbound_store) -> None:
        """Legacy mock-file mode also merges with inbound store."""
        mock_file = tmp_path / "mock_sms.json"
        svc = SMSService(
            mock_file=mock_file,
            from_number="+15551111111",
            inbound_store=inbound_store,
        )

        inbound_store.write(
            InboundSmsRecord(
                sid="SM_WH",
                from_number="+15550003333",
                to_number="+15551111111",
                body="Webhook in mock mode",
            )
        )

        messages = svc.receive(limit=10)
        assert len(messages) == 1
        assert messages[0].body == "Webhook in mock mode"
