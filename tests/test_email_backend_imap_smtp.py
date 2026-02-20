"""Tests for the IMAP/SMTP email backend.

All tests use mocks/fakes — no real network calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rex.email_backends.imap_smtp import ImapSmtpEmailBackend
from rex.email_backends.stub import StubEmailBackend

# ---------------------------------------------------------------------------
# Stub backend tests
# ---------------------------------------------------------------------------


class TestStubEmailBackend:
    """Tests for the StubEmailBackend."""

    def test_connect_missing_fixture(self, tmp_path):
        backend = StubEmailBackend(fixture_path=tmp_path / "missing.json")
        assert backend.connect() is True
        assert backend.is_connected is True
        assert backend.fetch_unread() == []

    def test_connect_loads_fixture(self, tmp_path):
        import json

        fixture = tmp_path / "emails.json"
        fixture.write_text(
            json.dumps(
                [
                    {
                        "id": "e1",
                        "from_addr": "a@b.com",
                        "subject": "Hello",
                        "snippet": "Hi there",
                        "received_at": "2026-01-01T10:00:00",
                        "labels": ["unread"],
                    }
                ]
            ),
            encoding="utf-8",
        )

        backend = StubEmailBackend(fixture_path=fixture)
        assert backend.connect() is True
        unread = backend.fetch_unread()
        assert len(unread) == 1
        assert unread[0].message_id == "e1"
        assert unread[0].from_addr == "a@b.com"

    def test_fetch_unread_limit(self, tmp_path):
        import json

        fixture = tmp_path / "emails.json"
        fixture.write_text(
            json.dumps(
                [
                    {
                        "id": f"e{i}",
                        "from_addr": "a@b.com",
                        "subject": f"Msg {i}",
                        "snippet": "",
                        "received_at": "2026-01-01T10:00:00",
                        "labels": ["unread"],
                    }
                    for i in range(5)
                ]
            ),
            encoding="utf-8",
        )

        backend = StubEmailBackend(fixture_path=fixture)
        backend.connect()
        assert len(backend.fetch_unread(limit=2)) == 2

    def test_send_stub(self, tmp_path):
        backend = StubEmailBackend(fixture_path=tmp_path / "none.json")
        result = backend.send(
            from_addr="me@x.com",
            to_addrs=["you@y.com"],
            subject="Test",
            body="Hello",
        )
        assert result.ok is True
        assert result.message_id == "stub-msg-id"
        assert len(backend.sent_messages) == 1

    def test_mark_as_read(self, tmp_path):
        import json

        fixture = tmp_path / "emails.json"
        fixture.write_text(
            json.dumps(
                [
                    {
                        "id": "e1",
                        "from_addr": "a@b.com",
                        "subject": "S",
                        "snippet": "",
                        "received_at": "2026-01-01T10:00:00",
                        "labels": ["unread"],
                    }
                ]
            ),
            encoding="utf-8",
        )
        backend = StubEmailBackend(fixture_path=fixture)
        backend.connect()
        assert backend.mark_as_read("e1") is True
        assert backend.fetch_unread() == []
        assert backend.mark_as_read("nonexistent") is False

    def test_list_mailboxes(self, tmp_path):
        backend = StubEmailBackend(fixture_path=tmp_path / "none.json")
        assert backend.list_mailboxes() == ["INBOX"]

    def test_disconnect(self, tmp_path):
        backend = StubEmailBackend(fixture_path=tmp_path / "none.json")
        backend.connect()
        assert backend.is_connected is True
        backend.disconnect()
        assert backend.is_connected is False


# ---------------------------------------------------------------------------
# ImapSmtp backend tests (all mocked)
# ---------------------------------------------------------------------------


class TestImapSmtpBackend:
    """Tests for the ImapSmtpEmailBackend with mocked IMAP/SMTP."""

    def _make_backend(self, imap_mock=None, smtp_mock=None):
        return ImapSmtpEmailBackend(
            imap_host="imap.example.com",
            imap_port=993,
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="user@example.com",
            password="secret",
            use_starttls=True,
            imap_client_factory=imap_mock,
            smtp_client_factory=smtp_mock,
        )

    def test_connect_success(self):
        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [b"1"])

        backend = self._make_backend(imap_mock=lambda: mock_imap)
        assert backend.connect() is True
        assert backend.is_connected is True
        mock_imap.login.assert_called_once_with("user@example.com", "secret")

    def test_connect_auth_failure(self):
        import imaplib

        mock_imap = MagicMock()
        mock_imap.login.side_effect = imaplib.IMAP4.error("auth failed")

        backend = self._make_backend(imap_mock=lambda: mock_imap)
        assert backend.connect() is False
        assert backend.is_connected is False

    def test_connect_network_failure(self):
        def factory():
            raise OSError("Connection refused")

        backend = self._make_backend(imap_mock=factory)
        assert backend.connect() is False

    def test_fetch_unread_not_connected(self):
        backend = self._make_backend()
        assert backend.fetch_unread() == []

    def test_fetch_unread_empty(self):
        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [b"1"])
        mock_imap.search.return_value = ("OK", [b""])

        backend = self._make_backend(imap_mock=lambda: mock_imap)
        backend.connect()
        assert backend.fetch_unread() == []

    def test_fetch_unread_parses_headers(self):
        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [b"1"])
        mock_imap.search.return_value = ("OK", [b"1"])

        raw_headers = (
            b"From: sender@example.com\r\n"
            b"To: me@example.com\r\n"
            b"Subject: Test Subject\r\n"
            b"Date: Mon, 20 Feb 2026 10:00:00 +0000\r\n"
            b"Message-ID: <msg-001@example.com>\r\n"
            b"\r\n"
        )
        mock_imap.fetch.return_value = ("OK", [(b"1 (RFC822.HEADER {200}", raw_headers)])

        backend = self._make_backend(imap_mock=lambda: mock_imap)
        backend.connect()

        envelopes = backend.fetch_unread(limit=10)
        assert len(envelopes) == 1
        env = envelopes[0]
        assert env.from_addr == "sender@example.com"
        assert env.subject == "Test Subject"
        assert "unread" in env.labels

    def test_send_success(self):
        mock_smtp = MagicMock()
        mock_smtp.ehlo.return_value = (250, b"OK")
        mock_smtp.starttls.return_value = (220, b"OK")
        mock_smtp.login.return_value = (235, b"OK")
        mock_smtp.send_message.return_value = {}
        mock_smtp.quit.return_value = (221, b"Bye")

        backend = self._make_backend(smtp_mock=lambda: mock_smtp)
        result = backend.send(
            from_addr="user@example.com",
            to_addrs=["recipient@example.com"],
            subject="Test Email",
            body="Hello, World!",
        )
        assert result.ok is True
        mock_smtp.login.assert_called_once_with("user@example.com", "secret")
        mock_smtp.send_message.assert_called_once()

    def test_send_auth_failure(self):
        import smtplib

        mock_smtp = MagicMock()
        mock_smtp.ehlo.return_value = (250, b"OK")
        mock_smtp.starttls.return_value = (220, b"OK")
        mock_smtp.login.side_effect = smtplib.SMTPAuthenticationError(535, b"bad creds")

        backend = self._make_backend(smtp_mock=lambda: mock_smtp)
        result = backend.send(
            from_addr="user@example.com",
            to_addrs=["recipient@example.com"],
            subject="Fail",
            body="Oops",
        )
        assert result.ok is False
        assert "authentication" in (result.error or "").lower()

    def test_send_smtp_error(self):
        import smtplib

        mock_smtp = MagicMock()
        mock_smtp.ehlo.return_value = (250, b"OK")
        mock_smtp.starttls.return_value = (220, b"OK")
        mock_smtp.login.return_value = (235, b"OK")
        mock_smtp.send_message.side_effect = smtplib.SMTPException("relay denied")

        backend = self._make_backend(smtp_mock=lambda: mock_smtp)
        result = backend.send(
            from_addr="user@example.com",
            to_addrs=["recipient@example.com"],
            subject="Fail",
            body="Oops",
        )
        assert result.ok is False
        assert result.error is not None

    def test_list_mailboxes(self):
        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [b"1"])
        mock_imap.list.return_value = (
            "OK",
            [b'(\\HasNoChildren) "/" "INBOX"', b'(\\HasNoChildren) "/" "Sent"'],
        )

        backend = self._make_backend(imap_mock=lambda: mock_imap)
        backend.connect()
        boxes = backend.list_mailboxes()
        assert "INBOX" in boxes
        assert "Sent" in boxes

    def test_mark_as_read(self):
        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [b"1"])
        mock_imap.search.return_value = ("OK", [b"42"])
        mock_imap.store.return_value = ("OK", [])

        backend = self._make_backend(imap_mock=lambda: mock_imap)
        backend.connect()
        assert backend.mark_as_read("<msg-001@example.com>") is True
        mock_imap.store.assert_called_once()

    def test_disconnect(self):
        mock_imap = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [b"1"])

        backend = self._make_backend(imap_mock=lambda: mock_imap)
        backend.connect()
        assert backend.is_connected is True
        backend.disconnect()
        assert backend.is_connected is False


# ---------------------------------------------------------------------------
# EmailBackend interface conformance
# ---------------------------------------------------------------------------


class TestBackendInterface:
    """Verify both backends implement the full EmailBackend interface."""

    @pytest.mark.parametrize(
        "method",
        ["connect", "fetch_unread", "list_mailboxes", "mark_as_read", "send", "disconnect"],
    )
    def test_stub_has_method(self, method):
        backend = StubEmailBackend()
        assert hasattr(backend, method)

    @pytest.mark.parametrize(
        "method",
        ["connect", "fetch_unread", "list_mailboxes", "mark_as_read", "send", "disconnect"],
    )
    def test_imap_smtp_has_method(self, method):
        backend = ImapSmtpEmailBackend(imap_host="x", smtp_host="x", username="x", password="x")
        assert hasattr(backend, method)
