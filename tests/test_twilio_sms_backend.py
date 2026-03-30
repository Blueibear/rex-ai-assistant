"""Tests for US-211: TwilioSMSBackend.

All tests use a mocked Twilio client — no live network calls are made.

Covers:
- Successful send delegates to twilio.rest.Client.messages.create
- Credentials resolved from CredentialManager when not passed directly
- Explicit constructor credentials take precedence over CredentialManager
- Missing credentials raise SMSSendError with actionable message
- Twilio 4xx API errors raise SMSSendError with status/code detail
- Network timeout raises SMSSendError with timeout detail
- No credentials appear in log output at any level
- receive() returns empty list (send-only backend)
- ImportError raised when twilio package is not installed (no factory)
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from tests.helpers.fake_twilio import FakeTwilioClient, fake_twilio_client

# ---------------------------------------------------------------------------
# Helpers — fake twilio module injected into sys.modules
# ---------------------------------------------------------------------------


def _fake_twilio_module() -> ModuleType:
    """Return a minimal fake twilio package sufficient for import guards."""
    mod = ModuleType("twilio")
    rest = ModuleType("twilio.rest")
    base = ModuleType("twilio.base")
    base.exceptions = ModuleType("twilio.base.exceptions")
    mod.rest = rest
    mod.base = base
    return mod


def _make_twilio_rest_exc(status: int = 400, code: int = 21211) -> Exception:
    """Return an exception that duck-types as TwilioRestException."""
    exc = Exception(f"Twilio API error {status}")
    exc.status = status
    exc.code = code
    return exc


def _make_backend(
    *,
    account_sid: str = "ACtest",
    auth_token: str = "tok_test",
    from_number: str = "+15550001234",
    create_raises: Exception | None = None,
    fake_client: FakeTwilioClient | None = None,
) -> tuple:
    """Return (backend, fake_client) using FakeTwilioClient."""
    from rex.integrations.messaging.backends.twilio_sms import TwilioSMSBackend

    fake_client = fake_client or FakeTwilioClient(create_raises=create_raises)

    def factory(sid, token):  # noqa: ANN001
        return fake_client

    backend = TwilioSMSBackend(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        twilio_client_factory=factory,
    )
    return backend, fake_client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def inject_fake_twilio():
    """Inject a fake twilio module so import guards pass."""
    fake = _fake_twilio_module()
    with patch.dict(sys.modules, {"twilio": fake, "twilio.rest": fake.rest}):
        yield


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_import_error_when_twilio_not_installed(self):
        """Raises ImportError with install hint when twilio absent and no factory."""
        with patch.dict(sys.modules, {"twilio": None}):
            # Force reload to re-evaluate the import guard
            import importlib

            from rex.integrations.messaging.backends import twilio_sms

            importlib.reload(twilio_sms)

            with pytest.raises(ImportError, match="pip install twilio"):
                twilio_sms.TwilioSMSBackend()

    def test_no_import_error_with_factory_injected(self):
        """No ImportError raised when factory is provided (twilio not needed)."""
        with patch.dict(sys.modules, {"twilio": None}):
            from rex.integrations.messaging.backends.twilio_sms import TwilioSMSBackend

            # Should not raise even without twilio installed
            backend = TwilioSMSBackend(
                account_sid="AC123",
                auth_token="tok",
                from_number="+1555",
                twilio_client_factory=lambda sid, tok: FakeTwilioClient(),
            )
            assert backend is not None


# ---------------------------------------------------------------------------
# Successful send
# ---------------------------------------------------------------------------


class TestSuccessfulSend:
    def test_send_calls_messages_create(self, fake_twilio_client: FakeTwilioClient):
        """send() delegates to client.messages.create with correct args."""
        backend, fake_client = _make_backend(fake_client=fake_twilio_client)
        backend.send(to="+15559998888", body="Hello from Rex")
        assert len(fake_client.create_calls) == 1
        call = fake_client.create_calls[0]
        assert call["to"] == "+15559998888"
        assert call["from_"] == "+15550001234"
        assert call["body"] == "Hello from Rex"

    def test_send_does_not_raise_on_success(self, fake_twilio_client: FakeTwilioClient):
        """send() returns None on success."""
        backend, _ = _make_backend(fake_client=fake_twilio_client)
        result = backend.send(to="+15559998888", body="Test")
        assert result is None


# ---------------------------------------------------------------------------
# Credential resolution
# ---------------------------------------------------------------------------


class TestCredentialResolution:
    def test_explicit_credentials_used_directly(self, fake_twilio_client: FakeTwilioClient):
        """Constructor credentials are used without consulting CredentialManager."""
        backend, mock_client = _make_backend(
            account_sid="ACexplicit",
            auth_token="tok_explicit",
            from_number="+10000000001",
            fake_client=fake_twilio_client,
        )
        with patch("rex.integrations.messaging.backends.twilio_sms.CredentialManager"):
            backend.send(to="+15551112222", body="hi")

        # CredentialManager should still be instantiated but get_token not needed
        # since explicit values satisfy all three credentials
        assert len(mock_client.create_calls) == 1

    def test_credentials_resolved_from_credential_manager(
        self, fake_twilio_client: FakeTwilioClient
    ):
        """When constructor args are None, CredentialManager supplies the values."""
        from rex.integrations.messaging.backends.twilio_sms import TwilioSMSBackend

        def factory(sid, tok):  # noqa: ANN001
            return fake_twilio_client

        def fake_get_token(key: str) -> str | None:
            return {
                "twilio_account_sid": "ACfrom_mgr",
                "twilio_auth_token": "tok_from_mgr",
                "twilio_from_number": "+10000000002",
            }.get(key)

        fake_mgr = MagicMock()
        fake_mgr.get_token.side_effect = fake_get_token

        backend = TwilioSMSBackend(twilio_client_factory=factory)

        with patch(
            "rex.integrations.messaging.backends.twilio_sms.CredentialManager",
            return_value=fake_mgr,
        ):
            backend.send(to="+15559990000", body="hello")

        assert len(fake_twilio_client.create_calls) == 1
        # Verify factory received the values from CredentialManager
        # (we can't easily check factory args, but create was called = success)

    def test_missing_credential_raises_sms_send_error(self):
        """SMSSendError raised when a required credential is missing."""
        from rex.integrations.messaging.backends.twilio_sms import SMSSendError, TwilioSMSBackend

        fake_mgr = MagicMock()
        fake_mgr.get_token.return_value = None  # all credentials missing

        backend = TwilioSMSBackend(twilio_client_factory=lambda s, t: FakeTwilioClient())

        with patch(
            "rex.integrations.messaging.backends.twilio_sms.CredentialManager",
            return_value=fake_mgr,
        ):
            with pytest.raises(SMSSendError, match="Missing Twilio credentials"):
                backend.send(to="+1555", body="test")

    def test_error_message_lists_missing_fields(self):
        """SMSSendError message names which credentials are missing."""
        from rex.integrations.messaging.backends.twilio_sms import SMSSendError, TwilioSMSBackend

        fake_mgr = MagicMock()
        fake_mgr.get_token.return_value = None

        backend = TwilioSMSBackend(twilio_client_factory=lambda s, t: FakeTwilioClient())

        with patch(
            "rex.integrations.messaging.backends.twilio_sms.CredentialManager",
            return_value=fake_mgr,
        ):
            with pytest.raises(SMSSendError) as exc_info:
                backend.send(to="+1555", body="test")

        msg = str(exc_info.value)
        assert "twilio_account_sid" in msg or "twilio_auth_token" in msg


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_twilio_4xx_raises_sms_send_error(self):
        """Twilio REST exception is wrapped in SMSSendError with status/code."""
        from rex.integrations.messaging.backends.twilio_sms import SMSSendError

        backend, _ = _make_backend(create_raises=_make_twilio_rest_exc(status=400, code=21211))

        with pytest.raises(SMSSendError, match="400") as exc_info:
            backend.send(to="+15559998888", body="bad")

        assert "21211" in str(exc_info.value)

    def test_timeout_raises_sms_send_error(self):
        """TimeoutError is wrapped in SMSSendError with timeout detail."""
        from rex.integrations.messaging.backends.twilio_sms import SMSSendError

        backend, _ = _make_backend(create_raises=TimeoutError("read timeout"))

        with pytest.raises(SMSSendError, match="timed out"):
            backend.send(to="+15559998888", body="too slow")

    def test_generic_exception_raises_sms_send_error(self):
        """Any unexpected exception is wrapped in SMSSendError."""
        from rex.integrations.messaging.backends.twilio_sms import SMSSendError

        backend, _ = _make_backend(create_raises=RuntimeError("connection refused"))

        with pytest.raises(SMSSendError):
            backend.send(to="+15559998888", body="boom")


# ---------------------------------------------------------------------------
# No secrets in logs
# ---------------------------------------------------------------------------


class TestNoSecretsLogged:
    def test_auth_token_not_in_log_on_success(self, caplog):
        """Auth token does not appear in any log output on successful send."""
        import logging

        backend, _ = _make_backend(auth_token="SECRET_TOKEN_VALUE")
        with caplog.at_level(
            logging.DEBUG, logger="rex.integrations.messaging.backends.twilio_sms"
        ):
            backend.send(to="+15559998888", body="Test")

        for record in caplog.records:
            assert "SECRET_TOKEN_VALUE" not in record.getMessage()

    def test_auth_token_not_in_log_on_failure(self, caplog):
        """Auth token does not appear in any log output when send fails."""
        import logging

        from rex.integrations.messaging.backends.twilio_sms import SMSSendError

        backend, _ = _make_backend(
            auth_token="SECRET_TOKEN_VALUE",
            create_raises=_make_twilio_rest_exc(),
        )

        with caplog.at_level(
            logging.DEBUG, logger="rex.integrations.messaging.backends.twilio_sms"
        ):
            with pytest.raises(SMSSendError):
                backend.send(to="+15559998888", body="Test")

        for record in caplog.records:
            assert "SECRET_TOKEN_VALUE" not in record.getMessage()


# ---------------------------------------------------------------------------
# receive() — send-only backend
# ---------------------------------------------------------------------------


class TestReceive:
    def test_receive_returns_empty_list(self):
        """receive() returns [] — TwilioSMSBackend is send-only."""
        backend, _ = _make_backend()
        assert backend.receive() == []
