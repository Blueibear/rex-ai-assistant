"""Tests for messaging backend adapters (stub and Twilio)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.messaging_backends.account_config import (
    MessagingAccountConfig,
    MessagingConfig,
    load_messaging_config,
)
from rex.messaging_backends.base import InboundSms, SmsBackend, SmsSendResult
from rex.messaging_backends.factory import create_sms_backend
from rex.messaging_backends.stub import StubSmsBackend

# --- Base interface tests ---


def test_sms_backend_is_abstract():
    """SmsBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SmsBackend()  # type: ignore[abstract]


def test_sms_send_result_ok():
    """SmsSendResult represents a successful send."""
    result = SmsSendResult(ok=True, message_sid="SM123")
    assert result.ok is True
    assert result.message_sid == "SM123"
    assert result.error is None


def test_sms_send_result_error():
    """SmsSendResult represents a failed send."""
    result = SmsSendResult(ok=False, error="Network error")
    assert result.ok is False
    assert result.error == "Network error"


def test_inbound_sms_defaults():
    """InboundSms gets a UTC timestamp by default."""
    msg = InboundSms(sid="S1", from_number="+1111", to_number="+2222", body="Hi")
    assert isinstance(msg.received_at, datetime)
    assert msg.received_at.tzinfo is not None


# --- StubSmsBackend tests ---


@pytest.fixture
def stub_fixture(tmp_path: Path) -> Path:
    """Temporary fixture file for stub backend."""
    return tmp_path / "mock_sms.json"


@pytest.fixture
def stub_backend(stub_fixture: Path) -> StubSmsBackend:
    """A StubSmsBackend wired to a temp fixture."""
    return StubSmsBackend(fixture_path=stub_fixture)


def test_stub_send_sms(stub_backend: StubSmsBackend):
    """Stub send returns ok and writes to fixture."""
    result = stub_backend.send_sms(to="+15551234567", body="Hello")
    assert result.ok is True
    assert result.message_sid is not None
    assert result.message_sid.startswith("stub_")
    assert len(stub_backend.sent_messages) == 1


def test_stub_send_sms_custom_from(stub_backend: StubSmsBackend):
    """Stub send accepts a custom from_number."""
    result = stub_backend.send_sms(to="+15551234567", body="Hello", from_number="+15559999999")
    assert result.ok is True
    sent = stub_backend.sent_messages
    assert sent[0]["from"] == "+15559999999"


def test_stub_fetch_inbound_empty(stub_backend: StubSmsBackend):
    """Stub fetch returns empty when no inbound messages."""
    messages = stub_backend.fetch_recent_inbound()
    assert messages == []


def test_stub_inject_and_fetch_inbound(stub_backend: StubSmsBackend):
    """Stub can inject inbound messages and retrieve them."""
    stub_backend.inject_inbound(from_number="+15551111111", body="Test inbound")
    stub_backend.inject_inbound(from_number="+15552222222", body="Second inbound")

    messages = stub_backend.fetch_recent_inbound(limit=10)
    assert len(messages) == 2
    assert all(isinstance(m, InboundSms) for m in messages)
    # Newest first
    assert messages[0].body == "Second inbound"


def test_stub_fetch_inbound_limit(stub_backend: StubSmsBackend):
    """Stub fetch respects the limit parameter."""
    for i in range(5):
        stub_backend.inject_inbound(from_number=f"+1555000000{i}", body=f"Msg {i}")

    messages = stub_backend.fetch_recent_inbound(limit=3)
    assert len(messages) == 3


def test_stub_persistence(stub_fixture: Path):
    """Messages survive across backend instances."""
    backend1 = StubSmsBackend(fixture_path=stub_fixture)
    backend1.send_sms(to="+15551234567", body="Persistent msg")

    backend2 = StubSmsBackend(fixture_path=stub_fixture)
    assert len(backend2.sent_messages) == 1
    assert backend2.sent_messages[0]["body"] == "Persistent msg"


def test_stub_fixture_auto_created(tmp_path: Path):
    """Stub backend creates its fixture file if missing."""
    fixture = tmp_path / "sub" / "dir" / "sms.json"
    assert not fixture.exists()
    StubSmsBackend(fixture_path=fixture)
    assert fixture.exists()


# --- Account config tests ---


def test_messaging_config_defaults():
    """Default config is stub mode with no accounts."""
    config = MessagingConfig()
    assert config.backend == "stub"
    assert config.default_account_id is None
    assert config.accounts == []


def test_messaging_config_get_account_empty():
    """get_account returns None when no accounts exist."""
    config = MessagingConfig()
    assert config.get_account() is None


def test_messaging_config_get_account_by_id():
    """get_account resolves explicit account ID."""
    config = MessagingConfig(
        accounts=[
            MessagingAccountConfig(
                id="primary",
                from_number="+15551234567",
                credential_ref="twilio:primary",
            ),
            MessagingAccountConfig(
                id="secondary",
                from_number="+15559876543",
                credential_ref="twilio:secondary",
            ),
        ]
    )
    acct = config.get_account("secondary")
    assert acct is not None
    assert acct.id == "secondary"
    assert acct.from_number == "+15559876543"


def test_messaging_config_get_account_default():
    """get_account falls back to default_account_id."""
    config = MessagingConfig(
        default_account_id="primary",
        accounts=[
            MessagingAccountConfig(
                id="primary",
                from_number="+15551234567",
                credential_ref="twilio:primary",
            ),
        ],
    )
    acct = config.get_account()
    assert acct is not None
    assert acct.id == "primary"


def test_messaging_config_get_account_fallback():
    """get_account falls back to first account when default is unset."""
    config = MessagingConfig(
        accounts=[
            MessagingAccountConfig(
                id="first",
                from_number="+15551234567",
                credential_ref="twilio:first",
            ),
        ],
    )
    acct = config.get_account()
    assert acct is not None
    assert acct.id == "first"


def test_messaging_config_get_account_missing_id():
    """get_account returns None for unknown account ID."""
    config = MessagingConfig(
        accounts=[
            MessagingAccountConfig(
                id="primary",
                from_number="+15551234567",
                credential_ref="twilio:primary",
            ),
        ],
    )
    assert config.get_account("nonexistent") is None


def test_messaging_config_list_account_ids():
    """list_account_ids returns all configured account IDs."""
    config = MessagingConfig(
        accounts=[
            MessagingAccountConfig(
                id="a",
                from_number="+1111",
                credential_ref="twilio:a",
            ),
            MessagingAccountConfig(
                id="b",
                from_number="+2222",
                credential_ref="twilio:b",
            ),
        ],
    )
    assert config.list_account_ids() == ["a", "b"]


def test_load_messaging_config_missing_section():
    """load_messaging_config returns defaults when section is absent."""
    config = load_messaging_config({})
    assert config.backend == "stub"
    assert config.accounts == []


def test_load_messaging_config_valid():
    """load_messaging_config parses a full config."""
    raw = {
        "messaging": {
            "backend": "twilio",
            "default_account_id": "main",
            "accounts": [
                {
                    "id": "main",
                    "label": "Main Twilio",
                    "from_number": "+15551234567",
                    "credential_ref": "twilio:main",
                }
            ],
        }
    }
    config = load_messaging_config(raw)
    assert config.backend == "twilio"
    assert len(config.accounts) == 1
    assert config.accounts[0].label == "Main Twilio"


def test_messaging_account_config_extra_forbid():
    """MessagingAccountConfig rejects unknown fields."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        MessagingAccountConfig(
            id="x",
            from_number="+1",
            credential_ref="ref",
            unknown_field="bad",
        )


# --- Factory tests ---


def test_factory_no_config(tmp_path: Path):
    """Factory returns stub when no config is provided."""
    backend = create_sms_backend(None, fixture_path=tmp_path / "sms.json")
    assert isinstance(backend, StubSmsBackend)


def test_factory_stub_config(tmp_path: Path):
    """Factory returns stub when config says stub."""
    raw = {"messaging": {"backend": "stub"}}
    backend = create_sms_backend(raw, fixture_path=tmp_path / "sms.json")
    assert isinstance(backend, StubSmsBackend)


def test_factory_twilio_missing_credentials(tmp_path: Path):
    """Factory falls back to stub when Twilio credentials are missing."""
    raw = {
        "messaging": {
            "backend": "twilio",
            "accounts": [
                {
                    "id": "main",
                    "from_number": "+15551234567",
                    "credential_ref": "twilio:main",
                }
            ],
        }
    }
    # CredentialManager will have no token for twilio:main
    with patch("rex.messaging_backends.factory.get_credential_manager") as mock_cm:
        mock_cm.return_value = MagicMock()
        mock_cm.return_value.get_token.return_value = None
        backend = create_sms_backend(raw, fixture_path=tmp_path / "sms.json")
    assert isinstance(backend, StubSmsBackend)


def test_factory_twilio_valid_credentials(tmp_path: Path):
    """Factory returns TwilioSmsBackend when credentials are available."""
    raw = {
        "messaging": {
            "backend": "twilio",
            "accounts": [
                {
                    "id": "main",
                    "from_number": "+15551234567",
                    "credential_ref": "twilio:main",
                }
            ],
        }
    }
    with patch("rex.messaging_backends.factory.get_credential_manager") as mock_cm:
        mock_cm.return_value = MagicMock()
        mock_cm.return_value.get_token.return_value = "ACtest123:authtoken456"
        backend = create_sms_backend(raw, fixture_path=tmp_path / "sms.json")

    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    assert isinstance(backend, TwilioSmsBackend)


def test_factory_twilio_no_accounts(tmp_path: Path):
    """Factory falls back to stub when Twilio has no accounts."""
    raw = {"messaging": {"backend": "twilio"}}
    backend = create_sms_backend(raw, fixture_path=tmp_path / "sms.json")
    assert isinstance(backend, StubSmsBackend)


# --- TwilioSmsBackend request construction tests ---


def test_twilio_backend_send_request_construction():
    """Verify Twilio send constructs correct API request."""
    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    backend = TwilioSmsBackend(
        account_sid="AC_test_sid",
        auth_token="test_auth_token",
        default_from="+15551234567",
    )

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"sid": "SM_test_123"}

    with patch("rex.messaging_backends.twilio_backend.requests.post") as mock_post:
        mock_post.return_value = mock_response
        result = backend.send_sms(to="+15559876543", body="Test message")

    assert result.ok is True
    assert result.message_sid == "SM_test_123"

    # Verify the request was made correctly
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    assert "Messages.json" in call_args[0][0]
    assert call_args[1]["data"]["To"] == "+15559876543"
    assert call_args[1]["data"]["From"] == "+15551234567"
    assert call_args[1]["data"]["Body"] == "Test message"
    assert call_args[1]["auth"] == ("AC_test_sid", "test_auth_token")


def test_twilio_backend_send_custom_from():
    """Verify Twilio send allows custom from_number override."""
    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    backend = TwilioSmsBackend(
        account_sid="AC_test",
        auth_token="token",
        default_from="+15551234567",
    )

    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"sid": "SM_456"}

    with patch("rex.messaging_backends.twilio_backend.requests.post") as mock_post:
        mock_post.return_value = mock_response
        backend.send_sms(to="+15559876543", body="Hi", from_number="+15550000000")

    assert mock_post.call_args[1]["data"]["From"] == "+15550000000"


def test_twilio_backend_send_error_response():
    """Verify Twilio send handles error response."""
    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    backend = TwilioSmsBackend(
        account_sid="AC_test",
        auth_token="token",
        default_from="+15551234567",
    )

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.json.return_value = {
        "code": 21211,
        "message": "Invalid 'To' Phone Number",
    }
    mock_response.text = '{"code":21211}'

    with patch("rex.messaging_backends.twilio_backend.requests.post") as mock_post:
        mock_post.return_value = mock_response
        result = backend.send_sms(to="invalid", body="Test")

    assert result.ok is False
    assert "21211" in (result.error or "")


def test_twilio_backend_send_timeout():
    """Verify Twilio send handles timeout."""
    import requests as req_lib

    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    backend = TwilioSmsBackend(
        account_sid="AC_test",
        auth_token="token",
        default_from="+15551234567",
    )

    with patch(
        "rex.messaging_backends.twilio_backend.requests.post",
        side_effect=req_lib.Timeout("timeout"),
    ):
        result = backend.send_sms(to="+15559876543", body="Test")

    assert result.ok is False
    assert "timed out" in (result.error or "").lower()


def test_twilio_backend_send_connection_error():
    """Verify Twilio send handles connection errors."""
    import requests as req_lib

    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    backend = TwilioSmsBackend(
        account_sid="AC_test",
        auth_token="token",
        default_from="+15551234567",
    )

    with patch(
        "rex.messaging_backends.twilio_backend.requests.post",
        side_effect=req_lib.ConnectionError("refused"),
    ):
        result = backend.send_sms(to="+15559876543", body="Test")

    assert result.ok is False
    assert "connection" in (result.error or "").lower()


def test_twilio_backend_fetch_inbound():
    """Verify Twilio fetch constructs correct API request."""
    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    backend = TwilioSmsBackend(
        account_sid="AC_test",
        auth_token="token",
        default_from="+15551234567",
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "messages": [
            {
                "sid": "SM_in_1",
                "from": "+15559876543",
                "to": "+15551234567",
                "body": "Hello",
                "date_created": "Tue, 01 Jan 2025 12:00:00 +0000",
            }
        ]
    }

    with patch("rex.messaging_backends.twilio_backend.requests.get") as mock_get:
        mock_get.return_value = mock_response
        messages = backend.fetch_recent_inbound(limit=5)

    assert len(messages) == 1
    assert messages[0].sid == "SM_in_1"
    assert messages[0].from_number == "+15559876543"
    assert messages[0].body == "Hello"

    # Verify the request
    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert "Messages.json" in call_args[0][0]
    assert call_args[1]["params"]["To"] == "+15551234567"


def test_twilio_backend_fetch_inbound_error():
    """Verify Twilio fetch returns empty on error."""
    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    backend = TwilioSmsBackend(
        account_sid="AC_test",
        auth_token="token",
        default_from="+15551234567",
    )

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"code": 20003, "message": "Auth error"}
    mock_response.text = "Unauthorized"

    with patch("rex.messaging_backends.twilio_backend.requests.get") as mock_get:
        mock_get.return_value = mock_response
        messages = backend.fetch_recent_inbound()

    assert messages == []


def test_twilio_backend_missing_credentials():
    """TwilioSmsBackend raises ValueError on empty credentials."""
    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    with pytest.raises(ValueError, match="credentials are required"):
        TwilioSmsBackend(
            account_sid="",
            auth_token="",
            default_from="+15551234567",
        )


def test_create_twilio_from_credentials_valid():
    """create_twilio_backend_from_credentials works with valid cred format."""
    from rex.messaging_backends.twilio_backend import create_twilio_backend_from_credentials

    mock_cm = MagicMock()
    mock_cm.get_token.return_value = "ACsid123:authtoken456"

    backend = create_twilio_backend_from_credentials(
        credential_manager=mock_cm,
        credential_ref="twilio:main",
        from_number="+15551234567",
    )

    from rex.messaging_backends.twilio_backend import TwilioSmsBackend

    assert isinstance(backend, TwilioSmsBackend)


def test_create_twilio_from_credentials_missing():
    """create_twilio_backend_from_credentials raises on missing creds."""
    from rex.messaging_backends.twilio_backend import create_twilio_backend_from_credentials

    mock_cm = MagicMock()
    mock_cm.get_token.return_value = None

    with pytest.raises(ValueError, match="No Twilio credentials"):
        create_twilio_backend_from_credentials(
            credential_manager=mock_cm,
            credential_ref="twilio:main",
            from_number="+15551234567",
        )


def test_create_twilio_from_credentials_bad_format():
    """create_twilio_backend_from_credentials raises on bad format."""
    from rex.messaging_backends.twilio_backend import create_twilio_backend_from_credentials

    mock_cm = MagicMock()
    mock_cm.get_token.return_value = "no-colon-here"

    with pytest.raises(ValueError, match="account_sid:auth_token"):
        create_twilio_backend_from_credentials(
            credential_manager=mock_cm,
            credential_ref="twilio:main",
            from_number="+15551234567",
        )
