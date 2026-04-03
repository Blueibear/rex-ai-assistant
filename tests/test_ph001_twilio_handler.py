"""Tests for US-PH-001: Twilio inbound webhook handler."""

from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def twilio_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake Twilio credentials in the environment."""
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACfake_account_sid")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "fake_auth_token_1234")
    monkeypatch.setenv("TWILIO_PHONE_NUMBER", "+15550001234")


@pytest.fixture()
def app_client(twilio_env):
    """Flask test client with telephony blueprint registered."""
    from flask import Flask

    from rex.telephony.twilio_handler import create_blueprint

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.register_blueprint(create_blueprint())
    return flask_app.test_client()


# ---------------------------------------------------------------------------
# is_configured
# ---------------------------------------------------------------------------


def test_is_configured_returns_false_when_no_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)

    from rex.telephony.twilio_handler import is_configured

    assert not is_configured()


def test_is_configured_returns_true_when_creds_set(twilio_env) -> None:
    from rex.telephony.twilio_handler import is_configured

    assert is_configured()


def test_is_configured_false_if_only_partial_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "ACfake")
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)

    from rex.telephony.twilio_handler import is_configured

    assert not is_configured()


# ---------------------------------------------------------------------------
# Inbound call route
# ---------------------------------------------------------------------------


def test_inbound_call_returns_xml(app_client) -> None:
    """POST /telephony/inbound/call returns TwiML XML."""
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/call",
            data={"CallSid": "CAtest", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    assert b"<?xml" in resp.data or b"<Response" in resp.data


def test_inbound_call_contains_greeting(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/call",
            data={"CallSid": "CAtest", "From": "+15559990000"},
        )
    assert b"Rex" in resp.data or b"Hi" in resp.data


def test_inbound_call_content_type_is_xml(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/call",
            data={},
        )
    assert "xml" in resp.content_type


def test_inbound_call_rejects_bad_signature(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=False):
        resp = app_client.post("/telephony/inbound/call", data={})
    assert resp.status_code == 403


def test_inbound_call_503_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)

    from flask import Flask

    from rex.telephony.twilio_handler import create_blueprint

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.register_blueprint(create_blueprint())
    client = flask_app.test_client()

    resp = client.post("/telephony/inbound/call", data={})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Inbound SMS route
# ---------------------------------------------------------------------------


def test_inbound_sms_returns_xml(app_client) -> None:
    with (
        patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),
        patch("rex.telephony.twilio_handler._generate_sms_reply", return_value="Hi!"),
        patch("rex.telephony.twilio_handler._send_sms"),
    ):
        resp = app_client.post(
            "/telephony/inbound/sms",
            data={"Body": "Hello Rex", "From": "+15559990000"},
        )
    assert resp.status_code == 200
    assert "xml" in resp.content_type


def test_inbound_sms_calls_generate_reply(app_client) -> None:
    with (
        patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),
        patch("rex.telephony.twilio_handler._generate_sms_reply", return_value="Hi!") as mock_gen,
        patch("rex.telephony.twilio_handler._send_sms"),
    ):
        app_client.post(
            "/telephony/inbound/sms",
            data={"Body": "What time is it?", "From": "+15559990000"},
        )
    mock_gen.assert_called_once_with("What time is it?")


def test_inbound_sms_calls_send_sms(app_client) -> None:
    with (
        patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True),
        patch("rex.telephony.twilio_handler._generate_sms_reply", return_value="It is noon."),
        patch("rex.telephony.twilio_handler._send_sms") as mock_send,
    ):
        app_client.post(
            "/telephony/inbound/sms",
            data={"Body": "Hello", "From": "+15559990000"},
        )
    mock_send.assert_called_once()
    _args, kwargs = mock_send.call_args
    assert kwargs["to"] == "+15559990000"
    assert kwargs["body"] == "It is noon."


def test_inbound_sms_empty_body_returns_204(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=True):
        resp = app_client.post(
            "/telephony/inbound/sms",
            data={"Body": "", "From": "+15559990000"},
        )
    assert resp.status_code == 204


def test_inbound_sms_rejects_bad_signature(app_client) -> None:
    with patch("rex.telephony.twilio_handler._validate_twilio_signature", return_value=False):
        resp = app_client.post(
            "/telephony/inbound/sms",
            data={"Body": "hi", "From": "+15559990000"},
        )
    assert resp.status_code == 403


def test_inbound_sms_503_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("TWILIO_PHONE_NUMBER", raising=False)

    from flask import Flask

    from rex.telephony.twilio_handler import create_blueprint

    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.register_blueprint(create_blueprint())
    client = flask_app.test_client()

    resp = client.post("/telephony/inbound/sms", data={"Body": "hi"})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# _validate_twilio_signature
# ---------------------------------------------------------------------------


def test_validate_signature_returns_true_when_twilio_missing(twilio_env) -> None:
    """If twilio package not installed, validation skips and returns True."""
    from flask import Flask

    flask_app = Flask(__name__)

    with flask_app.test_request_context("/telephony/inbound/call", method="POST"):
        with patch.dict("sys.modules", {"twilio": None, "twilio.request_validator": None}):
            from rex.telephony import twilio_handler

            result = twilio_handler._validate_twilio_signature("fake_token")
    assert result is True
