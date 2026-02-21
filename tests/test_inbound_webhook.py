"""Tests for the Twilio inbound SMS webhook endpoint.

All tests are offline and deterministic:
- No real network calls.
- Uses Flask test client.
- Uses ``tmp_path`` for the inbound store (no tracked repo files).
"""

from __future__ import annotations

import pytest
from flask import Flask

from rex.messaging_backends.inbound_store import InboundSmsStore
from rex.messaging_backends.inbound_webhook import create_inbound_sms_blueprint
from rex.messaging_backends.twilio_signature import compute_twilio_signature

_AUTH_TOKEN = "test_auth_token_12345"


@pytest.fixture()
def inbound_store(tmp_path):
    """Create a temp-backed inbound store."""
    return InboundSmsStore(db_path=tmp_path / "inbound_test.db")


@pytest.fixture()
def raw_config():
    """Sample runtime config with two messaging accounts."""
    return {
        "messaging": {
            "backend": "twilio",
            "default_account_id": "personal",
            "accounts": [
                {
                    "id": "personal",
                    "from_number": "+15551111111",
                    "credential_ref": "twilio:personal",
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
def app(inbound_store, raw_config):
    """Create a Flask test app with the inbound webhook blueprint."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    bp = create_inbound_sms_blueprint(
        auth_token=_AUTH_TOKEN,
        inbound_store=inbound_store,
        raw_config=raw_config,
        signature_verification=True,
    )
    flask_app.register_blueprint(bp)
    return flask_app


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()


def _make_twilio_params(
    *,
    message_sid: str = "SM0001",
    from_number: str = "+15559999999",
    to_number: str = "+15551111111",
    body: str = "Hello from test",
) -> dict[str, str]:
    """Build the form params that Twilio sends for inbound SMS."""
    return {
        "MessageSid": message_sid,
        "From": from_number,
        "To": to_number,
        "Body": body,
    }


def _sign_request(url: str, params: dict[str, str]) -> str:
    """Compute a valid Twilio signature for the given URL and params."""
    return compute_twilio_signature(_AUTH_TOKEN, url, params)


class TestWebhookSignatureVerification:
    """Tests that signature verification works correctly."""

    def test_valid_signature_accepted(self, client, inbound_store) -> None:
        """A request with a valid signature is accepted."""
        params = _make_twilio_params()
        # Flask test client uses http://localhost by default
        url = "http://localhost/webhooks/twilio/sms"
        sig = _sign_request(url, params)

        resp = client.post(
            "/webhooks/twilio/sms",
            data=params,
            headers={"X-Twilio-Signature": sig},
        )
        assert resp.status_code == 200
        assert b"<Response/>" in resp.data
        assert inbound_store.count() == 1

    def test_invalid_signature_rejected(self, client, inbound_store) -> None:
        """A request with a bad signature returns 403."""
        params = _make_twilio_params()
        resp = client.post(
            "/webhooks/twilio/sms",
            data=params,
            headers={"X-Twilio-Signature": "bad_signature"},
        )
        assert resp.status_code == 403
        assert inbound_store.count() == 0

    def test_missing_signature_rejected(self, client, inbound_store) -> None:
        """A request without a signature header returns 403."""
        params = _make_twilio_params()
        resp = client.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 403
        assert inbound_store.count() == 0

    def test_signature_with_wrong_url_rejected(self, client, inbound_store) -> None:
        """A signature computed for a different URL is rejected."""
        params = _make_twilio_params()
        sig = _sign_request("https://attacker.example.com/hook", params)
        resp = client.post(
            "/webhooks/twilio/sms",
            data=params,
            headers={"X-Twilio-Signature": sig},
        )
        assert resp.status_code == 403
        assert inbound_store.count() == 0

    def test_signature_with_tampered_body_rejected(self, client, inbound_store) -> None:
        """A signature computed for different params is rejected."""
        params = _make_twilio_params(body="Original")
        url = "http://localhost/webhooks/twilio/sms"
        sig = _sign_request(url, params)

        # Tamper with body
        params["Body"] = "Tampered"
        resp = client.post(
            "/webhooks/twilio/sms",
            data=params,
            headers={"X-Twilio-Signature": sig},
        )
        assert resp.status_code == 403
        assert inbound_store.count() == 0


class TestWebhookWithoutSignatureVerification:
    """Tests with signature verification disabled (local dev mode)."""

    @pytest.fixture()
    def nosig_app(self, inbound_store, raw_config):
        """Flask app with signature verification disabled."""
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        bp = create_inbound_sms_blueprint(
            auth_token=_AUTH_TOKEN,
            inbound_store=inbound_store,
            raw_config=raw_config,
            signature_verification=False,
        )
        flask_app.register_blueprint(bp)
        return flask_app

    @pytest.fixture()
    def nosig_client(self, nosig_app):
        return nosig_app.test_client()

    def test_accepted_without_signature(self, nosig_client, inbound_store) -> None:
        """When sig verification is off, requests are accepted without signature."""
        params = _make_twilio_params()
        resp = nosig_client.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 200
        assert inbound_store.count() == 1


class TestWebhookAccountRouting:
    """Tests for routing inbound messages to the correct account."""

    def _post_with_sig(self, client, params):
        """Helper to post with valid signature."""
        url = "http://localhost/webhooks/twilio/sms"
        sig = _sign_request(url, params)
        return client.post(
            "/webhooks/twilio/sms",
            data=params,
            headers={"X-Twilio-Signature": sig},
        )

    def test_routes_to_personal_account(self, client, inbound_store) -> None:
        """Message to +15551111111 routes to 'personal' account."""
        params = _make_twilio_params(to_number="+15551111111")
        resp = self._post_with_sig(client, params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        assert records[0].account_id == "personal"
        assert records[0].routed is True

    def test_routes_to_business_account(self, client, inbound_store) -> None:
        """Message to +15552222222 routes to 'business' account."""
        params = _make_twilio_params(to_number="+15552222222")
        resp = self._post_with_sig(client, params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        assert records[0].account_id == "business"
        assert records[0].routed is True

    def test_unrouted_when_no_matching_account(self, client, inbound_store) -> None:
        """Message to unknown number is stored as unrouted."""
        params = _make_twilio_params(to_number="+15559999999")
        resp = self._post_with_sig(client, params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        assert records[0].account_id is None
        assert records[0].routed is False

    def test_message_fields_persisted(self, client, inbound_store) -> None:
        """All Twilio form fields are stored correctly."""
        params = _make_twilio_params(
            message_sid="SM_TEST_123",
            from_number="+14155551234",
            to_number="+15551111111",
            body="Integration test body",
        )
        resp = self._post_with_sig(client, params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        rec = records[0]
        assert rec.sid == "SM_TEST_123"
        assert rec.from_number == "+14155551234"
        assert rec.to_number == "+15551111111"
        assert rec.body == "Integration test body"


class TestWebhookNoAccountsConfigured:
    """Tests when no messaging accounts are configured."""

    @pytest.fixture()
    def empty_config_app(self, inbound_store):
        """Flask app with empty accounts list."""
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        bp = create_inbound_sms_blueprint(
            auth_token=_AUTH_TOKEN,
            inbound_store=inbound_store,
            raw_config={"messaging": {"accounts": []}},
            signature_verification=False,
        )
        flask_app.register_blueprint(bp)
        return flask_app

    @pytest.fixture()
    def empty_client(self, empty_config_app):
        return empty_config_app.test_client()

    def test_all_messages_unrouted(self, empty_client, inbound_store) -> None:
        """With no accounts configured, all messages are unrouted."""
        params = _make_twilio_params()
        resp = empty_client.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 200

        records = inbound_store.query_recent()
        assert len(records) == 1
        assert records[0].account_id is None
        assert records[0].routed is False


class TestWebhookTwiMLResponse:
    """Tests for the TwiML response format."""

    @pytest.fixture()
    def nosig_app(self, inbound_store, raw_config):
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        bp = create_inbound_sms_blueprint(
            auth_token=_AUTH_TOKEN,
            inbound_store=inbound_store,
            raw_config=raw_config,
            signature_verification=False,
        )
        flask_app.register_blueprint(bp)
        return flask_app

    @pytest.fixture()
    def nosig_client(self, nosig_app):
        return nosig_app.test_client()

    def test_response_is_twiml(self, nosig_client) -> None:
        """Response has text/xml content type and valid TwiML."""
        params = _make_twilio_params()
        resp = nosig_client.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 200
        assert resp.content_type.startswith("text/xml")
        assert b"<Response/>" in resp.data
