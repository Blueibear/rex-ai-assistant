"""Tests for inbound SMS webhook startup wiring and rate limiting.

All tests are offline and deterministic:
- No real network calls or Twilio credentials.
- Uses Flask test client.
- Uses ``tmp_path`` for the inbound store (no tracked repo files).
- Uses mock CredentialManager to control token availability.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from flask import Flask

from rex.credentials import CredentialManager
from rex.messaging_backends.inbound_store import InboundSmsStore
from rex.messaging_backends.inbound_webhook import create_inbound_sms_blueprint
from rex.messaging_backends.twilio_signature import compute_twilio_signature
from rex.messaging_backends.webhook_wiring import register_inbound_sms_webhook

_AUTH_TOKEN = "test_auth_token_wiring_12345"


def _make_raw_config(
    *,
    enabled: bool = True,
    auth_token_ref: str = "twilio:inbound",
    store_path: str | None = None,
    rate_limit: str = "120 per minute",
) -> dict:
    """Build a minimal runtime config for the inbound section."""
    return {
        "messaging": {
            "backend": "twilio",
            "default_account_id": "primary",
            "accounts": [
                {
                    "id": "primary",
                    "from_number": "+15551111111",
                    "credential_ref": "twilio:primary",
                },
            ],
            "inbound": {
                "enabled": enabled,
                "auth_token_ref": auth_token_ref,
                "store_path": store_path,
                "retention_days": 90,
                "rate_limit": rate_limit,
            },
        }
    }


def _make_cred_manager(token: str | None = _AUTH_TOKEN) -> CredentialManager:
    """Create a CredentialManager that returns *token* for any service."""
    mgr = CredentialManager(credential_mapping={}, config_path="/dev/null")
    if token is not None:
        mgr.set_token("twilio:inbound", token)
    return mgr


def _make_twilio_params(
    *,
    message_sid: str = "SM0001",
    from_number: str = "+15559999999",
    to_number: str = "+15551111111",
    body: str = "Hello wiring test",
) -> dict[str, str]:
    return {
        "MessageSid": message_sid,
        "From": from_number,
        "To": to_number,
        "Body": body,
    }


def _sign_request(url: str, params: dict[str, str]) -> str:
    return compute_twilio_signature(_AUTH_TOKEN, url, params)


# ---------------------------------------------------------------------------
# Tests: register_inbound_sms_webhook
# ---------------------------------------------------------------------------


class TestWebhookWiringEnabled:
    """When inbound is enabled and credentials are available."""

    def test_blueprint_registered(self, tmp_path: Path) -> None:
        """The webhook blueprint is registered and the route exists."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        config = _make_raw_config(store_path=str(tmp_path / "inbound.db"))
        cred_mgr = _make_cred_manager()

        result = register_inbound_sms_webhook(app, config, cred_mgr)

        assert result is True
        # Check that the route exists
        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/webhooks/twilio/sms" in rules

    def test_webhook_accepts_valid_request(self, tmp_path: Path) -> None:
        """A properly-signed POST is accepted and the message is stored."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        config = _make_raw_config(store_path=str(tmp_path / "inbound.db"))
        cred_mgr = _make_cred_manager()

        register_inbound_sms_webhook(app, config, cred_mgr)

        client = app.test_client()
        params = _make_twilio_params()
        url = "http://localhost/webhooks/twilio/sms"
        sig = _sign_request(url, params)

        resp = client.post(
            "/webhooks/twilio/sms",
            data=params,
            headers={"X-Twilio-Signature": sig},
        )
        assert resp.status_code == 200
        assert b"<Response/>" in resp.data

    def test_webhook_rejects_invalid_signature(self, tmp_path: Path) -> None:
        """A request with a bad signature returns 403."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        config = _make_raw_config(store_path=str(tmp_path / "inbound.db"))
        cred_mgr = _make_cred_manager()

        register_inbound_sms_webhook(app, config, cred_mgr)

        client = app.test_client()
        params = _make_twilio_params()

        resp = client.post(
            "/webhooks/twilio/sms",
            data=params,
            headers={"X-Twilio-Signature": "bad_signature"},
        )
        assert resp.status_code == 403


class TestWebhookWiringDisabled:
    """When inbound is disabled in config."""

    def test_returns_false(self, tmp_path: Path) -> None:
        """register_inbound_sms_webhook returns False."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        config = _make_raw_config(
            enabled=False,
            store_path=str(tmp_path / "inbound.db"),
        )
        cred_mgr = _make_cred_manager()

        result = register_inbound_sms_webhook(app, config, cred_mgr)
        assert result is False

    def test_route_not_registered(self, tmp_path: Path) -> None:
        """No webhook route is registered."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        config = _make_raw_config(
            enabled=False,
            store_path=str(tmp_path / "inbound.db"),
        )
        cred_mgr = _make_cred_manager()

        register_inbound_sms_webhook(app, config, cred_mgr)

        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/webhooks/twilio/sms" not in rules


class TestWebhookWiringMissingToken:
    """When inbound is enabled but the auth token is not available."""

    def test_returns_false(self, tmp_path: Path) -> None:
        """register_inbound_sms_webhook returns False without a token."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        config = _make_raw_config(store_path=str(tmp_path / "inbound.db"))
        cred_mgr = _make_cred_manager(token=None)

        result = register_inbound_sms_webhook(app, config, cred_mgr)
        assert result is False

    def test_route_not_registered(self, tmp_path: Path) -> None:
        """No webhook route is registered when token is missing."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        config = _make_raw_config(store_path=str(tmp_path / "inbound.db"))
        cred_mgr = _make_cred_manager(token=None)

        register_inbound_sms_webhook(app, config, cred_mgr)

        rules = [rule.rule for rule in app.url_map.iter_rules()]
        assert "/webhooks/twilio/sms" not in rules


class TestWebhookWiringDefaultConfig:
    """When no config is passed, loads from default location."""

    def test_no_config_loads_default(self) -> None:
        """With None config, the function loads from load_config()."""
        app = Flask(__name__)
        app.config["TESTING"] = True

        # Default config has inbound disabled, so this should return False
        with patch(
            "rex.messaging_backends.webhook_wiring.load_config",
            return_value=_make_raw_config(enabled=False),
        ):
            result = register_inbound_sms_webhook(app, None)

        assert result is False


# ---------------------------------------------------------------------------
# Tests: Rate limiting
# ---------------------------------------------------------------------------


class TestWebhookRateLimiting:
    """Rate limiting on the inbound webhook endpoint."""

    def test_rate_limit_triggers_429(self, tmp_path: Path) -> None:
        """Exceeding the rate limit returns HTTP 429."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        # Use a very low limit to trigger quickly in tests
        config = _make_raw_config(
            store_path=str(tmp_path / "inbound.db"),
            rate_limit="3 per minute",
        )
        cred_mgr = _make_cred_manager()

        register_inbound_sms_webhook(app, config, cred_mgr)

        client = app.test_client()

        hit_429 = False
        for i in range(10):
            params = _make_twilio_params(message_sid=f"SM{i:04d}")
            url = "http://localhost/webhooks/twilio/sms"
            sig = _sign_request(url, params)
            resp = client.post(
                "/webhooks/twilio/sms",
                data=params,
                headers={"X-Twilio-Signature": sig},
            )
            if resp.status_code == 429:
                hit_429 = True
                break

        assert hit_429, "Expected 429 response after exceeding rate limit"

    def test_within_limit_succeeds(self, tmp_path: Path) -> None:
        """Requests within the rate limit are accepted."""
        app = Flask(__name__)
        app.config["TESTING"] = True
        config = _make_raw_config(
            store_path=str(tmp_path / "inbound.db"),
            rate_limit="100 per minute",
        )
        cred_mgr = _make_cred_manager()

        register_inbound_sms_webhook(app, config, cred_mgr)

        client = app.test_client()

        for i in range(5):
            params = _make_twilio_params(message_sid=f"SM{i:04d}")
            url = "http://localhost/webhooks/twilio/sms"
            sig = _sign_request(url, params)
            resp = client.post(
                "/webhooks/twilio/sms",
                data=params,
                headers={"X-Twilio-Signature": sig},
            )
            assert resp.status_code == 200


class TestWebhookWithoutLimiter:
    """Webhook works when Flask-Limiter is not installed."""

    def test_no_limiter_still_works(self, tmp_path: Path) -> None:
        """Blueprint works correctly without Flask-Limiter."""
        store = InboundSmsStore(db_path=tmp_path / "inbound.db")
        bp = create_inbound_sms_blueprint(
            auth_token=_AUTH_TOKEN,
            inbound_store=store,
            raw_config=_make_raw_config(),
            signature_verification=False,
            limiter=None,
            rate_limit_string="10 per minute",
        )

        app = Flask(__name__)
        app.config["TESTING"] = True
        app.register_blueprint(bp)

        client = app.test_client()
        params = _make_twilio_params()

        resp = client.post("/webhooks/twilio/sms", data=params)
        assert resp.status_code == 200
        assert b"<Response/>" in resp.data


# ---------------------------------------------------------------------------
# Tests: Doctor check
# ---------------------------------------------------------------------------


class TestDoctorInboundCheck:
    """Tests for the doctor.py inbound webhook check."""

    def test_disabled_reports_pass(self) -> None:
        """When inbound is disabled, the check passes."""
        with patch(
            "rex.config_manager.load_config",
            return_value=_make_raw_config(enabled=False),
        ):
            from scripts.doctor import _check_inbound_webhook

            results = _check_inbound_webhook()

        assert len(results) == 1
        label, ok, msg = results[0]
        assert label == "Inbound SMS webhook"
        assert ok is True
        assert "disabled" in msg

    def test_enabled_with_token_reports_pass(self) -> None:
        """When inbound is enabled and token is available, the check passes."""
        cred_mgr = _make_cred_manager()

        with (
            patch(
                "rex.config_manager.load_config",
                return_value=_make_raw_config(enabled=True),
            ),
            patch(
                "rex.credentials.get_credential_manager",
                return_value=cred_mgr,
            ),
        ):
            from scripts.doctor import _check_inbound_webhook

            results = _check_inbound_webhook()

        assert len(results) == 1
        label, ok, msg = results[0]
        assert label == "Inbound SMS webhook"
        assert ok is True
        assert "enabled" in msg
        assert "resolved" in msg

    def test_enabled_without_token_reports_warn(self) -> None:
        """When inbound is enabled but token is missing, the check warns."""
        cred_mgr = _make_cred_manager(token=None)

        with (
            patch(
                "rex.config_manager.load_config",
                return_value=_make_raw_config(enabled=True),
            ),
            patch(
                "rex.credentials.get_credential_manager",
                return_value=cred_mgr,
            ),
        ):
            from scripts.doctor import _check_inbound_webhook

            results = _check_inbound_webhook()

        assert len(results) == 1
        label, ok, msg = results[0]
        assert label == "Inbound SMS webhook"
        assert ok is False
        assert "not found" in msg
