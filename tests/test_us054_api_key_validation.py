"""Tests for US-054: API key validation.

Acceptance criteria:
- API keys validated on each request
- unauthorized rejected
- failures logged
- Typecheck passes
"""

from __future__ import annotations

import logging

import pytest

pytest.importorskip("flask")

from flask import Flask  # noqa: E402

from rex.api_key_auth import ApiKeyValidator, require_api_key, validate_api_key  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(env_var: str = "REX_API_KEY") -> Flask:
    """Build a minimal Flask app with one protected route."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    @app.route("/protected")
    @require_api_key(env_var=env_var, label="test")
    def protected():
        from flask import jsonify

        return jsonify({"ok": True})

    return app


# ---------------------------------------------------------------------------
# Criterion: validate_api_key — unit tests
# ---------------------------------------------------------------------------


class TestValidateApiKey:
    """Unit tests for the validate_api_key helper."""

    def test_matching_keys_accepted(self):
        """Correct key returns True."""
        assert validate_api_key("my-secret", "my-secret") is True

    def test_wrong_key_rejected(self):
        """Incorrect key returns False."""
        assert validate_api_key("wrong", "my-secret") is False

    def test_none_provided_rejected(self):
        """None provided key returns False."""
        assert validate_api_key(None, "my-secret") is False

    def test_empty_provided_rejected(self):
        """Empty string provided key returns False."""
        assert validate_api_key("", "my-secret") is False

    def test_none_expected_rejected(self):
        """None expected key returns False (no key configured)."""
        assert validate_api_key("my-secret", None) is False

    def test_both_none_rejected(self):
        """Both None returns False."""
        assert validate_api_key(None, None) is False

    def test_case_sensitive(self):
        """Comparison is case-sensitive."""
        assert validate_api_key("Secret", "secret") is False


# ---------------------------------------------------------------------------
# Criterion: API keys validated on each request
# ---------------------------------------------------------------------------


class TestApiKeyValidatedOnRequest:
    """Each request must be independently validated."""

    def test_valid_key_in_x_api_key_header_accepted(self, monkeypatch):
        """Request with correct X-API-Key header is accepted."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/protected", headers={"X-API-Key": "correct-key"})
        assert resp.status_code == 200

    def test_valid_key_in_authorization_header_accepted(self, monkeypatch):
        """Request with Bearer token in Authorization header is accepted."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/protected", headers={"Authorization": "Bearer correct-key"})
        assert resp.status_code == 200

    def test_valid_token_header_accepted(self, monkeypatch):
        """Request with Token prefix in Authorization header is accepted."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/protected", headers={"Authorization": "Token correct-key"})
        assert resp.status_code == 200

    def test_each_request_validated_independently(self, monkeypatch):
        """Two consecutive requests are each validated."""
        monkeypatch.setenv("REX_API_KEY", "my-key")
        app = _make_app()
        with app.test_client() as client:
            r1 = client.get("/protected", headers={"X-API-Key": "my-key"})
            r2 = client.get("/protected", headers={"X-API-Key": "my-key"})
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_custom_env_var_validated(self, monkeypatch):
        """require_api_key reads from the specified env var."""
        monkeypatch.setenv("REX_CUSTOM_KEY", "custom-secret")
        monkeypatch.delenv("REX_API_KEY", raising=False)
        app = _make_app(env_var="REX_CUSTOM_KEY")
        with app.test_client() as client:
            resp = client.get("/protected", headers={"X-API-Key": "custom-secret"})
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Criterion: unauthorized rejected
# ---------------------------------------------------------------------------


class TestUnauthorizedRejected:
    """Requests with missing or wrong keys must be rejected with 401."""

    def test_missing_key_returns_401(self, monkeypatch):
        """Request with no API key header returns 401."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/protected")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self, monkeypatch):
        """Request with incorrect key returns 401."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/protected", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_401_response_has_error_body(self, monkeypatch):
        """401 response includes a JSON error field."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/protected", headers={"X-API-Key": "bad"})
        data = resp.get_json()
        assert data is not None
        assert "error" in data

    def test_empty_key_returns_401(self, monkeypatch):
        """Request with empty X-API-Key value returns 401."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/protected", headers={"X-API-Key": ""})
        assert resp.status_code == 401

    def test_unconfigured_key_returns_401(self, monkeypatch):
        """No env var set → any key is rejected with 401."""
        monkeypatch.delenv("REX_API_KEY", raising=False)
        app = _make_app()
        with app.test_client() as client:
            resp = client.get("/protected", headers={"X-API-Key": "anything"})
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Criterion: failures logged
# ---------------------------------------------------------------------------


class TestFailuresLogged:
    """Failed authentication attempts must be logged at WARNING level."""

    def test_wrong_key_is_logged(self, monkeypatch, caplog):
        """Wrong key attempt produces a WARNING log entry."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with caplog.at_level(logging.WARNING, logger="rex.api_key_auth"):
            with app.test_client() as client:
                client.get("/protected", headers={"X-API-Key": "bad-key"})
        assert any(
            "unauthorized" in r.message.lower() or "invalid" in r.message.lower()
            for r in caplog.records
        )

    def test_missing_key_is_logged(self, monkeypatch, caplog):
        """Missing key attempt produces a WARNING log entry."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with caplog.at_level(logging.WARNING, logger="rex.api_key_auth"):
            with app.test_client() as client:
                client.get("/protected")
        assert len(caplog.records) > 0

    def test_unconfigured_env_is_logged(self, monkeypatch, caplog):
        """Missing env var produces a WARNING log entry."""
        monkeypatch.delenv("REX_API_KEY", raising=False)
        app = _make_app()
        with caplog.at_level(logging.WARNING, logger="rex.api_key_auth"):
            with app.test_client() as client:
                client.get("/protected", headers={"X-API-Key": "anything"})
        assert any(
            "unset" in r.message.lower() or "no api key" in r.message.lower()
            for r in caplog.records
        )

    def test_successful_request_not_logged_as_warning(self, monkeypatch, caplog):
        """Successful requests do not produce WARNING-level auth log entries."""
        monkeypatch.setenv("REX_API_KEY", "correct-key")
        app = _make_app()
        with caplog.at_level(logging.WARNING, logger="rex.api_key_auth"):
            with app.test_client() as client:
                client.get("/protected", headers={"X-API-Key": "correct-key"})
        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# ApiKeyValidator unit tests
# ---------------------------------------------------------------------------


class TestApiKeyValidator:
    """Unit tests for the ApiKeyValidator class."""

    def test_get_expected_key_reads_env(self, monkeypatch):
        """get_expected_key() returns value from environment."""
        monkeypatch.setenv("MY_KEY_VAR", "test-value")
        validator = ApiKeyValidator(env_var="MY_KEY_VAR")
        assert validator.get_expected_key() == "test-value"

    def test_get_expected_key_returns_none_when_unset(self, monkeypatch):
        """get_expected_key() returns None when env var is missing."""
        monkeypatch.delenv("MY_KEY_VAR", raising=False)
        validator = ApiKeyValidator(env_var="MY_KEY_VAR")
        assert validator.get_expected_key() is None

    def test_check_returns_true_for_valid_key(self, monkeypatch):
        """check() returns True for a matching key."""
        monkeypatch.setenv("REX_API_KEY", "valid-key")
        validator = ApiKeyValidator()
        assert validator.check("valid-key") is True

    def test_check_returns_false_for_invalid_key(self, monkeypatch):
        """check() returns False for a wrong key."""
        monkeypatch.setenv("REX_API_KEY", "valid-key")
        validator = ApiKeyValidator()
        assert validator.check("wrong-key") is False

    def test_check_returns_false_when_no_key_configured(self, monkeypatch):
        """check() returns False when no key is configured."""
        monkeypatch.delenv("REX_API_KEY", raising=False)
        validator = ApiKeyValidator()
        assert validator.check("anything") is False
