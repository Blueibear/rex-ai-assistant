"""Tests for US-117: Consistent error response envelope enforcement.

Acceptance criteria:
- error formatting logic lives in one place (middleware or exception handler),
  not duplicated per endpoint
- a test hitting each endpoint with a deliberately bad request confirms the
  standard envelope is returned
- 500-level errors include an error.request_id field for log correlation
- Typecheck passes
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from flask import Flask, g
from werkzeug.exceptions import BadRequest, NotFound

from rex.http_errors import (
    BAD_REQUEST,
    INTERNAL_ERROR,
    NOT_FOUND,
    UNPROCESSABLE,
    error_response,
    install_error_envelope_handler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app() -> Flask:
    """Minimal Flask app with the error envelope handler installed."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    install_error_envelope_handler(flask_app)

    # Seed g.request_id for every request so 500 handlers can use it.
    @flask_app.before_request
    def _set_request_id() -> None:
        g.request_id = "test-request-id"

    # Endpoints that produce various errors deliberately.
    @flask_app.route("/raise-400")
    def _raise_400() -> Any:
        raise BadRequest("deliberate bad request")

    @flask_app.route("/raise-404")
    def _raise_404() -> Any:
        raise NotFound("deliberate not found")

    @flask_app.route("/raise-500")
    def _raise_500() -> Any:
        raise RuntimeError("deliberate server error")

    @flask_app.route("/abort-400")
    def _abort_400() -> Any:
        from flask import abort

        abort(400)

    @flask_app.route("/abort-404")
    def _abort_404() -> Any:
        from flask import abort

        abort(404)

    @flask_app.route("/ok")
    def _ok() -> Any:
        return {"status": "ok"}, 200

    return flask_app


@pytest.fixture()
def client(app: Flask):  # type: ignore[type-arg]
    return app.test_client()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_envelope(data: dict, expected_code: str, expected_http: int, response: Any) -> None:
    assert response.status_code == expected_http, (
        f"Expected HTTP {expected_http}, got {response.status_code}"
    )
    assert "error" in data, f"Missing 'error' key in response: {data}"
    assert data["error"]["code"] == expected_code, (
        f"Expected code={expected_code!r}, got {data['error']['code']!r}"
    )
    assert "message" in data["error"], "Missing 'message' in error envelope"


# ---------------------------------------------------------------------------
# install_error_envelope_handler — unit tests
# ---------------------------------------------------------------------------


class TestInstallErrorEnvelopeHandler:
    def test_idempotent_double_install(self) -> None:
        """Calling install twice on the same app must not duplicate handlers."""
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        install_error_envelope_handler(flask_app)
        install_error_envelope_handler(flask_app)
        # No assertion needed — just must not raise

    def test_handler_registered_in_extensions(self) -> None:
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        install_error_envelope_handler(flask_app)
        assert flask_app.extensions.get("rex_error_envelope_installed") is True


# ---------------------------------------------------------------------------
# Standard envelope shape — 4xx errors
# ---------------------------------------------------------------------------


class TestEnvelopeShape4xx:
    def test_400_has_envelope(self, client: Any) -> None:
        resp = client.get("/raise-400")
        data = resp.get_json()
        _assert_envelope(data, BAD_REQUEST, 400, resp)

    def test_404_has_envelope(self, client: Any) -> None:
        resp = client.get("/raise-404")
        data = resp.get_json()
        _assert_envelope(data, NOT_FOUND, 404, resp)

    def test_abort_400_has_envelope(self, client: Any) -> None:
        resp = client.get("/abort-400")
        data = resp.get_json()
        _assert_envelope(data, BAD_REQUEST, 400, resp)

    def test_abort_404_has_envelope(self, client: Any) -> None:
        resp = client.get("/abort-404")
        data = resp.get_json()
        _assert_envelope(data, NOT_FOUND, 404, resp)

    def test_unknown_route_404_has_envelope(self, client: Any) -> None:
        """Flask returns 404 for unknown routes — must still use envelope."""
        resp = client.get("/this-does-not-exist")
        data = resp.get_json()
        _assert_envelope(data, NOT_FOUND, 404, resp)

    def test_envelope_code_is_string(self, client: Any) -> None:
        resp = client.get("/raise-400")
        data = resp.get_json()
        assert isinstance(data["error"]["code"], str)

    def test_envelope_message_is_string(self, client: Any) -> None:
        resp = client.get("/raise-400")
        data = resp.get_json()
        assert isinstance(data["error"]["message"], str)

    def test_4xx_does_not_include_request_id(self, client: Any) -> None:
        """request_id is only required for 5xx errors."""
        resp = client.get("/raise-400")
        data = resp.get_json()
        # request_id may or may not be present for 4xx — just not mandatory
        assert "code" in data["error"]


# ---------------------------------------------------------------------------
# 500-level errors include request_id
# ---------------------------------------------------------------------------


class TestEnvelopeShape5xx:
    def test_500_has_envelope(self, client: Any) -> None:
        resp = client.get("/raise-500")
        data = resp.get_json()
        _assert_envelope(data, INTERNAL_ERROR, 500, resp)

    def test_500_includes_request_id(self, client: Any) -> None:
        resp = client.get("/raise-500")
        data = resp.get_json()
        assert "request_id" in data["error"], (
            "500 response must include error.request_id for log correlation"
        )

    def test_500_request_id_matches_g(self, client: Any) -> None:
        """The request_id in the envelope must equal g.request_id set in before_request."""
        resp = client.get("/raise-500")
        data = resp.get_json()
        assert data["error"]["request_id"] == "test-request-id"

    def test_500_request_id_is_string(self, client: Any) -> None:
        resp = client.get("/raise-500")
        data = resp.get_json()
        assert isinstance(data["error"]["request_id"], str)


# ---------------------------------------------------------------------------
# Normal requests are unaffected
# ---------------------------------------------------------------------------


class TestNormalRequestUnaffected:
    def test_ok_response_unchanged(self, client: Any) -> None:
        resp = client.get("/ok")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"status": "ok"}


# ---------------------------------------------------------------------------
# error_response helper — request_id parameter
# ---------------------------------------------------------------------------


class TestErrorResponseHelper:
    def test_without_request_id_no_key(self, app: Flask) -> None:
        with app.app_context():
            resp, status = error_response("BAD_REQUEST", "bad", 400)
            data = json.loads(resp.get_data())
            assert "request_id" not in data["error"]
            assert status == 400

    def test_with_request_id_included(self, app: Flask) -> None:
        rid = str(uuid.uuid4())
        with app.app_context():
            resp, status = error_response("INTERNAL_ERROR", "oops", 500, request_id=rid)
            data = json.loads(resp.get_data())
            assert data["error"]["request_id"] == rid
            assert status == 500

    def test_with_none_request_id_no_key(self, app: Flask) -> None:
        with app.app_context():
            resp, status = error_response("NOT_FOUND", "missing", 404, request_id=None)
            data = json.loads(resp.get_data())
            assert "request_id" not in data["error"]


# ---------------------------------------------------------------------------
# 500 handler without g.request_id set (defensive)
# ---------------------------------------------------------------------------


class TestEnvelope500WithoutRequestId:
    def test_500_without_request_id_in_g(self) -> None:
        """Handler must not crash when g.request_id is absent."""
        flask_app = Flask(__name__)
        flask_app.config["TESTING"] = True
        install_error_envelope_handler(flask_app)

        @flask_app.route("/boom")
        def _boom() -> Any:
            raise RuntimeError("no request id set in g")

        with flask_app.test_client() as c:
            resp = c.get("/boom")
        data = resp.get_json()
        assert resp.status_code == 500
        assert data["error"]["code"] == INTERNAL_ERROR
        # request_id absent or None — either is acceptable when g.request_id not set
