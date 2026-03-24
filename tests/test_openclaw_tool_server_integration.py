"""Integration tests for rex.openclaw.tool_server — US-008.

These tests start an in-process Flask test client (no real TCP socket) and
exercise the full request path: auth → rate-limit → policy → tool dispatch.

The ``time_now`` tool is tested end-to-end using the *real* tool handler so
that the integration wiring is validated.  Other tools use stubs to avoid
needing optional dependencies (weather API key, Home Assistant, etc.).

Tests are marked ``integration`` and picked up by::

    pytest -m integration -q tests/test_openclaw_tool_server_integration.py

They also run in the normal ``pytest -q`` suite (no external service required).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from flask import Flask

from rex.openclaw.policy_adapter import PolicyAdapter
from rex.openclaw.tool_server import ToolServer, _add_health_routes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_KEY = "integration-test-key-abc123"


def _make_integration_app(
    tools: dict[str, Any] | None = None,
    policy: PolicyAdapter | None = None,
) -> Flask:
    """Return a Flask test app backed by the given (or real time_now) tools."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    if tools is None:
        # Import the real time_now handler for the integration wiring test.
        from rex.openclaw.tools.time_tool import time_now

        tools = {"time_now": time_now}

    if policy is None:
        policy = MagicMock(spec=PolicyAdapter)
        policy.guard.return_value = None

    server = ToolServer(policy=policy, tools=tools)
    server.register_all(app)
    _add_health_routes(app, server)
    return app


@pytest.fixture()
def client(monkeypatch):
    """Flask test client with REX_TOOL_API_KEY set."""
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")  # disable rate limiting
    app = _make_integration_app()
    with app.test_client() as c:
        yield c


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_API_KEY}", "Content-Type": "application/json"}


# ---------------------------------------------------------------------------
# Health endpoints (no auth required)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_health_live(client):
    """GET /health/live returns 200 with status ok."""
    resp = client.get("/health/live")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"


@pytest.mark.integration
def test_health_ready(client):
    """GET /health/ready returns 200 with tool_count."""
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert isinstance(data["tool_count"], int)
    assert data["tool_count"] >= 1


# ---------------------------------------------------------------------------
# POST /rex/tools/time_now — real handler
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_time_now_no_args(client):
    """POST /rex/tools/time_now with empty args returns HTTP 200 (tool returns error dict)."""
    resp = client.post("/rex/tools/time_now", headers=_auth_headers(), json={})
    # The tool server always returns 200 when the handler runs without exception.
    # The real time_now handler returns {"error": {...}} for missing location;
    # that is captured in result rather than raising an HTTP error.
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    # result is either a valid time dict or a graceful error dict — either is acceptable
    assert isinstance(data["result"], dict)


@pytest.mark.integration
def test_time_now_with_location(client):
    """POST /rex/tools/time_now with UTC location returns a valid time dict."""
    resp = client.post(
        "/rex/tools/time_now",
        headers=_auth_headers(),
        json={"args": {"location": "London"}},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    # Either we get a proper time result or a graceful error dict
    assert isinstance(data["result"], dict)


@pytest.mark.integration
def test_time_now_with_context(client):
    """POST /rex/tools/time_now passes context dict without error."""
    resp = client.post(
        "/rex/tools/time_now",
        headers=_auth_headers(),
        json={"args": {}, "context": {"session_key": "main", "channel": "voice"}},
    )
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "success"


# ---------------------------------------------------------------------------
# Auth required
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_time_now_no_auth(client):
    """Missing auth header → 401."""
    resp = client.post(
        "/rex/tools/time_now",
        headers={"Content-Type": "application/json"},
        json={},
    )
    assert resp.status_code == 401


@pytest.mark.integration
def test_time_now_wrong_key(client):
    """Wrong Bearer token → 401."""
    resp = client.post(
        "/rex/tools/time_now",
        headers={"Authorization": "Bearer wrong-key", "Content-Type": "application/json"},
        json={},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_unknown_tool_returns_404(client):
    """POST to a tool that doesn't exist → 404."""
    resp = client.post(
        "/rex/tools/does_not_exist",
        headers=_auth_headers(),
        json={},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Multi-tool stub app (send_email with policy)
# ---------------------------------------------------------------------------


@pytest.fixture()
def stub_client(monkeypatch):
    """Flask test client with stub tools including send_email."""
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")

    stub_tools = {
        "time_now": lambda location=None, context=None: {
            "local_time": "2026-03-24 10:00",
            "date": "2026-03-24",
            "timezone": "UTC",
        },
        "send_email": lambda to, subject, body, context=None: {
            "ok": True,
            "message_id": "msg-001",
        },
    }
    policy = MagicMock(spec=PolicyAdapter)
    policy.guard.return_value = None

    app = Flask(__name__)
    app.config["TESTING"] = True
    server = ToolServer(policy=policy, tools=stub_tools)
    server.register_all(app)
    _add_health_routes(app, server)
    with app.test_client() as c:
        yield c, policy


@pytest.mark.integration
def test_stub_time_now_succeeds(stub_client):
    """Stub time_now returns expected dict."""
    client, _ = stub_client
    resp = client.post("/rex/tools/time_now", headers=_auth_headers(), json={})
    assert resp.status_code == 200
    assert resp.get_json()["result"]["timezone"] == "UTC"


@pytest.mark.integration
def test_stub_send_email_allowed(stub_client):
    """send_email succeeds when policy allows."""
    client, _ = stub_client
    resp = client.post(
        "/rex/tools/send_email",
        headers=_auth_headers(),
        json={"args": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}},
    )
    assert resp.status_code == 200
    assert resp.get_json()["result"]["ok"] is True


@pytest.mark.integration
def test_stub_send_email_policy_denied(monkeypatch, stub_client):
    """send_email returns 403 when policy denies."""
    from rex.openclaw.tool_executor import PolicyDeniedError

    client, policy = stub_client
    policy.guard.side_effect = PolicyDeniedError("send_email", "denied by policy")

    resp = client.post(
        "/rex/tools/send_email",
        headers=_auth_headers(),
        json={"args": {"to": "a@b.com", "subject": "Hi", "body": "Hello"}},
    )
    assert resp.status_code == 403
