"""Unit tests for rex.openclaw.tool_server — US-007.

Tests cover:
- Successful tool invocation (time_now)
- Auth required (missing / invalid key → 401)
- Policy denial → 403
- Approval-required → 403
- Unknown tool → 404
- Invalid args (TypeError from handler) → 400
- Policy-gated tool (send_email) happy path
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from flask import Flask

from rex.openclaw.policy_adapter import PolicyAdapter
from rex.openclaw.tool_executor import ApprovalRequiredError, PolicyDeniedError
from rex.openclaw.tool_server import ToolServer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_API_KEY = "test-tool-api-key"  # pragma: allowlist secret


def _make_app(tools: dict | None = None, policy: PolicyAdapter | None = None) -> Flask:
    """Return a Flask test app with ToolServer registered."""
    app = Flask(__name__)
    app.config["TESTING"] = True

    if tools is None:
        # Minimal stub tools for unit testing
        tools = {
            "time_now": lambda location=None, context=None: {
                "local_time": "2026-03-24 10:00",
                "date": "2026-03-24",
                "timezone": "UTC",
            },
            "send_email": lambda to, subject, body, context=None: {
                "ok": True,
                "message_id": "msg-001",
                "error": None,
            },
        }

    if policy is None:
        policy = MagicMock(spec=PolicyAdapter)
        policy.guard.return_value = None  # allow all by default

    server = ToolServer(policy=policy, tools=tools)
    server.register_all(app)
    return app


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    # Disable rate limiting in tests
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    app = _make_app()
    return app.test_client()


@pytest.fixture()
def policy_mock():
    mock = MagicMock(spec=PolicyAdapter)
    mock.guard.return_value = None
    return mock


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_missing_api_key_returns_401(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    app = _make_app()
    c = app.test_client()
    resp = c.post("/rex/tools/time_now", json={"args": {}})
    assert resp.status_code == 401
    data = resp.get_json()
    assert data["error"]["code"] == "UNAUTHORIZED"


def test_invalid_api_key_returns_401(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    app = _make_app()
    c = app.test_client()
    resp = c.post(
        "/rex/tools/time_now",
        json={"args": {}},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401


def test_valid_bearer_token_accepted(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    app = _make_app()
    c = app.test_client()
    resp = c.post(
        "/rex/tools/time_now",
        json={"args": {}},
        headers={"Authorization": f"Bearer {_API_KEY}"},
    )
    assert resp.status_code == 200


def test_unconfigured_api_key_returns_401(monkeypatch):
    monkeypatch.delenv("REX_TOOL_API_KEY", raising=False)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    app = _make_app()
    c = app.test_client()
    resp = c.post(
        "/rex/tools/time_now",
        json={"args": {}},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# time_now success tests
# ---------------------------------------------------------------------------


def test_time_now_returns_success(client):
    resp = client.post(
        "/rex/tools/time_now",
        json={"args": {"location": "London"}},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert "local_time" in data["result"]


def test_time_now_empty_args_accepted(client):
    resp = client.post(
        "/rex/tools/time_now",
        json={},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"


def test_time_now_no_body_accepted(client):
    resp = client.post(
        "/rex/tools/time_now",
        headers={"X-API-Key": _API_KEY},
        content_type="application/json",
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------


def test_policy_denial_returns_403(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    policy = MagicMock(spec=PolicyAdapter)
    policy.guard.side_effect = PolicyDeniedError("send_email", "blocked by policy")
    app = _make_app(policy=policy)
    c = app.test_client()
    resp = c.post(
        "/rex/tools/send_email",
        json={"args": {"to": "x@example.com", "subject": "Hi", "body": "Hello"}},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 403
    data = resp.get_json()
    assert data["error"]["code"] == "FORBIDDEN"


def test_approval_required_returns_403(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    policy = MagicMock(spec=PolicyAdapter)
    policy.guard.side_effect = ApprovalRequiredError("send_email", "requires approval")
    app = _make_app(policy=policy)
    c = app.test_client()
    resp = c.post(
        "/rex/tools/send_email",
        json={"args": {"to": "x@example.com", "subject": "Hi", "body": "Hello"}},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 403
    data = resp.get_json()
    assert "Approval required" in data["error"]["message"]


def test_policy_is_called_for_each_tool(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    policy = MagicMock(spec=PolicyAdapter)
    policy.guard.return_value = None
    app = _make_app(policy=policy)
    c = app.test_client()
    c.post("/rex/tools/time_now", json={}, headers={"X-API-Key": _API_KEY})
    policy.guard.assert_called_once_with("time_now", metadata={})


# ---------------------------------------------------------------------------
# send_email happy path
# ---------------------------------------------------------------------------


def test_send_email_success(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    app = _make_app()
    c = app.test_client()
    resp = c.post(
        "/rex/tools/send_email",
        json={"args": {"to": "bob@example.com", "subject": "Test", "body": "Hi Bob"}},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["result"]["ok"] is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_unknown_tool_returns_404(client):
    resp = client.post(
        "/rex/tools/nonexistent_tool",
        json={"args": {}},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 404
    data = resp.get_json()
    assert data["error"]["code"] == "NOT_FOUND"


def test_invalid_args_type_returns_400(client):
    resp = client.post(
        "/rex/tools/time_now",
        json={"args": "not-a-dict"},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["code"] == "BAD_REQUEST"


def test_invalid_context_type_returns_400(client):
    resp = client.post(
        "/rex/tools/time_now",
        json={"args": {}, "context": "not-a-dict"},
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 400


def test_missing_required_arg_returns_400(monkeypatch):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    monkeypatch.setenv("REX_TOOL_RATE_LIMIT", "0")
    # send_email requires to, subject, body
    app = _make_app()
    c = app.test_client()
    resp = c.post(
        "/rex/tools/send_email",
        json={"args": {}},  # missing required args
        headers={"X-API-Key": _API_KEY},
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert data["error"]["code"] == "BAD_REQUEST"
    assert "Invalid arguments" in data["error"]["message"]


def test_register_all_logs_tools(monkeypatch, caplog):
    monkeypatch.setenv("REX_TOOL_API_KEY", _API_KEY)
    import logging

    with caplog.at_level(logging.INFO, logger="rex.openclaw.tool_server"):
        app = Flask(__name__)
        app.config["TESTING"] = True
        server = ToolServer(
            policy=MagicMock(spec=PolicyAdapter),
            tools={"time_now": lambda **kw: {}},
        )
        server.register_all(app)
    assert "ToolServer registered" in caplog.text
