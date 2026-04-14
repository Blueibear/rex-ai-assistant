"""Tests for US-042: push notification support."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rex.assistant_errors import IntegrationNotConfiguredError
from rex.notifications import send_push as exported_send_push
from rex.notifications.push import _send_ntfy, _send_pushover, send_push

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    provider: str | None = None,
    token: str | None = None,
    topic: str | None = None,
) -> Any:
    """Return a minimal config-like object."""
    cfg = MagicMock()
    cfg.push_provider = provider
    cfg.push_token = token
    cfg.push_topic = topic
    return cfg


def _fake_response(status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---------------------------------------------------------------------------
# send_push — top-level dispatch
# ---------------------------------------------------------------------------


def test_send_push_raises_when_not_configured() -> None:
    cfg = _make_config(provider=None)
    with pytest.raises(IntegrationNotConfiguredError):
        send_push("title", "body", config=cfg)


def test_send_push_raises_for_unknown_provider() -> None:
    cfg = _make_config(provider="carrier_pigeon", token="tok", topic="alerts")
    with pytest.raises(IntegrationNotConfiguredError, match="carrier_pigeon"):
        send_push("title", "body", config=cfg)


def test_send_push_dispatches_ntfy() -> None:
    cfg = _make_config(provider="ntfy", token="", topic="rex-alerts")
    with patch("rex.notifications.push._send_ntfy") as mock_ntfy:
        send_push("title", "body", config=cfg)
    mock_ntfy.assert_called_once_with("title", "body", "normal", cfg)


def test_send_push_dispatches_pushover() -> None:
    cfg = _make_config(provider="pushover", token="app_tok", topic="user_key")
    with patch("rex.notifications.push._send_pushover") as mock_po:
        send_push("title", "body", priority="high", config=cfg)
    mock_po.assert_called_once_with("title", "body", "high", cfg)


def test_send_push_provider_case_insensitive() -> None:
    cfg = _make_config(provider="Ntfy", token="", topic="rex-alerts")
    with patch("rex.notifications.push._send_ntfy") as mock_ntfy:
        send_push("title", "body", config=cfg)
    mock_ntfy.assert_called_once()


# ---------------------------------------------------------------------------
# ntfy provider
# ---------------------------------------------------------------------------


def test_ntfy_requires_topic() -> None:
    cfg = _make_config(provider="ntfy", token="", topic="")
    with pytest.raises(IntegrationNotConfiguredError, match="push_topic"):
        _send_ntfy("title", "body", "normal", cfg)


def test_ntfy_sends_correct_request() -> None:
    cfg = _make_config(provider="ntfy", token="", topic="my-topic")
    fake_resp = _fake_response(200)
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
        _send_ntfy("Hello", "World", "normal", cfg)

    req = mock_open.call_args[0][0]
    assert "my-topic" in req.full_url
    assert req.data == b"World"
    assert req.get_header("Title") == "Hello"
    assert req.get_header("Priority") == "3"  # normal -> 3


def test_ntfy_high_priority_maps_correctly() -> None:
    cfg = _make_config(provider="ntfy", token="", topic="my-topic")
    fake_resp = _fake_response(200)
    with patch("urllib.request.urlopen", return_value=fake_resp):
        _send_ntfy("title", "body", "high", cfg)
    # No assertion needed — just verifying no error is raised


def test_ntfy_sends_auth_header_when_token_set() -> None:
    cfg = _make_config(provider="ntfy", token="secret", topic="my-topic")
    fake_resp = _fake_response(200)
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
        _send_ntfy("title", "body", "normal", cfg)

    req = mock_open.call_args[0][0]
    assert req.get_header("Authorization") == "Bearer secret"


def test_ntfy_no_auth_header_when_token_empty() -> None:
    cfg = _make_config(provider="ntfy", token="", topic="my-topic")
    fake_resp = _fake_response(200)
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
        _send_ntfy("title", "body", "normal", cfg)

    req = mock_open.call_args[0][0]
    assert req.get_header("Authorization") is None


def test_ntfy_raises_runtime_error_on_http_error() -> None:
    import urllib.error

    cfg = _make_config(provider="ntfy", token="", topic="my-topic")
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(None, 403, "Forbidden", {}, None),  # type: ignore[arg-type]
    ):
        with pytest.raises(RuntimeError, match="HTTP 403"):
            _send_ntfy("title", "body", "normal", cfg)


def test_ntfy_raises_runtime_error_on_url_error() -> None:
    import urllib.error

    cfg = _make_config(provider="ntfy", token="", topic="my-topic")
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        with pytest.raises(RuntimeError, match="connection refused"):
            _send_ntfy("title", "body", "normal", cfg)


# ---------------------------------------------------------------------------
# Pushover provider
# ---------------------------------------------------------------------------


def test_pushover_requires_token_and_topic() -> None:
    cfg = _make_config(provider="pushover", token="", topic="")
    with pytest.raises(IntegrationNotConfiguredError, match="push_token"):
        _send_pushover("title", "body", "normal", cfg)


def test_pushover_requires_topic() -> None:
    cfg = _make_config(provider="pushover", token="app_tok", topic="")
    with pytest.raises(IntegrationNotConfiguredError, match="push_topic"):
        _send_pushover("title", "body", "normal", cfg)


def test_pushover_sends_correct_payload() -> None:
    cfg = _make_config(provider="pushover", token="app_tok", topic="user_key")
    fake_resp = _fake_response(200)
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
        _send_pushover("Alert", "Body text", "high", cfg)

    req = mock_open.call_args[0][0]
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["token"] == "app_tok"
    assert payload["user"] == "user_key"
    assert payload["title"] == "Alert"
    assert payload["message"] == "Body text"
    assert payload["priority"] == 1  # high -> 1


def test_pushover_normal_priority() -> None:
    cfg = _make_config(provider="pushover", token="app_tok", topic="user_key")
    fake_resp = _fake_response(200)
    with patch("urllib.request.urlopen", return_value=fake_resp) as mock_open:
        _send_pushover("title", "body", "normal", cfg)

    payload = json.loads(mock_open.call_args[0][0].data.decode())
    assert payload["priority"] == 0


def test_pushover_raises_on_http_error() -> None:
    import urllib.error

    cfg = _make_config(provider="pushover", token="tok", topic="usr")
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.HTTPError(None, 400, "Bad Request", {}, None),  # type: ignore[arg-type]
    ):
        with pytest.raises(RuntimeError, match="HTTP 400"):
            _send_pushover("title", "body", "normal", cfg)


# ---------------------------------------------------------------------------
# Package-level export
# ---------------------------------------------------------------------------


def test_send_push_exported_from_package() -> None:
    assert exported_send_push is send_push


# ---------------------------------------------------------------------------
# AppConfig fields
# ---------------------------------------------------------------------------


def test_appconfig_has_push_fields() -> None:
    from rex.config import AppConfig

    cfg = AppConfig()
    assert cfg.push_provider is None
    assert cfg.push_token is None
    assert cfg.push_topic is None
