"""Tests for US-045: cloud fallback when usage limit hit."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

from rex.model_router import ModelRouter

LOCAL = "llama3"
CLOUD = "gpt-4o"


def _router(**kwargs: object) -> ModelRouter:
    """Create a ModelRouter with Ollama availability stubbed out."""
    r = ModelRouter.__new__(ModelRouter)
    r._ollama_base_url = "http://localhost:11434"
    r._refresh_interval = 60
    r._available_ollama_models = {LOCAL, "llama3"}
    r._stop_event = __import__("threading").Event()
    r._refresh_thread = None
    r._cooldown_seconds = kwargs.get("cooldown_seconds", 3600)  # type: ignore[assignment]
    r._cloud_limited_until = 0.0
    r._notify_callback = kwargs.get("notify_callback", None)  # type: ignore[assignment]
    return r


# ---------------------------------------------------------------------------
# cloud_limit_hit
# ---------------------------------------------------------------------------


def test_cloud_limit_hit_429_activates_cooldown() -> None:
    r = _router()
    assert not r._cloud_in_cooldown()
    r.cloud_limit_hit(429)
    assert r._cloud_in_cooldown()


def test_cloud_limit_hit_402_activates_cooldown() -> None:
    r = _router()
    r.cloud_limit_hit(402)
    assert r._cloud_in_cooldown()


def test_cloud_limit_hit_other_status_ignored() -> None:
    r = _router()
    r.cloud_limit_hit(500)
    assert not r._cloud_in_cooldown()


def test_cloud_limit_hit_calls_notify_callback() -> None:
    callback = MagicMock()
    r = _router(notify_callback=callback)
    r.cloud_limit_hit(429)
    callback.assert_called_once()
    msg: str = callback.call_args[0][0]
    assert "Cloud limit reached" in msg
    assert "429" in msg


def test_cloud_limit_hit_no_callback_no_error() -> None:
    r = _router(notify_callback=None)
    r.cloud_limit_hit(429)  # should not raise


# ---------------------------------------------------------------------------
# _cloud_in_cooldown expiry
# ---------------------------------------------------------------------------


def test_cooldown_expires_after_period() -> None:
    r = _router(cooldown_seconds=1)
    r.cloud_limit_hit(429)
    assert r._cloud_in_cooldown()
    # Fast-forward by manipulating the internal deadline
    r._cloud_limited_until = time.monotonic() - 0.001
    assert not r._cloud_in_cooldown()


# ---------------------------------------------------------------------------
# route() — local_preferred mode with cooldown
# ---------------------------------------------------------------------------


def test_route_local_preferred_complex_uses_cloud_normally() -> None:
    r = _router()
    with patch.object(r, "_is_available", return_value=True):
        model = r.route(
            "analyze this",
            local_model=LOCAL,
            cloud_model=CLOUD,
            routing_mode="local_preferred",
            requires_tools=True,  # forces complex classification
        )
    assert model == CLOUD


def test_route_local_preferred_complex_falls_back_to_local_during_cooldown() -> None:
    r = _router()
    r.cloud_limit_hit(429)
    with patch.object(r, "_is_available", return_value=True):
        model = r.route(
            "analyze this",
            local_model=LOCAL,
            cloud_model=CLOUD,
            routing_mode="local_preferred",
            requires_tools=True,
        )
    assert model == LOCAL


def test_route_local_preferred_simple_uses_local_regardless_of_cooldown() -> None:
    r = _router()
    r.cloud_limit_hit(429)
    with patch.object(r, "_is_available", return_value=True):
        model = r.route(
            "hi",
            local_model=LOCAL,
            cloud_model=CLOUD,
            routing_mode="local_preferred",
        )
    assert model == LOCAL


# ---------------------------------------------------------------------------
# route() — cloud_only mode with cooldown
# ---------------------------------------------------------------------------


def test_route_cloud_only_returns_cloud_normally() -> None:
    r = _router()
    with patch.object(r, "_is_available", return_value=True):
        model = r.route(
            "hello",
            local_model=LOCAL,
            cloud_model=CLOUD,
            routing_mode="cloud_only",
        )
    assert model == CLOUD


def test_route_cloud_only_falls_back_to_local_during_cooldown() -> None:
    r = _router()
    r.cloud_limit_hit(429)
    with patch.object(r, "_is_available", return_value=True):
        model = r.route(
            "hello",
            local_model=LOCAL,
            cloud_model=CLOUD,
            routing_mode="cloud_only",
        )
    assert model == LOCAL


# ---------------------------------------------------------------------------
# route() — cloud resumes after cooldown
# ---------------------------------------------------------------------------


def test_route_resumes_cloud_after_cooldown_expires() -> None:
    r = _router()
    r.cloud_limit_hit(429)
    # Expire the cooldown manually
    r._cloud_limited_until = time.monotonic() - 0.001
    with patch.object(r, "_is_available", return_value=True):
        model = r.route(
            "analyze this",
            local_model=LOCAL,
            cloud_model=CLOUD,
            routing_mode="local_preferred",
            requires_tools=True,
        )
    assert model == CLOUD
