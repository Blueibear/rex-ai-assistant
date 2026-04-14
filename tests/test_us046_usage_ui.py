"""Tests for US-046: Cloud usage visibility in the dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# _is_cloud_model helper
# ---------------------------------------------------------------------------


def _reload_llm_usage():
    import importlib
    import rex.llm_usage as mod

    importlib.reload(mod)
    return mod


def test_is_cloud_model_openai():
    mod = _reload_llm_usage()
    assert mod._is_cloud_model("gpt-4o") is True
    assert mod._is_cloud_model("gpt-3.5-turbo") is True
    assert mod._is_cloud_model("text-davinci-003") is True
    assert mod._is_cloud_model("o1-preview") is True


def test_is_cloud_model_anthropic():
    mod = _reload_llm_usage()
    assert mod._is_cloud_model("claude-opus-4") is True
    assert mod._is_cloud_model("claude-sonnet-4") is True


def test_is_cloud_model_google():
    mod = _reload_llm_usage()
    assert mod._is_cloud_model("gemini-1.5-pro") is True


def test_is_cloud_model_local():
    mod = _reload_llm_usage()
    assert mod._is_cloud_model("llama3") is False
    assert mod._is_cloud_model("mistral") is False  # Ollama "mistral" (no dash suffix)
    assert mod._is_cloud_model("codellama") is False
    assert mod._is_cloud_model("phi3") is False
    assert mod._is_cloud_model("") is False


def test_is_cloud_model_mistral_cloud_vs_local():
    """'mistral-' prefix is cloud (Mistral API); bare 'mistral' is local Ollama."""
    mod = _reload_llm_usage()
    assert mod._is_cloud_model("mistral-large") is True  # cloud API
    assert mod._is_cloud_model("mistral") is False  # local Ollama


# ---------------------------------------------------------------------------
# usage_api_summary
# ---------------------------------------------------------------------------


def test_usage_api_summary_empty(tmp_path):
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        mod = _reload_llm_usage()
        result = mod.usage_api_summary()

    assert result["local"] == {"requests": 0, "tokens": 0}
    assert result["cloud"] == {"requests": 0, "tokens": 0}
    assert "by_period" in result
    for period in ("today", "week", "month"):
        assert result["by_period"][period]["local"] == {"requests": 0, "tokens": 0}
        assert result["by_period"][period]["cloud"] == {"requests": 0, "tokens": 0}


def test_usage_api_summary_local_only(tmp_path):
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        mod = _reload_llm_usage()
        mod.record_usage(model="llama3", prompt_tokens=10, completion_tokens=20)
        mod.record_usage(model="llama3", prompt_tokens=5, completion_tokens=10)
        result = mod.usage_api_summary()

    assert result["local"]["requests"] == 2
    assert result["local"]["tokens"] == 45  # (10+20) + (5+10)
    assert result["cloud"]["requests"] == 0
    assert result["cloud"]["tokens"] == 0


def test_usage_api_summary_cloud_only(tmp_path):
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        mod = _reload_llm_usage()
        mod.record_usage(model="gpt-4o", prompt_tokens=100, completion_tokens=50)
        result = mod.usage_api_summary()

    assert result["cloud"]["requests"] == 1
    assert result["cloud"]["tokens"] == 150
    assert result["local"]["requests"] == 0


def test_usage_api_summary_mixed(tmp_path):
    usage_file = tmp_path / "llm_usage.json"
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        mod = _reload_llm_usage()
        mod.record_usage(model="llama3", prompt_tokens=10, completion_tokens=20)
        mod.record_usage(model="gpt-4o", prompt_tokens=100, completion_tokens=200)
        result = mod.usage_api_summary()

    assert result["local"]["requests"] == 1
    assert result["local"]["tokens"] == 30
    assert result["cloud"]["requests"] == 1
    assert result["cloud"]["tokens"] == 300


def test_usage_api_summary_period_today(tmp_path):
    """Records with today's timestamp appear in the 'today' period bucket."""
    from datetime import UTC, datetime

    usage_file = tmp_path / "llm_usage.json"
    today_ts = datetime.now(UTC).isoformat()
    entry = json.dumps(
        {"model": "llama3", "prompt_tokens": 5, "completion_tokens": 10, "timestamp": today_ts}
    )
    usage_file.write_text(entry + "\n")

    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        mod = _reload_llm_usage()
        result = mod.usage_api_summary()

    assert result["by_period"]["today"]["local"]["requests"] == 1
    assert result["by_period"]["today"]["local"]["tokens"] == 15


def test_usage_api_summary_period_old_record(tmp_path):
    """Very old records do not appear in any period bucket."""
    usage_file = tmp_path / "llm_usage.json"
    old_ts = "2020-01-01T00:00:00+00:00"
    entry = json.dumps(
        {"model": "llama3", "prompt_tokens": 5, "completion_tokens": 10, "timestamp": old_ts}
    )
    usage_file.write_text(entry + "\n")

    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        mod = _reload_llm_usage()
        result = mod.usage_api_summary()

    # Old record still counts toward all-time totals
    assert result["local"]["requests"] == 1
    # But not in any period
    for period in ("today", "week", "month"):
        assert result["by_period"][period]["local"]["requests"] == 0


# ---------------------------------------------------------------------------
# Flask /api/usage endpoint
# ---------------------------------------------------------------------------


def _make_flask_client(tmp_path: Path):
    """Create a Flask test client with usage data in tmp_path."""
    usage_file = tmp_path / "data" / "llm_usage.json"
    usage_file.parent.mkdir(parents=True, exist_ok=True)

    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        import importlib
        import rex.llm_usage as mod

        importlib.reload(mod)
        mod.record_usage(model="llama3", prompt_tokens=10, completion_tokens=20)
        mod.record_usage(model="gpt-4o", prompt_tokens=50, completion_tokens=100)

    from rex.gui_app import _create_flask_app

    app = _create_flask_app(ui_enabled=False)
    app.config["TESTING"] = True
    return app.test_client(), usage_file


def test_api_usage_endpoint_returns_200(tmp_path):
    client, usage_file = _make_flask_client(tmp_path)
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        import importlib
        import rex.llm_usage as mod

        importlib.reload(mod)
        resp = client.get("/api/usage")

    assert resp.status_code == 200


def test_api_usage_endpoint_structure(tmp_path):
    client, usage_file = _make_flask_client(tmp_path)
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        import importlib
        import rex.llm_usage as mod

        importlib.reload(mod)
        resp = client.get("/api/usage")

    data = resp.get_json()
    assert "local" in data
    assert "cloud" in data
    assert "by_period" in data
    assert "today" in data["by_period"]
    assert "week" in data["by_period"]
    assert "month" in data["by_period"]

    for bucket in (data["local"], data["cloud"]):
        assert "requests" in bucket
        assert "tokens" in bucket


def test_api_usage_endpoint_local_cloud_counts(tmp_path):
    client, usage_file = _make_flask_client(tmp_path)
    with patch.dict(os.environ, {"REX_LLM_USAGE_PATH": str(usage_file)}):
        import importlib
        import rex.llm_usage as mod

        importlib.reload(mod)
        resp = client.get("/api/usage")

    data = resp.get_json()
    assert data["local"]["requests"] == 1
    assert data["local"]["tokens"] == 30  # 10+20
    assert data["cloud"]["requests"] == 1
    assert data["cloud"]["tokens"] == 150  # 50+100
