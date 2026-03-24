"""US-015: Local LLM provider acceptance tests.

Covers:
- local model reachable
- prompt execution works
- response returned
- Typecheck passes
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import pytest

from config import AppConfig
from llm_client import GenerationConfig, LanguageModel
from rex.assistant_errors import ConfigurationError
from rex.llm_client import OfflineTransformersStrategy, OllamaStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_gen_config() -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        seed=42,
    )


def _make_fake_ollama_module(response_content: str = "pong") -> types.ModuleType:
    """Return a fake `ollama` module with a Client class whose chat() returns *response_content*."""

    fake_response = {"message": {"content": response_content}}

    class FakeClient:
        def __init__(self, host: str = "") -> None:
            self.host = host

        def chat(self, model: str, messages: list, options: dict | None = None) -> dict:
            return fake_response

    fake_module = types.ModuleType("ollama")
    fake_module.Client = FakeClient  # type: ignore[attr-defined]
    return fake_module


# ---------------------------------------------------------------------------
# Criterion 1: local model reachable
# ---------------------------------------------------------------------------


def test_ollama_provider_initializes(monkeypatch):
    """OllamaStrategy should initialize when the ollama package is available."""
    fake_ollama = _make_fake_ollama_module()

    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        monkeypatch.setattr("rex.llm_client.OLLAMA_AVAILABLE", True)
        strategy = OllamaStrategy("llama3", base_url="http://localhost:11434")

    assert strategy.model_name == "llama3"
    assert strategy.base_url == "http://localhost:11434"


def test_ollama_provider_initializes_via_language_model(monkeypatch):
    """LanguageModel with provider=ollama should create an OllamaStrategy."""
    fake_ollama = _make_fake_ollama_module()

    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        monkeypatch.setattr("rex.llm_client.OLLAMA_AVAILABLE", True)
        cfg = AppConfig(llm_model="llama3", llm_provider="ollama")
        model = LanguageModel(cfg)

    assert model.provider == "ollama"
    assert isinstance(model.strategy, OllamaStrategy)


def test_offline_transformers_always_initializes():
    """OfflineTransformersStrategy requires no external dependencies."""
    strategy = OfflineTransformersStrategy("gpt2")
    assert strategy.model_name == "gpt2"


# ---------------------------------------------------------------------------
# Criterion 2: prompt execution works
# ---------------------------------------------------------------------------


def test_ollama_provider_executes_prompt(monkeypatch):
    """OllamaStrategy.generate() should call client.chat with the correct model."""
    calls: list[dict] = []
    fake_content = "Ollama says hi"

    class _TrackingClient:
        def __init__(self, host: str = "") -> None:
            pass

        def chat(self, model: str, messages: list, options: dict | None = None) -> dict:
            calls.append({"model": model, "messages": messages})
            return {"message": {"content": fake_content}}

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.Client = _TrackingClient  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        monkeypatch.setattr("rex.llm_client.OLLAMA_AVAILABLE", True)
        strategy = OllamaStrategy("llama3")

    strategy.generate("ping", _default_gen_config())

    assert len(calls) == 1
    assert calls[0]["model"] == "llama3"


def test_offline_transformers_executes_prompt():
    """OfflineTransformersStrategy.generate() should return a response string."""
    strategy = OfflineTransformersStrategy("gpt2")
    result = strategy.generate("hello", _default_gen_config())
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Criterion 3: response returned
# ---------------------------------------------------------------------------


def test_ollama_provider_returns_response(monkeypatch):
    """OllamaStrategy.generate() should return the content from Ollama."""
    fake_ollama = _make_fake_ollama_module("Local model is working")

    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        monkeypatch.setattr("rex.llm_client.OLLAMA_AVAILABLE", True)
        cfg = AppConfig(llm_model="llama3", llm_provider="ollama")
        model = LanguageModel(cfg)

    result = model.generate("Are you there?")
    assert result == "Local model is working"


def test_ollama_provider_returns_response_for_messages(monkeypatch):
    """OllamaStrategy.generate() should handle a messages list."""
    fake_ollama = _make_fake_ollama_module("Yes, I am here")

    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        monkeypatch.setattr("rex.llm_client.OLLAMA_AVAILABLE", True)
        cfg = AppConfig(llm_model="llama3", llm_provider="ollama")
        model = LanguageModel(cfg)

    result = model.generate(messages=[{"role": "user", "content": "Hello?"}])
    assert result == "Yes, I am here"


def test_offline_transformers_returns_non_empty_response():
    """OfflineTransformersStrategy.generate() should return a non-empty string."""
    strategy = OfflineTransformersStrategy("gpt2")
    result = strategy.generate("What time is it?", _default_gen_config())
    assert result.strip() != ""


def test_ollama_server_error_returns_friendly_message(monkeypatch):
    """OllamaStrategy should return a friendly message when the server is unreachable."""

    class _FailingClient:
        def __init__(self, host: str = "") -> None:
            pass

        def chat(self, model: str, messages: list, options: dict | None = None) -> dict:
            raise ConnectionError("Connection refused")

    fake_ollama = types.ModuleType("ollama")
    fake_ollama.Client = _FailingClient  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"ollama": fake_ollama}):
        monkeypatch.setattr("rex.llm_client.OLLAMA_AVAILABLE", True)
        strategy = OllamaStrategy("llama3")

    result = strategy.generate("ping", _default_gen_config())
    # Should return a friendly error string, not raise
    assert isinstance(result, str)
    assert "ollama" in result.lower() or "trouble" in result.lower()


def test_ollama_missing_package_raises_configuration_error(monkeypatch):
    """OllamaStrategy should raise ConfigurationError when ollama is not installed."""
    monkeypatch.setattr("rex.llm_client.OLLAMA_AVAILABLE", False)

    with pytest.raises(ConfigurationError, match="ollama"):
        OllamaStrategy("llama3")
