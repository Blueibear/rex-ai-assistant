"""US-014: Anthropic provider acceptance tests.

Covers:
- provider initializes
- prompt execution works
- response returned
- Typecheck passes
"""

from __future__ import annotations

import types

import pytest

from rex.assistant_errors import ConfigurationError
from config import AppConfig
from llm_client import LanguageModel


def _make_fake_anthropic_client(text: str = "hello from claude") -> types.SimpleNamespace:
    """Return a fake Anthropic client whose messages.create returns *text*."""

    class _Content:
        def __init__(self, t: str) -> None:
            self.text = t

    class _Response:
        def __init__(self) -> None:
            self.content = [_Content(text)]

    fake = types.SimpleNamespace()
    fake.messages = types.SimpleNamespace(create=lambda **_: _Response())
    return fake


# ---------------------------------------------------------------------------
# Criterion 1: provider initializes
# ---------------------------------------------------------------------------


def test_anthropic_provider_initializes(monkeypatch):
    """LanguageModel with provider=anthropic should initialize without error."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        LanguageModel, "_ensure_anthropic_client", lambda self: _make_fake_anthropic_client()
    )

    cfg = AppConfig(
        llm_model=None,
        llm_provider="anthropic",
        anthropic_model="claude-test",
    )
    model = LanguageModel(cfg)

    assert model.provider == "anthropic"
    assert model.model_name == "claude-test"


# ---------------------------------------------------------------------------
# Criterion 2: prompt execution works
# ---------------------------------------------------------------------------


def test_anthropic_provider_executes_prompt(monkeypatch):
    """generate() with a prompt should call the Anthropic API."""
    calls: list[dict] = []
    fake_client = _make_fake_anthropic_client("response")

    original_create = fake_client.messages.create

    def _tracking_create(**kwargs):
        calls.append(kwargs)
        return original_create(**kwargs)

    fake_client.messages.create = _tracking_create

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(LanguageModel, "_ensure_anthropic_client", lambda self: fake_client)

    cfg = AppConfig(llm_model=None, llm_provider="anthropic", anthropic_model="claude-test")
    model = LanguageModel(cfg)
    model.generate("Tell me something")

    assert len(calls) == 1
    assert calls[0]["model"] == "claude-test"


# ---------------------------------------------------------------------------
# Criterion 3: response returned
# ---------------------------------------------------------------------------


def test_anthropic_provider_returns_response(monkeypatch):
    """generate() should return the model's text content."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        LanguageModel,
        "_ensure_anthropic_client",
        lambda self: _make_fake_anthropic_client("Great answer!"),
    )

    cfg = AppConfig(llm_model=None, llm_provider="anthropic", anthropic_model="claude-test")
    model = LanguageModel(cfg)
    result = model.generate(messages=[{"role": "user", "content": "question"}])

    assert result == "Great answer!"


def test_anthropic_provider_handles_system_messages(monkeypatch):
    """System messages should be extracted and passed separately to the API."""
    received: list[dict] = []
    fake_client = _make_fake_anthropic_client("ok")

    def _capturing_create(**kwargs):
        received.append(kwargs)

        class _R:
            class _C:
                text = "ok"

            content = [_C()]

        return _R()

    fake_client.messages.create = _capturing_create

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(LanguageModel, "_ensure_anthropic_client", lambda self: fake_client)

    cfg = AppConfig(llm_model=None, llm_provider="anthropic", anthropic_model="claude-test")
    model = LanguageModel(cfg)
    model.generate(
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
    )

    assert len(received) == 1
    call = received[0]
    # System message should be passed as "system" kwarg, not inside messages list
    assert call.get("system") == "You are helpful."
    assert all(m["role"] != "system" for m in call["messages"])


# ---------------------------------------------------------------------------
# Missing API key raises ConfigurationError
# ---------------------------------------------------------------------------


def test_anthropic_missing_api_key_raises_configuration_error(monkeypatch):
    """ConfigurationError should be raised when the Anthropic API key is absent."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    cfg = AppConfig(llm_model=None, llm_provider="anthropic", anthropic_model="claude-test")
    model = LanguageModel(cfg)

    with pytest.raises(ConfigurationError, match="API key"):
        model._ensure_anthropic_client()
