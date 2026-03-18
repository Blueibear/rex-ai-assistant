"""US-013: OpenAI provider acceptance tests.

Covers:
- provider initializes
- prompt execution works
- response returned
- failure handled gracefully
"""

from __future__ import annotations

import types

import pytest

from config import AppConfig
from llm_client import LanguageModel
from rex.assistant_errors import ConfigurationError


def _make_fake_client(content: str = "hello") -> types.SimpleNamespace:
    """Return a fake OpenAI client that responds with *content*."""

    class _Choice:
        def __init__(self) -> None:
            self.message = types.SimpleNamespace(content=content, tool_calls=None)

    class _Response:
        def __init__(self) -> None:
            self.choices = [_Choice()]

    fake = types.SimpleNamespace()
    fake.chat = types.SimpleNamespace(
        completions=types.SimpleNamespace(create=lambda **_: _Response())
    )
    return fake


# ---------------------------------------------------------------------------
# Criterion 1: provider initializes
# ---------------------------------------------------------------------------


def test_openai_provider_initializes(monkeypatch):
    """LanguageModel with provider=openai should initialize without error."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(LanguageModel, "_ensure_openai_client", lambda self: _make_fake_client())

    cfg = AppConfig(llm_model=None, llm_provider="openai", openai_model="gpt-test")
    model = LanguageModel(cfg)

    assert model.provider == "openai"
    assert model.model_name == "gpt-test"


# ---------------------------------------------------------------------------
# Criterion 2: prompt execution works
# ---------------------------------------------------------------------------


def test_openai_provider_executes_prompt(monkeypatch):
    """generate() with a simple prompt should call the OpenAI API."""
    calls: list[dict] = []
    fake_client = _make_fake_client("world")

    original_create = fake_client.chat.completions.create

    def _tracking_create(**kwargs):
        calls.append(kwargs)
        return original_create(**kwargs)

    fake_client.chat.completions.create = _tracking_create

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(LanguageModel, "_ensure_openai_client", lambda self: fake_client)

    cfg = AppConfig(llm_model=None, llm_provider="openai", openai_model="gpt-test")
    model = LanguageModel(cfg)
    model.generate("Tell me a joke")

    assert len(calls) == 1, "Expected exactly one API call"
    assert calls[0]["model"] == "gpt-test"


# ---------------------------------------------------------------------------
# Criterion 3: response returned
# ---------------------------------------------------------------------------


def test_openai_provider_returns_response(monkeypatch):
    """generate() should return the model's content string."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        LanguageModel, "_ensure_openai_client", lambda self: _make_fake_client("Great answer!")
    )

    cfg = AppConfig(llm_model=None, llm_provider="openai", openai_model="gpt-test")
    model = LanguageModel(cfg)
    result = model.generate(messages=[{"role": "user", "content": "question"}])

    assert result == "Great answer!"


def test_openai_provider_returns_response_for_prompt(monkeypatch):
    """generate() with a plain prompt string should also return content."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        LanguageModel, "_ensure_openai_client", lambda self: _make_fake_client("Sure thing")
    )

    cfg = AppConfig(llm_model=None, llm_provider="openai", openai_model="gpt-test")
    model = LanguageModel(cfg)
    result = model.generate("Do something")

    assert result == "Sure thing"


# ---------------------------------------------------------------------------
# Criterion 4: failure handled gracefully
# ---------------------------------------------------------------------------


def test_openai_missing_api_key_raises_configuration_error(monkeypatch):
    """ConfigurationError should be raised when the API key is absent."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    cfg = AppConfig(llm_model=None, llm_provider="openai", openai_model="gpt-test")
    model = LanguageModel(cfg)

    with pytest.raises(ConfigurationError, match="API key"):
        model._ensure_openai_client()


def test_openai_api_error_propagates_as_exception(monkeypatch):
    """If the OpenAI API raises, the exception should propagate (not crash silently)."""

    def _failing_create(**_):
        raise RuntimeError("API unavailable")

    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_failing_create))
    )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(LanguageModel, "_ensure_openai_client", lambda self: fake_client)

    cfg = AppConfig(llm_model=None, llm_provider="openai", openai_model="gpt-test")
    model = LanguageModel(cfg)

    with pytest.raises(RuntimeError, match="API unavailable"):
        model.generate("hello")
