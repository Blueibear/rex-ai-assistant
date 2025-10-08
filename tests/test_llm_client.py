"""Tests for the language model abstraction."""

from __future__ import annotations

import types
import pytest

from config import AppConfig
from llm_client import (
    LanguageModel,
    register_strategy,
    TORCH_AVAILABLE,
    TRANSFORMERS_AVAILABLE,
)


def test_language_model_generates_text():
    """Test standard generation using a Hugging Face model (tiny GPT-2)."""
    cfg = AppConfig(
        wakeword="rex",
        llm_model="sshleifer/tiny-gpt2",
        llm_provider="transformers",
        llm_max_tokens=12,
        llm_temperature=0.0,
    )
    model = LanguageModel(cfg)
    prompt = "User: Hello there!\nAssistant:"
    completion = model.generate(prompt)

    assert isinstance(completion, str)
    assert completion.strip() != ""

    if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
        assert completion.startswith("(offline)")


def test_language_model_accepts_messages():
    """Ensure the model can handle structured chat messages."""
    cfg = AppConfig(llm_model="sshleifer/tiny-gpt2", llm_provider="transformers")
    model = LanguageModel(cfg)

    messages = [
        {"role": "system", "content": "You are cheerful."},
        {"role": "user", "content": "Say hi"},
    ]
    completion = model.generate(messages=messages)

    assert isinstance(completion, str)
    assert completion.strip() != ""


def test_language_model_rejects_empty_prompt():
    """Empty or whitespace prompts should raise a ValueError."""
    cfg = AppConfig(llm_model="sshleifer/tiny-gpt2", llm_provider="transformers")
    model = LanguageModel(cfg)

    with pytest.raises(ValueError):
        model.generate("   ")


def test_invalid_model_name_rejected():
    """Reject path traversal attempts in model names."""
    with pytest.raises(ValueError):
        AppConfig(llm_model="../../bad_model")


def test_openai_provider(monkeypatch):
    """Simulate OpenAI provider and validate correct parsing of response."""
    fake_client = types.SimpleNamespace()

    class _Response:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="hello"))]

    fake_client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **_: _Response()))

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(LanguageModel, "_ensure_openai_client", lambda self: fake_client)

    cfg = AppConfig(llm_model="openai:gpt-test", llm_provider="openai")
    model = LanguageModel(cfg)

    completion = model.generate(messages=[{"role": "user", "content": "hello"}])
    assert completion == "hello"


def test_language_model_custom_strategy():
    """Test injecting a custom backend via register_strategy."""

    class DummyStrategy:
        name = "dummy"

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate(self, prompt: str, config):
            return f"dummy::{prompt}::{config.max_new_tokens}"

    register_strategy("dummy", DummyStrategy)

    cfg = AppConfig(llm_provider="dummy", llm_model="xyz", llm_max_tokens=5)
    model = LanguageModel(cfg)
    result = model.generate("test")

    assert result == "dummy::test::5"
