"""Tests for the language model abstraction."""

from __future__ import annotations

import pytest

from config import AppConfig

import llm_client
from llm_client import LanguageModel, register_strategy


def test_language_model_generates_text(monkeypatch):
    """Ensure the configured transformer backend is invoked."""

    class DummyStrategy:
        name = "transformers"

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate(self, prompt: str, config):
            return f"{self.model_name}:{config.max_new_tokens}:{prompt}"

    monkeypatch.setitem(llm_client._STRATEGIES, "transformers", DummyStrategy)

    cfg = AppConfig(
        llm_backend="transformers",
        llm_model="sshleifer/tiny-gpt2",
        llm_max_tokens=12,
        llm_temperature=0.0,
    )
    model = LanguageModel(cfg)
    prompt = "User: Hello there!\nAssistant:"
    completion = model.generate(prompt)

    assert completion == "sshleifer/tiny-gpt2:12:User: Hello there!\nAssistant:"


def test_language_model_falls_back_when_strategy_init_fails(monkeypatch):
    """If a backend fails to initialise, the echo fallback should be used."""

    class BrokenStrategy:
        name = "transformers"

        def __init__(self, model_name: str) -> None:
            raise RuntimeError("backend unavailable")

    monkeypatch.setitem(llm_client._STRATEGIES, "transformers", BrokenStrategy)

    cfg = AppConfig(llm_backend="transformers", llm_model="fallback-model")
    model = LanguageModel(cfg)
    completion = model.generate("Prompt")

    assert completion.startswith("[fallback-model]")


def test_language_model_custom_strategy(monkeypatch):
    """Custom providers registered at runtime should be honoured."""

    class DummyStrategy:
        name = "dummy"

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate(self, prompt: str, config):
            return f"dummy::{prompt}::{config.max_new_tokens}"

    monkeypatch.setitem(llm_client._STRATEGIES, "dummy", DummyStrategy)
    register_strategy("dummy", DummyStrategy)

    cfg = AppConfig(llm_backend="dummy", llm_model="xyz", llm_max_tokens=5)
    model = LanguageModel(cfg)
    result = model.generate("test")

    assert result == "dummy::test::5"


def test_language_model_echo_backend():
    """Echo backend should mirror prompts when no heavy dependencies exist."""

    cfg = AppConfig(llm_backend="echo", llm_model="unit-test", llm_max_tokens=7)
    model = LanguageModel(cfg)
    completion = model.generate("Hello Rex!")

    assert completion.startswith("[unit-test]")


def test_language_model_rejects_empty_prompt():
    """Empty or whitespace prompts should raise a ValueError."""

    cfg = AppConfig(llm_backend="echo", llm_model="stub")
    model = LanguageModel(cfg)

    with pytest.raises(ValueError):
        model.generate("   ")
