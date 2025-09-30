"""Tests for the language model abstraction."""

from __future__ import annotations

import pytest

from config import AppConfig
from llm_client import LanguageModel, register_strategy
import llm_client as impl  # For monkeypatch fallback test


def test_language_model_generates_text():
    """Test standard generation using a Hugging Face model."""
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


def test_language_model_rejects_empty_prompt():
    """Prompt must not be empty or whitespace-only."""
    cfg = AppConfig(llm_model="sshleifer/tiny-gpt2", llm_provider="transformers")
    model = LanguageModel(cfg)

    with pytest.raises(ValueError):
        model.generate("   ")


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


def test_language_model_falls_back_when_transformers_missing(monkeypatch):
    """Ensure fallback to echo mode if transformers dependencies are unavailable."""
    monkeypatch.setattr(impl, "AutoTokenizer", None, raising=False)
    monkeypatch.setattr(impl, "AutoModelForCausalLM", None, raising=False)
    monkeypatch.setattr(impl, "hf_pipeline", None, raising=False)
    monkeypatch.setattr(impl, "torch", None, raising=False)

    cfg = AppConfig(llm_provider="transformers", llm_model="xyz")
    model = LanguageModel(cfg)
    completion = model.generate("Prompt")

    assert "[xyz]" in completion  # Fallback to echo mode

