"""Tests for the language model abstraction."""

from __future__ import annotations
import pytest

from llm_client import LanguageModel, register_strategy


def test_language_model_rejects_empty_prompt():
    """Empty or whitespace prompts should raise a ValueError."""
    model = LanguageModel(model_name="sshleifer/tiny-gpt2")
    with pytest.raises(ValueError):
        model.generate("   ")


def test_language_model_falls_back_when_transformers_missing(monkeypatch):
    """Ensure fallback to echo backend when transformers are unavailable."""
    import rex.llm_client as impl

    monkeypatch.setattr(impl, "AutoTokenizer", None, raising=False)
    monkeypatch.setattr(impl, "AutoModelForCausalLM", None, raising=False)
    monkeypatch.setattr(impl, "hf_pipeline", None, raising=False)

    model = LanguageModel(model_name="fallback-model", backend="transformers")
    completion = model.generate("Prompt")

    assert "[fallback-model]" in completion  # EchoStrategy should be used


def test_language_model_custom_strategy():
    """Test injecting a custom backend via register_strategy."""

    class DummyStrategy:
        name = "dummy"

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate(self, prompt: str, config):
            return f"dummy::{prompt}::{config.max_new_tokens}"

    register_strategy("dummy", DummyStrategy)

    model = LanguageModel(model_name="xyz", backend="dummy", max_new_tokens=5)
    result = model.generate("test")
    assert result == "dummy::test::5"
