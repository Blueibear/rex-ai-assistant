"""Tests for the language model abstraction."""

from __future__ import annotations

import pytest

import llm_client
import rex.llm_client as impl
from llm_client import LanguageModel


def test_language_model_generates_text():
    model = LanguageModel(
        model_name="sshleifer/tiny-gpt2",
        max_new_tokens=12,
        temperature=0.0,
    )
    prompt = "User: Hello there!\nAssistant:"
    completion = model.generate(prompt)

    assert isinstance(completion, str)
    assert completion.strip() != ""


def test_language_model_falls_back_when_transformers_missing(monkeypatch):
    monkeypatch.setattr(impl, "AutoTokenizer", None, raising=False)
    monkeypatch.setattr(impl, "AutoModelForCausalLM", None, raising=False)
    monkeypatch.setattr(impl, "hf_pipeline", None, raising=False)

    model = llm_client.LanguageModel(model_name="fallback-model")
    completion = model.generate("Prompt")

    assert "[fallback-model]" in completion


def test_language_model_respects_custom_backend(monkeypatch):
    class DummyStrategy:
        name = "dummy"

        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def generate(self, prompt: str, config):
            return f"dummy::{prompt}::{config.max_new_tokens}"

    llm_client.register_strategy("dummy", DummyStrategy)

    model = LanguageModel(model_name="whatever", backend="dummy", max_new_tokens=3)
    completion = model.generate("ping")

    assert completion == "dummy::ping::3"


def test_language_model_rejects_empty_prompt():
    model = LanguageModel(model_name="sshleifer/tiny-gpt2")

    with pytest.raises(ValueError):
        model.generate("   ")
