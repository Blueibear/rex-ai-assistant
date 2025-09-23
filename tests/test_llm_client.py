"""Tests for the lightweight transformer wrapper."""

from __future__ import annotations

import pytest

import llm_client
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
    monkeypatch.setattr(llm_client, "AutoTokenizer", None)
    monkeypatch.setattr(llm_client, "AutoModelForCausalLM", None)
    monkeypatch.setattr(llm_client, "hf_pipeline", None)

    model = llm_client.LanguageModel(model_name="fallback-model")
    completion = model.generate("Prompt")

    assert "[fallback-model]" in completion


def test_language_model_rejects_empty_prompt():
    model = LanguageModel(model_name="sshleifer/tiny-gpt2")

    with pytest.raises(ValueError):
        model.generate("   ")
