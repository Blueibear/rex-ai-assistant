"""Tests for the lightweight transformer wrapper."""

from __future__ import annotations

from llm_client import LanguageModel, TORCH_AVAILABLE, TRANSFORMERS_AVAILABLE


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

    if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
        assert completion.startswith("(offline)")
