"""Tests for the lightweight transformer wrapper."""

from __future__ import annotations

import types

import pytest

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


def test_language_model_accepts_messages():
    model = LanguageModel(
        model_name="sshleifer/tiny-gpt2",
        max_new_tokens=12,
        temperature=0.0,
    )

    messages = [
        {"role": "system", "content": "You are cheerful."},
        {"role": "user", "content": "Say hi"},
    ]
    completion = model.generate(messages=messages)

    assert isinstance(completion, str)
    assert completion.strip() != ""


def test_invalid_model_name_rejected():
    with pytest.raises(ValueError):
        LanguageModel(model_name="../../bad")


def test_openai_provider(monkeypatch):
    fake_client = types.SimpleNamespace()

    class _Response:
        def __init__(self):
            self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content="hello"))]

    fake_client.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **_: _Response()))

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setattr(LanguageModel, "_ensure_openai_client", lambda self: fake_client)

    model = LanguageModel(model_name="openai:gpt-test", provider="openai")

    completion = model.generate(messages=[{"role": "user", "content": "hello"}])

    assert completion == "hello"
