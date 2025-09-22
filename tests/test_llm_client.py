"""Tests for the language model abstraction layer."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from config import AppConfig
from llm_client import LanguageModel


def test_language_model_generates_text():
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
