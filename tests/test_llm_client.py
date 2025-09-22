"""Tests for the language model abstraction layer."""

from __future__ import annotations

import pytest

from rex.llm_client import EchoBackend, LLMClient, registry


def test_llm_client_uses_echo_backend():
    client = LLMClient(model_name="demo-model", backend="echo")
    output = client.generate("Hello world")
    assert "demo-model" in output


def test_unknown_backend_falls_back_to_echo(monkeypatch):
    class BrokenBackend(EchoBackend):
        name = "broken"

        def __init__(self, model_name: str) -> None:
            raise RuntimeError("backend initialisation failed")

    registry.register("broken", BrokenBackend)
    client = LLMClient(model_name="demo-model", backend="broken")
    output = client.generate("prompt")
    assert output.startswith("[demo-model]")


def test_registry_rejects_unknown_backend():
    with pytest.raises(RuntimeError):
        registry.create("does-not-exist", "model")
