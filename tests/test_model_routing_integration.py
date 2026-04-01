"""Integration tests for model routing wired into Assistant.generate_reply()."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock

import pytest

import rex.assistant as assistant_module
from rex.model_router import TaskCategory


# ---------------------------------------------------------------------------
# Minimal Settings stub
# ---------------------------------------------------------------------------


@dataclass
class _ModelRoutingStub:
    default: str = ""
    coding: str = ""
    reasoning: str = ""
    search: str = ""
    vision: str = ""
    fast: str = ""


@dataclass
class _SettingsStub:
    llm_provider: str = "transformers"
    llm_model: str = "stub-model"
    llm_max_tokens: int = 10
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_top_k: int = 50
    llm_seed: int = 42
    max_memory_items: int = 5
    transcripts_dir: str = "transcripts"
    persist_history: bool = False
    followups_enabled: bool = False
    ha_base_url: Optional[str] = None
    ha_token: Optional[str] = None
    user_id: str = "test"
    active_profile: str = "default"
    model_routing: _ModelRoutingStub = field(
        default_factory=lambda: _ModelRoutingStub(
            default="default-model",
            coding="coding-model",
        )
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CapturingLLM:
    """Minimal LLM stub that records the model_name at call time."""

    def __init__(self):
        self.model_name = "stub-model"
        self.calls: list[str] = []

    def generate(self, prompt, config=None):
        self.calls.append(self.model_name)
        return "stub reply"


def _make_assistant(monkeypatch, settings_obj, llm):
    """Build an Assistant with stubbed LanguageModel and settings."""

    class _LLMFactory:
        def __new__(cls, *args, **kwargs):
            return llm

    monkeypatch.setattr(assistant_module, "LanguageModel", _LLMFactory)
    return assistant_module.Assistant(
        settings_obj=settings_obj,
        transcripts_dir="transcripts",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_coding_message_uses_coding_model(monkeypatch, tmp_path):
    """A code-related message should route to the coding model."""
    settings = _SettingsStub(
        model_routing=_ModelRoutingStub(
            default="default-model",
            coding="coding-model",
        )
    )
    llm = _CapturingLLM()
    a = _make_assistant(monkeypatch, settings, llm)

    asyncio.run(a.generate_reply("Write a function to reverse a string"))

    assert llm.calls, "LLM was never called"
    assert llm.calls[0] == "coding-model", (
        f"Expected coding-model but LLM was called with model {llm.calls[0]!r}"
    )


def test_unconfigured_category_falls_back_to_default(monkeypatch, tmp_path):
    """A vision message with no vision model should fall back to routing default."""
    settings = _SettingsStub(
        model_routing=_ModelRoutingStub(
            default="default-model",
            coding="coding-model",
            # vision intentionally left empty
        )
    )
    llm = _CapturingLLM()
    a = _make_assistant(monkeypatch, settings, llm)

    asyncio.run(a.generate_reply("Describe this image for me"))

    assert llm.calls, "LLM was never called"
    assert llm.calls[0] == "default-model", (
        f"Expected default-model fallback but got {llm.calls[0]!r}"
    )


def test_model_restored_after_call(monkeypatch):
    """The LLM model_name is restored to its original value after generate_reply."""
    settings = _SettingsStub(
        model_routing=_ModelRoutingStub(
            default="default-model",
            coding="coding-model",
        )
    )
    llm = _CapturingLLM()
    original_model = llm.model_name
    a = _make_assistant(monkeypatch, settings, llm)

    asyncio.run(a.generate_reply("Write a function in Python"))

    assert llm.model_name == original_model, (
        "model_name was not restored after generate_reply"
    )


def test_no_routing_config_uses_existing_model(monkeypatch):
    """When model_routing is not configured, the LLM is called without override."""

    @dataclass
    class _NoRoutingSettings(_SettingsStub):
        model_routing: None = None  # type: ignore[assignment]

    settings = _NoRoutingSettings()
    llm = _CapturingLLM()
    llm.model_name = "my-configured-model"
    a = _make_assistant(monkeypatch, settings, llm)

    asyncio.run(a.generate_reply("Hello"))

    assert llm.model_name == "my-configured-model"
