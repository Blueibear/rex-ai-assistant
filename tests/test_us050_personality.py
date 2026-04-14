"""Tests for US-050: Personality system (backend).

Covers:
- Personality dataclass fields
- Built-in personalities: Professional, Friendly, Minimal
- get_personality() returns correct personality or falls back to Friendly
- list_personalities() returns all three built-ins
- Assistant._build_prompt() injects the active personality's system prompt
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Personality module tests
# ---------------------------------------------------------------------------


def test_personality_dataclass_fields():
    from rex.personality import Personality

    p = Personality(
        name="Test",
        system_prompt="You are a test assistant.",
        tone_keywords=["test"],
        greeting="Hello test.",
    )
    assert p.name == "Test"
    assert p.system_prompt == "You are a test assistant."
    assert p.tone_keywords == ["test"]
    assert p.greeting == "Hello test."


def test_list_personalities_returns_three():
    from rex.personality import list_personalities

    personalities = list_personalities()
    names = {p.name for p in personalities}
    assert {"Professional", "Friendly", "Minimal"} == names


def test_get_personality_friendly():
    from rex.personality import get_personality

    p = get_personality("Friendly")
    assert p.name == "Friendly"
    assert "warm" in p.system_prompt.lower() or "warm" in p.tone_keywords


def test_get_personality_professional():
    from rex.personality import get_personality

    p = get_personality("Professional")
    assert p.name == "Professional"
    assert len(p.system_prompt) > 0
    assert len(p.greeting) > 0


def test_get_personality_minimal():
    from rex.personality import get_personality

    p = get_personality("Minimal")
    assert p.name == "Minimal"
    assert len(p.system_prompt) > 0


def test_get_personality_unknown_falls_back_to_friendly():
    from rex.personality import DEFAULT_PERSONALITY, get_personality

    p = get_personality("NonExistentPersonality")
    assert p.name == DEFAULT_PERSONALITY


def test_default_personality_is_friendly():
    from rex.personality import DEFAULT_PERSONALITY

    assert DEFAULT_PERSONALITY == "Friendly"


# ---------------------------------------------------------------------------
# Assistant prompt injection test
# ---------------------------------------------------------------------------


def _make_assistant(personality: str = "Professional"):
    """Construct a minimal Assistant with a mocked LLM and given personality."""
    from rex.assistant import Assistant
    from rex.config import AppConfig

    cfg = AppConfig()
    cfg.personality = personality

    assistant = Assistant.__new__(Assistant)
    assistant._settings = cfg
    assistant._llm = MagicMock()
    assistant._history = []
    assistant._history_limit = 10
    assistant._plugins = []
    assistant._transcripts_dir = Path("/tmp/rex-test-transcripts")
    assistant._user_id = "test_user"
    assistant._history_store = None
    assistant._followup_engine = None
    assistant._pending_followup = None
    assistant._ha_bridge = None
    assistant._response_cache = None
    return assistant


def test_personality_prompt_injected_into_build_prompt():
    """Assistant._build_prompt() must include the personality system prompt."""
    from rex.personality import get_personality

    assistant = _make_assistant("Professional")

    with patch("rex.assistant.Assistant._build_system_context", return_value="[sys]"):
        prompt = assistant._build_prompt("hello", voice_mode=False, active_user_id=None)

    expected_snippet = get_personality("Professional").system_prompt
    assert expected_snippet in prompt, f"Expected personality prompt in:\n{prompt}"


def test_friendly_personality_prompt_injected():
    from rex.personality import get_personality

    assistant = _make_assistant("Friendly")

    with patch("rex.assistant.Assistant._build_system_context", return_value="[sys]"):
        prompt = assistant._build_prompt("hello")

    assert get_personality("Friendly").system_prompt in prompt


def test_minimal_personality_prompt_injected():
    from rex.personality import get_personality

    assistant = _make_assistant("Minimal")

    with patch("rex.assistant.Assistant._build_system_context", return_value="[sys]"):
        prompt = assistant._build_prompt("hi")

    assert get_personality("Minimal").system_prompt in prompt


def test_per_user_personality_overrides_config(tmp_path: Path):
    """Personality stored in user preferences takes precedence over AppConfig."""
    import json

    from rex.personality import get_personality

    # Create a fake Memory profile with personality = "Minimal"
    user_id = "test-user-abc"
    profile_dir = tmp_path / user_id
    profile_dir.mkdir()
    (profile_dir / "core.json").write_text(
        json.dumps({"preferences": {"personality": "Minimal"}}),
        encoding="utf-8",
    )

    assistant = _make_assistant("Professional")  # config says Professional
    assistant._user_id = user_id

    with (
        patch("rex.assistant.Assistant._build_system_context", return_value="[sys]"),
        patch("rex.identity.get_user_profile") as mock_profile,
    ):
        mock_profile.return_value = {"preferences": {"personality": "Minimal"}}
        prompt = assistant._build_prompt("hello", active_user_id=user_id)

    assert get_personality("Minimal").system_prompt in prompt
