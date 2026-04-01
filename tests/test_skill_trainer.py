"""Tests for US-SK-003: natural language skill creation via chat.

Covers:
- detect_skill_creation_intent() pattern matching
- SkillTrainer.handle_if_training_request() happy path
- Script file created and contains SKILL_METADATA
- Skill registered in SkillRegistry and callable
- Non-training messages return None
- Integration: assistant generate_reply() intercepts training request
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rex.skills.registry import SkillRegistry
from rex.skills.trainer import SkillTrainer, detect_skill_creation_intent

# ---------------------------------------------------------------------------
# detect_skill_creation_intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message,expected_fragment",
    [
        ("teach yourself to tell me the current battery level", "battery level"),
        ("learn how to check the weather forecast", "weather forecast"),
        ("add a skill that reads my emails aloud", "reads my emails aloud"),
        ("create a skill for showing my calendar", "showing my calendar"),
        ("remember how to start the coffee machine", "coffee machine"),
        ("make a skill that greets me by name", "greets me by name"),
        # With trailing punctuation
        ("teach yourself to find nearby restaurants.", "restaurants"),
    ],
)
def test_detect_creation_intent_match(message: str, expected_fragment: str) -> None:
    intent = detect_skill_creation_intent(message)
    assert intent is not None, f"Expected match for: {message!r}"
    assert expected_fragment.lower() in intent.lower()


@pytest.mark.parametrize(
    "message",
    [
        "what is the weather today",
        "tell me a joke",
        "turn off the lights",
        "how are you",
        "set a timer for 5 minutes",
    ],
)
def test_detect_creation_intent_no_match(message: str) -> None:
    assert detect_skill_creation_intent(message) is None


# ---------------------------------------------------------------------------
# SkillTrainer — unit tests
# ---------------------------------------------------------------------------


def test_handle_returns_none_for_non_training_message(tmp_path: Path) -> None:
    registry = SkillRegistry(skills_path=tmp_path / "skills.json")
    trainer = SkillTrainer(skills_dir=tmp_path / "skills")
    result = trainer.handle_if_training_request("what is the weather today", registry)
    assert result is None


def test_handle_creates_skill_script(tmp_path: Path) -> None:
    registry = SkillRegistry(skills_path=tmp_path / "skills.json")
    skills_dir = tmp_path / "skills"
    trainer = SkillTrainer(skills_dir=skills_dir)

    response = trainer.handle_if_training_request(
        "teach yourself to tell me the current battery level", registry
    )

    assert response is not None
    # A .py file should have been created
    scripts = list(skills_dir.glob("*.py"))
    assert len(scripts) == 1, f"Expected 1 script, found: {[s.name for s in scripts]}"


def test_handle_script_contains_skill_metadata(tmp_path: Path) -> None:
    registry = SkillRegistry(skills_path=tmp_path / "skills.json")
    skills_dir = tmp_path / "skills"
    trainer = SkillTrainer(skills_dir=skills_dir)

    trainer.handle_if_training_request(
        "teach yourself to tell me the current battery level", registry
    )

    script = next(skills_dir.glob("*.py"))
    content = script.read_text(encoding="utf-8")
    assert "SKILL_METADATA" in content
    assert "battery level" in content.lower()
    assert "triggers" in content


def test_handle_registers_skill_in_registry(tmp_path: Path) -> None:
    registry = SkillRegistry(skills_path=tmp_path / "skills.json")
    skills_dir = tmp_path / "skills"
    trainer = SkillTrainer(skills_dir=skills_dir)

    trainer.handle_if_training_request(
        "teach yourself to tell me the current battery level", registry
    )

    skills = registry.list_skills()
    assert len(skills) == 1
    skill = skills[0]
    assert "battery level" in skill.name.lower()
    assert skill.enabled is True
    assert len(skill.trigger_patterns) >= 1


def test_handle_returns_confirmation_message(tmp_path: Path) -> None:
    registry = SkillRegistry(skills_path=tmp_path / "skills.json")
    trainer = SkillTrainer(skills_dir=tmp_path / "skills")

    response = trainer.handle_if_training_request(
        "teach yourself to tell me the current battery level", registry
    )

    assert response is not None
    assert "I've learned how to" in response
    assert "battery level" in response.lower()
    assert "You can trigger it by saying" in response


def test_handle_script_has_run_function(tmp_path: Path) -> None:
    """The generated script must define a run() function and be importable."""
    registry = SkillRegistry(skills_path=tmp_path / "skills.json")
    skills_dir = tmp_path / "skills"
    trainer = SkillTrainer(skills_dir=skills_dir)

    trainer.handle_if_training_request(
        "teach yourself to tell me the current battery level", registry
    )

    script = next(skills_dir.glob("*.py"))
    spec = importlib.util.spec_from_file_location("generated_skill", script)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    assert hasattr(mod, "run"), "Generated skill script must define run()"
    result = mod.run("test input")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Integration: Assistant.generate_reply() intercepts training requests
# ---------------------------------------------------------------------------


def _make_assistant(tmp_path: Path) -> object:
    """Build a minimal Assistant with skill trainer wired in."""
    from rex.assistant import Assistant

    assistant = Assistant.__new__(Assistant)
    assistant._settings = MagicMock()
    assistant._settings.model_routing = None
    assistant._settings.ollama_base_url = "http://localhost:11434"
    assistant._settings.default_timezone = None
    assistant._settings.default_location = None
    assistant._settings.persist_history = False
    assistant._llm = MagicMock()
    assistant._llm.generate.return_value = "Mock reply."
    assistant._llm.model_name = "test-model"
    assistant._history = []
    assistant._history_limit = 50
    assistant._plugins = []
    assistant._transcripts_dir = tmp_path / "transcripts"
    assistant._user_id = "default"
    assistant._history_store = None
    assistant._prune_timer = None
    assistant._followup_engine = None
    assistant._pending_followup = None
    assistant._ha_bridge = None
    assistant._tool_router_fn = lambda completion, ctx, model_call: completion

    from rex.model_router import ModelRouter

    assistant._router = ModelRouter()

    skills_dir = tmp_path / "skills"
    assistant._skill_registry = SkillRegistry(skills_path=tmp_path / "skills.json")
    assistant._skill_trainer = SkillTrainer(skills_dir=skills_dir)

    # asyncio.Lock must be created inside an event loop; use a lazy property approach
    import asyncio

    assistant._followup_lock = asyncio.Lock()

    return assistant


def test_assistant_intercepts_training_request(tmp_path: Path) -> None:
    """generate_reply() must return a confirmation, not an LLM response."""
    assistant = _make_assistant(tmp_path)

    response = asyncio.run(
        assistant.generate_reply("teach yourself to tell me the current battery level")
    )

    # Should get a skill creation confirmation, not the mock LLM reply
    assert "I've learned how to" in response
    # LLM must NOT have been called
    assistant._llm.generate.assert_not_called()


def test_assistant_skill_created_and_registered(tmp_path: Path) -> None:
    """After generate_reply(), skill script must exist and be in the registry."""
    assistant = _make_assistant(tmp_path)
    skills_dir = tmp_path / "skills"

    asyncio.run(assistant.generate_reply("teach yourself to tell me the current battery level"))

    # Script file created
    scripts = list(skills_dir.glob("*.py"))
    assert len(scripts) == 1

    # Skill registered
    skills = assistant._skill_registry.list_skills()
    assert len(skills) == 1
    assert skills[0].enabled is True


def test_assistant_non_training_message_uses_llm(tmp_path: Path) -> None:
    """Normal messages must still reach the LLM."""
    assistant = _make_assistant(tmp_path)

    response = asyncio.run(assistant.generate_reply("what is the weather today"))

    assert response == "Mock reply."
    assistant._llm.generate.assert_called_once()
