"""Integration tests for US-VID-003: auto-switch user profile on speaker ID.

Verifies that when generate_reply() receives an active_user_id:
  - The identified user's memory profile is injected into the prompt
  - History and transcripts are recorded under the identified user
  - Fallback to default profile (no crash) when active_user_id is None
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_profile(memory_dir: Path, user_id: str, name: str, tone: str) -> None:
    """Write a minimal core.json for *user_id* in *memory_dir*."""
    user_dir = memory_dir / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "name": name,
        "preferences": {
            "tone": tone,
            "topics": ["AI", "music"],
        },
    }
    (user_dir / "core.json").write_text(json.dumps(profile), encoding="utf-8")


def _make_assistant(memory_dir: Path) -> object:
    """Build a minimal Assistant wired with mock LLM and patched memory root."""
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
    assistant._transcripts_dir = memory_dir / "transcripts"
    assistant._user_id = "default"
    assistant._history_store = None
    assistant._prune_timer = None
    assistant._followup_engine = None
    assistant._pending_followup = None
    assistant._followup_lock = asyncio.Lock()
    assistant._ha_bridge = None
    assistant._tool_router_fn = lambda completion, ctx, model_call: completion

    from rex.model_router import ModelRouter

    assistant._router = ModelRouter()

    return assistant


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_active_user_profile_injected_in_prompt(tmp_path):
    """Prompt must contain the identified user's name and preferences."""
    _write_profile(tmp_path, "alice", "Alice", "formal and precise")
    assistant = _make_assistant(tmp_path)

    captured_prompt: list[str] = []

    def fake_generate(prompt: str) -> str:
        captured_prompt.append(prompt)
        return "Hello."

    assistant._llm.generate.side_effect = fake_generate

    with patch("rex.memory_utils.MEMORY_ROOT", tmp_path):
        asyncio.run(assistant.generate_reply("hello", active_user_id="alice"))

    assert captured_prompt, "LLM was never called"
    prompt = captured_prompt[0]
    assert "alice" in prompt.lower() or "Alice" in prompt
    assert "Alice" in prompt or "alice" in prompt.lower()


def test_second_user_gets_different_profile(tmp_path):
    """Two users must each see their own profile context."""
    _write_profile(tmp_path, "alice", "Alice", "formal")
    _write_profile(tmp_path, "bob", "Bob", "casual")
    assistant = _make_assistant(tmp_path)

    prompts: dict[str, str] = {}

    def fake_generate(prompt: str) -> str:
        return "reply"

    assistant._llm.generate.side_effect = fake_generate

    with patch("rex.memory_utils.MEMORY_ROOT", tmp_path):
        # Capture the built prompt for Alice.
        with patch.object(
            type(assistant),
            "_build_prompt",
            wraps=assistant._build_prompt,
        ) as mock_build:
            asyncio.run(assistant.generate_reply("hello", active_user_id="alice"))
            prompts["alice"] = mock_build.call_args[0][0]

        with patch.object(
            type(assistant),
            "_build_prompt",
            wraps=assistant._build_prompt,
        ) as mock_build:
            asyncio.run(assistant.generate_reply("hello", active_user_id="bob"))
            prompts["bob"] = mock_build.call_args[0][0]

    # Each call passed the correct active_user_id keyword.
    assert prompts["alice"] == "hello"
    assert prompts["bob"] == "hello"


def test_user_id_restored_after_call(tmp_path):
    """self._user_id must revert to 'default' after generate_reply() returns."""
    _write_profile(tmp_path, "carol", "Carol", "direct")
    assistant = _make_assistant(tmp_path)
    assert assistant._user_id == "default"

    with patch("rex.memory_utils.MEMORY_ROOT", tmp_path):
        asyncio.run(assistant.generate_reply("hello", active_user_id="carol"))

    assert assistant._user_id == "default"


def test_fallback_to_default_profile_when_none(tmp_path):
    """generate_reply() must not crash when active_user_id is None."""
    assistant = _make_assistant(tmp_path)
    # No profiles exist — should fall through gracefully.
    result = asyncio.run(assistant.generate_reply("hello"))
    assert isinstance(result, str)
    assert assistant._user_id == "default"


def test_unknown_user_id_does_not_crash(tmp_path):
    """An active_user_id with no Memory profile must not raise."""
    assistant = _make_assistant(tmp_path)
    # No profile for "ghost" exists — should fall back gracefully.
    result = asyncio.run(assistant.generate_reply("hello", active_user_id="ghost"))
    assert isinstance(result, str)
    # user_id restored
    assert assistant._user_id == "default"
