"""Smoke tests for the OpenClaw Rex agent — US-P2-008 and US-P2-009.

US-P2-008: start agent, send "What time is it?", confirm non-empty response.
US-P2-009: send prompt, verify persona markers in response.

The LLM is mocked so these tests run without loading model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rex.openclaw.agent import RexAgent


def _make_agent(llm_reply: str = "I am Rex, your assistant.") -> RexAgent:
    """Return a RexAgent backed by a mock LLM that returns *llm_reply*."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = llm_reply
    return RexAgent(llm=mock_llm, system_prompt="You are Rex, a helpful AI assistant.")


# ---------------------------------------------------------------------------
# US-P2-008: basic agent response smoke test
# ---------------------------------------------------------------------------


class TestAgentBasicResponse:
    def test_respond_returns_non_empty_string(self):
        """Agent returns a non-empty string for any non-empty prompt."""
        agent = _make_agent("It is 14:30 in Edinburgh.")
        result = agent.respond("What time is it?")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_respond_what_time_is_it(self):
        """Smoke test: send 'What time is it?' and get a non-empty response."""
        agent = _make_agent("The current time is 2:30 PM.")
        result = agent.respond("What time is it?")
        assert result  # truthy — non-empty string

    def test_respond_calls_llm_once(self):
        """Each respond() call invokes the LLM exactly once."""
        agent = _make_agent("Hello!")
        agent.respond("What time is it?")
        agent.llm.generate.assert_called_once()

    def test_respond_passes_user_prompt_in_messages(self):
        """The user prompt appears in the messages list passed to the LLM."""
        agent = _make_agent()
        agent.respond("What time is it?")

        call_args = agent.llm.generate.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_messages = [m for m in messages if m["role"] == "user"]
        assert any("What time is it?" in m["content"] for m in user_messages)

    def test_respond_includes_system_message(self):
        """A system-role message is included in every LLM call."""
        agent = _make_agent()
        agent.respond("Hello?")

        call_args = agent.llm.generate.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_messages = [m for m in messages if m["role"] == "system"]
        assert len(system_messages) >= 1

    def test_respond_raises_on_empty_prompt(self):
        """respond() raises ValueError for blank/empty prompts."""
        agent = _make_agent()
        with pytest.raises(ValueError):
            agent.respond("")

    def test_respond_raises_on_whitespace_prompt(self):
        """respond() raises ValueError for whitespace-only prompts."""
        agent = _make_agent()
        with pytest.raises(ValueError):
            agent.respond("   ")

    def test_agent_initialises_without_error(self):
        """RexAgent can be created without passing any arguments."""
        # Uses global config — patch load_config to avoid file I/O
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hi"
        agent = RexAgent(llm=mock_llm)
        assert agent is not None
        assert agent.agent_name  # non-empty
        assert agent.system_prompt  # non-empty


# ---------------------------------------------------------------------------
# US-P2-009: persona verification smoke test
# ---------------------------------------------------------------------------


class TestAgentPersonaVerification:
    def test_system_prompt_contains_agent_name(self):
        """The agent's system prompt references the wakeword-derived name."""
        from rex.config import AppConfig

        cfg = AppConfig(wakeword="rex")
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hello"
        agent = RexAgent(llm=mock_llm, config=cfg)
        assert "Rex" in agent.system_prompt

    def test_system_prompt_injected_in_llm_call(self):
        """system_prompt content is present in the system message sent to LLM."""
        agent = _make_agent()
        agent.respond("Tell me about yourself.")

        call_args = agent.llm.generate.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_content = " ".join(m["content"] for m in messages if m["role"] == "system")
        assert "Rex" in system_content

    def test_custom_system_prompt_used(self):
        """Explicit system_prompt overrides the default persona."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "As Aria, I can help you."
        agent = RexAgent(llm=mock_llm, system_prompt="You are Aria, a different assistant.")
        agent.respond("Who are you?")

        call_args = agent.llm.generate.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        system_content = " ".join(m["content"] for m in messages if m["role"] == "system")
        assert "Aria" in system_content

    def test_agent_name_derived_from_wakeword(self):
        """agent_name is derived from the wakeword in AppConfig."""
        from rex.config import AppConfig

        cfg = AppConfig(wakeword="rex")
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hi"
        agent = RexAgent(llm=mock_llm, config=cfg)
        assert agent.agent_name == "Rex"

    def test_persona_derived_from_config_location(self):
        """Location in AppConfig appears in the system prompt."""
        from rex.config import AppConfig

        cfg = AppConfig(wakeword="rex", default_location="Edinburgh, Scotland")
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hi"
        agent = RexAgent(llm=mock_llm, config=cfg)
        assert "Edinburgh" in agent.system_prompt


# ---------------------------------------------------------------------------
# US-P3-006: memory persistence across interactions
# ---------------------------------------------------------------------------


class TestAgentMemoryPersistence:
    """Verify that conversation history is saved and replayed across respond() calls."""

    def _agent_with_memory(self, tmp_path, replies: list[str]) -> RexAgent:
        """Return an agent backed by a mock LLM and a tmp_path-scoped MemoryAdapter."""
        from rex.openclaw.memory_adapter import MemoryAdapter

        mock_llm = MagicMock()
        mock_llm.generate.side_effect = replies
        adapter = MemoryAdapter(memory_root=str(tmp_path))
        return RexAgent(
            llm=mock_llm,
            system_prompt="You are Rex.",
            memory_adapter=adapter,
        )

    def test_history_saved_after_respond(self, tmp_path):
        """After one respond() with a user_key, history contains user and assistant turns."""
        from rex.openclaw.memory_adapter import MemoryAdapter

        adapter = MemoryAdapter(memory_root=str(tmp_path))
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "It is 3pm."
        agent = RexAgent(llm=mock_llm, system_prompt="You are Rex.", memory_adapter=adapter)

        agent.respond("What time is it?", user_key="alice")
        history = adapter.load_recent("alice")

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["text"] == "What time is it?"
        assert history[1]["role"] == "assistant"
        assert history[1]["text"] == "It is 3pm."

    def test_history_replayed_in_second_call(self, tmp_path):
        """On a second respond() call the first exchange is included in LLM messages."""
        agent = self._agent_with_memory(tmp_path, ["It is 3pm.", "You asked about the time."])

        agent.respond("What time is it?", user_key="bob")
        agent.respond("What did I ask?", user_key="bob")

        second_call_args = agent.llm.generate.call_args_list[1]
        messages = second_call_args.kwargs.get("messages") or second_call_args.args[0]
        contents = [m["content"] for m in messages]

        assert "What time is it?" in contents
        assert "It is 3pm." in contents
        assert "What did I ask?" in contents

    def test_no_history_without_user_key(self, tmp_path):
        """Without a user_key, no history is stored."""
        from rex.openclaw.memory_adapter import MemoryAdapter

        adapter = MemoryAdapter(memory_root=str(tmp_path))
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Hello!"
        agent = RexAgent(llm=mock_llm, system_prompt="You are Rex.", memory_adapter=adapter)

        agent.respond("Hello?")  # no user_key
        # Memory directory should be empty — no user subdirectory created
        user_dirs = list(tmp_path.iterdir())
        assert user_dirs == []

    def test_different_users_have_separate_histories(self, tmp_path):
        """History for user A does not bleed into messages for user B."""
        agent = self._agent_with_memory(tmp_path, ["Reply to A.", "Reply to B."])

        agent.respond("Message from A.", user_key="user-a")
        agent.respond("Message from B.", user_key="user-b")

        second_call_args = agent.llm.generate.call_args_list[1]
        messages = second_call_args.kwargs.get("messages") or second_call_args.args[0]
        contents = " ".join(m["content"] for m in messages)

        assert "Message from A." not in contents
        assert "Message from B." in contents

    def test_history_accumulates_across_three_turns(self, tmp_path):
        """Three consecutive turns all appear in the messages for the fourth call."""
        agent = self._agent_with_memory(tmp_path, ["Reply 1", "Reply 2", "Reply 3", "Reply 4"])

        for i in range(1, 4):
            agent.respond(f"Turn {i}", user_key="carol")

        agent.respond("Turn 4", user_key="carol")
        last_call = agent.llm.generate.call_args_list[3]
        messages = last_call.kwargs.get("messages") or last_call.args[0]
        contents = " ".join(m["content"] for m in messages)

        assert "Turn 1" in contents
        assert "Turn 2" in contents
        assert "Turn 3" in contents
        assert "Turn 4" in contents
