"""Tests for the conversation memory helper."""

from __future__ import annotations

import pytest

from conversation_memory import ConversationMemory


def test_memory_summarises_when_over_capacity():
    memory = ConversationMemory(max_turns=2, summary_trigger=1)

    memory.add_turn("First question", "First answer")
    memory.add_turn("Second question", "Second answer")
    memory.add_turn("Third question", "Third answer")

    assert len(memory.turns) <= 4
    assert "Previous exchanges" in memory.summary
    assert "First question" in memory.summary


def test_build_messages_requires_user_text():
    memory = ConversationMemory()

    with pytest.raises(ValueError):
        memory.build_messages(persona="", profile_context="", user_text="")

    memory.add_turn("Hello", "Hi")
    messages = memory.build_messages(
        persona="You are helpful.",
        profile_context="User is named Alex.",
        user_text="What's up?",
    )

    roles = [message["role"] for message in messages]
    assert roles[0] == "system"
    assert any(msg["role"] == "user" for msg in messages)
