"""Lightweight conversation memory with heuristic summarisation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class MemoryTurn:
    """Represents a single utterance in the conversation."""

    role: str
    content: str


class ConversationMemory:
    """Maintain recent conversation turns with compact summaries."""

    def __init__(
        self,
        *,
        max_turns: int = 6,
        summary_trigger: int = 4,
    ) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1")
        if summary_trigger < 1:
            raise ValueError("summary_trigger must be at least 1")
        self.max_turns = max_turns
        self.summary_trigger = summary_trigger
        self._turns: deque[MemoryTurn] = deque()
        self._summary: str = ""

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def add_turn(self, user_text: str, assistant_text: str) -> None:
        """Record a user/assistant exchange and update summaries."""

        user_clean = (user_text or "").strip()
        assistant_clean = (assistant_text or "").strip()

        if user_clean:
            self._turns.append(MemoryTurn("user", user_clean))
        if assistant_clean:
            self._turns.append(MemoryTurn("assistant", assistant_clean))

        if len(self._turns) // 2 > self.max_turns:
            self._summarise_history()

    def reset(self) -> None:
        """Clear stored history and summaries."""

        self._turns.clear()
        self._summary = ""

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def summary(self) -> str:
        return self._summary

    @property
    def turns(self) -> list[MemoryTurn]:
        return list(self._turns)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def build_messages(
        self,
        *,
        persona: str,
        profile_context: str,
        user_text: str,
    ) -> list[dict[str, str]]:
        """Return structured chat messages for the active interaction."""

        if not user_text or not user_text.strip():
            raise ValueError("user_text must not be empty")

        messages: list[dict[str, str]] = []
        if persona.strip():
            messages.append({"role": "system", "content": persona.strip()})
        if self._summary:
            messages.append({"role": "system", "content": self._summary})
        if profile_context.strip():
            messages.append({"role": "system", "content": profile_context.strip()})
        for turn in self._turns:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": user_text.strip()})
        return messages

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _summarise_history(self) -> None:
        keep_turns = max(1, self.summary_trigger) * 2
        if len(self._turns) <= keep_turns:
            return

        stale = list(self._turns)[:-keep_turns]
        if not stale:
            return

        summary_lines: list[str] = []
        i = 0
        while i < len(stale):
            user = stale[i].content if i < len(stale) else ""
            assistant = ""
            if i + 1 < len(stale) and stale[i + 1].role == "assistant":
                assistant = stale[i + 1].content
                i += 2
            else:
                i += 1
            if not user and not assistant:
                continue
            snippet_user = (user[:80] + "…") if len(user) > 80 else user
            snippet_assistant = (
                (assistant[:80] + "…") if len(assistant) > 80 else assistant or "(no reply)"
            )
            summary_lines.append(
                f"- User asked about '{snippet_user}' and the assistant replied '{snippet_assistant}'."
            )

        if summary_lines:
            summary_text = "Previous exchanges:\n" + "\n".join(summary_lines)
            if self._summary:
                self._summary = f"{self._summary}\n{summary_text}"
            else:
                self._summary = summary_text

        self._turns = deque(list(self._turns)[-keep_turns:])


__all__ = ["ConversationMemory", "MemoryTurn"]
