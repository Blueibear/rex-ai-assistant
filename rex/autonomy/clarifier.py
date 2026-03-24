"""Clarifier for the Rex autonomy engine.

When a :class:`~rex.autonomy.goal_graph.Goal` is flagged as ambiguous by the
:class:`~rex.autonomy.goal_parser.GoalParser`, the :class:`Clarifier` determines
whether clarification is needed and generates a single targeted question for the
user to answer.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from rex.autonomy.goal_graph import Goal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class ClarifierBackend(Protocol):
    """Protocol for AI backends accepted by :class:`Clarifier`."""

    def generate(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[dict[str, str]] | None = None,
    ) -> str: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Clarifier
# ---------------------------------------------------------------------------


class Clarifier:
    """Checks ambiguity and generates clarification questions for goals.

    Args:
        backend: AI backend (must implement :class:`ClarifierBackend`).
            If *None*, a :class:`~rex.llm_client.LanguageModel` is
            instantiated with default settings on first use.
    """

    def __init__(self, backend: ClarifierBackend | None = None) -> None:
        self._backend: ClarifierBackend | None = backend

    # ------------------------------------------------------------------
    # Lazy backend initialisation
    # ------------------------------------------------------------------

    def _get_backend(self) -> ClarifierBackend:
        if self._backend is None:
            from rex.llm_client import LanguageModel  # lazy import

            self._backend = LanguageModel()
        return self._backend

    # ------------------------------------------------------------------
    # Ambiguity check
    # ------------------------------------------------------------------

    def needs_clarification(self, goal: Goal) -> bool:
        """Return ``True`` if *goal* requires clarification before planning.

        A goal requires clarification when it was flagged ambiguous by
        :class:`~rex.autonomy.goal_parser.GoalParser` (``goal.ambiguous is True``).

        Args:
            goal: The goal to check.

        Returns:
            ``True`` if the goal is ambiguous, ``False`` otherwise.
        """
        return goal.ambiguous

    # ------------------------------------------------------------------
    # Question generation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_question_prompt(goal: Goal) -> str:
        """Build the prompt that asks the LLM for a clarifying question."""
        return (
            "You are a clarification assistant. A user's goal is ambiguous and "
            "you must generate one clear, concise question to resolve the ambiguity.\n\n"
            f"## Ambiguous goal\n{goal.description}\n\n"
            "## Instructions\n"
            "Write a single clarifying question that:\n"
            "  - Is either yes/no or provides 2–3 specific options\n"
            "  - Directly addresses the ambiguity in the goal\n"
            "  - Is phrased politely for an end user\n\n"
            "Respond with only the question. No preamble, no explanation."
        )

    def generate_question(self, goal: Goal) -> str:
        """Generate a clarifying question for *goal*.

        Args:
            goal: The ambiguous goal to clarify.

        Returns:
            A single clarifying question as a plain string.
        """
        prompt = self._build_question_prompt(goal)
        logger.debug("Clarifier: generating question for goal=%r", goal.id)

        backend = self._get_backend()
        question = backend.generate(messages=[{"role": "user", "content": prompt}])

        logger.debug("Clarifier: question=%r", question)
        return question.strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Clarifier",
    "ClarifierBackend",
]
