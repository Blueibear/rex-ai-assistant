"""GoalParser for the Rex autonomy engine.

Parses a free-form user message into a :class:`~rex.autonomy.goal_graph.GoalGraph`
by asking the LLM to identify discrete goals and their dependencies.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any, Protocol

from rex.autonomy.goal_graph import Goal, GoalGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class GoalParsingError(Exception):
    """Raised when the LLM response cannot be parsed into a :class:`GoalGraph`."""


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class GoalParserBackend(Protocol):
    """Protocol for AI backends accepted by :class:`GoalParser`."""

    def generate(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[dict[str, str]] | None = None,
    ) -> str: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# GoalParser
# ---------------------------------------------------------------------------


class GoalParser:
    """Uses an LLM to parse a user message into a :class:`~rex.autonomy.goal_graph.GoalGraph`.

    Args:
        backend: AI backend (must implement :class:`GoalParserBackend`).
            If *None*, a :class:`~rex.llm_client.LanguageModel` is
            instantiated with default settings on first use.
    """

    def __init__(self, backend: GoalParserBackend | None = None) -> None:
        self._backend: GoalParserBackend | None = backend

    # ------------------------------------------------------------------
    # Lazy backend initialisation
    # ------------------------------------------------------------------

    def _get_backend(self) -> GoalParserBackend:
        if self._backend is None:
            from rex.llm_client import LanguageModel  # lazy import

            self._backend = LanguageModel()
        return self._backend

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(user_input: str) -> str:
        """Construct the goal-extraction prompt."""
        return (
            "You are a goal-extraction assistant. Your job is to identify all distinct "
            "goals in the user's message, determine which goals depend on others, and "
            "return a structured list.\n\n"
            "## User message\n"
            f"{user_input}\n\n"
            "## Instructions\n"
            "Return ONLY a JSON array of goal objects. Each object must have exactly:\n"
            "  - id          (string) — short unique slug, e.g. 'book_flight'\n"
            "  - description (string) — one sentence describing this specific goal\n"
            "  - depends_on  (array of strings) — IDs of goals that must finish first "
            "(empty array if none)\n"
            "  - ambiguous   (boolean) — true if the goal is unclear or could be "
            "interpreted multiple ways, false otherwise\n\n"
            "Rules:\n"
            "  - If the message contains only one goal, return a single-element array.\n"
            "  - If a goal is ambiguous, set ambiguous to true.\n"
            "  - Do not include any text outside the JSON array.\n"
            "  - Respond with valid JSON only."
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_fences(raw: str) -> str:
        """Remove optional markdown code-fence wrapping from *raw*."""
        stripped = raw.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        inner = [line for line in lines[1:] if line.strip() != "```"]
        return "\n".join(inner).strip()

    def _parse_response(self, raw: str) -> GoalGraph:
        """Parse LLM JSON output into a :class:`~rex.autonomy.goal_graph.GoalGraph`.

        Args:
            raw: Raw text returned by the LLM.

        Returns:
            A :class:`~rex.autonomy.goal_graph.GoalGraph` with one or more goals.

        Raises:
            GoalParsingError: If *raw* is not valid JSON, not a list, or empty.
        """
        cleaned = self._strip_fences(raw)

        try:
            data: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GoalParsingError(
                f"GoalParser: LLM response is not valid JSON: {exc}\nRaw: {raw!r}"
            ) from exc

        if not isinstance(data, list):
            raise GoalParsingError(
                f"GoalParser: expected a JSON array, got {type(data).__name__}. "
                f"Raw: {raw!r}"
            )

        if len(data) == 0:
            raise GoalParsingError(
                "GoalParser: LLM returned an empty goal list — cannot build a GoalGraph."
            )

        goals: list[Goal] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise GoalParsingError(
                    f"GoalParser: goal {i} is not a JSON object: {item!r}"
                )

            goal_id = str(item.get("id", "")).strip()
            description = str(item.get("description", "")).strip()
            raw_deps: Any = item.get("depends_on", [])

            if not goal_id:
                raise GoalParsingError(
                    f"GoalParser: goal {i} is missing 'id' field: {item!r}"
                )
            if not description:
                description = f"Goal {goal_id}"

            depends_on: list[str]
            if isinstance(raw_deps, list):
                depends_on = [str(d) for d in raw_deps]
            else:
                logger.warning(
                    "GoalParser: goal %d 'depends_on' is not a list — defaulting to []", i
                )
                depends_on = []

            ambiguous = bool(item.get("ambiguous", False))

            goals.append(
                Goal(
                    id=goal_id,
                    description=description,
                    depends_on=depends_on,
                    ambiguous=ambiguous,
                )
            )

        return GoalGraph(goals)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def parse(self, user_input: str) -> GoalGraph:
        """Parse *user_input* into a :class:`~rex.autonomy.goal_graph.GoalGraph`.

        Args:
            user_input: Free-form text from the user describing what they want
                to accomplish.  May contain one or more goals.

        Returns:
            A :class:`~rex.autonomy.goal_graph.GoalGraph` with one or more
            :class:`~rex.autonomy.goal_graph.Goal` objects.

        Raises:
            GoalParsingError: If the LLM response cannot be parsed.
        """
        prompt = self._build_prompt(user_input)
        logger.debug("GoalParser: extracting goals from user input")

        backend = self._get_backend()
        raw = backend.generate(messages=[{"role": "user", "content": prompt}])

        logger.debug("GoalParser: raw LLM response: %r", raw)
        graph = self._parse_response(raw)
        logger.debug("GoalParser: extracted %d goal(s)", len(graph.goals))
        return graph


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "GoalParser",
    "GoalParserBackend",
    "GoalParsingError",
]
