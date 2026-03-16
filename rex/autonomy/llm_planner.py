"""LLM-based planner for the Rex autonomy engine.

``LLMPlanner`` converts a natural-language goal into an executable
:class:`~rex.autonomy.models.Plan` by calling Rex's AI backend with a
structured prompt and parsing the JSON response.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel

from rex.autonomy.models import Plan, PlanStep, PlannerProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class PlanningError(Exception):
    """Raised when the LLM response cannot be parsed into a valid Plan."""


class ToolDefinition(BaseModel):
    """Describes a single tool available to the planner."""

    name: str
    description: str
    args_schema: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class LLMBackend(Protocol):
    """Protocol for AI backends accepted by :class:`LLMPlanner`.

    Any object whose ``generate`` method accepts *messages* as a keyword
    argument and returns ``str`` satisfies this interface.  Rex's own
    :class:`~rex.llm_client.LanguageModel` satisfies it.
    """

    def generate(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[dict[str, str]] | None = None,
    ) -> str: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# LLMPlanner
# ---------------------------------------------------------------------------


class LLMPlanner(PlannerProtocol):
    """Planner that delegates goal decomposition to an LLM.

    The planner builds a structured system prompt listing available tools
    and the current context, asks the LLM to return a JSON array of steps,
    then parses the response into a :class:`~rex.autonomy.models.Plan`.

    Args:
        tools: Available tools the planner may reference in the plan.
        backend: AI backend instance (must implement :class:`LLMBackend`).
            If *None*, a :class:`~rex.llm_client.LanguageModel` is
            instantiated with default settings on first use.
    """

    def __init__(
        self,
        tools: list[ToolDefinition],
        backend: LLMBackend | None = None,
    ) -> None:
        self._tools: list[ToolDefinition] = tools
        self._backend: LLMBackend | None = backend

    # ------------------------------------------------------------------
    # Lazy backend initialisation
    # ------------------------------------------------------------------

    def _get_backend(self) -> LLMBackend:
        if self._backend is None:
            from rex.llm_client import LanguageModel  # lazy import

            self._backend = LanguageModel()
        return self._backend

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, goal: str, context: dict[str, Any]) -> str:
        """Construct the planning prompt sent to the LLM."""
        tools_block = (
            "\n".join(
                f"- {t.name}: {t.description}" for t in self._tools
            )
            or "(none)"
        )
        context_block = (
            json.dumps(context, indent=2, default=str) if context else "{}"
        )
        return (
            "You are a planning assistant. Your job is to break a user goal "
            "into an ordered list of tool calls that together achieve the goal.\n\n"
            "## Available tools\n"
            f"{tools_block}\n\n"
            "## Current context\n"
            f"{context_block}\n\n"
            "## Goal\n"
            f"{goal}\n\n"
            "## Instructions\n"
            "Return ONLY a JSON array of steps. Each step must be a JSON object "
            "with exactly these keys:\n"
            "  - tool        (string) — name of the tool to call\n"
            "  - args        (object) — keyword arguments for the tool (may be {})\n"
            "  - description (string) — one-sentence human-readable explanation\n\n"
            "Do not include any text outside the JSON array. "
            "If the goal cannot be broken down further, return a single-step array. "
            "Respond with valid JSON only."
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

    def _parse_response(self, raw: str, goal: str) -> Plan:
        """Parse LLM JSON output into a :class:`~rex.autonomy.models.Plan`.

        Args:
            raw: Raw text returned by the LLM.
            goal: The original planning goal (used to populate ``Plan.goal``).

        Returns:
            A :class:`~rex.autonomy.models.Plan` with one or more steps.

        Raises:
            PlanningError: If *raw* is not valid JSON, is not a list, or the
                list is empty.
        """
        cleaned = self._strip_fences(raw)

        try:
            data: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlanningError(
                f"LLM response is not valid JSON: {exc}\nRaw response: {raw!r}"
            ) from exc

        if not isinstance(data, list):
            raise PlanningError(
                f"Expected a JSON array of steps, got {type(data).__name__}. "
                f"Raw response: {raw!r}"
            )

        if len(data) == 0:
            raise PlanningError(
                "LLM returned an empty step list — cannot create a Plan with zero steps."
            )

        steps: list[PlanStep] = []
        tool_names = {t.name for t in self._tools}

        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise PlanningError(
                    f"Step {i} is not a JSON object: {item!r}"
                )

            tool_name = str(item.get("tool", "")).strip()
            raw_args: Any = item.get("args", {})
            description = str(item.get("description", "")).strip()

            if not tool_name:
                raise PlanningError(
                    f"Step {i} is missing a 'tool' field: {item!r}"
                )

            args: dict[str, Any]
            if isinstance(raw_args, dict):
                args = dict(raw_args)
            else:
                logger.warning(
                    "LLMPlanner: step %d 'args' is not a dict — defaulting to {}",
                    i,
                )
                args = {}

            if not description:
                description = f"Step {i + 1}: call {tool_name}"

            # Warn if tool is not in the registered list, but keep the step.
            if tool_names and tool_name not in tool_names:
                logger.warning(
                    "LLMPlanner: step %d references unknown tool %r — "
                    "step kept but may fail at execution time",
                    i,
                    tool_name,
                )

            steps.append(
                PlanStep(
                    id=f"step-{i + 1}",
                    tool=tool_name,
                    args=args,
                    description=description,
                )
            )

        return Plan(id=str(uuid.uuid4()), goal=goal, steps=steps)

    # ------------------------------------------------------------------
    # PlannerProtocol implementation
    # ------------------------------------------------------------------

    def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Convert *goal* into an executable :class:`~rex.autonomy.models.Plan`.

        Args:
            goal: Natural-language description of what the user wants to achieve.
            context: Arbitrary key/value context (available tools, user
                preferences, session state, etc.).

        Returns:
            A :class:`~rex.autonomy.models.Plan` with one or more
            :class:`~rex.autonomy.models.PlanStep` objects.

        Raises:
            PlanningError: If the LLM response cannot be parsed into a
                valid plan.
        """
        prompt = self._build_prompt(goal, context)
        logger.debug("LLMPlanner: sending planning prompt for goal=%r", goal)

        backend = self._get_backend()
        raw = backend.generate(messages=[{"role": "user", "content": prompt}])

        logger.debug("LLMPlanner: raw LLM response: %r", raw)
        return self._parse_response(raw, goal)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "LLMPlanner",
    "LLMBackend",
    "PlanningError",
    "ToolDefinition",
]
