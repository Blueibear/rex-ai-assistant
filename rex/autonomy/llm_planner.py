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
        model: Optional model identifier forwarded to :class:`~rex.llm_client.LanguageModel`
            when the backend is lazily created.  Has no effect when *backend*
            is explicitly provided.  Defaults to ``""`` (use the configured
            default model).
    """

    def __init__(
        self,
        tools: list[ToolDefinition],
        backend: LLMBackend | None = None,
        *,
        model: str = "",
    ) -> None:
        self._tools: list[ToolDefinition] = tools
        self._backend: LLMBackend | None = backend
        self._model: str = model

    # ------------------------------------------------------------------
    # Lazy backend initialisation
    # ------------------------------------------------------------------

    def _get_backend(self) -> LLMBackend:
        if self._backend is None:
            from rex.llm_client import LanguageModel  # lazy import

            if self._model:
                self._backend = LanguageModel(model=self._model)
            else:
                self._backend = LanguageModel()
        return self._backend

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self, goal: str, context: dict[str, Any], feedback_summary: str = ""
    ) -> str:
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
        feedback_section = (
            f"## Past Execution Patterns\n{feedback_summary}\n\n"
            if feedback_summary
            else ""
        )
        return (
            "You are a planning assistant. Your job is to break a user goal "
            "into an ordered list of tool calls that together achieve the goal.\n\n"
            "## Available tools\n"
            f"{tools_block}\n\n"
            "## Current context\n"
            f"{context_block}\n\n"
            f"{feedback_section}"
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

        return self._parse_step_list(data, goal, id_prefix="step")

    # ------------------------------------------------------------------
    # Helpers shared between plan() and plan_with_alternatives()
    # ------------------------------------------------------------------

    def _parse_step_list(
        self, items: Any, goal: str, id_prefix: str = "step"
    ) -> Plan:
        """Convert a raw JSON list of step dicts into a :class:`Plan`.

        Args:
            items: A parsed JSON value expected to be a non-empty list of dicts.
            goal: The planning goal (stored on the returned Plan).
            id_prefix: Prefix for generated step IDs (e.g. ``"step"`` or
                ``"alt0-step"``).

        Returns:
            A :class:`~rex.autonomy.models.Plan` with one or more steps.

        Raises:
            PlanningError: If *items* is not a non-empty list of dicts.
        """
        if not isinstance(items, list):
            raise PlanningError(
                f"Expected a JSON array of steps, got {type(items).__name__}."
            )
        if len(items) == 0:
            raise PlanningError(
                "LLM returned an empty step list — cannot create a Plan with zero steps."
            )

        steps: list[PlanStep] = []
        tool_names = {t.name for t in self._tools}

        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise PlanningError(f"Step {i} is not a JSON object: {item!r}")

            tool_name = str(item.get("tool", "")).strip()
            raw_args: Any = item.get("args", {})
            description = str(item.get("description", "")).strip()

            if not tool_name:
                raise PlanningError(f"Step {i} is missing a 'tool' field: {item!r}")

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

            if tool_names and tool_name not in tool_names:
                logger.warning(
                    "LLMPlanner: step %d references unknown tool %r — "
                    "step kept but may fail at execution time",
                    i,
                    tool_name,
                )

            steps.append(
                PlanStep(
                    id=f"{id_prefix}-{i + 1}",
                    tool=tool_name,
                    args=args,
                    description=description,
                )
            )

        return Plan(id=str(uuid.uuid4()), goal=goal, steps=steps)

    # ------------------------------------------------------------------
    # Alternatives prompt and parser
    # ------------------------------------------------------------------

    def _build_alternatives_prompt(self, goal: str, context: dict[str, Any]) -> str:
        """Construct a prompt that asks the LLM for three alternative plans."""
        tools_block = (
            "\n".join(f"- {t.name}: {t.description}" for t in self._tools) or "(none)"
        )
        context_block = (
            json.dumps(context, indent=2, default=str) if context else "{}"
        )
        return (
            "You are a planning assistant. Your job is to produce THREE different "
            "plans to achieve a goal using the available tools.\n\n"
            "## Available tools\n"
            f"{tools_block}\n\n"
            "## Current context\n"
            f"{context_block}\n\n"
            "## Goal\n"
            f"{goal}\n\n"
            "## Instructions\n"
            "Return ONLY a JSON array containing EXACTLY THREE sub-arrays. "
            "Each sub-array is a separate plan:\n"
            "  - First sub-array: your primary (preferred) approach\n"
            "  - Second sub-array: first alternative approach\n"
            "  - Third sub-array: second alternative approach\n\n"
            "Each step in each plan must be a JSON object with exactly:\n"
            "  - tool        (string)\n"
            "  - args        (object, may be {})\n"
            "  - description (string)\n\n"
            "Format: [[primary_steps...], [alt1_steps...], [alt2_steps...]]\n"
            "Do not include any text outside the JSON array. "
            "Respond with valid JSON only."
        )

    def _parse_alternatives_response(self, raw: str, goal: str) -> list[Plan]:
        """Parse LLM response containing three plan arrays.

        Args:
            raw: Raw text returned by the LLM.
            goal: The planning goal.

        Returns:
            A list of exactly three :class:`~rex.autonomy.models.Plan` objects.

        Raises:
            PlanningError: If *raw* cannot be parsed into three valid plans.
        """
        cleaned = self._strip_fences(raw)

        try:
            outer: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlanningError(
                f"LLM alternatives response is not valid JSON: {exc}\nRaw: {raw!r}"
            ) from exc

        if not isinstance(outer, list) or len(outer) != 3:
            raise PlanningError(
                f"Expected JSON array of exactly 3 plan arrays, got "
                f"{type(outer).__name__} of length "
                f"{len(outer) if isinstance(outer, list) else '?'}.\nRaw: {raw!r}"
            )

        return [
            self._parse_step_list(inner, goal, id_prefix=f"alt{idx}-step")
            for idx, inner in enumerate(outer)
        ]

    # ------------------------------------------------------------------
    # plan_with_alternatives
    # ------------------------------------------------------------------

    def plan_with_alternatives(self, goal: str, context: dict[str, Any]) -> list[Plan]:
        """Generate a primary plan plus two alternatives for *goal*.

        Args:
            goal: Natural-language description of what the user wants to achieve.
            context: Arbitrary key/value context forwarded to the LLM.

        Returns:
            A list of exactly three :class:`~rex.autonomy.models.Plan` objects:
            the primary plan followed by two alternatives.

        Raises:
            PlanningError: If the LLM response cannot be parsed.
        """
        prompt = self._build_alternatives_prompt(goal, context)
        logger.debug("LLMPlanner: requesting alternatives for goal=%r", goal)

        backend = self._get_backend()
        raw = backend.generate(messages=[{"role": "user", "content": prompt}])

        logger.debug("LLMPlanner: raw alternatives response: %r", raw)
        return self._parse_alternatives_response(raw, goal)

    # ------------------------------------------------------------------
    # PlannerProtocol implementation
    # ------------------------------------------------------------------

    def plan(
        self,
        goal: str,
        context: dict[str, Any],
        *,
        feedback_summary: str = "",
    ) -> Plan:
        """Convert *goal* into an executable :class:`~rex.autonomy.models.Plan`.

        Args:
            goal: Natural-language description of what the user wants to achieve.
            context: Arbitrary key/value context (available tools, user
                preferences, session state, etc.).
            feedback_summary: Optional plain-text summary of past execution
                history.  When non-empty it is included in the prompt under a
                ``## Past Execution Patterns`` section so the LLM can learn
                from prior runs.

        Returns:
            A :class:`~rex.autonomy.models.Plan` with one or more
            :class:`~rex.autonomy.models.PlanStep` objects.

        Raises:
            PlanningError: If the LLM response cannot be parsed into a
                valid plan.
        """
        prompt = self._build_prompt(goal, context, feedback_summary)
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
