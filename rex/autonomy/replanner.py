"""Replanner for the Rex autonomy engine.

When a plan step fails, :class:`Replanner` calls the LLM with context about
the failure and asks for a revised set of steps to complete the original goal.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

from rex.autonomy.models import Plan, PlanStep, StepStatus

if TYPE_CHECKING:
    from rex.autonomy.llm_planner import LLMPlanner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol (mirrors LLMPlanner's protocol)
# ---------------------------------------------------------------------------


class ReplannerBackend(Protocol):
    """Protocol for AI backends accepted by :class:`Replanner`."""

    def generate(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[dict[str, str]] | None = None,
    ) -> str: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Replanner
# ---------------------------------------------------------------------------


class Replanner:
    """Produces a revised :class:`~rex.autonomy.models.Plan` after a step failure.

    The replanner sends context about the original goal, steps already
    completed, the failed step, and the error to the LLM and parses the
    response into a new set of steps.

    When *planner* is supplied (an :class:`~rex.autonomy.llm_planner.LLMPlanner`
    instance), the first :meth:`replan` call uses
    :meth:`~rex.autonomy.llm_planner.LLMPlanner.plan_with_alternatives` to
    generate three alternative plans.  Subsequent calls return each alternative
    in turn.  This enables the runner to try up to two fallback approaches
    before declaring failure.

    Args:
        backend: AI backend (must implement :class:`ReplannerBackend`).
            If *None*, a :class:`~rex.llm_client.LanguageModel` is
            instantiated with default settings on first use.
        planner: Optional :class:`~rex.autonomy.llm_planner.LLMPlanner`
            used to generate alternative plans via
            :meth:`~rex.autonomy.llm_planner.LLMPlanner.plan_with_alternatives`.
            When provided, the Replanner cycles through the generated
            alternatives rather than calling the backend directly.
    """

    def __init__(
        self,
        backend: ReplannerBackend | None = None,
        planner: LLMPlanner | None = None,
    ) -> None:
        self._backend: ReplannerBackend | None = backend
        self._planner: LLMPlanner | None = planner
        # Populated on first replan() call when _planner is set.
        self._alternatives: list[Plan] = []
        self._alt_index: int = 0

    # ------------------------------------------------------------------
    # Lazy backend initialisation
    # ------------------------------------------------------------------

    def _get_backend(self) -> ReplannerBackend:
        if self._backend is None:
            from rex.llm_client import LanguageModel  # lazy import

            self._backend = LanguageModel()
        return self._backend

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_replan_prompt(
        self,
        original_plan: Plan,
        failed_step: PlanStep,
        error_context: str,
    ) -> str:
        """Build the replanning prompt sent to the LLM."""
        completed = [s for s in original_plan.steps if s.status == StepStatus.SUCCESS]
        completed_block = (
            "\n".join(f"- {s.id}: {s.description} (result: {s.result!r})" for s in completed)
            or "(none)"
        )
        return (
            "You are a replanning assistant. A plan step has failed and you must "
            "produce a revised set of steps to complete the original goal.\n\n"
            f"## Original goal\n{original_plan.goal}\n\n"
            f"## Steps completed so far\n{completed_block}\n\n"
            "## Failed step\n"
            f"- ID: {failed_step.id}\n"
            f"- Tool: {failed_step.tool}\n"
            f"- Description: {failed_step.description}\n"
            f"- Error: {error_context}\n\n"
            "## Instructions\n"
            "Return ONLY a JSON array of steps for the REMAINING work (do not "
            "re-include already-completed steps). Each step must be a JSON object "
            "with exactly these keys:\n"
            "  - tool        (string) — name of the tool to call\n"
            "  - args        (object) — keyword arguments for the tool (may be {})\n"
            "  - description (string) — one-sentence human-readable explanation\n\n"
            "Do not include any text outside the JSON array. "
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
            ValueError: If *raw* is not valid JSON, not a list, or empty.
        """
        cleaned = self._strip_fences(raw)

        try:
            data: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Replanner: LLM response is not valid JSON: {exc}\nRaw: {raw!r}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                f"Replanner: expected a JSON array, got {type(data).__name__}. " f"Raw: {raw!r}"
            )

        if len(data) == 0:
            raise ValueError(
                "Replanner: LLM returned an empty step list — cannot replan with zero steps."
            )

        steps: list[PlanStep] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Replanner: step {i} is not a JSON object: {item!r}")

            tool_name = str(item.get("tool", "")).strip()
            raw_args: Any = item.get("args", {})
            description = str(item.get("description", "")).strip()

            if not tool_name:
                raise ValueError(f"Replanner: step {i} is missing 'tool' field: {item!r}")

            args: dict[str, Any]
            if isinstance(raw_args, dict):
                args = dict(raw_args)
            else:
                logger.warning("Replanner: step %d 'args' is not a dict — defaulting to {}", i)
                args = {}

            if not description:
                description = f"Replanned step {i + 1}: call {tool_name}"

            steps.append(
                PlanStep(
                    id=f"replan-step-{i + 1}",
                    tool=tool_name,
                    args=args,
                    description=description,
                )
            )

        return Plan(id=str(uuid.uuid4()), goal=goal, steps=steps)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def replan(
        self,
        original_plan: Plan,
        failed_step: PlanStep,
        error_context: str,
    ) -> Plan:
        """Generate a revised plan after a step failure.

        Args:
            original_plan: The plan that was being executed when the failure
                occurred.  Used to extract the goal and completed steps.
            failed_step: The step that failed.
            error_context: The error message or description of the failure.

        Returns:
            A new :class:`~rex.autonomy.models.Plan` containing steps that
            pick up where the original plan left off.

        Raises:
            ValueError: If the LLM response cannot be parsed.
        """
        # If a planner is configured, use plan_with_alternatives to generate
        # a pool of alternatives on the first call, then return them in order.
        if self._planner is not None:
            if not self._alternatives:
                logger.debug(
                    "Replanner: generating alternatives for goal=%r via plan_with_alternatives",
                    original_plan.goal,
                )
                self._alternatives = self._planner.plan_with_alternatives(original_plan.goal, {})
                self._alt_index = 0

            if self._alt_index < len(self._alternatives):
                n = self._alt_index + 1
                plan = self._alternatives[self._alt_index]
                self._alt_index += 1
                logger.info(
                    "Trying alternative plan %d/2 for goal '%s'",
                    n,
                    original_plan.goal,
                )
                return plan

        # Fall back to single-LLM replan (original behaviour).
        prompt = self._build_replan_prompt(original_plan, failed_step, error_context)
        logger.debug(
            "Replanner: sending replan prompt for goal=%r after step %s failed",
            original_plan.goal,
            failed_step.id,
        )

        backend = self._get_backend()
        raw = backend.generate(messages=[{"role": "user", "content": prompt}])

        logger.debug("Replanner: raw LLM response: %r", raw)
        return self._parse_response(raw, original_plan.goal)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "Replanner",
    "ReplannerBackend",
]
