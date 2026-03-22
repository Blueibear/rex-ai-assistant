# OPENCLAW-WRAP: This module will be wrapped around OpenClaw. Preserve public API.

"""Rule-based planner for the Rex autonomy engine.

.. deprecated::
    ``RulePlanner`` is superseded by :class:`~rex.autonomy.llm_planner.LLMPlanner`
    and retained only for backwards compatibility.  Use the LLM planner instead.
    This module will be removed in a future release.
"""

from __future__ import annotations

import logging
import uuid
import warnings
from typing import Any

from rex.autonomy.models import Plan, PlannerProtocol, PlanStep

logger = logging.getLogger(__name__)

_DEPRECATION_MSG = (
    "RulePlanner is deprecated and will be removed in a future release. "
    "Use LLMPlanner (planner='llm') instead."
)


class RulePlanner(PlannerProtocol):
    """Simple rule-based planner.

    .. deprecated::
        Use :class:`~rex.autonomy.llm_planner.LLMPlanner` instead.

    Produces a single-step plan that describes the goal as a ``no_op`` tool
    call.  This stub exists to allow gradual migration away from rule-based
    planning and to preserve any callers that reference this class directly.
    """

    def __init__(self) -> None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)

    def plan(self, goal: str, context: dict[str, Any]) -> Plan:
        """Return a minimal single-step plan for *goal*.

        Args:
            goal: Natural-language description of the goal.
            context: Ignored by the rule-based planner.

        Returns:
            A :class:`~rex.autonomy.models.Plan` with one ``no_op`` step.
        """
        logger.warning("RulePlanner is deprecated; upgrade to LLMPlanner. Goal: %r", goal)
        step = PlanStep(
            id="step-1",
            tool="no_op",
            args={},
            description=f"Rule-based stub for goal: {goal}",
        )
        return Plan(id=str(uuid.uuid4()), goal=goal, steps=[step])


__all__ = ["RulePlanner"]
