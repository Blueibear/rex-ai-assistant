"""Autonomy dispatch entry point for Rex.

:func:`create_planner` returns either an :class:`~rex.autonomy.llm_planner.LLMPlanner`
(default) or a :class:`~rex.autonomy.rule_planner.RulePlanner` depending on the
``planner`` key in ``config/autonomy.json``.

:func:`run` is the high-level convenience function — build a planner and call
``.plan(goal, context)`` in one shot.

Configuration
-------------
Add a ``"planner"`` key to ``config/autonomy.json``:

.. code-block:: json

    {
      "planner": "llm"
    }

Valid values are ``"llm"`` (default) and ``"rule"``.  Any other value logs a
warning and falls back to ``"llm"``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Literal

from rex.autonomy.llm_planner import LLMPlanner, ToolDefinition
from rex.autonomy.models import Plan, PlannerProtocol, PlanStatus, PlanStep, StepStatus
from rex.autonomy.replanner import Replanner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_AUTONOMY_CONFIG = Path("config/autonomy.json")
PlannerKey = Literal["llm", "rule"]
_VALID_PLANNER_KEYS: frozenset[str] = frozenset({"llm", "rule"})


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _load_planner_key(config_path: Path | None = None) -> PlannerKey:
    """Read the ``planner`` key from the autonomy config file.

    Args:
        config_path: Override path to ``autonomy.json``.  Defaults to
            ``config/autonomy.json``.

    Returns:
        ``"llm"`` or ``"rule"`` — always a valid :data:`PlannerKey`.
    """
    path = config_path or _DEFAULT_AUTONOMY_CONFIG
    raw_key: str = "llm"

    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                data: Any = json.load(fh)
            raw_key = str(data.get("planner", "llm")).lower()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "runner: could not read planner key from %s — using 'llm'. Error: %s",
                path,
                exc,
            )

    if raw_key not in _VALID_PLANNER_KEYS:
        logger.warning(
            "runner: unknown planner key %r in %s — falling back to 'llm'",
            raw_key,
            path,
        )
        raw_key = "llm"

    return raw_key  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_planner(
    tools: list[ToolDefinition] | None = None,
    *,
    planner_key: PlannerKey | None = None,
    config_path: Path | None = None,
) -> PlannerProtocol:
    """Instantiate the planner selected by config (or *planner_key*).

    Args:
        tools: Tool definitions exposed to the LLM planner.  Ignored by the
            rule-based planner.  Defaults to an empty list.
        planner_key: Override the config-file selection.  Pass ``"llm"`` or
            ``"rule"`` to skip config-file loading.
        config_path: Override path to ``autonomy.json`` when *planner_key* is
            not supplied.

    Returns:
        A :class:`~rex.autonomy.models.PlannerProtocol` implementation.
    """
    if tools is None:
        tools = []

    key: PlannerKey = planner_key or _load_planner_key(config_path)

    if key == "rule":
        from rex.autonomy.rule_planner import RulePlanner  # lazy import

        logger.info("runner: using RulePlanner (deprecated)")
        return RulePlanner()

    # Default: LLMPlanner
    logger.info("runner: using LLMPlanner")
    return LLMPlanner(tools=tools)


# ---------------------------------------------------------------------------
# High-level dispatch
# ---------------------------------------------------------------------------


def run(
    goal: str,
    context: dict[str, Any] | None = None,
    *,
    tools: list[ToolDefinition] | None = None,
    planner_key: PlannerKey | None = None,
    config_path: Path | None = None,
) -> Plan:
    """Plan and return a :class:`~rex.autonomy.models.Plan` for *goal*.

    This is the primary entry point for the autonomy engine.  It creates the
    appropriate planner (defaulting to LLMPlanner) and calls ``.plan()``.

    Args:
        goal: Natural-language description of what to achieve.
        context: Optional key/value context forwarded to the planner.
        tools: Tool definitions available to the planner.
        planner_key: Override planner selection (``"llm"`` or ``"rule"``).
        config_path: Override path to ``autonomy.json``.

    Returns:
        A non-empty :class:`~rex.autonomy.models.Plan`.
    """
    effective_context: dict[str, Any] = context or {}
    planner = create_planner(
        tools=tools,
        planner_key=planner_key,
        config_path=config_path,
    )
    logger.debug("runner: planning goal=%r with %s", goal, type(planner).__name__)
    return planner.plan(goal, effective_context)


# ---------------------------------------------------------------------------
# Plan executor
# ---------------------------------------------------------------------------


def execute_plan(
    plan: Plan,
    tools: dict[str, Callable[..., str]],
    *,
    replanner: Replanner | None = None,
    max_replan_attempts: int = 2,
) -> Plan:
    """Execute a :class:`~rex.autonomy.models.Plan` step by step.

    Each step's status is updated in real time as execution proceeds:
    ``pending`` → ``running`` → ``success`` (with *result*) or ``failed``
    (with *error*).

    When *replanner* is supplied, a failed step triggers a call to
    :meth:`~rex.autonomy.replanner.Replanner.replan`.  The revised steps
    replace ``plan.steps`` and execution restarts with the new steps.
    This repeats up to *max_replan_attempts* times; after that the plan is
    marked ``failed``.

    Without a *replanner* (the default), the original behaviour is preserved:
    all steps run, failures are recorded, and the plan is marked ``failed``
    or ``completed`` at the end.

    Args:
        plan: The plan to execute.  Its steps are mutated in place.
        tools: Mapping from tool name to a callable that accepts the step's
            ``args`` dict as keyword arguments and returns a string result.
        replanner: Optional :class:`~rex.autonomy.replanner.Replanner` to
            call when a step fails.  Defaults to ``None`` (no replanning).
        max_replan_attempts: Maximum number of replan cycles per run.
            Defaults to ``2``.

    Returns:
        The same *plan* object with updated statuses.
    """
    plan.status = PlanStatus.RUNNING
    replan_count = 0

    while True:
        any_failed = False
        first_failed_step: PlanStep | None = None

        for step in plan.steps:
            if step.status in (StepStatus.SKIPPED, StepStatus.SUCCESS):
                logger.debug("runner: step %s skipped or already succeeded", step.id)
                continue

            step.status = StepStatus.RUNNING
            logger.debug("runner: executing step %s (tool=%s)", step.id, step.tool)

            tool_fn = tools.get(step.tool)
            if tool_fn is None:
                step.status = StepStatus.FAILED
                step.error = f"Unknown tool: {step.tool!r}"
                logger.error("runner: step %s failed — %s", step.id, step.error)
                any_failed = True
                if first_failed_step is None:
                    first_failed_step = step
                continue

            try:
                step.result = tool_fn(**step.args)
                step.status = StepStatus.SUCCESS
                logger.debug("runner: step %s succeeded", step.id)
            except Exception as exc:  # noqa: BLE001
                step.status = StepStatus.FAILED
                step.error = str(exc)
                logger.error("runner: step %s failed — %s", step.id, exc)
                any_failed = True
                if first_failed_step is None:
                    first_failed_step = step

        if not any_failed:
            plan.status = PlanStatus.COMPLETED
            return plan

        # At least one step failed — attempt replanning if configured.
        if (
            replanner is not None
            and replan_count < max_replan_attempts
            and first_failed_step is not None
        ):
            replan_count += 1
            logger.info(
                "runner: step %s failed, replanning (attempt %d/%d)",
                first_failed_step.id,
                replan_count,
                max_replan_attempts,
            )
            try:
                new_plan = replanner.replan(
                    original_plan=plan,
                    failed_step=first_failed_step,
                    error_context=first_failed_step.error or "",
                )
                plan.steps = new_plan.steps
                continue
            except Exception as exc:  # noqa: BLE001
                logger.error("runner: replanning call failed — %s", exc)

        plan.status = PlanStatus.FAILED
        return plan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "create_planner",
    "execute_plan",
    "run",
    "PlannerKey",
    "Replanner",
]
