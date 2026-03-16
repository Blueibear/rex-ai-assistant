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

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Literal, cast

from rex.autonomy.feedback import FeedbackAnalyzer
from rex.autonomy.history import ExecutionRecord, HistoryStore
from rex.autonomy.llm_planner import LLMPlanner, ToolDefinition
from rex.autonomy.models import Plan, PlannerProtocol, PlanStatus, PlanStep, StepStatus
from rex.autonomy.replanner import Replanner
from rex.autonomy.retry import retry_step

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
    feedback_analyzer: FeedbackAnalyzer | None = None,
    history_store: HistoryStore | None = None,
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
        feedback_analyzer: Optional :class:`~rex.autonomy.feedback.FeedbackAnalyzer`
            used to generate a feedback summary from history before planning.
            Requires *history_store* to be supplied as well.
        history_store: Optional :class:`~rex.autonomy.history.HistoryStore` used
            by *feedback_analyzer* to fetch past execution records.

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

    feedback_summary = ""
    if feedback_analyzer is not None and history_store is not None:
        try:
            feedback_summary = feedback_analyzer.summarize(goal, history_store)
            logger.debug("runner: feedback summary length=%d chars", len(feedback_summary))
        except Exception as exc:  # noqa: BLE001
            logger.warning("runner: feedback summarize failed — %s", exc)

    if feedback_summary:
        try:
            return cast(LLMPlanner, planner).plan(
                goal, effective_context, feedback_summary=feedback_summary
            )
        except TypeError:
            logger.debug("runner: planner does not support feedback_summary — skipping")
    return planner.plan(goal, effective_context)


# ---------------------------------------------------------------------------
# History helpers
# ---------------------------------------------------------------------------


def _outcome_from_plan(plan: Plan) -> str:
    """Derive the OutcomeType string from a completed plan."""
    if plan.status == PlanStatus.COMPLETED:
        return "success"
    any_succeeded = any(s.status == StepStatus.SUCCESS for s in plan.steps)
    return "partial" if any_succeeded else "failed"


def _write_history(
    store: HistoryStore,
    plan: Plan,
    duration_s: float,
    replan_count: int,
) -> None:
    """Persist an :class:`ExecutionRecord` to *store* without raising."""
    failed_errors = [s.error for s in plan.steps if s.error]
    error_summary: str | None = "; ".join(failed_errors) if failed_errors else None
    record = ExecutionRecord(
        goal=plan.goal,
        plan=plan,
        outcome=_outcome_from_plan(plan),  # type: ignore[arg-type]
        duration_s=duration_s,
        replan_count=replan_count,
        error_summary=error_summary,
    )
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(store.append(record))
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("runner: history write failed — %s", exc)


# ---------------------------------------------------------------------------
# Plan executor
# ---------------------------------------------------------------------------


def execute_plan(
    plan: Plan,
    tools: dict[str, Callable[..., str]],
    *,
    replanner: Replanner | None = None,
    max_replan_attempts: int = 2,
    history_store: HistoryStore | None = None,
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
        history_store: Optional :class:`~rex.autonomy.history.HistoryStore` to
            write an :class:`~rex.autonomy.history.ExecutionRecord` after each
            run.  Write failures are logged as warnings and do not propagate.

    Returns:
        The same *plan* object with updated statuses.
    """
    plan.status = PlanStatus.RUNNING
    replan_count = 0
    _start = time.monotonic()

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
                _fn: Callable[..., str] = tool_fn
                _kw: dict[str, Any] = dict(step.args)

                def _call(_f: Callable[..., str] = _fn, _k: dict[str, Any] = _kw) -> str:
                    return _f(**_k)

                step.result = retry_step(_call, step.id)
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
            if history_store is not None:
                _write_history(history_store, plan, time.monotonic() - _start, replan_count)
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
        if history_store is not None:
            _write_history(history_store, plan, time.monotonic() - _start, replan_count)
        return plan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "create_planner",
    "execute_plan",
    "run",
    "HistoryStore",
    "PlannerKey",
    "Replanner",
    "retry_step",
]
