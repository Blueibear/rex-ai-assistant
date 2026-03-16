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
from typing import Any, Literal

from rex.autonomy.llm_planner import LLMPlanner, ToolDefinition
from rex.autonomy.models import Plan, PlannerProtocol

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
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "create_planner",
    "run",
    "PlannerKey",
]
