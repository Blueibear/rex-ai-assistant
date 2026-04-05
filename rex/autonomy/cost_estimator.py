"""CostEstimator for the Rex autonomy engine.

Provides a lightweight pre-execution cost estimate for a :class:`~rex.autonomy.models.Plan`
so callers can gate execution against a configurable budget.

The estimate is intentionally conservative (heuristic-based, not LLM-queried) to keep
the hot path fast and free from additional network calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import BaseModel

from rex.autonomy.models import Plan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing constants (defaults — can be overridden via CostEstimator.__init__)
# ---------------------------------------------------------------------------

# Approximate tokens per word (English average).
_TOKENS_PER_WORD: float = 1.33
# Overhead tokens per step (system prompt, planner context, etc.).
_BASE_TOKENS_PER_STEP: int = 100
# Claude Sonnet input price (USD / token) as of 2025.
_DEFAULT_LOW_PRICE_PER_TOKEN: float = 3.0 / 1_000_000
# Claude Sonnet output price (USD / token) — used as conservative upper bound.
_DEFAULT_HIGH_PRICE_PER_TOKEN: float = 15.0 / 1_000_000


# ---------------------------------------------------------------------------
# CostEstimate data model
# ---------------------------------------------------------------------------


class CostEstimate(BaseModel):
    """Heuristic cost estimate for a :class:`~rex.autonomy.models.Plan`.

    Attributes:
        low_usd: Lower-bound cost estimate in USD (based on input token pricing).
        high_usd: Upper-bound cost estimate in USD (based on output token pricing).
        step_count: Number of steps the estimate covers.
    """

    low_usd: float
    high_usd: float
    step_count: int


# ---------------------------------------------------------------------------
# CostEstimator
# ---------------------------------------------------------------------------


class CostEstimator:
    """Estimates the execution cost of a :class:`~rex.autonomy.models.Plan`.

    The estimate is derived from the number and description length of plan steps.
    It does **not** make any LLM calls.

    Args:
        low_price_per_token: Price per token for the lower bound (default: Claude
            Sonnet input price ≈ $3 / 1M tokens).
        high_price_per_token: Price per token for the upper bound (default: Claude
            Sonnet output price ≈ $15 / 1M tokens).
    """

    def __init__(
        self,
        *,
        low_price_per_token: float = _DEFAULT_LOW_PRICE_PER_TOKEN,
        high_price_per_token: float = _DEFAULT_HIGH_PRICE_PER_TOKEN,
    ) -> None:
        self._low_price = low_price_per_token
        self._high_price = high_price_per_token

    def estimate(self, plan: Plan) -> CostEstimate:
        """Return a heuristic cost estimate for *plan*.

        Each step's token count is estimated from its description word count plus
        a fixed per-step base overhead.  Totals are multiplied by the configured
        low and high per-token prices.

        Args:
            plan: The plan to estimate.

        Returns:
            A :class:`CostEstimate` with ``low_usd``, ``high_usd``, and
            ``step_count``.
        """
        total_tokens = 0
        for step in plan.steps:
            word_count = len(step.description.split())
            tokens = int(word_count * _TOKENS_PER_WORD) + _BASE_TOKENS_PER_STEP
            total_tokens += tokens

        low = total_tokens * self._low_price
        high = total_tokens * self._high_price
        estimate = CostEstimate(low_usd=low, high_usd=high, step_count=len(plan.steps))
        logger.debug(
            "CostEstimator: plan=%s steps=%d tokens≈%d low=$%.6f high=$%.6f",
            plan.id,
            estimate.step_count,
            total_tokens,
            low,
            high,
        )
        return estimate


# ---------------------------------------------------------------------------
# Default budget-exceeded prompt (used when no callback is supplied)
# ---------------------------------------------------------------------------


def _default_budget_prompt(estimate: CostEstimate, budget_usd: float) -> bool:
    """Ask the user via stdin whether to proceed past budget.

    Returns ``True`` if the user approves, ``False`` otherwise.
    """
    answer = input(
        f"Estimated cost: ${estimate.high_usd:.4f} (high) exceeds budget "
        f"${budget_usd:.4f}. Proceed? [y/N]: "
    )
    return answer.strip().lower() == "y"


# ---------------------------------------------------------------------------
# Runner integration helper
# ---------------------------------------------------------------------------


def check_budget(
    estimator: CostEstimator,
    plan: Plan,
    budget_usd: float,
    on_budget_exceeded: Callable[[CostEstimate], bool] | None = None,
) -> bool:
    """Estimate plan cost and gate on *budget_usd*.

    Args:
        estimator: The :class:`CostEstimator` to use.
        plan: Plan to estimate.
        budget_usd: Maximum allowed high-end cost.  Pass ``0`` to skip the
            check (unlimited budget).
        on_budget_exceeded: Optional callback ``(estimate) -> bool`` called
            when ``estimate.high_usd > budget_usd``.  Return ``True`` to
            proceed, ``False`` to abort.  Defaults to a stdin prompt.

    Returns:
        ``True`` if execution should proceed, ``False`` if it should be
        aborted.
    """
    if budget_usd <= 0:
        return True

    estimate = estimator.estimate(plan)
    if estimate.high_usd <= budget_usd:
        logger.debug(
            "CostEstimator: estimate $%.6f ≤ budget $%.6f — proceeding",
            estimate.high_usd,
            budget_usd,
        )
        return True

    logger.info(
        "CostEstimator: estimate $%.6f exceeds budget $%.6f — requesting approval",
        estimate.high_usd,
        budget_usd,
    )
    if on_budget_exceeded is not None:
        return on_budget_exceeded(estimate)
    return _default_budget_prompt(estimate, budget_usd)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "CostEstimate",
    "CostEstimator",
    "check_budget",
]
