"""PreferenceLearner — updates :class:`UserPreferenceProfile` after each plan run.

After each successful execution the learner updates three rolling signals:

* ``active_hours`` — the hour-of-day bucket for the current execution is
  appended (or kept if already present) to record when the user is active.
* ``avg_budget_usd`` — a rolling average of per-plan spend is maintained using
  an exponentially-weighted moving average (α = 0.2).
* ``common_goal_patterns`` — the goal description from the execution record is
  prepended to the list.  Duplicates are removed (keeping the most-recent
  occurrence), and the list is capped at 20 entries (LRU eviction).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from rex.autonomy.history import ExecutionRecord
from rex.autonomy.preferences import UserPreferenceProfile

logger = logging.getLogger(__name__)

# Weight for exponentially-weighted moving average of avg_budget_usd.
_EMA_ALPHA: float = 0.2
# Maximum number of entries kept in common_goal_patterns.
_MAX_PATTERNS: int = 20


class PreferenceLearner:
    """Updates a :class:`UserPreferenceProfile` from an :class:`ExecutionRecord`.

    The learner is intentionally stateless — it reads from *record* and
    *profile* and returns a **new** profile object.  Callers are responsible
    for persisting the result via :class:`~rex.autonomy.preferences.PreferenceStore`.
    """

    def update(
        self,
        record: ExecutionRecord,
        profile: UserPreferenceProfile,
    ) -> UserPreferenceProfile:
        """Return an updated copy of *profile* based on *record*.

        Args:
            record: The :class:`~rex.autonomy.history.ExecutionRecord` from a
                completed plan run.
            profile: The current :class:`UserPreferenceProfile` to update.

        Returns:
            A new :class:`UserPreferenceProfile` with updated fields.
        """
        updated = profile.model_copy(deep=True)

        # 1. active_hours — append current hour if not already present.
        current_hour: int = datetime.now(UTC).hour
        if current_hour not in updated.active_hours:
            updated.active_hours.append(current_hour)
            logger.debug(
                "preference_learner: added active_hour=%d (total=%d)",
                current_hour,
                len(updated.active_hours),
            )

        # 2. avg_budget_usd — exponentially-weighted moving average.
        if updated.avg_budget_usd == 0.0:
            updated.avg_budget_usd = record.total_cost_usd
        else:
            updated.avg_budget_usd = (
                _EMA_ALPHA * record.total_cost_usd + (1.0 - _EMA_ALPHA) * updated.avg_budget_usd
            )
        logger.debug(
            "preference_learner: avg_budget_usd updated to %.6f",
            updated.avg_budget_usd,
        )

        # 3. common_goal_patterns — LRU dedup + cap at _MAX_PATTERNS.
        goal = record.goal.strip()
        if goal:
            # Remove existing occurrence so the new entry moves to the front.
            patterns = [p for p in updated.common_goal_patterns if p != goal]
            patterns.insert(0, goal)
            # Enforce maximum length (LRU eviction from the tail).
            updated.common_goal_patterns = patterns[:_MAX_PATTERNS]
            logger.debug(
                "preference_learner: common_goal_patterns updated (size=%d)",
                len(updated.common_goal_patterns),
            )

        updated.last_updated = datetime.now(UTC)
        return updated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["PreferenceLearner"]
