"""FeedbackAnalyzer for the Rex autonomy engine.

Distils the most recent execution history into a short plain-text summary that
the LLM planner can include as additional context so it learns from past runs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Protocol

from rex.autonomy.history import ExecutionRecord, HistoryStore
from rex.autonomy.models import StepStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class FeedbackBackend(Protocol):
    """Protocol for AI backends accepted by :class:`FeedbackAnalyzer`."""

    def generate(
        self,
        prompt: str | None = None,
        *,
        messages: Sequence[dict[str, str]] | None = None,
    ) -> str: ...  # pragma: no cover


# ---------------------------------------------------------------------------
# FeedbackAnalyzer
# ---------------------------------------------------------------------------


class FeedbackAnalyzer:
    """Summarises execution history into a short paragraph for the planner.

    Args:
        backend: AI backend (must implement :class:`FeedbackBackend`).
            If *None*, a :class:`~rex.llm_client.LanguageModel` is
            instantiated with default settings on first use.
    """

    def __init__(self, backend: FeedbackBackend | None = None) -> None:
        self._backend: FeedbackBackend | None = backend

    # ------------------------------------------------------------------
    # Lazy backend initialisation
    # ------------------------------------------------------------------

    def _get_backend(self) -> FeedbackBackend:
        if self._backend is None:
            from rex.llm_client import LanguageModel  # lazy import

            self._backend = LanguageModel()
        return self._backend

    # ------------------------------------------------------------------
    # Async bridge
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_recent(store: HistoryStore, n: int = 10) -> list[ExecutionRecord]:
        """Synchronously retrieve the *n* most recent records from *store*."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(store.recent(n=n))
        finally:
            loop.close()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(goal: str, records: list[ExecutionRecord]) -> str:
        """Build the LLM prompt from *goal* and *records*."""
        lines: list[str] = []
        for r in records:
            tools_used = ", ".join(s.tool for s in r.plan.steps)
            failed_tools = (
                ", ".join(s.tool for s in r.plan.steps if s.status == StepStatus.FAILED) or "none"
            )
            lines.append(
                f"- Goal: {r.goal!r} | Outcome: {r.outcome} | "
                f"Tools: {tools_used} | Failed tools: {failed_tools} | "
                f"Replans: {r.replan_count}"
                + (f" | Error: {r.error_summary}" if r.error_summary else "")
            )
        history_block = "\n".join(lines)
        return (
            "You are a planning analyst. Below is a recent execution history. "
            "Write a single concise paragraph (≤ 200 words) that summarises:\n"
            "1. Common failure patterns and which tools tend to fail.\n"
            "2. Which tools tend to succeed.\n"
            "3. Any prior goals similar to the current goal that succeeded.\n\n"
            f"## Current goal\n{goal}\n\n"
            f"## Execution history (newest first)\n{history_block}\n\n"
            "Respond with only the summary paragraph. No lists, no headings."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def summarize(self, goal: str, history_store: HistoryStore) -> str:
        """Return a short feedback summary derived from the recent history.

        Args:
            goal: The current planning goal (used to find similar prior goals).
            history_store: The :class:`~rex.autonomy.history.HistoryStore` to
                query.

        Returns:
            A plain-text summary paragraph (≤ 200 words), or an empty string
            if there are no history records.
        """
        records = self._fetch_recent(history_store, n=10)
        if not records:
            logger.debug("FeedbackAnalyzer: history is empty — returning empty summary")
            return ""

        prompt = self._build_prompt(goal, records)
        logger.debug("FeedbackAnalyzer: requesting summary for goal=%r", goal)

        backend = self._get_backend()
        summary = backend.generate(messages=[{"role": "user", "content": prompt}])

        logger.debug("FeedbackAnalyzer: summary length=%d words", len(summary.split()))
        return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FeedbackAnalyzer",
    "FeedbackBackend",
]
