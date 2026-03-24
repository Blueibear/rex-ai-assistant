"""EmailTriageEngine: LLM-based priority scoring and categorization for emails.

Scores each :class:`~rex.integrations.models.EmailMessage` with a priority
level (``low`` / ``medium`` / ``high`` / ``critical``) and a category tag
(e.g. ``action-required``, ``newsletter``, ``receipt``, ``personal``).

Results are cached in-memory (keyed by message ``id``) to avoid re-scoring
unchanged messages within the same process lifetime.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol, cast, runtime_checkable

from rex.integrations.models import EmailMessage, PriorityLevel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TriageBackend(Protocol):
    """Structural protocol for any object that can generate text."""

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 128,
    ) -> str:
        """Return a generated text response."""
        ...


# ---------------------------------------------------------------------------
# EmailTriageEngine
# ---------------------------------------------------------------------------

_VALID_PRIORITIES: set[str] = {"low", "medium", "high", "critical"}

_VALID_CATEGORIES: set[str] = {
    "action-required",
    "newsletter",
    "receipt",
    "personal",
    "notification",
    "social",
    "promotion",
    "spam",
    "update",
    "other",
}


class EmailTriageEngine:
    """Score and categorize email messages using an LLM backend.

    Args:
        backend: Optional :class:`TriageBackend` to use for scoring.  When
            ``None``, the engine lazily imports :class:`rex.llm.LanguageModel`
            on first use.
    """

    def __init__(self, backend: TriageBackend | None = None) -> None:
        self._backend = backend
        # Cache keyed by message id → (priority, category)
        self._cache: dict[str, tuple[PriorityLevel, str]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def triage(self, messages: list[EmailMessage]) -> list[EmailMessage]:
        """Return *messages* with the ``priority`` field populated.

        Messages whose ``id`` is already in the session cache are returned
        immediately without calling the LLM again.

        Args:
            messages: List of email messages to score.

        Returns:
            The same list with ``priority`` mutated in-place (returns new
            ``EmailMessage`` copies via Pydantic ``model_copy``).
        """
        result: list[EmailMessage] = []
        for msg in messages:
            priority, _category = self._score(msg)
            result.append(msg.model_copy(update={"priority": priority}))
        return result

    def triage_with_categories(
        self, messages: list[EmailMessage]
    ) -> list[tuple[EmailMessage, str]]:
        """Return *(message_with_priority, category)* pairs.

        Args:
            messages: List of email messages to score.

        Returns:
            List of (updated EmailMessage, category string) tuples.
        """
        result: list[tuple[EmailMessage, str]] = []
        for msg in messages:
            priority, category = self._score(msg)
            result.append((msg.model_copy(update={"priority": priority}), category))
        return result

    def clear_cache(self) -> None:
        """Clear the in-memory triage cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score(self, msg: EmailMessage) -> tuple[PriorityLevel, str]:
        """Return (priority, category) for *msg*, using cache when available."""
        if msg.id in self._cache:
            logger.debug("Triage cache hit for message id=%s", msg.id)
            return self._cache[msg.id]

        priority, category = self._call_llm(msg)
        self._cache[msg.id] = (priority, category)
        return priority, category

    def _build_prompt(self, msg: EmailMessage) -> str:
        snippet = (msg.body_text[:300] + "…") if len(msg.body_text) > 300 else msg.body_text
        return (
            "You are an email triage assistant. Given the following email details, "
            "return a JSON object with two fields:\n"
            '  "priority": one of "low", "medium", "high", "critical"\n'
            '  "category": one of "action-required", "newsletter", "receipt", '
            '"personal", "notification", "social", "promotion", "spam", "update", "other"\n\n'
            "Rules:\n"
            '- "critical": immediate action required, security alerts, outages\n'
            '- "high": action required within the day, important deadlines\n'
            '- "medium": should read soon, FYI items\n'
            '- "low": newsletters, receipts, automated notifications\n\n'
            f"From: {msg.sender}\n"
            f"Subject: {msg.subject}\n"
            f"Snippet: {snippet}\n\n"
            "Respond with ONLY valid JSON, no markdown fences."
        )

    def _call_llm(self, msg: EmailMessage) -> tuple[PriorityLevel, str]:
        """Call the LLM backend and parse the triage response."""
        backend = self._get_backend()
        prompt = self._build_prompt(msg)
        try:
            raw = backend.generate(
                [{"role": "user", "content": prompt}],
                max_tokens=64,
            )
            return self._parse_response(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Triage LLM call failed for message id=%s: %s — defaulting to low/other",
                msg.id,
                exc,
            )
            return "low", "other"

    def _parse_response(self, raw: str) -> tuple[PriorityLevel, str]:
        """Parse the LLM JSON response into (priority, category)."""
        text = raw.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(line for line in lines if not line.startswith("```")).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Triage response is not valid JSON: %r", raw[:200])
            return "low", "other"

        if not isinstance(data, dict):
            logger.warning("Triage response is not a JSON object: %r", raw[:200])
            return "low", "other"

        priority_raw = str(data.get("priority", "low")).lower()
        category_raw = str(data.get("category", "other")).lower()

        priority = cast(PriorityLevel, priority_raw) if priority_raw in _VALID_PRIORITIES else "low"
        category = category_raw if category_raw in _VALID_CATEGORIES else "other"

        return priority, category

    def _get_backend(self) -> TriageBackend:
        if self._backend is not None:
            return self._backend
        try:
            from rex.llm import LanguageModel

            self._backend = LanguageModel()
            return self._backend
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "No triage backend provided and LanguageModel could not be loaded."
            ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["EmailTriageEngine", "TriageBackend"]
