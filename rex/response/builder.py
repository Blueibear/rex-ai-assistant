"""Response assembly for assistant replies (US-017).

Extracted from ``rex.assistant.Assistant.generate_reply``.  Handles
response cache lookup/write, TTS text normalization, suggestion
generation, and follow-up extraction so that ``generate_reply`` reads
as a thin orchestration spec.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTS cleaning helpers
# ---------------------------------------------------------------------------

_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_BOLD_ITALIC = re.compile(r"\*{1,3}(.+?)\*{1,3}", re.DOTALL)
_MD_INLINE_CODE = re.compile(r"`{1,3}(.+?)`{1,3}", re.DOTALL)
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _clean_for_tts(text: str) -> str:
    """Strip common markdown formatting for TTS readback."""
    text = _MD_HEADER.sub("", text)
    text = _MD_BOLD_ITALIC.sub(r"\1", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    return text.strip()


# ---------------------------------------------------------------------------
# FinalResponse
# ---------------------------------------------------------------------------


@dataclass
class FinalResponse:
    """Fully assembled response returned by :class:`ResponseBuilder`."""

    text: str
    tts_text: str
    suggestions: list[str] = field(default_factory=list)
    followups: list[str] = field(default_factory=list)
    cache_hit: bool = False


# ---------------------------------------------------------------------------
# ResponseBuilder
# ---------------------------------------------------------------------------


class ResponseBuilder:
    """Assemble the final response from an action result.

    Handles (in order):

    1. Response cache write (``rex.response_cache.ResponseCache``)
    2. TTS text normalization (strip markdown formatting)
    3. Suggestion surfacing (``rex.suggestions.engine.SuggestionEngine``)
    4. Follow-up extraction (``rex.followup_engine.FollowupEngine``)

    Args:
        settings:         App config/settings object.
        response_cache:   Optional :class:`~rex.response_cache.ResponseCache` instance.
        suggestion_engine: Optional :class:`~rex.suggestions.engine.SuggestionEngine`.
        followup_engine:  Optional follow-up engine instance.
    """

    def __init__(
        self,
        *,
        settings: Any = None,
        response_cache: Any = None,
        suggestion_engine: Any = None,
        followup_engine: Any = None,
    ) -> None:
        self._settings = settings
        self._cache = response_cache
        self._suggestion_engine = suggestion_engine
        self._followup_engine = followup_engine

    # ------------------------------------------------------------------
    # Cache lookup (pre-dispatch fast path)
    # ------------------------------------------------------------------

    def check_cache(self, transcript: str, *, user_id: str | None = None) -> str | None:
        """Return a cached response for *transcript*, or ``None`` on miss/bypass.

        Delegates to :meth:`~rex.response_cache.ResponseCache.get`.  When
        *user_id* is given the lookup is confined to that user's cache
        partition so one user never receives another user's cached answer.
        Returns ``None`` when no cache is configured.
        """
        if self._cache is None:
            return None
        if user_id is None:
            return self._cache.get(transcript)  # type: ignore[no-any-return]
        return self._cache.get(transcript, user_id=user_id)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def build(
        self,
        action_result: Any,
        context: Any,
        *,
        transcript: str = "",
        user_id: str | None = None,
    ) -> FinalResponse:
        """Build a :class:`FinalResponse` from an :class:`~rex.actions.dispatcher.ActionResult`.

        Args:
            action_result: Result from :class:`~rex.actions.dispatcher.ActionDispatcher`.
            context:       :class:`~rex.context.builder.ContextPackage` from the context builder.
            transcript:    The original user transcript; used as the cache key for the PUT.
            user_id:       Owner of the cache entry; confines the PUT to that
                           user's partition (issue #303).

        Returns:
            A fully populated :class:`FinalResponse`.
        """
        text = action_result.response
        tts_text = _clean_for_tts(text)
        suggestions = self._get_suggestions(user_id)
        followups = self._get_followups()

        # Cache PUT: store result so identical future queries skip the LLM.
        if self._cache is not None and transcript:
            if user_id is None:
                self._cache.put(transcript, text)
            else:
                self._cache.put(transcript, text, user_id=user_id)

        return FinalResponse(
            text=text,
            tts_text=tts_text,
            suggestions=suggestions,
            followups=followups,
            cache_hit=False,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_suggestions(self, user_id: str | None = None) -> list[str]:
        """Return *user_id*'s pending suggestion text from the suggestion engine.

        Pending suggestions are per-user (issue #303); without a *user_id* no
        suggestion is surfaced (fail closed).
        """
        engine = self._suggestion_engine
        if engine is None or not user_id:
            return []
        try:
            spoken = engine.pending_spoken_text(user_id)
            if spoken:
                return [str(spoken)]
        except Exception as exc:
            logger.debug("Failed to get suggestions from engine: %s", exc)
        return []

    def _get_followups(self) -> list[str]:
        """Return formatted follow-up prompts from the follow-up engine."""
        engine = self._followup_engine
        if engine is None:
            return []
        if not hasattr(engine, "format_followups"):
            return []
        try:
            followups_text = engine.format_followups()
            if followups_text:
                return [str(followups_text)]
        except Exception as exc:
            logger.debug("format_followups failed: %s", exc)
        return []


__all__ = ["FinalResponse", "ResponseBuilder", "_clean_for_tts"]
