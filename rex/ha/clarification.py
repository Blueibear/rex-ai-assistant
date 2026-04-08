"""Clarification system for ambiguous Home Assistant commands."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .device_aliases import AliasResolver

# Commands where the "device" is a pronoun (missing context)
_PRONOUN_DEVICE_PATTERN = re.compile(
    r"\bturn\s+(?:the\s+)?(?:it|this|that)\s+(?:on|off)\b"
    r"|\b(?:turn\s+(?:on|off)|activate|lock|unlock|start|run)\s+(?:the\s+)?(?:it|this|that)\b",
    re.IGNORECASE,
)

# Patterns to extract the device name from a command transcript
_ENTITY_EXTRACT_PATTERNS = [
    re.compile(r"\bturn\s+on\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)", re.IGNORECASE),
    re.compile(r"\bturn\s+off\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)", re.IGNORECASE),
    re.compile(r"\b(?:activate|start|run)\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)", re.IGNORECASE),
    re.compile(r"\b(?:lock|unlock)\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+)", re.IGNORECASE),
    re.compile(r"\bset\s+(?:the\s+)?(?P<entity>[a-z0-9\s]+?)\s+to\s+\d", re.IGNORECASE),
]

# Two candidates within this confidence spread are considered equally plausible
_AMBIGUITY_DELTA = 0.15


def _extract_entity_query(transcript: str) -> str | None:
    """Return the device name token from *transcript*, or ``None`` if not found."""
    for pattern in _ENTITY_EXTRACT_PATTERNS:
        m = pattern.search(transcript)
        if m:
            return m.group("entity").strip()
    return None


class ClarificationHandler:
    """Detects ambiguous or incomplete HA commands and returns clarification questions.

    Two scenarios trigger a clarification response:

    1. **Missing device** – the transcript uses a pronoun ("turn it on") where
       a specific device name is required.
    2. **Ambiguous entity** – the device name matches multiple aliases with
       similar confidence (within ``_AMBIGUITY_DELTA``).
    """

    def __init__(self, resolver: AliasResolver | None = None) -> None:
        self._resolver = resolver

    def check(self, transcript: str) -> str | None:
        """Return a clarification question if the transcript is ambiguous, else ``None``."""
        # Check for pronoun-based missing device first (fast path, no resolver needed)
        if _PRONOUN_DEVICE_PATTERN.search(transcript):
            return "Which device would you like to control?"

        # Check for ambiguous entity name resolution
        if self._resolver is not None:
            entity_query = _extract_entity_query(transcript)
            if entity_query:
                candidates = self._resolver.resolve_all(entity_query)
                if len(candidates) >= 2:
                    top_alias, _top_entity, top_conf = candidates[0]
                    second_alias, _second_entity, second_conf = candidates[1]
                    if top_conf - second_conf <= _AMBIGUITY_DELTA:
                        return f"Did you mean {top_alias} or {second_alias}?"

        return None
