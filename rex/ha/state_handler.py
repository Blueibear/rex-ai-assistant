"""Handler that intercepts device-state questions and answers them via HA.

Detects natural-language queries like "is the kitchen light on?",
resolves the device name through :class:`~rex.ha.device_aliases.AliasResolver`,
queries the real-time state from Home Assistant, and returns a short
spoken answer.  Returns ``None`` when the transcript does not match so
normal LLM routing continues.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Patterns that indicate a device-state question
_STATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"\bis (?:the |my )?(.+?)\s+(on|off|open|closed|locked|unlocked|playing|paused)\??$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat(?:'s| is)(?: the)? (?:state|status)(?: of)?(?: the| my)? (.+?)\??$",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bcheck(?: the)?(?: status| state)? of(?: the| my)? (.+?)\??$",
        re.IGNORECASE,
    ),
]


def _extract_device_name(transcript: str) -> str | None:
    """Return the device name phrase extracted from *transcript*, or ``None``."""
    for pattern in _STATE_PATTERNS:
        m = pattern.search(transcript)
        if m:
            # Group 1 is always the device name portion
            return m.group(1).strip().rstrip("?").strip()
    return None


class DeviceStateHandler:
    """Answer real-time device-state questions using Home Assistant.

    Args:
        base_url: HA base URL (e.g. ``http://homeassistant.local:8123``).
        token: HA long-lived access token.
        aliases_path: Optional override for the device aliases file path.
    """

    def __init__(
        self,
        base_url: str | None,
        token: str | None,
        aliases_path: str | None = None,
    ) -> None:
        from .device_aliases import AliasResolver

        self._base_url = base_url or ""
        self._token = token or ""
        self._resolver = AliasResolver(aliases_path=aliases_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(self, transcript: str) -> str | None:
        """Return a spoken answer if *transcript* is a device-state question.

        Returns ``None`` when the transcript is not a state query so that
        normal LLM routing continues.
        """
        device_name = _extract_device_name(transcript)
        if device_name is None:
            return None

        resolved = self._resolver.resolve(device_name)
        if resolved is None:
            logger.debug("device_state_handler: no alias match for %r", device_name)
            return None

        entity_id, confidence = resolved
        logger.debug(
            "device_state_handler: %r -> %s (confidence=%.2f)",
            device_name,
            entity_id,
            confidence,
        )

        if not self._base_url or not self._token:
            return "Home Assistant is not set up."

        from .device_state import get_device_state

        state_info = get_device_state(entity_id, self._base_url, self._token)
        if state_info is None:
            return f"I couldn't find the device {device_name} in Home Assistant."

        friendly_name = state_info["attributes"].get("friendly_name") or device_name
        state = state_info["state"]
        return f"The {friendly_name} is {state}."
