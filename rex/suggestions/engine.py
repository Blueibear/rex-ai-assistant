"""Proactive suggestion engine — surfaces one automation suggestion per session.

Workflow:
1. Caller passes a list of detected patterns (from :func:`detect_patterns`).
2. :meth:`SuggestionEngine.get_suggestion` returns ``(key, spoken_text)`` for
   the first eligible pattern, or ``None`` if nothing is due.
3. The assistant speaks *spoken_text* and waits for the user to reply.
4. On "yes" → :meth:`handle_yes` saves the automation entry.
   On "no thanks" → :meth:`handle_dismiss` records the dismissal so the same
   pattern is skipped for the next 30 days.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DISMISS_WINDOW_DAYS: int = 30
_DEFAULT_DISMISSED_PATH = Path("data/dismissed_suggestions.json")
_DEFAULT_AUTOMATIONS_PATH = Path("data/automations.json")

# Lowercased words/phrases that count as acceptance
_ACCEPT_WORDS: frozenset[str] = frozenset(
    {"yes", "yeah", "yep", "sure", "ok", "okay", "do it", "automate it", "please do"}
)
# Lowercased words/phrases that count as dismissal
_DISMISS_WORDS: frozenset[str] = frozenset(
    {"no", "no thanks", "nope", "not now", "skip", "dismiss", "cancel", "don't"}
)


class SuggestionEngine:
    """Surfaces proactive automation suggestions, one per session.

    Args:
        dismissed_path: Path to JSON file that persists dismissed pattern keys
            with their dismissal timestamps.  Created on first write.
        automations_path: Path to JSON file that persists accepted automations.
            Created on first write.
    """

    def __init__(
        self,
        dismissed_path: Path | str | None = None,
        automations_path: Path | str | None = None,
    ) -> None:
        self._dismissed_path = Path(dismissed_path or _DEFAULT_DISMISSED_PATH)
        self._automations_path = Path(automations_path or _DEFAULT_AUTOMATIONS_PATH)
        # At most one suggestion per session
        self._suggested_this_session: bool = False
        # Stores the active pending suggestion while waiting for user response
        self._pending: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    @property
    def has_pending(self) -> bool:
        """True when a suggestion has been spoken and awaits a user response."""
        return self._pending is not None

    def is_dismissed(self, key: str, window_days: int = _DISMISS_WINDOW_DAYS) -> bool:
        """Return ``True`` if *key* was dismissed within *window_days* days."""
        dismissed = self._load_dismissed()
        ts = dismissed.get(key)
        if ts is None:
            return False
        age_days = (time.time() - float(ts)) / 86400.0
        return age_days < window_days

    def is_accept(self, transcript: str) -> bool:
        """Return ``True`` if *transcript* looks like a yes/accept response."""
        return transcript.strip().lower() in _ACCEPT_WORDS

    def is_dismiss(self, transcript: str) -> bool:
        """Return ``True`` if *transcript* looks like a no/dismiss response."""
        t = transcript.strip().lower()
        return t in _DISMISS_WORDS or t.startswith("no ")

    # ------------------------------------------------------------------
    # Suggestion lifecycle
    # ------------------------------------------------------------------

    def get_suggestion(
        self,
        patterns: list[dict[str, Any]],
    ) -> tuple[str, str] | None:
        """Return ``(key, spoken_text)`` for the first eligible pattern.

        Returns ``None`` when:
        - A suggestion was already made this session.
        - All patterns have been dismissed within the last 30 days.
        - *patterns* is empty.

        Side effect: marks *_suggested_this_session* and sets *_pending* so that
        a subsequent call to :meth:`handle_yes` or :meth:`handle_dismiss` knows
        which pattern is being answered.
        """
        if self._suggested_this_session or not patterns:
            return None

        for pattern in patterns:
            key = _pattern_key(pattern)
            if self.is_dismissed(key):
                continue
            spoken = _build_spoken_text(pattern)
            self._suggested_this_session = True
            self._pending = {
                "key": key,
                "spoken_text": spoken,
                "automation": pattern.get("suggested_automation", ""),
            }
            return key, spoken

        return None

    def handle_yes(self) -> str:
        """Accept the pending suggestion and persist the automation entry.

        Returns a confirmation string suitable for TTS.  No-op when there is no
        pending suggestion.
        """
        pending = self._pending
        if pending is None:
            return "No pending suggestion to accept."

        key = pending["key"]
        automation = pending["automation"]
        self._pending = None

        try:
            automations: list[dict[str, Any]] = []
            if self._automations_path.exists():
                raw = json.loads(self._automations_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    automations = raw
            automations.append(
                {
                    "key": key,
                    "automation": automation,
                    "created_at": time.time(),
                }
            )
            self._automations_path.parent.mkdir(parents=True, exist_ok=True)
            self._automations_path.write_text(json.dumps(automations, indent=2), encoding="utf-8")
            logger.info("Automation saved: %s", automation)
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.warning("Could not save automation: %s", exc)

        return "Great, I've set that up for you!"

    def handle_dismiss(self) -> str:
        """Dismiss the pending suggestion for 30 days.

        Returns a confirmation string suitable for TTS.  No-op when there is no
        pending suggestion.
        """
        pending = self._pending
        if pending is None:
            return "No pending suggestion."

        key = pending["key"]
        self._pending = None

        dismissed = self._load_dismissed()
        dismissed[key] = time.time()
        self._save_dismissed(dismissed)
        logger.info("Suggestion dismissed: %s", key)

        return "Got it, I won't suggest that again for a while."

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dismissed(self) -> dict[str, float]:
        try:
            if self._dismissed_path.exists():
                data = json.loads(self._dismissed_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {str(k): float(v) for k, v in data.items()}
        except Exception as exc:
            logger.warning("Could not load dismissed suggestions: %s", exc)
        return {}

    def _save_dismissed(self, dismissed: dict[str, float]) -> None:
        try:
            self._dismissed_path.parent.mkdir(parents=True, exist_ok=True)
            self._dismissed_path.write_text(json.dumps(dismissed, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not save dismissed suggestions: %s", exc)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _pattern_key(pattern: dict[str, Any]) -> str:
    """Derive a stable string key from a pattern result dict."""
    entity_id: str = pattern.get("entity_id", "")
    service: str = pattern.get("service", "")
    if not entity_id or not service:
        # Fall back to parsing suggested_automation
        # Format: "Automate: turn_on light.kitchen_ceiling daily at 07:00"
        parts = (pattern.get("suggested_automation") or "").split()
        if len(parts) >= 3:
            service = parts[1]
            entity_id = parts[2]
    return f"{entity_id}:{service}"


def _build_spoken_text(pattern: dict[str, Any]) -> str:
    """Build a natural-language suggestion from a pattern result dict."""
    entity_id: str = pattern.get("entity_id", "")
    service: str = pattern.get("service", "")
    start_hour = pattern.get("start_hour")

    if not entity_id or not service or start_hour is None:
        # Fall back to parsing suggested_automation
        automation = pattern.get("suggested_automation", "")
        parts = automation.split()
        # "Automate: turn_on light.kitchen_ceiling daily at 07:00"
        if len(parts) >= 5:
            service = parts[1]
            entity_id = parts[2]
            try:
                start_hour = int(parts[-1].split(":")[0])
            except (ValueError, IndexError):
                start_hour = 0

    # "light.kitchen_ceiling" → "kitchen ceiling"
    if "." in entity_id:
        entity_name = entity_id.split(".", 1)[1].replace("_", " ")
    else:
        entity_name = entity_id.replace("_", " ")

    # "turn_on" → "turn on"
    friendly_service = service.replace("_", " ")

    # 7 → "7am", 14 → "2pm", 0 → "midnight", 12 → "noon"
    hour_int = int(start_hour) if start_hour is not None else 0
    if hour_int == 0:
        time_label = "midnight"
    elif hour_int < 12:
        time_label = f"{hour_int}am"
    elif hour_int == 12:
        time_label = "noon"
    else:
        time_label = f"{hour_int - 12}pm"

    return (
        f"I noticed you {friendly_service} the {entity_name} at {time_label} most days. "
        f"Want me to automate that?"
    )
