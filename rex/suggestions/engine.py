"""Proactive suggestion engine — surfaces one automation suggestion per session.

Workflow:
1. Caller passes a list of detected patterns (from :func:`detect_patterns`)
   together with the ``user_id`` whose command history produced them.
2. :meth:`SuggestionEngine.get_suggestion` returns ``(key, spoken_text)`` for
   the first eligible pattern, or ``None`` if nothing is due.
3. The assistant speaks *spoken_text* and waits for the user to reply.
4. On "yes" → :meth:`handle_yes` saves the automation entry.
   On "no thanks" → :meth:`handle_dismiss` records the dismissal so the same
   pattern is skipped for the next 30 days.

Per-user isolation (issue #303): all session state (pending suggestion,
one-per-session flag) and persisted dismissals are keyed by ``user_id`` so one
user can never see, accept, or dismiss another user's suggestion.  A missing
or invalid ``user_id`` fails closed: no suggestion is surfaced and no pending
state is consumed.

Persisted file formats:

- ``dismissed_suggestions.json`` — ``{"users": {"<user_id>": {"<key>": ts}}}``.
  The legacy flat format ``{"<key>": ts}`` predates per-user scoping and is
  read as belonging to the ``"default"`` user (never silently shared with
  other users).
- ``automations.json`` — list of entries; entries now carry a ``"user_id"``
  field recording who accepted the suggestion.  Legacy entries without the
  field are left untouched and treated as unattributed.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from rex.proactivity.models import ProactiveCandidate

logger = logging.getLogger(__name__)

_DISMISS_WINDOW_DAYS: int = 30
_DEFAULT_DISMISSED_PATH = Path("data/dismissed_suggestions.json")
_DEFAULT_AUTOMATIONS_PATH = Path("data/automations.json")

# Top-level key of the per-user dismissed-suggestions file format.  Pattern
# keys always contain a ":" (``"<entity_id>:<service>"``) so a bare "users"
# key can never collide with a legacy flat entry.
_DISMISSED_USERS_KEY = "users"

# User that legacy (pre-per-user) dismissal entries are attributed to.
_LEGACY_DISMISSED_USER = "default"

# Lowercased words/phrases that count as acceptance
_ACCEPT_WORDS: frozenset[str] = frozenset(
    {"yes", "yeah", "yep", "sure", "ok", "okay", "do it", "automate it", "please do"}
)
# Lowercased words/phrases that count as dismissal
_DISMISS_WORDS: frozenset[str] = frozenset(
    {"no", "no thanks", "nope", "not now", "skip", "dismiss", "cancel", "don't"}
)


class SuggestionEngine:
    """Surfaces proactive automation suggestions, one per user per session.

    All stateful methods take an explicit ``user_id`` and operate only on that
    user's slice of state, mirroring :class:`rex.followup_engine.FollowupEngine`.

    Args:
        dismissed_path: Path to JSON file that persists dismissed pattern keys
            per user with their dismissal timestamps.  Created on first write.
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
        # At most one suggestion per user per session, keyed by user_id
        self._suggested_this_session: dict[str, bool] = {}
        # Active pending suggestion awaiting a response, keyed by user_id
        self._pending: dict[str, dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Identity guard
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_user(user_id: str | None) -> str | None:
        """Return a validated user ID, or ``None`` to fail closed.

        Missing, empty, or malformed identities must never fall through to a
        default or another user's state (issue #303).
        """
        if not isinstance(user_id, str) or not user_id:
            return None
        # Lazy import: rex.identity loads rex.config.settings at module import
        # time; deferring keeps this module free of config side effects.
        from rex.identity import validate_user_id

        try:
            return validate_user_id(user_id)
        except ValueError:
            logger.warning("suggestions: ignoring invalid user_id %r", user_id)
            return None

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def has_pending(self, user_id: str | None) -> bool:
        """True when a suggestion was spoken to *user_id* and awaits their response."""
        uid = self._valid_user(user_id)
        return uid is not None and uid in self._pending

    def pending_spoken_text(self, user_id: str | None) -> str | None:
        """Return the spoken text of *user_id*'s pending suggestion, if any."""
        uid = self._valid_user(user_id)
        if uid is None:
            return None
        pending = self._pending.get(uid)
        if pending is None:
            return None
        spoken = pending.get("spoken_text")
        return str(spoken) if spoken else None

    def pending_contextual_text(self, user_id: str | None) -> str | None:
        """Return pending contextual proactive text only for its owning user."""
        uid = self._valid_user(user_id)
        if uid is None:
            return None
        pending = self._pending.get(uid)
        if pending is None or pending.get("kind") != "contextual":
            return None
        spoken = pending.get("spoken_text")
        return str(spoken) if spoken else None

    def is_dismissed(
        self,
        key: str,
        user_id: str | None,
        window_days: int = _DISMISS_WINDOW_DAYS,
    ) -> bool:
        """Return ``True`` if *user_id* dismissed *key* within *window_days* days.

        An invalid *user_id* fails closed by reporting the key as dismissed,
        so nothing is surfaced to an unidentified caller.
        """
        uid = self._valid_user(user_id)
        if uid is None:
            return True
        dismissed = self._load_dismissed(uid)
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
        user_id: str | None,
    ) -> tuple[str, str] | None:
        """Return ``(key, spoken_text)`` for the first eligible pattern.

        Returns ``None`` when:
        - *user_id* is missing or invalid (fail closed).
        - A suggestion was already made to *user_id* this session.
        - All patterns were dismissed by *user_id* within the last 30 days.
        - *patterns* is empty.

        Side effect: marks *user_id* in *_suggested_this_session* and stores
        their *_pending* entry so a subsequent :meth:`handle_yes` or
        :meth:`handle_dismiss` for the same user knows which pattern is being
        answered.
        """
        uid = self._valid_user(user_id)
        if uid is None:
            return None
        if self._suggested_this_session.get(uid) or not patterns:
            return None

        for pattern in patterns:
            key = _pattern_key(pattern)
            if self.is_dismissed(key, uid):
                continue
            spoken = _build_spoken_text(pattern)
            self._suggested_this_session[uid] = True
            self._pending[uid] = {
                "kind": "automation",
                "key": key,
                "spoken_text": spoken,
                "automation": pattern.get("suggested_automation", ""),
            }
            return key, spoken

        return None

    def get_contextual_suggestion(
        self,
        candidates: list[ProactiveCandidate] | tuple[ProactiveCandidate, ...],
        *,
        user_id: str | None,
    ) -> tuple[str, str] | None:
        """Surface one eligible contextual candidate using existing suppression state."""
        uid = self._valid_user(user_id)
        if uid is None or self._suggested_this_session.get(uid):
            return None
        ordered = sorted(candidates, key=lambda item: (-item.score, item.key))
        for candidate in ordered:
            if candidate.user_id != uid or self.is_dismissed(candidate.key, uid):
                continue
            self._suggested_this_session[uid] = True
            self._pending[uid] = {
                "kind": "contextual",
                "key": candidate.key,
                "spoken_text": candidate.spoken_text,
                "suggested_action": candidate.suggested_action,
            }
            return candidate.key, candidate.spoken_text
        return None

    def handle_yes(self, user_id: str | None) -> str:
        """Accept *user_id*'s pending suggestion and persist the automation entry.

        Returns a confirmation string suitable for TTS.  No-op when *user_id*
        is invalid or has no pending suggestion — another user's pending state
        is never consumed.
        """
        uid = self._valid_user(user_id)
        if uid is None:
            return "No pending suggestion to accept."
        pending = self._pending.pop(uid, None)
        if pending is None:
            return "No pending suggestion to accept."
        if pending.get("kind") == "contextual":
            return "Okay, noted."

        key = pending["key"]
        automation = pending["automation"]

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
                    "user_id": uid,
                    "created_at": time.time(),
                }
            )
            self._automations_path.parent.mkdir(parents=True, exist_ok=True)
            self._automations_path.write_text(json.dumps(automations, indent=2), encoding="utf-8")
            logger.info("Automation saved for user %s: %s", uid, automation)
        except Exception as exc:  # pragma: no cover - defensive I/O guard
            logger.warning("Could not save automation: %s", exc)

        return "Great, I've set that up for you!"

    def handle_dismiss(self, user_id: str | None) -> str:
        """Dismiss *user_id*'s pending suggestion for 30 days.

        Returns a confirmation string suitable for TTS.  No-op when *user_id*
        is invalid or has no pending suggestion — another user's pending state
        is never consumed, and the dismissal only applies to *user_id*.
        """
        uid = self._valid_user(user_id)
        if uid is None:
            return "No pending suggestion."
        pending = self._pending.pop(uid, None)
        if pending is None:
            return "No pending suggestion."

        key = pending["key"]

        dismissed = self._load_dismissed(uid)
        dismissed[key] = time.time()
        self._save_dismissed(uid, dismissed)
        logger.info("Suggestion dismissed by user %s: %s", uid, key)

        return "Got it, I won't suggest that again for a while."

    def reset_session(self, user_id: str | None) -> None:
        """Clear the one-per-session flag for *user_id* (e.g. on a new session)."""
        uid = self._valid_user(user_id)
        if uid is not None:
            self._suggested_this_session.pop(uid, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_dismissed_all(self) -> dict[str, dict[str, float]]:
        """Load the full per-user dismissal map, migrating the legacy format.

        Legacy flat files (``{"<key>": ts}``) predate per-user scoping and are
        attributed to the ``"default"`` user rather than shared across users.
        """
        try:
            if self._dismissed_path.exists():
                data = json.loads(self._dismissed_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    users = data.get(_DISMISSED_USERS_KEY)
                    if isinstance(users, dict):
                        return {
                            str(user): {str(k): float(v) for k, v in entries.items()}
                            for user, entries in users.items()
                            if isinstance(entries, dict)
                        }
                    if data:
                        legacy = {str(k): float(v) for k, v in data.items()}
                        return {_LEGACY_DISMISSED_USER: legacy}
        except Exception as exc:
            logger.warning("Could not load dismissed suggestions: %s", exc)
        return {}

    def _load_dismissed(self, user_id: str) -> dict[str, float]:
        return self._load_dismissed_all().get(user_id, {})

    def _save_dismissed(self, user_id: str, dismissed: dict[str, float]) -> None:
        try:
            all_users = self._load_dismissed_all()
            all_users[user_id] = dismissed
            self._dismissed_path.parent.mkdir(parents=True, exist_ok=True)
            self._dismissed_path.write_text(
                json.dumps({_DISMISSED_USERS_KEY: all_users}, indent=2),
                encoding="utf-8",
            )
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
