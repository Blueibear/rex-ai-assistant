"""Follow-up engine for Rex AI Assistant.

This module integrates cues into conversations, making Rex feel more personal
by following up on calendar events, reminders, and other activities.

The follow-up engine:
- Injects cue prompts into conversations at the start
- Tracks which cues have been asked per session
- Rate-limits cue injection (default 1 per session)
- Generates cues from calendar events when conversations start

Usage:
    from rex.followup_engine import FollowupEngine, get_followup_engine

    engine = get_followup_engine()

    # At the start of a conversation session
    cue_prompt = engine.get_followup_prompt(user_id="default")
    if cue_prompt:
        # Include in the assistant's opening or first response
        print(f"Rex might ask: {cue_prompt}")

    # After the user responds to a cue
    engine.mark_current_cue_asked()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _get_config() -> dict[str, Any]:
    """Load followup configuration with safe defaults."""
    try:
        from rex.config_manager import load_config

        config = load_config()
        return config.get("conversation", {}).get("followups", {})
    except Exception:
        return {}


class FollowupEngine:
    """Engine for managing follow-up cue injection into conversations.

    Provides functionality to:
    - Get a follow-up prompt for a conversation session
    - Track cues asked within a session
    - Rate-limit cue injection
    - Generate cues from calendar events

    The engine maintains session state to avoid asking the same user
    multiple cues in a single conversation.

    Example:
        engine = FollowupEngine()
        engine.start_session("default")
        prompt = engine.get_followup_prompt("default")
        if prompt:
            # Use the prompt in conversation
            engine.mark_current_cue_asked()
    """

    def __init__(self) -> None:
        """Initialize the follow-up engine."""
        # Session state: tracks cues asked per user session
        self._session_cues_asked: dict[str, int] = {}
        self._current_cue_id: dict[str, Optional[str]] = {}

    def _is_enabled(self) -> bool:
        """Check if follow-ups are enabled in config."""
        config = _get_config()
        return config.get("enabled", True)

    def _max_per_session(self) -> int:
        """Get the maximum cues per session from config."""
        config = _get_config()
        return config.get("max_per_session", 1)

    def _lookback_hours(self) -> int:
        """Get the lookback hours from config."""
        config = _get_config()
        return config.get("lookback_hours", 72)

    def _expire_hours(self) -> int:
        """Get the expire hours from config."""
        config = _get_config()
        return config.get("expire_hours", 168)

    def start_session(self, user_id: str) -> None:
        """Start a new conversation session for a user.

        Resets the cue count for the session and generates new cues
        from calendar events.

        Args:
            user_id: The user ID starting the session.
        """
        self._session_cues_asked[user_id] = 0
        self._current_cue_id[user_id] = None

        # Generate cues from calendar if enabled
        if self._is_enabled():
            self._generate_calendar_cues(user_id)

    def _generate_calendar_cues(self, user_id: str) -> None:
        """Generate follow-up cues from recent calendar events.

        Args:
            user_id: The user ID to generate cues for.
        """
        try:
            from rex.calendar_service import get_calendar_service

            calendar = get_calendar_service()
            calendar.generate_followup_cues(
                user_id=user_id,
                lookback_hours=self._lookback_hours(),
                expire_hours=self._expire_hours(),
            )
        except Exception as e:
            logger.debug(f"Could not generate calendar cues: {e}")

    def get_followup_prompt(
        self,
        user_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        """Get a follow-up prompt for the user if available.

        Returns a cue prompt if:
        - Follow-ups are enabled
        - Rate limit not exceeded for session
        - Pending cues exist for the user

        The cue is NOT marked as asked until mark_current_cue_asked() is called.

        Args:
            user_id: The user ID to get a prompt for.
            now: Current time (defaults to UTC now).

        Returns:
            A follow-up prompt string, or None if no cue available.
        """
        if not self._is_enabled():
            return None

        # Check rate limit
        asked_count = self._session_cues_asked.get(user_id, 0)
        if asked_count >= self._max_per_session():
            return None

        try:
            from rex.cue_store import get_cue_store

            store = get_cue_store()
            pending = store.list_pending_cues(
                user_id=user_id,
                now=now,
                limit=1,
                window_hours=self._lookback_hours(),
            )

            if not pending:
                return None

            cue = pending[0]
            self._current_cue_id[user_id] = cue.cue_id
            return cue.prompt

        except Exception as e:
            logger.warning(f"Failed to get followup prompt: {e}")
            return None

    def get_followup_context(
        self,
        user_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> Optional[dict[str, Any]]:
        """Get follow-up context including prompt and metadata.

        Similar to get_followup_prompt but returns more context
        for advanced use cases.

        Args:
            user_id: The user ID to get context for.
            now: Current time (defaults to UTC now).

        Returns:
            A dict with 'prompt', 'cue_id', 'title', 'source_type', or None.
        """
        if not self._is_enabled():
            return None

        asked_count = self._session_cues_asked.get(user_id, 0)
        if asked_count >= self._max_per_session():
            return None

        try:
            from rex.cue_store import get_cue_store

            store = get_cue_store()
            pending = store.list_pending_cues(
                user_id=user_id,
                now=now,
                limit=1,
                window_hours=self._lookback_hours(),
            )

            if not pending:
                return None

            cue = pending[0]
            self._current_cue_id[user_id] = cue.cue_id
            return {
                "prompt": cue.prompt,
                "cue_id": cue.cue_id,
                "title": cue.title,
                "source_type": cue.source_type,
                "source_id": cue.source_id,
            }

        except Exception as e:
            logger.warning(f"Failed to get followup context: {e}")
            return None

    def mark_current_cue_asked(self, user_id: str) -> bool:
        """Mark the current cue as asked.

        Should be called after the follow-up prompt has been delivered
        to the user and they have responded.

        Args:
            user_id: The user ID whose cue to mark.

        Returns:
            True if a cue was marked, False otherwise.
        """
        cue_id = self._current_cue_id.get(user_id)
        if cue_id is None:
            return False

        try:
            from rex.cue_store import get_cue_store

            store = get_cue_store()
            if store.mark_asked(cue_id):
                self._session_cues_asked[user_id] = (
                    self._session_cues_asked.get(user_id, 0) + 1
                )
                self._current_cue_id[user_id] = None
                return True
            return False

        except Exception as e:
            logger.warning(f"Failed to mark cue as asked: {e}")
            return False

    def has_pending_cue(self, user_id: str) -> bool:
        """Check if there's a pending cue for the user.

        Args:
            user_id: The user ID to check.

        Returns:
            True if there's at least one pending cue.
        """
        if not self._is_enabled():
            return False

        try:
            from rex.cue_store import get_cue_store

            store = get_cue_store()
            pending = store.list_pending_cues(
                user_id=user_id,
                limit=1,
                window_hours=self._lookback_hours(),
            )
            return len(pending) > 0

        except Exception:
            return False

    def get_session_cues_asked(self, user_id: str) -> int:
        """Get the number of cues asked in the current session.

        Args:
            user_id: The user ID to check.

        Returns:
            Number of cues asked this session.
        """
        return self._session_cues_asked.get(user_id, 0)

    def inject_followup_into_prompt(
        self,
        user_id: str,
        base_prompt: str,
        *,
        now: Optional[datetime] = None,
    ) -> str:
        """Inject a follow-up cue into a system/assistant prompt.

        This is a convenience method for modifying prompts to include
        follow-up behavior.

        Args:
            user_id: The user ID.
            base_prompt: The original prompt/system message.
            now: Current time (defaults to UTC now).

        Returns:
            Modified prompt with follow-up instruction, or original if no cue.
        """
        followup = self.get_followup_prompt(user_id, now=now)
        if followup is None:
            return base_prompt

        followup_instruction = (
            f"\n\nEarly in this conversation, naturally ask the user: "
            f'"{followup}" '
            f"Make it feel like friendly small talk, not a checklist."
        )

        return base_prompt + followup_instruction


# Global instance
_followup_engine: Optional[FollowupEngine] = None


def get_followup_engine() -> FollowupEngine:
    """Get the global follow-up engine instance."""
    global _followup_engine
    if _followup_engine is None:
        _followup_engine = FollowupEngine()
    return _followup_engine


def set_followup_engine(engine: FollowupEngine) -> None:
    """Set the global follow-up engine instance (for testing)."""
    global _followup_engine
    _followup_engine = engine


__all__ = [
    "FollowupEngine",
    "get_followup_engine",
    "set_followup_engine",
]
