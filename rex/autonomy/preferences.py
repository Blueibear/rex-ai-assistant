"""User preference profile and persistent store for the Rex autonomy engine.

Provides:
- :class:`UserPreferenceProfile` — a Pydantic model capturing patterns in how
  the user invokes Rex.
- :class:`PreferenceStore` — a JSON-backed store for loading and saving a
  :class:`UserPreferenceProfile`.

The backing file is created automatically at ``~/.rex/preferences.json`` on
first save.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_PREFS_PATH = Path.home() / ".rex" / "preferences.json"


# ---------------------------------------------------------------------------
# UserPreferenceProfile
# ---------------------------------------------------------------------------


class UserPreferenceProfile(BaseModel):
    """Captured patterns describing how a user invokes Rex.

    Attributes:
        preferred_autonomy_mode: Most-used autonomy mode (e.g. ``"manual"``,
            ``"supervised"``, ``"full-auto"``).
        preferred_model: Most-used LLM model identifier.
        common_goal_patterns: Frequently issued goal descriptions or keywords.
        active_hours: Hours of the day (0–23) during which the user is
            typically active.
        avg_budget_usd: Rolling average cost-per-plan in USD.
        last_updated: UTC datetime when the profile was last written.
    """

    preferred_autonomy_mode: str = "manual"
    preferred_model: str = ""
    common_goal_patterns: list[str] = Field(default_factory=list)
    active_hours: list[int] = Field(default_factory=list)
    avg_budget_usd: float = 0.0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# PreferenceStore
# ---------------------------------------------------------------------------


class PreferenceStore:
    """JSON-backed store for a :class:`UserPreferenceProfile`.

    Args:
        prefs_path: Path to the JSON file.  Defaults to
            ``~/.rex/preferences.json``.  The parent directory is created
            automatically on :meth:`save`.
    """

    def __init__(self, prefs_path: Path | None = None) -> None:
        self._path: Path = prefs_path or _DEFAULT_PREFS_PATH

    def load(self) -> UserPreferenceProfile:
        """Load the preference profile from disk.

        If the file does not exist or cannot be parsed, a default
        :class:`UserPreferenceProfile` is returned and a debug message is
        logged.

        Returns:
            A :class:`UserPreferenceProfile` (may be default).
        """
        if not self._path.exists():
            logger.debug("PreferenceStore: %s not found — returning defaults", self._path)
            return UserPreferenceProfile()
        try:
            raw = self._path.read_text(encoding="utf-8")
            return UserPreferenceProfile.model_validate_json(raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "PreferenceStore: failed to parse %s — returning defaults. Error: %s",
                self._path,
                exc,
            )
            return UserPreferenceProfile()

    def save(self, profile: UserPreferenceProfile) -> None:
        """Persist *profile* to disk.

        Args:
            profile: The :class:`UserPreferenceProfile` to write.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        logger.debug("PreferenceStore: saved profile to %s", self._path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PreferenceStore",
    "UserPreferenceProfile",
]
