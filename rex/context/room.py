"""Room context module for scoping commands to the user's current room."""

from __future__ import annotations


class RoomContext:
    """Tracks which room the user is currently in.

    Priority order for ``current_room``:
    1. explicit — set directly by caller (highest priority)
    2. speaker_origin — derived from audio input device or MQTT topic
    3. last_active — most recently active UI context
    4. config_default — value from application config (lowest priority)
    """

    def __init__(self, config_default: str | None = None) -> None:
        self._explicit: str | None = None
        self._speaker_origin: str | None = None
        self._last_active: str | None = None
        self._config_default: str | None = config_default

    # -- setters ----------------------------------------------------------

    def set_explicit(self, room: str | None) -> None:
        """Set an explicit room override (highest priority)."""
        self._explicit = room

    def clear_explicit(self) -> None:
        """Clear the explicit override so lower-priority sources take effect."""
        self._explicit = None

    def set_speaker_origin(self, room: str | None) -> None:
        """Set the room derived from the speaker/audio-input origin."""
        self._speaker_origin = room

    def set_last_active(self, room: str | None) -> None:
        """Set the most recently active UI context room."""
        self._last_active = room

    def set_config_default(self, room: str | None) -> None:
        """Update the config-level default room."""
        self._config_default = room

    # -- read-only resolution ---------------------------------------------

    @property
    def current_room(self) -> str | None:
        """Return the highest-priority non-None room name."""
        for source in (
            self._explicit,
            self._speaker_origin,
            self._last_active,
            self._config_default,
        ):
            if source is not None:
                return source
        return None
