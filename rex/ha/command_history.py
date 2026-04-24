"""FIFO command history for Home Assistant undo support.

Stores the last N executed HA commands with timestamps so that a subsequent
"undo" request can reverse the most recent reversible action.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

# Services that can be reversed and their inverse
_INVERSE_SERVICES: dict[str, str] = {
    "turn_on": "turn_off",
    "turn_off": "turn_on",
    "lock": "unlock",
    "unlock": "lock",
    "open_cover": "close_cover",
    "close_cover": "open_cover",
}


@dataclass
class CommandEntry:
    """A single executed HA command stored for potential undo."""

    entity_id: str
    domain: str
    service: str
    data: dict[str, Any]
    description: str
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def inverse_service(self) -> str | None:
        """Return the service name that would reverse this command, or None."""
        return _INVERSE_SERVICES.get(self.service)


class CommandHistory:
    """FIFO queue of the last *max_size* executed HA commands.

    Args:
        max_size: Maximum number of commands to retain (default 5).
        undo_window: Seconds within which undo is allowed (default 30).
    """

    def __init__(self, max_size: int = 5, undo_window: float = 30.0) -> None:
        self._entries: deque[CommandEntry] = deque(maxlen=max_size)
        self.undo_window = undo_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        *,
        entity_id: str,
        domain: str,
        service: str,
        data: dict[str, Any],
        description: str,
    ) -> None:
        """Record a successfully executed command."""
        self._entries.append(
            CommandEntry(
                entity_id=entity_id,
                domain=domain,
                service=service,
                data=dict(data),
                description=description,
            )
        )

    def peek_undo_candidate(self) -> CommandEntry | None:
        """Return the most recent reversible command within the undo window.

        Returns ``None`` if there are no commands, the last command is not
        reversible, or it was executed more than *undo_window* seconds ago.
        """
        if not self._entries:
            return None
        entry = self._entries[-1]
        if entry.inverse_service is None:
            return None
        age = time.monotonic() - entry.timestamp
        if age >= self.undo_window:
            return None
        return entry

    def pop_undo_candidate(self) -> CommandEntry | None:
        """Like :meth:`peek_undo_candidate` but removes the entry on success."""
        candidate = self.peek_undo_candidate()
        if candidate is not None:
            self._entries.pop()
        return candidate

    def recent_entity_ids(self) -> list[str]:
        """Return entity IDs from stored history, oldest first."""
        return [e.entity_id for e in self._entries]

    def __len__(self) -> int:
        return len(self._entries)
