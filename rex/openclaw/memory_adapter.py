"""OpenClaw memory adapter — US-P3-002.

Provides a thin adapter layer that routes Rex conversation-history operations
to OpenClaw's storage backend when available, falling back to Rex's own
file-based implementation (``rex.memory``) when OpenClaw is not installed.

Typical usage::

    from rex.openclaw.memory_adapter import MemoryAdapter

    adapter = MemoryAdapter()

    # Append a turn to a user's history
    adapter.append_entry("alice", {"role": "user", "text": "Hello!"})

    # Read recent history
    turns = adapter.load_recent("alice", limit=20)

    # Trim an in-memory history list to the configured window
    trimmed = adapter.trim_history(turns, limit=10)
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from rex.memory import (
    append_history_entry as _rex_append,
)
from rex.memory import (
    load_recent_history as _rex_load_recent,
)
from rex.memory import (
    trim_history as _rex_trim,
)

logger = logging.getLogger(__name__)


class MemoryAdapter:
    """Adapter between Rex conversation memory and OpenClaw storage.

    When ``openclaw`` is installed, write/read operations are delegated to
    OpenClaw's storage backend (stub — filled in once the API is confirmed,
    see PRD §8.3).  When OpenClaw is absent, all operations delegate directly
    to Rex's file-based ``rex.memory`` helpers.

    Args:
        memory_root: Override the default Rex memory root directory.  Passed
            through to the Rex fallback helpers.  Ignored when OpenClaw
            storage is active.
    """

    def __init__(self, memory_root: str | None = None) -> None:
        self._memory_root = memory_root
        logger.debug("MemoryAdapter: using Rex file-based storage")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trim_history(
        self,
        history: Iterable[dict[str, Any]],
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return the most recent *limit* entries from *history*.

        Delegates to OpenClaw's history-trimming primitive when available,
        otherwise calls :func:`rex.memory.trim_history`.

        Args:
            history: Iterable of conversation turn dicts.
            limit: Maximum number of turns to retain.  When ``None``, the
                configured ``max_memory_items`` setting is used.

        Returns:
            A list of at most *limit* turn dicts, most-recent last.
        """
        return _rex_trim(history, limit=limit)

    def append_entry(
        self,
        user_key: str,
        entry: dict[str, str],
        *,
        max_turns: int | None = None,
    ) -> None:
        """Persist a conversation turn for *user_key*.

        Delegates to OpenClaw's storage write API when available, otherwise
        calls :func:`rex.memory.append_history_entry`.

        Args:
            user_key: Identifier for the user whose history is being updated.
            entry: Dict with ``"role"`` and ``"text"`` keys (required).
                A ``"timestamp"`` key is added automatically if absent.
            max_turns: Override for the maximum retained turns.  When
                ``None``, the configured ``memory_max_turns`` setting applies.
        """
        kwargs: dict[str, Any] = {}
        if self._memory_root is not None:
            kwargs["memory_root"] = self._memory_root
        if max_turns is not None:
            kwargs["max_turns"] = max_turns

        _rex_append(user_key, entry, **kwargs)

    def load_recent(
        self,
        user_key: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """Return the most recent conversation history for *user_key*.

        Delegates to OpenClaw's storage read API when available, otherwise
        calls :func:`rex.memory.load_recent_history`.

        Args:
            user_key: Identifier for the user whose history is being read.
            limit: Maximum number of turns to return.  When ``None``, all
                stored turns are returned.

        Returns:
            List of turn dicts (``{"role": ..., "text": ..., ...}``), oldest
            first.  Returns an empty list when no history exists.
        """
        kwargs: dict[str, Any] = {}
        if self._memory_root is not None:
            kwargs["memory_root"] = self._memory_root
        if limit is not None:
            kwargs["limit"] = limit

        return _rex_load_recent(user_key, **kwargs)
