"""OpenClaw memory adapter — US-P3-002.

Provides a thin adapter layer that manages Rex conversation-history using
Rex's own file-based implementation (``rex.memory``) as the source of truth.

Session persistence with OpenClaw is achieved automatically: ``RexAgent.respond()``
sends a stable ``user`` field in every ``/v1/chat/completions`` request, which
causes OpenClaw to maintain its own per-channel session state server-side.
No direct HTTP calls are needed from this adapter — Rex keeps local history
for voice/text interactions; OpenClaw keeps its own per-channel history.

Typical usage::

    from rex.openclaw.memory_adapter import MemoryAdapter

    adapter = MemoryAdapter()

    # Append a turn to a user's history (always writes locally)
    adapter.append_entry("alice", {"role": "user", "text": "Hello!"})

    # Read recent history (always reads locally)
    turns = adapter.load_recent("alice", limit=20)

    # Trim an in-memory history list to the configured window (local only)
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

    Rex is the local source of truth for voice conversation history.
    All read/write operations use Rex's file-based ``rex.memory`` helpers.

    OpenClaw session persistence is handled automatically by ``RexAgent.respond()``,
    which sends a stable ``user`` field in every ``/v1/chat/completions`` request.
    This causes OpenClaw to maintain its own per-channel session state server-side
    without any explicit storage API calls from this adapter.

    Args:
        memory_root: Override the default Rex memory root directory.  Passed
            through to the Rex file-based helpers.
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

        Always operates locally via :func:`rex.memory.trim_history`.
        No HTTP calls are made.

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
        """Persist a conversation turn for *user_key* in local Rex memory.

        Always writes to Rex's local file-based storage via
        :func:`rex.memory.append_history_entry`.

        OpenClaw session state is updated automatically on the next
        ``RexAgent.respond()`` call: the ``user`` field sent in
        ``/v1/chat/completions`` requests causes OpenClaw to maintain its own
        per-channel session server-side (dual-write: local now + OpenClaw on
        next completion).  No explicit HTTP call is made from this method.

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

        Always reads from Rex's local file-based storage via
        :func:`rex.memory.load_recent_history`.  Rex is the source of truth
        for voice conversation history; OpenClaw maintains its own per-channel
        history separately.

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
