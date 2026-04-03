"""Shopping list voice intent handler (US-SL-002).

Detects shopping list utterances before they reach the LLM so Rex can respond
immediately without spending tokens on a trivial task.

Supported patterns
------------------
Add intent:
  "add milk to the shopping list"
  "add milk, eggs, and butter to my list"
  "put oat milk on the list"
  "I need bread"

Read intent:
  "what's on my shopping list?"
  "what is on the list?"
  "read my shopping list"
  "show me my list"
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

_ADD_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"^(?:please\s+)?add\s+(.+?)\s+to\s+(?:the\s+|my\s+)?(?:shopping\s+)?list$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?put\s+(.+?)\s+on\s+(?:the\s+|my\s+)?(?:shopping\s+)?list$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?:please\s+)?i\s+need\s+(.+?)$",
        re.IGNORECASE,
    ),
]

_READ_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"what(?:'s| is)\s+on\s+(?:my\s+|the\s+)?(?:shopping\s+)?list", re.IGNORECASE),
    re.compile(
        r"(?:read|show|list)\s+(?:me\s+)?(?:my\s+|the\s+)?(?:shopping\s+)?list", re.IGNORECASE
    ),
    re.compile(r"(?:what\s+)?do\s+i\s+need\s+(?:to\s+buy|from\s+the\s+store)", re.IGNORECASE),
]

# Separates items in a list utterance: commas, "and", semicolons
_ITEM_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+|;\s*", re.IGNORECASE)


def _extract_items(raw: str) -> list[str]:
    """Split a raw item string into individual normalised item names."""
    parts = _ITEM_SPLIT_RE.split(raw)
    items = [p.strip().rstrip(".!?").strip() for p in parts]
    return [i for i in items if i]


def _match_add(transcript: str) -> str | None:
    """Return the raw items string if transcript matches an add pattern, else None."""
    for pattern in _ADD_PATTERNS:
        m = pattern.match(transcript.strip())
        if m:
            return m.group(1)
    return None


def _match_read(transcript: str) -> bool:
    """Return True if transcript matches a read/list pattern."""
    for pattern in _READ_PATTERNS:
        if pattern.search(transcript.strip()):
            return True
    return False


# ---------------------------------------------------------------------------
# Handler class
# ---------------------------------------------------------------------------


class ShoppingListHandler:
    """Handles shopping list voice intents before they reach the LLM.

    Parameters
    ----------
    shopping_list:
        A ``ShoppingList`` instance shared with the rest of the application.
    """

    def __init__(self, shopping_list) -> None:  # avoid circular import
        self._list = shopping_list

    def handle(self, transcript: str, *, user_id: str = "default") -> str | None:
        """Return a response string if *transcript* is a shopping list command, else None.

        Parameters
        ----------
        transcript:
            The user's spoken (transcribed) text.
        user_id:
            The active user identifier used as ``added_by`` for new items.
        """
        # --- add intent ---
        raw_items = _match_add(transcript)
        if raw_items is not None:
            items = _extract_items(raw_items)
            if not items:
                return None
            added: list[str] = []
            for item_name in items:
                self._list.add_item(item_name, added_by=user_id)
                added.append(item_name)
            logger.debug("[shopping] added %d item(s) for user %r: %s", len(added), user_id, added)
            if len(added) == 1:
                return f"Added {added[0]} to your shopping list."
            items_str = ", ".join(added[:-1]) + f", and {added[-1]}"
            return f"Added {items_str} to your shopping list."

        # --- read intent ---
        if _match_read(transcript):
            unchecked = self._list.list_items(include_checked=False)
            if not unchecked:
                return "Your shopping list is empty."
            if len(unchecked) == 1:
                item = unchecked[0]
                qty = (
                    f"{item.quantity:g} {item.unit}".strip() if item.unit else f"{item.quantity:g}"
                )
                return f"You have {qty} {item.name} on your list."
            parts = []
            for item in unchecked:
                qty = (
                    f"{item.quantity:g} {item.unit}".strip() if item.unit else f"{item.quantity:g}"
                )
                parts.append(f"{qty} {item.name}".strip())
            items_str = ", ".join(parts[:-1]) + f", and {parts[-1]}"
            return f"You need: {items_str}."

        return None


__all__ = ["ShoppingListHandler"]
