"""Shopping list data model and persistence (US-SL-001).

All users share a single list; items are tagged with ``added_by`` so the voice
layer can attribute additions to the identified speaker.  The list is persisted
as JSON at ``data/shopping_list.json`` (path configurable at construction time).
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("data") / "shopping_list.json"


def _utcnow() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(UTC).isoformat()


@dataclass
class ShoppingItem:
    """A single item on the shopping list."""

    id: str
    name: str
    quantity: float
    unit: str
    added_by: str
    checked: bool = False
    added_at: str = field(default_factory=_utcnow)
    checked_at: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> ShoppingItem:
        return cls(
            id=data["id"],
            name=data["name"],
            quantity=float(data.get("quantity", 1)),
            unit=data.get("unit", ""),
            added_by=data.get("added_by", "default"),
            checked=bool(data.get("checked", False)),
            added_at=data.get("added_at", _utcnow()),
            checked_at=data.get("checked_at"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


class ShoppingList:
    """Persistent, thread-safe shopping list.

    Parameters
    ----------
    path:
        Path to the JSON backing file.  Defaults to ``data/shopping_list.json``.
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._lock = Lock()
        self._items: list[ShoppingItem] = []
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load items from the backing file (silently creates empty list if absent)."""
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._items = [ShoppingItem.from_dict(d) for d in data.get("items", [])]
        except Exception as exc:
            logger.warning("ShoppingList: failed to load %s: %s", self._path, exc)
            self._items = []

    def _save(self) -> None:
        """Write items to the backing file."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"items": [item.to_dict() for item in self._items]}
            self._path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            logger.error("ShoppingList: failed to save %s: %s", self._path, exc)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add_item(
        self,
        name: str,
        *,
        quantity: float = 1.0,
        unit: str = "",
        added_by: str = "default",
    ) -> ShoppingItem:
        """Add a new item to the list and return it.

        If an unchecked item with the same normalised name already exists, its
        quantity is incremented instead of creating a duplicate.
        """
        normalised = name.strip().lower()
        with self._lock:
            for item in self._items:
                if item.name.strip().lower() == normalised and not item.checked:
                    item.quantity += quantity
                    self._save()
                    return item

            new_item = ShoppingItem(
                id=str(uuid.uuid4()),
                name=name.strip(),
                quantity=quantity,
                unit=unit,
                added_by=added_by,
            )
            self._items.append(new_item)
            self._save()
            return new_item

    def check_item(self, item_id: str) -> bool:
        """Mark item as checked (purchased).  Returns True if found."""
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    item.checked = True
                    item.checked_at = _utcnow()
                    self._save()
                    return True
        return False

    def uncheck_item(self, item_id: str) -> bool:
        """Mark item as unchecked.  Returns True if found."""
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    item.checked = False
                    item.checked_at = None
                    self._save()
                    return True
        return False

    def remove_item(self, item_id: str) -> bool:
        """Remove item permanently.  Returns True if found."""
        with self._lock:
            before = len(self._items)
            self._items = [i for i in self._items if i.id != item_id]
            if len(self._items) < before:
                self._save()
                return True
        return False

    def list_items(
        self,
        *,
        include_checked: bool = True,
        added_by: str | None = None,
    ) -> list[ShoppingItem]:
        """Return items, optionally filtered by checked state or user."""
        with self._lock:
            items = list(self._items)

        if not include_checked:
            items = [i for i in items if not i.checked]
        if added_by is not None:
            items = [i for i in items if i.added_by == added_by]
        return items

    def clear_checked(self) -> int:
        """Remove all checked items.  Returns the number removed."""
        with self._lock:
            before = len(self._items)
            self._items = [i for i in self._items if not i.checked]
            removed = before - len(self._items)
            if removed:
                self._save()
        return removed

    def get_item(self, item_id: str) -> ShoppingItem | None:
        """Return the item with *item_id*, or None if not found."""
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    return item
        return None


__all__ = ["ShoppingItem", "ShoppingList"]
