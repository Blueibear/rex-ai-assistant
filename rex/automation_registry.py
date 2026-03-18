"""
Automation Registry for Rex AI Assistant.

Stores and retrieves automations (trigger + action pairs) persistently.
Automations are saved as JSON and survive restarts.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_STORAGE = Path("data/automations/automations.json")


@dataclass
class Automation:
    """A single automation definition."""

    automation_id: str
    name: str
    trigger: dict[str, Any]
    action: dict[str, Any]
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "automation_id": self.automation_id,
            "name": self.name,
            "trigger": self.trigger,
            "action": self.action,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Automation:
        return cls(
            automation_id=str(data.get("automation_id") or str(uuid.uuid4())),
            name=str(data.get("name") or "Unnamed Automation"),
            trigger=dict(data.get("trigger") or {}),
            action=dict(data.get("action") or {}),
            enabled=bool(data.get("enabled", True)),
            created_at=str(data.get("created_at") or datetime.now(timezone.utc).isoformat()),
            updated_at=str(data.get("updated_at") or datetime.now(timezone.utc).isoformat()),
            metadata=dict(data.get("metadata") or {}),
        )


class AutomationRegistry:
    """
    Persistent store for automation definitions.

    Automations are stored in a JSON file and loaded on startup,
    so they persist across restarts.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self._path = storage_path or _DEFAULT_STORAGE
        self._lock = threading.RLock()
        self._automations: dict[str, Automation] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to read automations file %s: %s", self._path, exc)
            return

        items: list[dict[str, Any]] = raw if isinstance(raw, list) else []
        loaded = 0
        with self._lock:
            self._automations.clear()
            for item in items:
                if not isinstance(item, dict):
                    continue
                auto = Automation.from_dict(item)
                self._automations[auto.automation_id] = auto
                loaded += 1
        logger.info("Loaded %d automation(s) from %s", loaded, self._path)

    def _save(self) -> None:
        with self._lock:
            payload = [a.to_dict() for a in self._automations.values()]
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save automations to %s: %s", self._path, exc)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def store(
        self,
        name: str,
        trigger: dict[str, Any],
        action: dict[str, Any],
        *,
        automation_id: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> Automation:
        """Create and persist a new automation. Returns the stored Automation."""
        auto = Automation(
            automation_id=str(automation_id or uuid.uuid4()),
            name=name,
            trigger=dict(trigger),
            action=dict(action),
            enabled=enabled,
            metadata=dict(metadata or {}),
        )
        with self._lock:
            self._automations[auto.automation_id] = auto
        self._save()
        logger.info("Stored automation '%s' (%s)", name, auto.automation_id)
        return auto

    def get(self, automation_id: str) -> Automation | None:
        """Retrieve an automation by ID."""
        with self._lock:
            return self._automations.get(automation_id)

    def get_by_name(self, name: str) -> Automation | None:
        """Retrieve the first automation matching the given name."""
        with self._lock:
            for auto in self._automations.values():
                if auto.name == name:
                    return auto
        return None

    def list_all(self, *, include_disabled: bool = True) -> list[Automation]:
        """Return all stored automations."""
        with self._lock:
            autos = list(self._automations.values())
        if not include_disabled:
            autos = [a for a in autos if a.enabled]
        return autos

    def update(self, automation_id: str, **updates: Any) -> Automation | None:
        """Update fields on an existing automation. Returns the updated object or None."""
        with self._lock:
            auto = self._automations.get(automation_id)
            if auto is None:
                return None
            for key, value in updates.items():
                if hasattr(auto, key):
                    setattr(auto, key, value)
            auto.updated_at = datetime.now(timezone.utc).isoformat()
        self._save()
        return auto

    def remove(self, automation_id: str) -> bool:
        """Delete an automation. Returns True if it existed."""
        with self._lock:
            existed = automation_id in self._automations
            if existed:
                del self._automations[automation_id]
        if existed:
            self._save()
        return existed

    def clear(self) -> None:
        """Remove all automations."""
        with self._lock:
            self._automations.clear()
        self._save()


# ------------------------------------------------------------------
# Global registry
# ------------------------------------------------------------------

_REGISTRY: AutomationRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_automation_registry() -> AutomationRegistry:
    """Return the global AutomationRegistry instance."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = AutomationRegistry()
    return _REGISTRY


def set_automation_registry(registry: AutomationRegistry | None) -> None:
    """Replace the global registry (primarily for testing)."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = registry


__all__ = [
    "Automation",
    "AutomationRegistry",
    "get_automation_registry",
    "set_automation_registry",
]
