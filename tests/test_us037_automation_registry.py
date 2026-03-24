"""
US-037: Automation Registry

Acceptance Criteria:
- automations stored
- automations retrieved
- persistence works
- Typecheck passes
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rex.automation_registry import (
    Automation,
    AutomationRegistry,
    get_automation_registry,
    set_automation_registry,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_registry():
    """Isolate global registry state per test."""
    set_automation_registry(None)
    yield
    set_automation_registry(None)


@pytest.fixture()
def registry(tmp_path: Path) -> AutomationRegistry:
    return AutomationRegistry(storage_path=tmp_path / "automations.json")


TRIGGER = {"type": "schedule", "cron": "0 8 * * *"}
ACTION = {"type": "notify", "message": "Good morning!"}


# ---------------------------------------------------------------------------
# Automation stored
# ---------------------------------------------------------------------------


def test_store_returns_automation(registry: AutomationRegistry) -> None:
    auto = registry.store("morning_greeting", TRIGGER, ACTION)
    assert isinstance(auto, Automation)
    assert auto.name == "morning_greeting"


def test_stored_automation_has_id(registry: AutomationRegistry) -> None:
    auto = registry.store("greet", TRIGGER, ACTION)
    assert auto.automation_id
    assert len(auto.automation_id) > 0


def test_store_with_explicit_id(registry: AutomationRegistry) -> None:
    auto = registry.store("greet", TRIGGER, ACTION, automation_id="my-id-123")
    assert auto.automation_id == "my-id-123"


def test_store_preserves_trigger_and_action(registry: AutomationRegistry) -> None:
    auto = registry.store("greet", TRIGGER, ACTION)
    assert auto.trigger == TRIGGER
    assert auto.action == ACTION


def test_store_enabled_default(registry: AutomationRegistry) -> None:
    auto = registry.store("greet", TRIGGER, ACTION)
    assert auto.enabled is True


def test_store_disabled(registry: AutomationRegistry) -> None:
    auto = registry.store("greet", TRIGGER, ACTION, enabled=False)
    assert auto.enabled is False


def test_store_with_metadata(registry: AutomationRegistry) -> None:
    meta = {"source": "test", "priority": 5}
    auto = registry.store("greet", TRIGGER, ACTION, metadata=meta)
    assert auto.metadata["source"] == "test"
    assert auto.metadata["priority"] == 5


def test_multiple_automations_stored(registry: AutomationRegistry) -> None:
    registry.store("a1", TRIGGER, ACTION)
    registry.store("a2", TRIGGER, ACTION)
    registry.store("a3", TRIGGER, ACTION)
    assert len(registry.list_all()) == 3


# ---------------------------------------------------------------------------
# Automations retrieved
# ---------------------------------------------------------------------------


def test_get_by_id(registry: AutomationRegistry) -> None:
    stored = registry.store("greet", TRIGGER, ACTION)
    retrieved = registry.get(stored.automation_id)
    assert retrieved is not None
    assert retrieved.automation_id == stored.automation_id


def test_get_missing_returns_none(registry: AutomationRegistry) -> None:
    assert registry.get("nonexistent-id") is None


def test_get_by_name(registry: AutomationRegistry) -> None:
    registry.store("greet", TRIGGER, ACTION)
    retrieved = registry.get_by_name("greet")
    assert retrieved is not None
    assert retrieved.name == "greet"


def test_get_by_name_missing_returns_none(registry: AutomationRegistry) -> None:
    assert registry.get_by_name("does_not_exist") is None


def test_list_all_returns_all(registry: AutomationRegistry) -> None:
    registry.store("a", TRIGGER, ACTION)
    registry.store("b", TRIGGER, ACTION)
    names = {a.name for a in registry.list_all()}
    assert "a" in names
    assert "b" in names


def test_list_all_exclude_disabled(registry: AutomationRegistry) -> None:
    registry.store("enabled", TRIGGER, ACTION, enabled=True)
    registry.store("disabled", TRIGGER, ACTION, enabled=False)
    active = registry.list_all(include_disabled=False)
    assert all(a.enabled for a in active)
    assert len(active) == 1


def test_remove_automation(registry: AutomationRegistry) -> None:
    stored = registry.store("greet", TRIGGER, ACTION)
    removed = registry.remove(stored.automation_id)
    assert removed is True
    assert registry.get(stored.automation_id) is None


def test_remove_missing_returns_false(registry: AutomationRegistry) -> None:
    assert registry.remove("ghost-id") is False


def test_update_automation(registry: AutomationRegistry) -> None:
    stored = registry.store("greet", TRIGGER, ACTION)
    updated = registry.update(stored.automation_id, enabled=False)
    assert updated is not None
    assert updated.enabled is False


def test_update_missing_returns_none(registry: AutomationRegistry) -> None:
    assert registry.update("ghost-id", enabled=False) is None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persists_to_file(tmp_path: Path) -> None:
    storage = tmp_path / "automations.json"
    reg = AutomationRegistry(storage_path=storage)
    reg.store("persist_me", TRIGGER, ACTION)
    assert storage.exists()


def test_persisted_json_is_valid(tmp_path: Path) -> None:
    storage = tmp_path / "automations.json"
    reg = AutomationRegistry(storage_path=storage)
    reg.store("alpha", TRIGGER, ACTION)
    data = json.loads(storage.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["name"] == "alpha"


def test_reloads_on_new_instance(tmp_path: Path) -> None:
    storage = tmp_path / "automations.json"
    reg1 = AutomationRegistry(storage_path=storage)
    auto = reg1.store("survive", TRIGGER, ACTION)

    reg2 = AutomationRegistry(storage_path=storage)
    retrieved = reg2.get(auto.automation_id)
    assert retrieved is not None
    assert retrieved.name == "survive"


def test_reload_preserves_trigger_and_action(tmp_path: Path) -> None:
    storage = tmp_path / "automations.json"
    reg1 = AutomationRegistry(storage_path=storage)
    auto = reg1.store("check_fields", TRIGGER, ACTION)

    reg2 = AutomationRegistry(storage_path=storage)
    retrieved = reg2.get(auto.automation_id)
    assert retrieved is not None
    assert retrieved.trigger == TRIGGER
    assert retrieved.action == ACTION


def test_reload_preserves_enabled_state(tmp_path: Path) -> None:
    storage = tmp_path / "automations.json"
    reg1 = AutomationRegistry(storage_path=storage)
    auto = reg1.store("disabled_auto", TRIGGER, ACTION, enabled=False)

    reg2 = AutomationRegistry(storage_path=storage)
    retrieved = reg2.get(auto.automation_id)
    assert retrieved is not None
    assert retrieved.enabled is False


def test_empty_registry_creates_no_file(tmp_path: Path) -> None:
    storage = tmp_path / "automations.json"
    AutomationRegistry(storage_path=storage)
    assert not storage.exists()


def test_clear_removes_all(registry: AutomationRegistry) -> None:
    registry.store("a", TRIGGER, ACTION)
    registry.store("b", TRIGGER, ACTION)
    registry.clear()
    assert registry.list_all() == []


# ---------------------------------------------------------------------------
# Global registry helpers
# ---------------------------------------------------------------------------


def test_global_registry_returns_instance(tmp_path: Path) -> None:
    reg = AutomationRegistry(storage_path=tmp_path / "g.json")
    set_automation_registry(reg)
    assert get_automation_registry() is reg


def test_global_registry_singleton(tmp_path: Path) -> None:
    set_automation_registry(None)
    r1 = get_automation_registry()
    r2 = get_automation_registry()
    assert r1 is r2
