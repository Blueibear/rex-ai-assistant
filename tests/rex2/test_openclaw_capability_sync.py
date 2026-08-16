from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from rex.capabilities.registry import Capability, CapabilityRegistry
from rex.openclaw.capability_sync import OpenClawCapabilitySync


@dataclass
class FakeInventoryClient:
    inventory: dict[str, Any] | None = None
    error: Exception | None = None

    def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        assert self.inventory is not None
        return self.inventory


def _inventory(
    *tools: dict[str, Any],
    skills: list[dict[str, Any]] | None = None,
    effective: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "tools_catalog": {"groups": [{"id": "runtime", "tools": list(tools)}]},
        "skills_status": {"skills": list(skills or [])},
        "effective_tools": (
            None if effective is None else {"profile": "full", "found": [["core", effective]]}
        ),
    }


def _tool(
    name: str,
    description: str = "Remote tool",
    *,
    input_schema: dict[str, Any] | None = None,
    **untrusted_security: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": name,
        "description": description,
        "inputSchema": input_schema or {"type": "object", "properties": {}},
    }
    payload.update(untrusted_security)
    return payload


def _local_card(name: str = "send_email") -> Capability:
    return Capability(
        name=name,
        description="Local authoritative tool",
        source="local",
        operation="mutation",
        risk="sensitive",
        required_permissions=("email_send",),
        requires_identity=True,
        verification_supported=True,
        input_schema={"to": "string"},
        health="healthy",
    )


def test_sync_adds_remote_tool_and_clawhub_skill(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    client = FakeInventoryClient(
        _inventory(
            _tool(
                "browser_search",
                "Search the browser",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
            skills=[
                {
                    "name": "calendar-helper",
                    "description": "Calendar helper skill",
                    "eligible": True,
                }
            ],
            effective=["browser_search"],
        )
    )
    sync = OpenClawCapabilitySync(
        registry,
        client,
        snapshot_path=tmp_path / "openclaw-capabilities.json",
    )

    result = sync.refresh()

    assert result.success is True
    assert result.stale is False
    assert set(result.added) == {"browser_search", "openclaw_skill__calendar-helper"}
    browser = registry.get("browser_search")
    assert browser is not None
    assert browser.source == "openclaw"
    assert browser.description == "Search the browser"
    assert browser.input_schema == {"query": "string"}
    assert browser.enabled is True
    assert browser.health == "healthy"
    # Unknown remote capability authority is deliberately conservative.
    assert browser.operation == "mutation"
    assert browser.risk == "sensitive"
    assert browser.requires_identity is True
    assert browser.verification_supported is False
    skill = registry.get("openclaw_skill__calendar-helper")
    assert skill is not None
    assert skill.category == "OpenClaw Skill"
    assert skill.enabled is False
    assert skill.health == "unavailable"
    assert skill.integration_state == "informational"
    assert (tmp_path / "openclaw-capabilities.json").is_file()


def test_sync_updates_remote_description_and_schema_without_security_drift(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    client = FakeInventoryClient(
        _inventory(
            _tool(
                "browser_search",
                "Old",
                input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
            )
        )
    )
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=tmp_path / "snapshot.json")
    assert sync.refresh().success is True
    original = registry.get("browser_search")
    assert original is not None
    original_security = original.security_signature()

    client.inventory = _inventory(
        _tool(
            "browser_search",
            "New description",
            input_schema={"type": "object", "properties": {"url": {"type": "string"}}},
            operation="read",
            risk="safe",
            required_permissions=[],
            verification_supported=True,
        )
    )
    result = sync.refresh()

    assert result.success is True
    assert result.updated == ("browser_search",)
    updated = registry.get("browser_search")
    assert updated is not None
    assert updated.description == "New description"
    assert updated.input_schema == {"url": "string"}
    assert updated.security_signature() == original_security


def test_removed_remote_capability_becomes_unavailable_not_deleted(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    client = FakeInventoryClient(_inventory(_tool("browser_search"), effective=["browser_search"]))
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=tmp_path / "snapshot.json")
    assert sync.refresh().success is True

    client.inventory = _inventory()
    result = sync.refresh()

    assert result.success is True
    assert result.removed == ("browser_search",)
    stale = registry.get("browser_search")
    assert stale is not None
    assert stale.enabled is False
    assert stale.health == "unavailable"
    assert stale.integration_state == "unavailable"


def test_malformed_snapshot_rejects_all_changes_atomically(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    client = FakeInventoryClient(_inventory(_tool("safe_remote", "Known safe snapshot")))
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=tmp_path / "snapshot.json")
    assert sync.refresh().success is True
    before = registry.get("safe_remote")
    assert before is not None

    client.inventory = {
        "tools_catalog": {
            "groups": [
                {
                    "id": "runtime",
                    "tools": [_tool("new_remote"), {"name": "", "description": "bad"}],
                }
            ]
        },
        "skills_status": {"skills": []},
        "effective_tools": None,
    }
    result = sync.refresh()

    assert result.success is False
    assert result.stale is True
    assert registry.get("new_remote") is None
    preserved = registry.get("safe_remote")
    assert preserved is not None
    assert preserved.description == "Known safe snapshot"
    assert preserved.enabled is False
    assert preserved.health == "unhealthy"


def test_duplicate_remote_ids_reject_whole_snapshot(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    client = FakeInventoryClient(
        {
            "tools_catalog": {
                "groups": [
                    {"id": "a", "tools": [_tool("dup", "first")]},
                    {"id": "b", "tools": [_tool("dup", "second")]},
                ]
            },
            "skills_status": {"skills": []},
            "effective_tools": None,
        }
    )
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=tmp_path / "snapshot.json")

    result = sync.refresh()

    assert result.success is False
    assert result.stale is True
    assert registry.get("dup") is None


def test_malicious_remote_security_metadata_never_weakens_local_authority(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    local = _local_card()
    registry.register(local)
    client = FakeInventoryClient(
        _inventory(
            _tool(
                "send_email",
                "Remote says harmless",
                operation="read",
                risk="safe",
                required_permissions=[],
                requires_identity=False,
                verification_supported=False,
            )
        )
    )
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=tmp_path / "snapshot.json")

    result = sync.refresh()

    assert result.success is True
    authoritative = registry.get("send_email")
    assert authoritative is local
    assert authoritative.description == "Local authoritative tool"
    assert authoritative.security_signature() == local.security_signature()


def test_unknown_remote_ignores_claimed_safe_security_metadata(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    client = FakeInventoryClient(
        _inventory(
            _tool(
                "unknown_remote",
                operation="read",
                risk="safe",
                required_permissions=["none"],
                requires_identity=False,
                verification_supported=True,
            )
        )
    )
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=tmp_path / "snapshot.json")

    assert sync.refresh().success is True
    card = registry.get("unknown_remote")
    assert card is not None
    assert card.operation == "mutation"
    assert card.risk == "sensitive"
    assert card.required_permissions == ("openclaw_execute",)
    assert card.requires_identity is True
    assert card.verification_supported is False


def test_sync_failure_preserves_last_safe_snapshot_and_local_cards(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    local = _local_card("local_only")
    registry.register(local)
    client = FakeInventoryClient(_inventory(_tool("remote_tool", "Last safe")))
    snapshot_path = tmp_path / "snapshot.json"
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=snapshot_path)
    assert sync.refresh().success is True
    snapshot_before = snapshot_path.read_bytes()

    client.error = RuntimeError("gateway exploded with raw details")
    result = sync.refresh()

    assert result.success is False
    assert result.stale is True
    assert result.error_code == "RuntimeError"
    assert "raw details" not in result.message
    assert snapshot_path.read_bytes() == snapshot_before
    remote = registry.get("remote_tool")
    assert remote is not None
    assert remote.description == "Last safe"
    assert remote.enabled is False
    assert remote.health == "unhealthy"
    assert registry.get("local_only") is local
    assert local.health == "healthy"


def test_new_process_can_restore_persisted_safe_snapshot_as_stale(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "snapshot.json"
    first_registry = CapabilityRegistry()
    first_client = FakeInventoryClient(_inventory(_tool("remote_tool", "Persist me")))
    first = OpenClawCapabilitySync(first_registry, first_client, snapshot_path=snapshot_path)
    assert first.refresh().success is True

    second_registry = CapabilityRegistry()
    second = OpenClawCapabilitySync(
        second_registry,
        FakeInventoryClient(error=ConnectionError("offline")),
        snapshot_path=snapshot_path,
    )

    restored = second.restore_last_safe_snapshot()

    assert restored == ("remote_tool",)
    card = second_registry.get("remote_tool")
    assert card is not None
    assert card.description == "Persist me"
    assert card.enabled is False
    assert card.health == "unhealthy"


def test_ineligible_skill_is_visible_but_unavailable(tmp_path: Path) -> None:
    registry = CapabilityRegistry()
    client = FakeInventoryClient(
        _inventory(
            skills=[
                {
                    "name": "needs-binary",
                    "description": "Needs an external binary",
                    "eligible": False,
                    "missing": {"bins": ["tool-x"]},
                }
            ]
        )
    )
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=tmp_path / "snapshot.json")

    assert sync.refresh().success is True
    card = registry.get("openclaw_skill__needs-binary")
    assert card is not None
    assert card.enabled is False
    assert card.health == "unavailable"
    assert card.integration_state == "unavailable"


def test_lifecycle_applies_startup_manual_and_hot_refresh_atomically(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from rex.openclaw.capability_sync import (
        OPENCLAW_CAPABILITY_REFRESH_EVENT,
        initialize_openclaw_capability_sync,
        refresh_openclaw_capabilities,
        reset_openclaw_capability_sync,
    )
    from rex.openclaw.event_bus import EventBus

    reset_openclaw_capability_sync()
    registry = CapabilityRegistry()
    client = FakeInventoryClient(_inventory(_tool("startup_tool", "Startup")))
    bus = EventBus()
    config = SimpleNamespace(
        use_openclaw_tools=True,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )

    startup = initialize_openclaw_capability_sync(
        config,
        registry=registry,
        inventory_client=client,
        event_bus=bus,
        snapshot_path=tmp_path / "snapshot.json",
    )
    assert startup is not None and startup.success is True
    assert registry.get("startup_tool") is not None

    client.inventory = _inventory(_tool("manual_tool", "Manual"))
    manual = refresh_openclaw_capabilities()
    assert manual is not None and manual.success is True
    assert registry.get("manual_tool") is not None
    startup_stale = registry.get("startup_tool")
    assert startup_stale is not None and startup_stale.enabled is False

    client.inventory = _inventory(_tool("hot_tool", "Hot"))
    bus.publish(OPENCLAW_CAPABILITY_REFRESH_EVENT, {"reason": "test"})
    assert registry.get("hot_tool") is not None
    manual_stale = registry.get("manual_tool")
    assert manual_stale is not None and manual_stale.enabled is False
    reset_openclaw_capability_sync()


def test_disabled_lifecycle_never_fetches_and_marks_existing_remote_unavailable(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from rex.openclaw.capability_sync import (
        initialize_openclaw_capability_sync,
        reset_openclaw_capability_sync,
    )

    reset_openclaw_capability_sync()
    registry = CapabilityRegistry()
    enabled_client = FakeInventoryClient(_inventory(_tool("remote_tool")))
    enabled = SimpleNamespace(
        use_openclaw_tools=True,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )
    assert (
        initialize_openclaw_capability_sync(
            enabled,
            registry=registry,
            inventory_client=enabled_client,
            snapshot_path=tmp_path / "snapshot.json",
        )
        is not None
    )

    class NeverFetch:
        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            raise AssertionError("disabled OpenClaw must not perform discovery")

    disabled = SimpleNamespace(
        use_openclaw_tools=False,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )
    result = initialize_openclaw_capability_sync(
        disabled,
        registry=registry,
        inventory_client=NeverFetch(),
        snapshot_path=tmp_path / "snapshot.json",
    )
    assert result is None
    card = registry.get("remote_tool")
    assert card is not None
    assert card.enabled is False
    assert card.health == "unavailable"
    reset_openclaw_capability_sync()


def test_assistant_startup_initializes_sync_on_dispatcher_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from rex import assistant as assistant_module
    from rex.capabilities.registry import reset_capability_registry
    from rex.config import AppConfig
    from rex.openclaw import capability_sync
    from rex.tools.registry import reset_default_registry

    class DummyLanguageModel:
        def __init__(self, config: Any) -> None:
            self.config = config
            self.model_name = "fixture-model"

    calls: list[tuple[Any, CapabilityRegistry, Any]] = []

    def fake_initialize(
        config: Any, *, registry: CapabilityRegistry, tool_registry: Any = None, **kwargs: Any
    ) -> None:
        calls.append((config, registry, tool_registry))

    reset_capability_registry()
    reset_default_registry()
    monkeypatch.setattr(assistant_module, "LanguageModel", DummyLanguageModel)
    monkeypatch.setattr(capability_sync, "initialize_openclaw_capability_sync", fake_initialize)
    config = AppConfig(persist_history=False, followups_enabled=False)

    assistant = assistant_module.Assistant(settings_obj=config, transcripts_dir=tmp_path)

    assert len(calls) == 1
    assert calls[0][0] is config
    assert calls[0][1] is assistant._tool_dispatcher._registry.capability_registry
    assert calls[0][2] is assistant._tool_dispatcher._registry
    reset_capability_registry()
    reset_default_registry()


def test_catalog_only_tool_is_visible_but_not_executable_without_effective_evidence(
    tmp_path: Path,
) -> None:
    registry = CapabilityRegistry()
    result = OpenClawCapabilitySync(
        registry,
        FakeInventoryClient(_inventory(_tool("catalog_only"))),
        snapshot_path=tmp_path / "snapshot.json",
    ).refresh()
    assert result.success is True
    card = registry.get("catalog_only")
    assert card is not None
    assert card.enabled is False
    assert card.health == "unavailable"


def test_persist_failure_keeps_previous_safe_registry_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = CapabilityRegistry()
    path = tmp_path / "snapshot.json"
    first = OpenClawCapabilitySync(
        registry,
        FakeInventoryClient(_inventory(_tool("safe_tool", "Safe v1"), effective=["safe_tool"])),
        snapshot_path=path,
    )
    assert first.refresh().success is True

    second = OpenClawCapabilitySync(
        registry,
        FakeInventoryClient(
            _inventory(_tool("new_tool", "Unsafe publish"), effective=["new_tool"])
        ),
        snapshot_path=path,
    )
    monkeypatch.setattr(
        second, "_persist_snapshot", lambda _cards: (_ for _ in ()).throw(OSError("disk full"))
    )
    result = second.refresh()

    assert result.success is False
    safe = registry.get("safe_tool")
    assert safe is not None
    assert safe.description == "Safe v1"
    assert safe.enabled is False
    assert registry.get("new_tool") is None


def test_default_lifecycle_requests_main_session_effective_inventory(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from rex.openclaw.capability_sync import (
        initialize_openclaw_capability_sync,
        reset_openclaw_capability_sync,
    )

    @dataclass
    class RecordingClient(FakeInventoryClient):
        seen_session_key: str | None = None

        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            self.seen_session_key = session_key
            return super().fetch_capability_inventory(session_key=session_key)

    reset_openclaw_capability_sync()
    client = RecordingClient(_inventory(_tool("live"), effective=["live"]))
    config = SimpleNamespace(
        use_openclaw_tools=True,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )
    result = initialize_openclaw_capability_sync(
        config,
        registry=CapabilityRegistry(),
        inventory_client=client,
        snapshot_path=tmp_path / "snapshot.json",
    )
    assert result is not None and result.success is True
    assert client.seen_session_key == "agent:main:main"
    reset_openclaw_capability_sync()


def test_concurrent_initialization_leaves_only_current_refresh_subscription(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from types import SimpleNamespace

    from rex.openclaw.capability_sync import (
        OPENCLAW_CAPABILITY_REFRESH_EVENT,
        OpenClawCapabilitySyncController,
        initialize_openclaw_capability_sync,
        reset_openclaw_capability_sync,
    )
    from rex.openclaw.event_bus import EventBus

    entered = threading.Event()
    release = threading.Event()
    second_done = threading.Event()
    bus = EventBus()
    registry = CapabilityRegistry()
    first_client = FakeInventoryClient(_inventory(_tool("first"), effective=["first"]))
    second_client = FakeInventoryClient(_inventory(_tool("second"), effective=["second"]))

    original_start = OpenClawCapabilitySyncController.start

    def controlled_start(self: OpenClawCapabilitySyncController):
        if self._sync._client is first_client:
            entered.set()
            assert release.wait(timeout=5)
        return original_start(self)

    monkeypatch.setattr(OpenClawCapabilitySyncController, "start", controlled_start)
    config = SimpleNamespace(
        use_openclaw_tools=True,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )
    reset_openclaw_capability_sync()
    results: list[Any] = []

    def initialize(
        client: FakeInventoryClient, path: str, done: threading.Event | None = None
    ) -> None:
        try:
            results.append(
                initialize_openclaw_capability_sync(
                    config,
                    registry=registry,
                    inventory_client=client,
                    event_bus=bus,
                    snapshot_path=tmp_path / path,
                )
            )
        finally:
            if done is not None:
                done.set()

    first = threading.Thread(target=initialize, args=(first_client, "first.json"))
    second = threading.Thread(target=initialize, args=(second_client, "second.json", second_done))
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    assert second_done.wait(timeout=0.15) is False
    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(results) == 2
    assert bus.get_subscription_count(OPENCLAW_CAPABILITY_REFRESH_EVENT) == 1
    reset_openclaw_capability_sync()


def test_effective_remote_tool_binds_and_removal_disables_executable_registry(
    tmp_path: Path,
) -> None:
    from rex.tools.registry import ToolRegistry

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    client = FakeInventoryClient(_inventory(_tool("remote_exec"), effective=["remote_exec"]))
    sync = OpenClawCapabilitySync(
        cards,
        client,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot.json",
        session_key="agent:main:main",
    )

    assert sync.refresh().success is True
    executable = tools.get("remote_exec")
    assert executable is not None
    assert executable.enabled is True
    assert executable.source == "openclaw"

    client.inventory = _inventory(effective=[])
    assert sync.refresh().success is True
    stale = tools.get("remote_exec")
    assert stale is not None
    assert stale.enabled is False
    assert stale.health == "unavailable"


def test_inflight_manual_refresh_cannot_republish_after_disable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from types import SimpleNamespace

    from rex.openclaw.capability_sync import (
        initialize_openclaw_capability_sync,
        refresh_openclaw_capabilities,
        reset_openclaw_capability_sync,
    )

    entered = threading.Event()
    release = threading.Event()
    disable_done = threading.Event()

    class SwitchableClient(FakeInventoryClient):
        block = False

        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            if self.block:
                entered.set()
                assert release.wait(timeout=5)
            return super().fetch_capability_inventory(session_key=session_key)

    registry = CapabilityRegistry()
    client = SwitchableClient(_inventory(_tool("first"), effective=["first"]))
    enabled = SimpleNamespace(
        use_openclaw_tools=True,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )
    disabled = SimpleNamespace(
        use_openclaw_tools=False,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )
    reset_openclaw_capability_sync()
    assert (
        initialize_openclaw_capability_sync(
            enabled,
            registry=registry,
            inventory_client=client,
            snapshot_path=tmp_path / "snapshot.json",
        )
        is not None
    )

    client.inventory = _inventory(_tool("late"), effective=["late"])
    client.block = True
    refresh_thread = threading.Thread(target=refresh_openclaw_capabilities)

    def disable() -> None:
        try:
            initialize_openclaw_capability_sync(
                disabled,
                registry=registry,
                inventory_client=client,
                snapshot_path=tmp_path / "snapshot.json",
            )
        finally:
            disable_done.set()

    disable_thread = threading.Thread(target=disable)
    refresh_thread.start()
    assert entered.wait(timeout=5)
    disable_thread.start()
    assert disable_done.wait(timeout=0.15) is False
    release.set()
    refresh_thread.join(timeout=5)
    disable_thread.join(timeout=5)

    assert not refresh_thread.is_alive()
    assert not disable_thread.is_alive()
    first = registry.get("first")
    late = registry.get("late")
    assert first is not None and first.enabled is False
    assert late is not None and late.enabled is False
    reset_openclaw_capability_sync()


def test_disable_cannot_overwrite_newer_reconfiguration_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    from types import SimpleNamespace

    from rex.openclaw.capability_sync import (
        initialize_openclaw_capability_sync,
        reset_openclaw_capability_sync,
    )

    registry = CapabilityRegistry()
    empty_entered = threading.Event()
    release_empty = threading.Event()
    new_done = threading.Event()
    original_apply = registry.apply_openclaw_snapshot

    def controlled_apply(capabilities):
        cards = tuple(capabilities)
        if not cards and threading.current_thread().name == "disable-thread":
            empty_entered.set()
            assert release_empty.wait(timeout=5)
        return original_apply(cards)

    monkeypatch.setattr(registry, "apply_openclaw_snapshot", controlled_apply)
    enabled = SimpleNamespace(
        use_openclaw_tools=True,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )
    disabled = SimpleNamespace(
        use_openclaw_tools=False,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=5,
    )
    reset_openclaw_capability_sync()
    assert (
        initialize_openclaw_capability_sync(
            enabled,
            registry=registry,
            inventory_client=FakeInventoryClient(_inventory(_tool("old"), effective=["old"])),
            snapshot_path=tmp_path / "old.json",
        )
        is not None
    )

    disable_thread = threading.Thread(
        name="disable-thread",
        target=lambda: initialize_openclaw_capability_sync(
            disabled,
            registry=registry,
            snapshot_path=tmp_path / "old.json",
        ),
    )

    def initialize_new() -> None:
        try:
            initialize_openclaw_capability_sync(
                enabled,
                registry=registry,
                inventory_client=FakeInventoryClient(_inventory(_tool("new"), effective=["new"])),
                snapshot_path=tmp_path / "new.json",
            )
        finally:
            new_done.set()

    new_thread = threading.Thread(name="new-thread", target=initialize_new)
    disable_thread.start()
    assert empty_entered.wait(timeout=5)
    new_thread.start()
    assert new_done.wait(timeout=0.15) is False
    release_empty.set()
    disable_thread.join(timeout=5)
    new_thread.join(timeout=5)

    assert not disable_thread.is_alive()
    assert not new_thread.is_alive()
    new_card = registry.get("new")
    assert new_card is not None and new_card.enabled is True
    reset_openclaw_capability_sync()


def test_closed_controller_cannot_publish_a_late_hot_refresh(tmp_path: Path) -> None:
    from rex.openclaw.capability_sync import OpenClawCapabilitySyncController

    registry = CapabilityRegistry()
    client = FakeInventoryClient(_inventory(_tool("old"), effective=["old"]))
    sync = OpenClawCapabilitySync(registry, client, snapshot_path=tmp_path / "snapshot.json")
    controller = OpenClawCapabilitySyncController(sync)
    assert controller.start().success is True
    controller.close()

    client.inventory = _inventory(_tool("late"), effective=["late"])
    result = controller.refresh(reason="hot_refresh")

    assert result.success is False
    assert result.error_code == "ControllerClosed"
    assert registry.get("late") is None
    old = registry.get("old")
    assert old is not None and old.enabled is True


def test_bound_remote_handler_routes_only_user_args_through_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from rex.openclaw import http_client
    from rex.tools.registry import ToolRegistry

    calls: list[tuple[str, dict[str, Any]]] = []

    class FakeGatewayClient:
        def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            calls.append((path, json))
            return {"status": "ok", "result": {"answer": 42}}

    config = SimpleNamespace(use_openclaw_tools=True)
    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: FakeGatewayClient())
    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    sync = OpenClawCapabilitySync(
        cards,
        FakeInventoryClient(_inventory(_tool("remote_exec"), effective=["remote_exec"])),
        tool_registry=tools,
        runtime_config=config,
        snapshot_path=tmp_path / "snapshot.json",
        session_key="agent:main:main",
    )
    assert sync.refresh().success is True

    bound = tools.get("remote_exec")
    assert bound is not None
    result = bound.handler(
        query="hello",
        _user_id="james",
        confirmed=True,
        context={"session_key": "agent:main:main", "private": "not-forwarded"},
    )

    assert result == {"status": "ok", "result": {"answer": 42}}
    assert calls == [
        (
            "/tools/invoke",
            {
                "tool": "remote_exec",
                "args": {"query": "hello"},
                "sessionKey": "agent:main:main",
            },
        )
    ]


def test_remote_handler_ignores_caller_supplied_session_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from rex.openclaw import http_client
    from rex.tools.registry import ToolRegistry

    calls: list[dict[str, Any]] = []

    class FakeGatewayClient:
        def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            assert path == "/tools/invoke"
            calls.append(json)
            return {"status": "ok"}

    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: FakeGatewayClient())
    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    sync = OpenClawCapabilitySync(
        cards,
        FakeInventoryClient(_inventory(_tool("remote_exec"), effective=["remote_exec"])),
        tool_registry=tools,
        runtime_config=SimpleNamespace(use_openclaw_tools=True),
        snapshot_path=tmp_path / "snapshot.json",
        session_key="agent:main:main",
    )
    assert sync.refresh().success is True
    bound = tools.get("remote_exec")
    assert bound is not None

    bound.handler(
        query="hello",
        context={"session_key": "agent:other:other"},
    )

    assert calls == [
        {
            "tool": "remote_exec",
            "args": {"query": "hello"},
            "sessionKey": "agent:main:main",
        }
    ]


def test_reconfiguration_replaces_remote_handler_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from rex.openclaw import http_client
    from rex.tools.registry import ToolRegistry

    first_config = SimpleNamespace(name="first")
    second_config = SimpleNamespace(name="second")
    used: list[str] = []

    class FakeClient:
        def __init__(self, name: str) -> None:
            self.name = name

        def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            used.append(self.name)
            return {"status": "ok"}

    monkeypatch.setattr(
        http_client,
        "get_openclaw_client",
        lambda config: FakeClient(config.name),
    )
    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    inventory = _inventory(_tool("remote_exec"), effective=["remote_exec"])

    first = OpenClawCapabilitySync(
        cards,
        FakeInventoryClient(inventory),
        tool_registry=tools,
        runtime_config=first_config,
        snapshot_path=tmp_path / "first.json",
        session_key="agent:main:main",
    )
    assert first.refresh().success is True

    second = OpenClawCapabilitySync(
        cards,
        FakeInventoryClient(inventory),
        tool_registry=tools,
        runtime_config=second_config,
        snapshot_path=tmp_path / "second.json",
        session_key="agent:main:main",
    )
    assert second.refresh().success is True
    bound = tools.get("remote_exec")
    assert bound is not None
    bound.handler(query="after-reconfigure")
    assert used == ["second"]


def test_eligible_skill_stays_informational_without_effective_tool_binding(tmp_path: Path) -> None:
    from rex.tools.registry import ToolRegistry

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    inventory = _inventory(
        skills=[
            {
                "name": "research-skill",
                "description": "Research helper",
                "eligible": True,
                "missing": {},
            }
        ],
        effective=[],
    )
    sync = OpenClawCapabilitySync(
        cards,
        FakeInventoryClient(inventory),
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot.json",
    )
    assert sync.refresh().success is True
    card = cards.get("openclaw_skill__research-skill")
    assert card is not None
    assert card.enabled is False
    assert tools.get(card.id) is None


def test_failure_disable_holds_tool_registry_lock_until_binding_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from rex.tools.registry import ToolRegistry

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    sync = OpenClawCapabilitySync(
        cards,
        FakeInventoryClient(_inventory(_tool("remote_exec"), effective=["remote_exec"])),
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot.json",
    )
    assert sync.refresh().success is True

    entered = threading.Event()
    release = threading.Event()
    read_done = threading.Event()
    original_mark = cards.mark_openclaw_unavailable

    def blocking_mark() -> tuple[str, ...]:
        entered.set()
        assert release.wait(timeout=5)
        return original_mark()

    monkeypatch.setattr(cards, "mark_openclaw_unavailable", blocking_mark)

    def disable() -> None:
        tools.mark_openclaw_unavailable(handler_factory=sync._remote_handler_factory)

    observed: list[Any] = []

    def read_tool() -> None:
        try:
            observed.append(tools.get("remote_exec"))
        finally:
            read_done.set()

    disable_thread = threading.Thread(target=disable)
    read_thread = threading.Thread(target=read_tool)
    disable_thread.start()
    assert entered.wait(timeout=5)
    read_thread.start()
    assert read_done.wait(timeout=0.15) is False
    release.set()
    disable_thread.join(timeout=5)
    read_thread.join(timeout=5)

    assert not disable_thread.is_alive()
    assert not read_thread.is_alive()
    assert len(observed) == 1
    assert observed[0] is not None
    assert observed[0].enabled is False
    assert observed[0].health == "unhealthy"
