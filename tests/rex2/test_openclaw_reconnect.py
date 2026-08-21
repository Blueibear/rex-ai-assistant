from __future__ import annotations

from typing import Any

import pytest

from rex.capabilities.registry import CapabilityRegistry
from rex.openclaw.capability_sync import (
    OpenClawCapabilitySync,
    OpenClawSyncResult,
    get_openclaw_reconnect_controller,
    initialize_openclaw_capability_sync,
    reset_openclaw_capability_sync,
)
from rex.openclaw.errors import (
    OpenClawConnectionError,
    OpenClawOutcomeUnknownError,
    OpenClawUnavailableError,
)
from rex.openclaw.event_bus import EventBus
from rex.openclaw.reconnect import (
    OPENCLAW_HEALTH_CHANGED_EVENT,
    OpenClawReconnectController,
    OpenClawReconnectState,
)
from rex.tools.registry import Tool, ToolRegistry


def _fresh_success() -> OpenClawSyncResult:
    return OpenClawSyncResult(success=True, stale=False)


def test_disconnect_disables_remote_authority_immediately() -> None:
    disabled: list[str] = []
    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": False},
        resync=_fresh_success,
        mark_unavailable=lambda: disabled.append("disabled"),
        auto_reconnect=False,
    )

    controller.mark_disconnected("transport_failure")

    assert controller.state is OpenClawReconnectState.DISCONNECTED
    assert disabled == ["disabled"]
    assert controller.ready_for_dispatch is False


def test_health_recovery_requires_fresh_capability_resync() -> None:
    calls: list[str] = []
    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=lambda: calls.append("resync") or _fresh_success(),
        mark_unavailable=lambda: calls.append("unavailable"),
        auto_reconnect=False,
    )
    controller.mark_disconnected("transport_failure")

    recovered = controller.run_until_recovered(max_attempts=1)

    assert recovered is True
    assert calls == ["unavailable", "resync"]
    assert controller.state is OpenClawReconnectState.READY
    assert controller.ready_for_dispatch is True


def test_health_up_but_stale_resync_stays_disconnected() -> None:
    delays: list[float] = []
    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=lambda: OpenClawSyncResult(success=False, stale=True, error_code="schema"),
        mark_unavailable=lambda: None,
        auto_reconnect=False,
        wait_fn=lambda delay: delays.append(delay) or False,
    )
    controller.mark_disconnected("transport_failure")

    recovered = controller.run_until_recovered(max_attempts=1)

    assert recovered is False
    assert controller.state is OpenClawReconnectState.DISCONNECTED
    assert delays == []


def test_reconnect_backoff_is_bounded_and_never_hot_loops() -> None:
    delays: list[float] = []
    attempts = 0

    def probe() -> dict[str, Any]:
        nonlocal attempts
        attempts += 1
        return {"available": False, "error": "must-not-be-published"}

    controller = OpenClawReconnectController(
        health_probe=probe,
        resync=_fresh_success,
        mark_unavailable=lambda: None,
        auto_reconnect=False,
        base_delay_seconds=1.0,
        max_delay_seconds=4.0,
        wait_fn=lambda delay: delays.append(delay) or False,
    )
    controller.mark_disconnected("transport_failure")

    assert controller.run_until_recovered(max_attempts=5) is False
    assert attempts == 5
    assert delays == [1.0, 2.0, 4.0, 4.0]
    assert controller.state is OpenClawReconnectState.DISCONNECTED


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def publish(self, event_name: str, payload: dict[str, Any]) -> None:
        self.events.append((event_name, payload))


def test_health_transition_events_are_content_free() -> None:
    bus = _RecordingBus()
    controller = OpenClawReconnectController(
        health_probe=lambda: {
            "available": False,
            "error_type": "OpenClawConnectionError",
            "error": "secret-bearing upstream text",
            "details": {"token": "do-not-publish"},
        },
        resync=_fresh_success,
        mark_unavailable=lambda: None,
        event_bus=bus,
        auto_reconnect=False,
        wait_fn=lambda _delay: False,
    )
    controller.mark_disconnected("transport_failure")
    controller.run_until_recovered(max_attempts=1)

    assert bus.events
    assert all(name == OPENCLAW_HEALTH_CHANGED_EVENT for name, _payload in bus.events)
    allowed = {"state", "reason_code", "attempt", "next_delay_seconds"}
    assert all(set(payload) <= allowed for _name, payload in bus.events)
    serialized = repr(bus.events)
    assert "secret-bearing" not in serialized
    assert "do-not-publish" not in serialized


class _InventoryClient:
    def __init__(self, inventory: dict[str, Any]) -> None:
        self.inventory = inventory

    def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
        return self.inventory


def _inventory(
    *tool_ids: str,
    effective: list[str] | None = None,
    schema: dict[str, str] | None = None,
) -> dict[str, Any]:
    properties = {name: {"type": kind} for name, kind in (schema or {"query": "string"}).items()}
    return {
        "tools_catalog": {
            "tools": [
                {
                    "id": tool_id,
                    "description": f"Remote {tool_id}",
                    "inputSchema": {"type": "object", "properties": properties},
                }
                for tool_id in tool_ids
            ]
        },
        "effective_tools": {"profile": "full", "found": [["core", effective or []]]},
        "skills_status": {"skills": []},
    }


class _GatewayClient:
    def __init__(self) -> None:
        self.calls = 0

    def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        return {"status": "completed", "path": path, "request": json}


def _local_tool() -> Tool:
    return Tool(
        name="local_clock",
        description="Local clock",
        capability_tags=["clock"],
        requires_config=[],
        handler=lambda: {"time": "now"},
        operation="read",
        risk="safe",
        enabled=True,
        health="healthy",
    )


def test_outage_before_dispatch_blocks_stale_remote_handler_and_keeps_local_tool(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rex.openclaw import http_client

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    tools.register(_local_tool())
    inventory_client = _InventoryClient(_inventory("remote_old", effective=["remote_old"]))
    gateway = _GatewayClient()
    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: gateway)
    sync = OpenClawCapabilitySync(
        cards,
        inventory_client,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot.json",
        session_key="agent:main:main",
    )
    assert sync.refresh().success is True
    stale_handler = tools.get("remote_old").handler  # type: ignore[union-attr]
    reconnect = OpenClawReconnectController(
        health_probe=lambda: {"available": False},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(reconnect)

    reconnect.mark_disconnected("transport_failure")

    assert tools.get("remote_old").enabled is False  # type: ignore[union-attr]
    assert tools.get("local_clock").enabled is True  # type: ignore[union-attr]
    with pytest.raises(OpenClawUnavailableError):
        stale_handler(query="hello")
    assert gateway.calls == 0


def test_recovery_resync_applies_schema_change_and_stale_removal_before_ready(
    tmp_path: Any,
) -> None:
    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    tools.register(_local_tool())
    inventory_client = _InventoryClient(_inventory("remote_old", effective=["remote_old"]))
    sync = OpenClawCapabilitySync(
        cards,
        inventory_client,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot.json",
        session_key="agent:main:main",
    )
    assert sync.refresh().success is True
    reconnect = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(reconnect)
    reconnect.mark_disconnected("transport_failure")
    inventory_client.inventory = _inventory(
        "remote_new",
        effective=["remote_new"],
        schema={"url": "string", "limit": "integer"},
    )

    assert reconnect.run_until_recovered(max_attempts=1) is True
    assert reconnect.state is OpenClawReconnectState.READY

    old = tools.get("remote_old")
    new = tools.get("remote_new")
    local = tools.get("local_clock")
    assert old is not None and old.enabled is False and old.health == "unavailable"
    assert new is not None and new.enabled is True and new.health == "healthy"
    assert new.input_schema == {"limit": "integer", "url": "string"}
    assert local is not None and local.enabled is True


def test_transport_failure_marks_remote_binding_disconnected_and_unavailable(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rex.openclaw import http_client

    class FailingGateway:
        def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            raise OpenClawConnectionError("http://127.0.0.1:18789", ConnectionError("down"))

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    sync = OpenClawCapabilitySync(
        cards,
        _InventoryClient(_inventory("remote_write", effective=["remote_write"])),
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot.json",
        session_key="agent:main:main",
    )
    assert sync.refresh().success is True
    reconnect = OpenClawReconnectController(
        health_probe=lambda: {"available": False},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(reconnect)
    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: FailingGateway())

    with pytest.raises(OpenClawOutcomeUnknownError):
        tools.get("remote_write").handler(value="x")  # type: ignore[union-attr]

    assert reconnect.state is OpenClawReconnectState.DISCONNECTED
    remote = tools.get("remote_write")
    assert remote is not None and remote.enabled is False and remote.health == "unhealthy"


def test_process_initializer_attaches_one_reconnect_controller(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from rex.openclaw import gateway_rpc, http_client

    inventory = _inventory("remote_live", effective=["remote_live"])

    class FakeHealthClient:
        def health(self) -> dict[str, Any]:
            return {"available": True, "status": "up", "details": {"ignored": True}}

    monkeypatch.setattr(
        gateway_rpc,
        "OpenClawGatewayRpcClient",
        lambda *args, **kwargs: _InventoryClient(inventory),
    )
    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: FakeHealthClient())
    config = SimpleNamespace(
        use_openclaw_tools=True,
        openclaw_gateway_url="http://127.0.0.1:18789",
        openclaw_gateway_token="fixture-token",
        openclaw_gateway_timeout=1,
    )
    tools = ToolRegistry(capability_registry=CapabilityRegistry())
    reset_openclaw_capability_sync()
    result = initialize_openclaw_capability_sync(
        config,
        tool_registry=tools,
        event_bus=EventBus(),
        snapshot_path=tmp_path / "snapshot.json",
        reconnect_auto=False,
    )

    assert result is not None and result.success is True
    reconnect = get_openclaw_reconnect_controller()
    assert reconnect is not None
    assert reconnect.state is OpenClawReconnectState.READY
    assert tools.get("remote_live") is not None

    reset_openclaw_capability_sync()
    assert get_openclaw_reconnect_controller() is None


def test_concurrent_disconnect_reports_start_only_one_reconnect_worker() -> None:
    import threading

    disable_entered = threading.Event()
    release_disable = threading.Event()
    probe_entered = threading.Event()
    release_probe = threading.Event()
    disable_calls = 0
    probe_calls = 0

    def disable() -> None:
        nonlocal disable_calls
        disable_calls += 1
        if disable_calls == 1:
            disable_entered.set()
            assert release_disable.wait(timeout=5)

    def probe() -> dict[str, Any]:
        nonlocal probe_calls
        probe_calls += 1
        probe_entered.set()
        release_probe.wait(timeout=5)
        return {"available": False}

    controller = OpenClawReconnectController(
        health_probe=probe,
        resync=_fresh_success,
        mark_unavailable=disable,
        auto_reconnect=True,
        wait_fn=lambda _delay: True,
    )

    caller_errors: list[BaseException] = []

    def report_disconnect() -> None:
        try:
            controller.mark_disconnected()
        except BaseException as exc:
            caller_errors.append(exc)

    first = threading.Thread(target=report_disconnect)
    second = threading.Thread(target=report_disconnect)
    first.start()
    assert disable_entered.wait(timeout=5)
    second.start()
    second.join(timeout=5)
    release_disable.set()
    first.join(timeout=5)
    assert probe_entered.wait(timeout=5)
    release_probe.set()
    controller.close()

    assert caller_errors == []
    assert probe_calls == 1


def test_disconnect_during_resync_prevents_stale_recovery_from_becoming_ready() -> None:
    import threading

    resync_entered = threading.Event()
    release_resync = threading.Event()

    def resync() -> OpenClawSyncResult:
        resync_entered.set()
        assert release_resync.wait(timeout=5)
        return _fresh_success()

    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=resync,
        mark_unavailable=lambda: None,
        auto_reconnect=False,
    )
    controller.mark_disconnected("first_failure")
    outcome: list[bool] = []
    worker = threading.Thread(
        target=lambda: outcome.append(controller.run_until_recovered(max_attempts=1))
    )
    worker.start()
    assert resync_entered.wait(timeout=5)

    controller.mark_disconnected("second_failure")
    release_resync.set()
    worker.join(timeout=5)

    assert outcome == [False]
    assert controller.state is OpenClawReconnectState.DISCONNECTED
    assert controller.ready_for_dispatch is False


def test_close_during_reserved_worker_does_not_start_or_join_unstarted_thread() -> None:
    import threading

    disable_entered = threading.Event()
    release_disable = threading.Event()
    caller_errors: list[BaseException] = []
    probe_calls = 0

    def disable() -> None:
        disable_entered.set()
        assert release_disable.wait(timeout=5)

    def probe() -> dict[str, Any]:
        nonlocal probe_calls
        probe_calls += 1
        return {"available": False}

    controller = OpenClawReconnectController(
        health_probe=probe,
        resync=_fresh_success,
        mark_unavailable=disable,
        auto_reconnect=True,
    )

    def report_disconnect() -> None:
        try:
            controller.mark_disconnected()
        except BaseException as exc:
            caller_errors.append(exc)

    caller = threading.Thread(target=report_disconnect)
    caller.start()
    assert disable_entered.wait(timeout=5)

    controller.close()
    release_disable.set()
    caller.join(timeout=5)

    assert caller_errors == []
    assert controller.state is OpenClawReconnectState.CLOSED
    assert probe_calls == 0


def test_reconnect_backoff_does_not_overflow_after_many_attempts() -> None:
    delays: list[float] = []
    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": False},
        resync=_fresh_success,
        mark_unavailable=lambda: None,
        auto_reconnect=False,
        base_delay_seconds=1.0,
        max_delay_seconds=4.0,
        wait_fn=lambda delay: delays.append(delay) or False,
    )
    controller.mark_disconnected("transport_failure")

    assert controller.run_until_recovered(max_attempts=1100) is False
    assert len(delays) == 1099
    assert delays[:4] == [1.0, 2.0, 4.0, 4.0]
    assert delays[-1] == 4.0


def test_concurrent_recovery_callers_share_single_probe_and_resync() -> None:
    import threading

    probe_entered = threading.Event()
    release_probe = threading.Event()
    probe_calls = 0
    resync_calls = 0
    outcomes: list[bool] = []

    def probe() -> dict[str, Any]:
        nonlocal probe_calls
        probe_calls += 1
        probe_entered.set()
        assert release_probe.wait(timeout=5)
        return {"available": True}

    def resync() -> OpenClawSyncResult:
        nonlocal resync_calls
        resync_calls += 1
        return _fresh_success()

    controller = OpenClawReconnectController(
        health_probe=probe,
        resync=resync,
        mark_unavailable=lambda: None,
        auto_reconnect=False,
    )
    controller.mark_disconnected("transport_failure")

    callers = [
        threading.Thread(
            target=lambda: outcomes.append(controller.run_until_recovered(max_attempts=1))
        )
        for _ in range(2)
    ]
    callers[0].start()
    assert probe_entered.wait(timeout=5)
    callers[1].start()
    release_probe.set()
    for caller in callers:
        caller.join(timeout=5)

    assert sorted(outcomes) == [True, True]
    assert probe_calls == 1
    assert resync_calls == 1
    assert controller.state is OpenClawReconnectState.READY


def test_disconnect_between_client_resolution_and_dispatch_blocks_stale_handler(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from rex.openclaw import http_client

    client_resolved = threading.Event()
    release_client = threading.Event()
    post_calls = 0

    class Gateway:
        def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            nonlocal post_calls
            post_calls += 1
            return {"ok": True}

    gateway = Gateway()

    def resolve_client(_config: Any) -> Gateway:
        client_resolved.set()
        assert release_client.wait(timeout=5)
        return gateway

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    sync = OpenClawCapabilitySync(
        cards,
        _InventoryClient(_inventory("remote_write", effective=["remote_write"])),
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-race.json",
        session_key="agent:main:main",
    )
    assert sync.refresh().success is True
    reconnect = OpenClawReconnectController(
        health_probe=lambda: {"available": False},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(reconnect)
    monkeypatch.setattr(http_client, "get_openclaw_client", resolve_client)

    handler = tools.get("remote_write")
    assert handler is not None
    outcome: list[BaseException | dict[str, Any]] = []

    def invoke() -> None:
        try:
            outcome.append(handler.handler(value="x"))
        except BaseException as exc:
            outcome.append(exc)

    caller = threading.Thread(target=invoke)
    caller.start()
    assert client_resolved.wait(timeout=5)
    reconnect.mark_disconnected("newer_outage")
    release_client.set()
    caller.join(timeout=5)

    assert post_calls == 0
    assert len(outcome) == 1
    assert isinstance(outcome[0], OpenClawUnavailableError)
    assert reconnect.ready_for_dispatch is False


def test_transport_failure_closes_dispatch_gate_before_waiting_caller_can_dispatch(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from rex.openclaw import http_client

    first_entered = threading.Event()
    release_first = threading.Event()
    calls = 0

    class Gateway:
        def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_entered.set()
                assert release_first.wait(timeout=5)
                raise OpenClawConnectionError("http://127.0.0.1:18789", ConnectionError("lost"))
            return {"status": "completed"}

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    sync = OpenClawCapabilitySync(
        cards,
        _InventoryClient(_inventory("remote_write", effective=["remote_write"])),
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-gate.json",
        session_key="agent:main:main",
    )
    assert sync.refresh().success is True
    reconnect = OpenClawReconnectController(
        health_probe=lambda: {"available": False},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(reconnect)
    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: Gateway())
    handler = tools.get("remote_write")
    assert handler is not None
    outcomes: list[BaseException | dict[str, Any]] = []

    def invoke() -> None:
        try:
            outcomes.append(handler.handler(value="x"))
        except BaseException as exc:
            outcomes.append(exc)

    first = threading.Thread(target=invoke)
    second = threading.Thread(target=invoke)
    first.start()
    assert first_entered.wait(timeout=5)
    second.start()
    release_first.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert calls == 1
    assert len(outcomes) == 2
    assert any(isinstance(item, OpenClawOutcomeUnknownError) for item in outcomes)
    assert any(isinstance(item, OpenClawUnavailableError) for item in outcomes)
    assert reconnect.state is OpenClawReconnectState.DISCONNECTED


def test_manual_refresh_failure_closes_gate_for_preselected_handler(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rex.openclaw import http_client

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    inventory = _InventoryClient(_inventory("remote_write", effective=["remote_write"]))
    gateway = _GatewayClient()
    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: gateway)
    sync = OpenClawCapabilitySync(
        cards,
        inventory,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-refresh-fail.json",
        session_key="agent:main:main",
    )
    assert sync.refresh().success is True
    stale_handler = tools.get("remote_write").handler  # type: ignore[union-attr]
    reconnect = OpenClawReconnectController(
        health_probe=lambda: {"available": False},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(reconnect)

    class BrokenInventory:
        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            raise ValueError("malformed remote inventory")

    sync._client = BrokenInventory()  # type: ignore[assignment]
    result = sync.refresh()

    assert result.success is False and result.stale is True
    assert reconnect.state is OpenClawReconnectState.DISCONNECTED
    with pytest.raises(OpenClawUnavailableError):
        stale_handler(value="x")
    assert gateway.calls == 0


def test_outage_after_ready_before_worker_exit_reserves_successor_worker() -> None:
    import threading

    recovered = threading.Event()
    release_first_worker = threading.Event()
    successor_probe = threading.Event()
    probe_calls = 0

    class PausingController(OpenClawReconnectController):
        def run_until_recovered(self, *, max_attempts: int | None = None) -> bool:
            result = super().run_until_recovered(max_attempts=max_attempts)
            if result and not recovered.is_set():
                recovered.set()
                assert release_first_worker.wait(timeout=5)
            return result

    def probe() -> dict[str, Any]:
        nonlocal probe_calls
        probe_calls += 1
        if probe_calls == 1:
            return {"available": True}
        successor_probe.set()
        return {"available": False}

    controller = PausingController(
        health_probe=probe,
        resync=_fresh_success,
        mark_unavailable=lambda: None,
        auto_reconnect=True,
        wait_fn=lambda _delay: True,
    )
    controller.mark_disconnected("first_outage")
    assert recovered.wait(timeout=5)
    assert controller.state is OpenClawReconnectState.READY

    controller.mark_disconnected("second_outage")
    assert controller.state is OpenClawReconnectState.DISCONNECTED
    release_first_worker.set()

    assert successor_probe.wait(timeout=5)
    controller.close()
    assert probe_calls == 2


def test_handler_retained_before_outage_cannot_regain_authority_after_resync(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rex.openclaw import http_client

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    inventory = _InventoryClient(_inventory("remote_old", effective=["remote_old"]))
    gateway = _GatewayClient()
    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: gateway)
    sync = OpenClawCapabilitySync(
        cards,
        inventory,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-stale-handler.json",
    )
    assert sync.refresh().success is True
    stale_handler = tools.get("remote_old").handler  # type: ignore[union-attr]
    reconnect = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(reconnect)

    reconnect.mark_disconnected("transport_failure")
    inventory.inventory = _inventory("remote_new", effective=["remote_new"])
    assert reconnect.run_until_recovered(max_attempts=1) is True
    assert reconnect.ready_for_dispatch is True

    with pytest.raises(OpenClawUnavailableError):
        stale_handler(query="hello")
    assert gateway.calls == 0

    current = tools.get("remote_new")
    assert current is not None and current.enabled is True
    assert current.handler(query="hello")["status"] == "completed"
    assert gateway.calls == 1


def test_successful_hot_refresh_cannot_race_stale_handler_past_new_snapshot(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from rex.openclaw import http_client

    client_resolution_started = threading.Event()
    release_client = threading.Event()
    gateway = _GatewayClient()

    def delayed_client(_config: Any) -> _GatewayClient:
        client_resolution_started.set()
        assert release_client.wait(timeout=5)
        return gateway

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    inventory = _InventoryClient(_inventory("remote_old", effective=["remote_old"]))
    sync = OpenClawCapabilitySync(
        cards,
        inventory,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-hot-refresh-race.json",
    )
    assert sync.refresh().success is True
    reconnect = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(reconnect)
    stale_handler = tools.get("remote_old").handler  # type: ignore[union-attr]
    monkeypatch.setattr(http_client, "get_openclaw_client", delayed_client)
    outcome: list[BaseException | dict[str, Any]] = []

    def invoke_old() -> None:
        try:
            outcome.append(stale_handler(query="hello"))
        except BaseException as exc:
            outcome.append(exc)

    caller = threading.Thread(target=invoke_old)
    caller.start()
    assert client_resolution_started.wait(timeout=5)

    inventory.inventory = _inventory("remote_new", effective=["remote_new"])
    assert sync.refresh().success is True
    release_client.set()
    caller.join(timeout=5)

    assert gateway.calls == 0
    assert len(outcome) == 1
    assert isinstance(outcome[0], OpenClawUnavailableError)
    assert tools.get("remote_new") is not None


def test_delayed_outage_projection_cannot_disable_freshly_recovered_authority() -> None:
    import threading
    import time

    projection_started = threading.Event()
    release_projection = threading.Event()
    recovery_done = threading.Event()
    binding_enabled = True

    def mark_unavailable() -> None:
        nonlocal binding_enabled
        projection_started.set()
        assert release_projection.wait(timeout=5)
        binding_enabled = False

    def resync() -> OpenClawSyncResult:
        nonlocal binding_enabled
        binding_enabled = True
        return _fresh_success()

    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=resync,
        mark_unavailable=mark_unavailable,
        auto_reconnect=False,
    )

    disconnect_thread = threading.Thread(target=controller.mark_disconnected)
    disconnect_thread.start()
    assert projection_started.wait(timeout=5)

    def recover() -> None:
        assert controller.run_until_recovered(max_attempts=1) is True
        recovery_done.set()

    recovery_thread = threading.Thread(target=recover)
    recovery_thread.start()
    time.sleep(0.05)

    release_projection.set()
    disconnect_thread.join(timeout=5)
    recovery_thread.join(timeout=5)

    assert recovery_done.is_set()
    assert controller.state is OpenClawReconnectState.READY
    assert binding_enabled is True


def test_newer_outage_during_resync_cannot_republish_stale_remote_authority(
    tmp_path: Any,
) -> None:
    import threading

    second_fetch_started = threading.Event()
    release_second_fetch = threading.Event()

    class BlockingInventory:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 2:
                second_fetch_started.set()
                assert release_second_fetch.wait(timeout=5)
            return _inventory("remote", effective=["remote"])

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    inventory = BlockingInventory()
    sync = OpenClawCapabilitySync(
        cards,
        inventory,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-stale-resync.json",
    )
    assert sync.refresh().success is True
    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(controller)
    controller.mark_disconnected("first_outage")

    recovered: list[bool] = []
    worker = threading.Thread(
        target=lambda: recovered.append(controller.run_until_recovered(max_attempts=1))
    )
    worker.start()
    assert second_fetch_started.wait(timeout=5)

    controller.mark_disconnected("newer_outage")
    stale = tools.get("remote")
    assert stale is not None and stale.enabled is False

    release_second_fetch.set()
    worker.join(timeout=5)

    current = tools.get("remote")
    assert recovered == [False]
    assert controller.state is OpenClawReconnectState.DISCONNECTED
    assert current is not None and current.enabled is False
    assert current.health == "unhealthy"


def test_superseding_disconnect_does_not_strand_reserved_reconnect_worker() -> None:
    import threading

    first_projection_entered = threading.Event()
    release_first_projection = threading.Event()
    probe_started = threading.Event()

    class PausingProjectionController(OpenClawReconnectController):
        def _project_unavailable_and_start(self, reservation: Any) -> None:
            if reservation.worker is not None and not first_projection_entered.is_set():
                first_projection_entered.set()
                assert release_first_projection.wait(timeout=5)
            super()._project_unavailable_and_start(reservation)

    controller = PausingProjectionController(
        health_probe=lambda: probe_started.set() or {"available": False},
        resync=_fresh_success,
        mark_unavailable=lambda: None,
        auto_reconnect=True,
        wait_fn=lambda _delay: True,
    )

    first = threading.Thread(target=lambda: controller.mark_disconnected("first"))
    first.start()
    assert first_projection_entered.wait(timeout=5)

    controller.mark_disconnected("second")
    release_first_projection.set()
    first.join(timeout=5)

    assert probe_started.wait(timeout=2)
    controller.close()


def test_manual_refresh_started_before_outage_cannot_publish_after_newer_generation(
    tmp_path: Any,
) -> None:
    import threading

    refresh_started = threading.Event()
    release_refresh = threading.Event()

    class BlockingInventory:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 2:
                refresh_started.set()
                assert release_refresh.wait(timeout=5)
            return _inventory("remote", effective=["remote"])

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    inventory = BlockingInventory()
    sync = OpenClawCapabilitySync(
        cards,
        inventory,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-stale-manual-refresh.json",
    )
    assert sync.refresh().success is True
    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": False},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(controller)
    result: list[OpenClawSyncResult] = []
    worker = threading.Thread(target=lambda: result.append(sync.refresh()))
    worker.start()
    assert refresh_started.wait(timeout=5)

    controller.mark_disconnected("newer_outage")
    assert tools.get("remote") is not None and tools.get("remote").enabled is False  # type: ignore[union-attr]

    release_refresh.set()
    worker.join(timeout=5)

    current = tools.get("remote")
    assert current is not None and current.enabled is False
    assert current.health == "unhealthy"
    assert controller.state is OpenClawReconnectState.DISCONNECTED
    assert result and result[0].success is False and result[0].stale is True


def test_health_probe_from_superseded_generation_cannot_publish_recovery() -> None:
    import threading

    probe_started = threading.Event()
    release_probe = threading.Event()
    resync_calls = 0

    def health_probe() -> dict[str, Any]:
        probe_started.set()
        assert release_probe.wait(timeout=5)
        return {"available": True}

    def resync() -> OpenClawSyncResult:
        nonlocal resync_calls
        resync_calls += 1
        return _fresh_success()

    controller = OpenClawReconnectController(
        health_probe=health_probe,
        resync=resync,
        mark_unavailable=lambda: None,
        auto_reconnect=False,
    )
    controller.mark_disconnected("first_outage")

    recovered: list[bool] = []
    worker = threading.Thread(
        target=lambda: recovered.append(controller.run_until_recovered(max_attempts=1))
    )
    worker.start()
    assert probe_started.wait(timeout=5)

    controller.mark_disconnected("newer_outage")
    release_probe.set()
    worker.join(timeout=5)

    assert recovered == [False]
    assert controller.state is OpenClawReconnectState.DISCONNECTED
    assert resync_calls == 0


def test_obsolete_refresh_failure_cannot_disable_newer_recovered_authority(tmp_path: Any) -> None:
    import threading

    manual_started = threading.Event()
    release_manual = threading.Event()

    class Inventory:
        def __init__(self) -> None:
            self.calls = 0

        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 2:
                manual_started.set()
                assert release_manual.wait(timeout=5)
            return _inventory("remote", effective=["remote"])

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    inventory = Inventory()
    sync = OpenClawCapabilitySync(
        cards,
        inventory,
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-obsolete-refresh.json",
    )
    assert sync.refresh().success is True
    controller = OpenClawReconnectController(
        health_probe=lambda: {"available": True},
        resync=sync.refresh,
        mark_unavailable=sync.mark_unavailable,
        auto_reconnect=False,
    )
    sync.attach_reconnect_controller(controller)
    manual_results: list[OpenClawSyncResult] = []
    manual = threading.Thread(target=lambda: manual_results.append(sync.refresh()))
    manual.start()
    assert manual_started.wait(timeout=5)

    controller.mark_disconnected("outage")
    assert controller.run_until_recovered(max_attempts=1) is True
    assert controller.state is OpenClawReconnectState.READY
    recovered = tools.get("remote")
    assert recovered is not None and recovered.enabled is True

    release_manual.set()
    manual.join(timeout=5)

    current = tools.get("remote")
    assert manual_results and manual_results[0].success is False
    assert manual_results[0].stale is True
    assert controller.state is OpenClawReconnectState.READY
    assert current is not None and current.enabled is True and current.health == "healthy"
