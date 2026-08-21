from __future__ import annotations

from typing import Any

import pytest

from rex.openclaw.errors import OpenClawOutcomeUnknownError
from rex.tools.execution import ToolExecutionLifecycle, ToolOutcome
from rex.tools.registry import Tool


def _mutation_tool(handler: Any, *, verifier: Any = None) -> Tool:
    return Tool(
        name="remote_mutation",
        description="Remote mutation",
        capability_tags=["remote"],
        requires_config=[],
        handler=handler,
        source="openclaw",
        operation="mutation",
        risk="sensitive",
        requires_identity=True,
        verifier=verifier,
        required_permissions=("openclaw_execute",),
        enabled=True,
        health="healthy",
    )


def _context() -> dict[str, Any]:
    return {
        "user_id": "james",
        "confirmed": True,
        "granted_permissions": {"openclaw_execute"},
    }


def test_remote_mutation_outcome_unknown_is_attempted_unverified_without_retry() -> None:
    calls = 0

    def handler(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise OpenClawOutcomeUnknownError()

    result = ToolExecutionLifecycle().execute(
        _mutation_tool(handler),
        {"value": "x"},
        _context(),
    )

    assert calls == 1
    assert result.status == ToolOutcome.ATTEMPTED_UNVERIFIED.value
    assert result.success is False
    assert result.lifecycle is not None
    assert result.lifecycle.state.value == "unverified"
    assert "could not verify" in result.detail.lower()


def test_remote_self_declared_verified_cannot_promote_mutation() -> None:
    result = ToolExecutionLifecycle().execute(
        _mutation_tool(lambda **_kwargs: {"status": "verified", "receipt": "remote-claim"}),
        {"value": "x"},
        _context(),
    )

    assert result.status == ToolOutcome.ATTEMPTED_UNVERIFIED.value
    assert result.success is False


def test_rex_verifier_can_promote_remote_mutation_after_independent_postcondition() -> None:
    def verifier(_args: dict[str, Any], output: Any) -> bool:
        return isinstance(output, dict) and output.get("receipt") == "trusted-postcondition"

    result = ToolExecutionLifecycle().execute(
        _mutation_tool(
            lambda **_kwargs: {
                "status": "verified",
                "receipt": "trusted-postcondition",
            },
            verifier=verifier,
        ),
        {"value": "x"},
        _context(),
    )

    assert result.status == ToolOutcome.VERIFIED.value
    assert result.success is True
    assert result.lifecycle is not None
    assert result.lifecycle.state.value == "verified"


def test_outcome_unknown_read_uses_existing_transient_retry_policy() -> None:
    calls = 0

    def handler(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OpenClawOutcomeUnknownError()
        return {"value": "ok"}

    read_tool = Tool(
        name="remote_read",
        description="Remote read",
        capability_tags=["remote"],
        requires_config=[],
        handler=handler,
        source="openclaw",
        operation="read",
        risk="safe",
        enabled=True,
        health="healthy",
    )
    result = ToolExecutionLifecycle().execute(read_tool, {}, {})

    assert calls == 2
    assert result.status == ToolOutcome.COMPLETED.value
    assert result.success is True


def test_dynamic_openclaw_transport_loss_becomes_attempted_unverified(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rex.capabilities.registry import CapabilityRegistry
    from rex.openclaw import http_client
    from rex.openclaw.capability_sync import OpenClawCapabilitySync
    from rex.openclaw.errors import OpenClawConnectionError
    from rex.openclaw.reconnect import OpenClawReconnectController
    from rex.tools.registry import ToolRegistry

    class Inventory:
        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            return {
                "tools_catalog": {
                    "tools": [
                        {
                            "id": "remote_write",
                            "description": "Remote write",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"value": {"type": "string"}},
                            },
                        }
                    ]
                },
                "effective_tools": {
                    "profile": "full",
                    "found": [["core", ["remote_write"]]],
                },
                "skills_status": {"skills": []},
            }

    class Gateway:
        def post(self, path: str, *, json: dict[str, Any]) -> dict[str, Any]:
            raise OpenClawConnectionError(
                "http://127.0.0.1:18789", ConnectionError("connection lost")
            )

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    sync = OpenClawCapabilitySync(
        cards,
        Inventory(),
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
    monkeypatch.setattr(http_client, "get_openclaw_client", lambda _config: Gateway())

    remote = tools.get("remote_write")
    assert remote is not None
    result = ToolExecutionLifecycle().execute(
        remote,
        {"value": "x"},
        _context(),
    )

    assert result.status == ToolOutcome.ATTEMPTED_UNVERIFIED.value
    assert result.success is False
    assert reconnect.ready_for_dispatch is False


def test_tools_invoke_transport_can_disable_client_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from requests.exceptions import ConnectionError as RequestsConnectionError

    from rex.openclaw import http_client
    from rex.openclaw.errors import OpenClawConnectionError
    from rex.openclaw.http_client import OpenClawClient

    client = OpenClawClient(
        "http://127.0.0.1:18789",
        "fixture-token",
        timeout=1,
        max_retries=3,
    )
    calls = 0

    def fail_request(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise RequestsConnectionError("connection lost")

    monkeypatch.setattr(client._session, "request", fail_request)
    monkeypatch.setattr(http_client, "_wait_retry", lambda _delay: None)

    with pytest.raises(OpenClawConnectionError):
        client.post("/tools/invoke", json={"tool": "write", "args": {}})

    assert calls == 1


def test_tools_invoke_5xx_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rex.openclaw.errors import OpenClawAPIError
    from rex.openclaw.http_client import OpenClawClient

    client = OpenClawClient(
        "http://127.0.0.1:18789",
        "fixture-token",
        timeout=1,
        max_retries=3,
    )
    response = type(
        "Response",
        (),
        {
            "status_code": 503,
            "text": "service unavailable",
            "headers": {},
            "content": b"service unavailable",
        },
    )()
    calls = 0

    def fail_response(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return response

    monkeypatch.setattr(client._session, "request", fail_response)

    with pytest.raises(OpenClawAPIError):
        client.post("/tools/invoke", json={"tool": "write", "args": {}})

    assert calls == 1


def test_tools_invoke_malformed_success_body_becomes_outcome_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rex.openclaw.errors import OpenClawOutcomeUnknownError
    from rex.openclaw.http_client import OpenClawClient

    client = OpenClawClient("http://127.0.0.1:18789", "fixture-token", timeout=1)

    class Response:
        status_code = 200
        text = "not-json"
        headers: dict[str, str] = {}
        content = b"not-json"

        def json(self) -> dict[str, Any]:
            raise ValueError("malformed JSON")

    monkeypatch.setattr(client._session, "request", lambda *_args, **_kwargs: Response())

    with pytest.raises(OpenClawOutcomeUnknownError):
        client.post("/tools/invoke", json={"tool": "write", "args": {}})


def test_dynamic_non_object_response_is_attempted_unverified(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rex.capabilities.registry import CapabilityRegistry
    from rex.openclaw import http_client
    from rex.openclaw.capability_sync import OpenClawCapabilitySync
    from rex.openclaw.reconnect import OpenClawReconnectController
    from rex.tools.registry import ToolRegistry

    class Inventory:
        def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
            return {
                "tools_catalog": {"tools": [{"id": "remote_write", "description": "write"}]},
                "effective_tools": {"found": [["core", ["remote_write"]]]},
                "skills_status": {"skills": []},
            }

    class Gateway:
        def post(self, path: str, *, json: dict[str, Any]) -> Any:
            return ["unexpected", "response"]

    cards = CapabilityRegistry()
    tools = ToolRegistry(capability_registry=cards)
    sync = OpenClawCapabilitySync(
        cards,
        Inventory(),
        tool_registry=tools,
        runtime_config=object(),
        snapshot_path=tmp_path / "snapshot-non-object.json",
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
    remote = tools.get("remote_write")
    assert remote is not None

    result = ToolExecutionLifecycle().execute(
        remote,
        {},
        _context(),
    )

    assert result.status == ToolOutcome.ATTEMPTED_UNVERIFIED.value
    assert result.success is False


def test_outcome_unknown_mutation_can_be_verified_by_trusted_postcondition() -> None:
    calls: list[tuple[dict[str, Any], Any]] = []

    def handler(**_kwargs: Any) -> dict[str, Any]:
        raise OpenClawOutcomeUnknownError()

    def verifier(args: dict[str, Any], output: Any) -> bool:
        calls.append((args, output))
        return args == {"value": "x"} and output is None

    tool = Tool(
        name="remote_write_verified_after_unknown",
        description="Remote write with independent postcondition",
        capability_tags=["remote"],
        requires_config=[],
        handler=handler,
        source="openclaw",
        operation="mutation",
        risk="sensitive",
        requires_identity=True,
        verifier=verifier,
        enabled=True,
        health="healthy",
    )

    result = ToolExecutionLifecycle().execute(tool, {"value": "x"}, _context())

    assert result.status == ToolOutcome.VERIFIED.value
    assert result.success is True
    assert "verified" in result.detail.lower()
    assert "outcome is unknown" not in result.detail.lower()
    assert calls == [({"value": "x"}, None)]
