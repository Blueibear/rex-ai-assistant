from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from requests.exceptions import ConnectionError as RequestsConnectionError

from rex.openclaw.errors import OpenClawAPIError, OpenClawConnectionError
from rex.openclaw.http_client import OpenClawClient
from rex.openclaw.tool_bridge import ToolBridge


def _response(status: int, payload: dict | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    response.text = "service unavailable" if status >= 500 else ""
    response.content = b"..." if payload is not None else b""
    response.json.return_value = payload or {}
    return response


def test_healthz_reports_gateway_up() -> None:
    client = OpenClawClient("http://openclaw.test:18789", "token", max_retries=0)
    with patch.object(
        client._session,
        "request",
        return_value=_response(200, {"status": "ok"}),
    ) as request:
        health = client.health()

    assert health["available"] is True
    assert health["status"] == "up"
    assert health["details"] == {"status": "ok"}
    assert health["latency_ms"] >= 0
    assert request.call_args.args[:2] == ("GET", "http://openclaw.test:18789/healthz")


def test_healthz_retries_connection_failure_only_to_configured_bound() -> None:
    client = OpenClawClient("http://openclaw.test:18789", "token", max_retries=2)
    with patch.object(
        client._session,
        "request",
        side_effect=RequestsConnectionError("refused"),
    ) as request:
        with patch("rex.openclaw.http_client.time.sleep") as sleep:
            health = client.health()

    assert health["available"] is False
    assert health["status"] == "down"
    assert health["error_type"] == "OpenClawConnectionError"
    assert request.call_count == 3
    assert sleep.call_count == 2


def test_healthz_recovers_on_later_probe() -> None:
    client = OpenClawClient("http://openclaw.test:18789", "token", max_retries=0)
    with patch.object(
        client._session,
        "request",
        side_effect=[
            RequestsConnectionError("refused"),
            _response(200, {"status": "ok"}),
        ],
    ):
        first = client.health()
        second = client.health()

    assert first["available"] is False
    assert second["available"] is True


def test_tool_gateway_failure_falls_back_with_structured_warning_and_recovers(caplog) -> None:
    config = SimpleNamespace(use_openclaw_tools=True)
    client = MagicMock()
    client.post.side_effect = [
        OpenClawConnectionError("http://openclaw.test:18789", Exception("refused")),
        {"status": "success", "result": "remote"},
    ]
    bridge = ToolBridge(config=config)
    local = {"status": "ok", "result": "local"}

    with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=local) as local_execute:
            with caplog.at_level(logging.WARNING, logger="rex.openclaw.tool_bridge"):
                first = bridge.execute_tool({"tool": "time_now", "args": {}}, {})
                second = bridge.execute_tool({"tool": "time_now", "args": {}}, {})

    assert first == local
    assert second == {"status": "success", "result": "remote"}
    local_execute.assert_called_once()
    fallback = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "openclaw.tool_fallback"
    )
    assert fallback.tool_name == "time_now"
    assert fallback.failure == "OpenClawConnectionError"


def test_exhausted_server_retries_fall_back_locally_with_warning(caplog) -> None:
    config = SimpleNamespace(use_openclaw_tools=True)
    client = MagicMock()
    client.post.side_effect = OpenClawAPIError(503, "service unavailable")
    bridge = ToolBridge(config=config)
    local = {"status": "ok", "result": "local"}

    with patch("rex.openclaw.tool_bridge.get_openclaw_client", return_value=client):
        with patch("rex.openclaw.tool_bridge._execute_tool", return_value=local):
            with caplog.at_level(logging.WARNING, logger="rex.openclaw.tool_bridge"):
                result = bridge.execute_tool({"tool": "weather_now", "args": {}}, {})

    assert result == local
    fallback = next(
        record
        for record in caplog.records
        if getattr(record, "event", None) == "openclaw.tool_fallback"
    )
    assert fallback.tool_name == "weather_now"
    assert fallback.failure == "OpenClawAPIError:503"
