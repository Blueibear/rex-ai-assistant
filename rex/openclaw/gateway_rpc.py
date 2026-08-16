"""Least-privilege OpenClaw Gateway control-plane discovery client.

US-113 uses the Gateway WebSocket protocol only for read-only capability
inventory. Execution continues through the existing HTTP ``/tools/invoke``
path. The backend-mode device exemption is accepted only for direct loopback
connections authenticated with the configured shared gateway token.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Callable
from contextlib import suppress
from typing import Any, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from rex.openclaw.errors import (
    OpenClawConfigError,
    OpenClawConnectionError,
    OpenClawProtocolError,
)

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = 4
_MAX_FRAMES_PER_REQUEST = 128
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


class _GatewaySocket(Protocol):
    def recv(self) -> str: ...

    def send(self, data: str) -> Any: ...

    def close(self) -> Any: ...


Connector = Callable[[str, float], _GatewaySocket]


def _gateway_ws_url(base_url: str) -> str:
    parsed = urlsplit(base_url.strip())
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.hostname:
        raise OpenClawConfigError("gateway URL must be an absolute HTTP(S) or WS(S) URL")
    if parsed.hostname.lower() not in _LOOPBACK_HOSTS:
        raise OpenClawConfigError(
            "read-only backend capability discovery currently requires a loopback OpenClaw gateway; "
            "remote gateways require a paired device identity"
        )
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    path = parsed.path.rstrip("/")
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


def _default_connector(url: str, timeout: float) -> _GatewaySocket:
    try:
        import websocket
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise OpenClawConfigError(
            "websocket-client is required for OpenClaw capability discovery"
        ) from exc
    return cast(_GatewaySocket, websocket.create_connection(url, timeout=timeout))


class OpenClawGatewayRpcClient:
    """Authenticated ``operator.read`` client for Gateway inventory RPC."""

    def __init__(
        self,
        base_url: str,
        auth_token: str,
        *,
        timeout: float = 5.0,
        connector: Connector | None = None,
    ) -> None:
        if not auth_token:
            raise OpenClawConfigError("gateway token is required for capability discovery")
        self._ws_url = _gateway_ws_url(base_url)
        self._auth_token = auth_token
        self._timeout = max(0.1, float(timeout))
        self._connector = connector or _default_connector

    def fetch_capability_inventory(self, *, session_key: str | None = None) -> dict[str, Any]:
        """Return tool catalog, skill status, and optional session-effective tools."""
        sock: _GatewaySocket | None = None
        try:
            try:
                sock = self._connector(self._ws_url, self._timeout)
            except Exception as exc:
                # Never copy transport exception text into the user-visible error;
                # it may contain the token, headers, or remote free-form text.
                safe_cause = RuntimeError(type(exc).__name__)
                raise OpenClawConnectionError(self._ws_url, safe_cause) from exc
            self._connect(sock)
            catalog = self._request(sock, "tools.catalog", {})
            skills = self._request(sock, "skills.status", {})
            effective = None
            if session_key:
                effective = self._request(sock, "tools.effective", {"sessionKey": session_key})
            return {
                "tools_catalog": catalog,
                "skills_status": skills,
                "effective_tools": effective,
            }
        finally:
            if sock is not None:
                with suppress(Exception):
                    sock.close()

    def _connect(self, sock: _GatewaySocket) -> None:
        challenge = self._recv_json(sock)
        if (
            challenge.get("type") != "event"
            or challenge.get("event") != "connect.challenge"
            or not isinstance(challenge.get("payload"), dict)
            or not str(challenge["payload"].get("nonce", "")).strip()
        ):
            raise OpenClawProtocolError("expected connect.challenge")

        request_id = uuid.uuid4().hex
        payload = {
            "type": "req",
            "id": request_id,
            "method": "connect",
            "params": {
                "minProtocol": _PROTOCOL_VERSION,
                "maxProtocol": _PROTOCOL_VERSION,
                "client": {
                    "id": "gateway-client",
                    "version": "askrex-1",
                    "platform": sys.platform,
                    "mode": "backend",
                },
                "role": "operator",
                "scopes": ["operator.read"],
                "caps": [],
                "commands": [],
                "permissions": {},
                "auth": {"token": self._auth_token},
            },
        }
        sock.send(json.dumps(payload, separators=(",", ":")))
        response = self._wait_for_response(sock, request_id, "connect")
        hello = response.get("payload")
        if not isinstance(hello, dict) or hello.get("type") != "hello-ok":
            raise OpenClawProtocolError("connect did not return hello-ok")
        if hello.get("protocol") != _PROTOCOL_VERSION:
            raise OpenClawProtocolError("gateway protocol version mismatch")
        auth = hello.get("auth")
        if not isinstance(auth, dict):
            raise OpenClawProtocolError("hello-ok missing auth scope evidence")
        scopes = auth.get("scopes")
        if not isinstance(scopes, list) or "operator.read" not in scopes:
            raise OpenClawProtocolError("gateway did not grant operator.read")

    def _request(self, sock: _GatewaySocket, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        sock.send(
            json.dumps(
                {"type": "req", "id": request_id, "method": method, "params": params},
                separators=(",", ":"),
            )
        )
        response = self._wait_for_response(sock, request_id, method)
        payload = response.get("payload")
        if not isinstance(payload, dict):
            raise OpenClawProtocolError(f"{method} returned a non-object payload")
        return payload

    def _wait_for_response(
        self, sock: _GatewaySocket, request_id: str, method: str
    ) -> dict[str, Any]:
        for _ in range(_MAX_FRAMES_PER_REQUEST):
            frame = self._recv_json(sock)
            if frame.get("type") != "res" or frame.get("id") != request_id:
                # Events and unrelated responses are allowed between request/response frames.
                continue
            if frame.get("ok") is not True:
                error = frame.get("error")
                code = "UNKNOWN"
                if isinstance(error, dict) and isinstance(error.get("code"), str):
                    code = error["code"][:80]
                raise OpenClawProtocolError(f"{method} failed with code {code}")
            return frame
        raise OpenClawProtocolError(f"{method} response frame limit exceeded")

    @staticmethod
    def _recv_json(sock: _GatewaySocket) -> dict[str, Any]:
        try:
            raw = sock.recv()
            value = json.loads(raw)
        except Exception as exc:
            raise OpenClawProtocolError("gateway returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise OpenClawProtocolError("gateway frame must be an object")
        return value


__all__ = ["OpenClawGatewayRpcClient"]
