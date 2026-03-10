"""HTTP client for the Rex Windows computer agent API.

This client implements the agent API contract (Cycle 5.1 client-side only).
The agent server itself is not part of this cycle (that is Cycle 5.3).

Agent API contract
------------------
All requests include the ``X-Auth-Token`` header for authentication.

- ``GET /health``
    Response 200 JSON: ``{"status": "ok"}``

- ``GET /status``
    Response 200 JSON with host info::

        {
          "hostname": "DESKTOP-123",
          "os": "Windows 10",
          "user": "alice",
          "time": "2026-02-23T14:30:00"
        }

- ``POST /run``
    Request body::

        {"command": "whoami", "args": [], "cwd": null}

    Response 200 JSON::

        {"exit_code": 0, "stdout": "alice\\n", "stderr": ""}

Security notes
--------------
- Tokens are **never** logged.  Only safe labels (computer id, hostname) appear
  in log output.
- This client makes no attempt to validate TLS certificates beyond what the
  underlying HTTP library does.  For production use, ensure the agent uses TLS
  and configure certificate verification appropriately.

Dependencies
------------
Uses ``requests`` if available; falls back to ``urllib.request`` from stdlib.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

# Auth header name used by this client and expected by the agent server.
AUTH_HEADER = "X-Auth-Token"


@dataclass
class HealthResult:
    """Result of a ``GET /health`` call."""

    ok: bool
    raw: dict[str, Any]
    error: str | None = None


@dataclass
class StatusResult:
    """Result of a ``GET /status`` call."""

    ok: bool
    hostname: str = ""
    os: str = ""
    user: str = ""
    time: str = ""
    raw: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class RunResult:
    """Result of a ``POST /run`` call."""

    ok: bool
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""
    error: str | None = None


class AgentClient:
    """HTTP client that speaks the Rex agent API.

    Parameters
    ----------
    base_url:
        Base URL of the agent (e.g. ``"http://127.0.0.1:7777"``).
    token:
        Auth token sent as ``X-Auth-Token`` header.
    connect_timeout:
        Seconds before a connection attempt times out.
    read_timeout:
        Seconds before a read operation times out.
    computer_id:
        Logical label used in log messages (not the token — safe to log).
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        computer_id: str = "",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        # Safe label for logging — computer id and hostname only, never the token.
        self._label = computer_id or urlparse(base_url).hostname or base_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> HealthResult:
        """Call ``GET /health`` on the agent.

        Returns:
            :class:`HealthResult` with ``ok=True`` when the agent is reachable
            and reports a healthy status.
        """
        url = self._url("/health")
        logger.debug("health check for %s -> %s", self._label, url)
        try:
            data = self._get(url)
            ok = isinstance(data, dict) and data.get("status") == "ok"
            return HealthResult(ok=ok, raw=data if isinstance(data, dict) else {})
        except Exception as exc:  # noqa: BLE001
            logger.warning("health check failed for %s: %s", self._label, exc)
            return HealthResult(ok=False, raw={}, error=str(exc))

    def status(self) -> StatusResult:
        """Call ``GET /status`` on the agent.

        Returns:
            :class:`StatusResult` with host details, or ``ok=False`` on failure.
        """
        url = self._url("/status")
        logger.debug("status request for %s -> %s", self._label, url)
        try:
            data = self._get(url)
            if not isinstance(data, dict):
                return StatusResult(ok=False, error="Unexpected response format")
            return StatusResult(
                ok=True,
                hostname=str(data.get("hostname", "")),
                os=str(data.get("os", "")),
                user=str(data.get("user", "")),
                time=str(data.get("time", "")),
                raw=data,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("status request failed for %s: %s", self._label, exc)
            return StatusResult(ok=False, error=str(exc))

    def run(
        self,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
    ) -> RunResult:
        """Call ``POST /run`` on the agent to execute a command.

        Parameters
        ----------
        command:
            The command name to run (e.g. ``"whoami"``).
        args:
            Optional list of arguments.
        cwd:
            Optional working directory on the remote computer.

        Returns:
            :class:`RunResult` with stdout/stderr and exit code.
        """
        url = self._url("/run")
        # Only log the command name — args might contain sensitive values.
        logger.debug("run command=%r on %s", command, self._label)
        payload: dict[str, Any] = {"command": command, "args": args or [], "cwd": cwd}
        try:
            data = self._post(url, payload)
            if not isinstance(data, dict):
                return RunResult(ok=False, error="Unexpected response format")
            exit_code = int(data.get("exit_code", -1))
            return RunResult(
                ok=exit_code == 0,
                exit_code=exit_code,
                stdout=str(data.get("stdout", "")),
                stderr=str(data.get("stderr", "")),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("run command=%r failed on %s: %s", command, self._label, exc)
            return RunResult(ok=False, error=str(exc))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return urljoin(self._base_url + "/", path.lstrip("/"))

    def _headers(self) -> dict[str, str]:
        return {
            AUTH_HEADER: self._token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _get(self, url: str) -> Any:
        """Perform a GET request and return parsed JSON.

        Tries ``requests`` first; falls back to ``urllib.request``.
        """
        try:
            return self._get_requests(url)
        except ImportError:
            return self._get_urllib(url)

    def _post(self, url: str, payload: dict[str, Any]) -> Any:
        """Perform a POST request and return parsed JSON.

        Tries ``requests`` first; falls back to ``urllib.request``.
        """
        try:
            return self._post_requests(url, payload)
        except ImportError:
            return self._post_urllib(url, payload)

    # --- requests backend ---

    def _get_requests(self, url: str) -> Any:
        import requests

        resp = requests.get(
            url,
            headers=self._headers(),
            timeout=(self._connect_timeout, self._read_timeout),
        )
        resp.raise_for_status()
        return resp.json()

    def _post_requests(self, url: str, payload: dict[str, Any]) -> Any:
        import requests

        resp = requests.post(
            url,
            json=payload,
            headers=self._headers(),
            timeout=(self._connect_timeout, self._read_timeout),
        )
        resp.raise_for_status()
        return resp.json()

    # --- urllib backend ---

    def _get_urllib(self, url: str) -> Any:
        import urllib.request

        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(  # noqa: S310
            req, timeout=max(self._connect_timeout, self._read_timeout)
        ) as resp:
            return json.loads(resp.read().decode())

    def _post_urllib(self, url: str, payload: dict[str, Any]) -> Any:
        import urllib.request

        body = json.dumps(payload).encode()
        req = urllib.request.Request(url, data=body, headers=self._headers(), method="POST")
        with urllib.request.urlopen(  # noqa: S310
            req, timeout=max(self._connect_timeout, self._read_timeout)
        ) as resp:
            return json.loads(resp.read().decode())


__all__ = [
    "AgentClient",
    "AUTH_HEADER",
    "HealthResult",
    "RunResult",
    "StatusResult",
]
