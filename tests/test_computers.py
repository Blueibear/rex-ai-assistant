"""Offline tests for the rex.computers package (Cycle 5.1 client foundation).

All tests run without network access.  The agent HTTP layer is replaced by a
tiny in-process fake server fixture or simple mock.

Coverage targets
----------------
- Config parsing and validation for ``computers[]``
- Allowlist enforcement blocks disallowed commands without any network call
- ``status`` call uses fake server, includes ``X-Auth-Token`` header, handles timeout
- ``run`` call uses fake server, includes ``X-Auth-Token`` header, returns parsed result
- Disabled computer is handled correctly
- Missing token produces a clear error
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import MagicMock

import pytest

from rex.cli import cmd_pc
from rex.computers.client import AUTH_HEADER, AgentClient
from rex.computers.config import (
    ComputerAllowlists,
    ComputerConfig,
    ComputersConfig,
    load_computers_config,
)
from rex.computers.service import (
    AllowlistDeniedError,
    ComputerDisabledError,
    ComputerInfo,
    ComputerNotFoundError,
    ComputerService,
    MissingTokenError,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_config(
    *,
    computer_id: str = "desktop",
    label: str = "Main Desktop",
    base_url: str = "http://127.0.0.1:7777",
    auth_token_ref: str = "PC_DESKTOP_TOKEN",
    enabled: bool = True,
    commands: list[str] | None = None,
) -> ComputerConfig:
    """Build a :class:`ComputerConfig` for testing."""
    if commands is None:
        commands = ["whoami", "dir", "ipconfig", "systeminfo"]
    return ComputerConfig(
        id=computer_id,
        label=label,
        base_url=base_url,
        auth_token_ref=auth_token_ref,
        enabled=enabled,
        allowlists=ComputerAllowlists(commands=commands),
    )


def _make_service(
    computers: list[ComputerConfig],
    *,
    token: str | None = "test-secret-token",
) -> ComputerService:
    """Build a :class:`ComputerService` backed by a mock CredentialManager."""
    creds = MagicMock()
    creds.get_token.return_value = token
    config = ComputersConfig(computers=computers)
    return ComputerService(computers_config=config, credential_manager=creds)


# ---------------------------------------------------------------------------
# Tiny in-process fake HTTP server fixture
# ---------------------------------------------------------------------------


class _FakeAgentHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler simulating the agent API."""

    # Class-level state shared across requests in a test
    captured_headers: dict[str, str] = {}
    captured_body: bytes = b""
    response_map: dict[str, tuple[int, dict[str, Any]]] = {}

    def log_message(self, *args: Any) -> None:  # suppress server logs in tests
        pass

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        _FakeAgentHandler.captured_headers = dict(self.headers)
        status, data = _FakeAgentHandler.response_map.get(self.path, (404, {"error": "not found"}))
        self._send_json(status, data)

    def do_POST(self) -> None:  # noqa: N802
        _FakeAgentHandler.captured_headers = dict(self.headers)
        length = int(self.headers.get("Content-Length", 0))
        _FakeAgentHandler.captured_body = self.rfile.read(length)
        status, data = _FakeAgentHandler.response_map.get(self.path, (404, {"error": "not found"}))
        self._send_json(status, data)


@pytest.fixture()
def fake_agent():
    """Start a real local HTTP server simulating the agent API.

    Yields a dict with ``base_url``, ``handler_class``, and a
    ``set_response(path, status, data)`` helper.
    """
    server = HTTPServer(("127.0.0.1", 0), _FakeAgentHandler)
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}"

    # Reset class-level state
    _FakeAgentHandler.captured_headers = {}
    _FakeAgentHandler.captured_body = b""
    _FakeAgentHandler.response_map = {}

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def set_response(path: str, status: int, data: dict[str, Any]) -> None:
        _FakeAgentHandler.response_map[path] = (status, data)

    yield {
        "base_url": base_url,
        "port": port,
        "set_response": set_response,
    }

    server.shutdown()


# ===========================================================================
# Config parsing tests
# ===========================================================================


class TestComputerConfig:
    def test_valid_config_parses(self) -> None:
        cfg = _make_config()
        assert cfg.id == "desktop"
        assert cfg.label == "Main Desktop"
        assert cfg.base_url == "http://127.0.0.1:7777"
        assert cfg.auth_token_ref == "PC_DESKTOP_TOKEN"
        assert cfg.enabled is True
        assert "whoami" in cfg.allowlists.commands

    def test_base_url_strips_trailing_slash(self) -> None:
        cfg = _make_config(base_url="http://127.0.0.1:7777/")
        assert not cfg.base_url.endswith("/")

    def test_base_url_rejects_non_http(self) -> None:
        with pytest.raises(Exception, match="http or https"):
            _make_config(base_url="ftp://somehost/")

    def test_base_url_rejects_no_netloc(self) -> None:
        with pytest.raises(Exception, match="netloc"):
            _make_config(base_url="http:///path")

    def test_disabled_computer(self) -> None:
        cfg = _make_config(enabled=False)
        assert cfg.enabled is False

    def test_is_command_allowed_true(self) -> None:
        cfg = _make_config(commands=["whoami", "dir"])
        assert cfg.is_command_allowed("whoami") is True

    def test_is_command_allowed_false(self) -> None:
        cfg = _make_config(commands=["whoami"])
        assert cfg.is_command_allowed("del") is False

    def test_empty_allowlist(self) -> None:
        cfg = _make_config(commands=[])
        assert cfg.is_command_allowed("whoami") is False

    def test_extra_fields_forbidden(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ComputerConfig.model_validate(
                {
                    "id": "x",
                    "base_url": "http://127.0.0.1",
                    "auth_token_ref": "REF",
                    "unknown_field": "bad",
                }
            )


class TestComputersConfig:
    def test_get_computer_found(self) -> None:
        cfg = _make_config()
        computers_config = ComputersConfig(computers=[cfg])
        result = computers_config.get_computer("desktop")
        assert result is not None
        assert result.id == "desktop"

    def test_get_computer_not_found(self) -> None:
        computers_config = ComputersConfig(computers=[_make_config()])
        assert computers_config.get_computer("nonexistent") is None

    def test_list_enabled_filters_disabled(self) -> None:
        enabled = _make_config(computer_id="a", enabled=True)
        disabled = _make_config(computer_id="b", enabled=False)
        computers_config = ComputersConfig(computers=[enabled, disabled])
        result = computers_config.list_enabled()
        assert len(result) == 1
        assert result[0].id == "a"

    def test_list_all_includes_disabled(self) -> None:
        enabled = _make_config(computer_id="a", enabled=True)
        disabled = _make_config(computer_id="b", enabled=False)
        computers_config = ComputersConfig(computers=[enabled, disabled])
        assert len(computers_config.list_all()) == 2


class TestLoadComputersConfig:
    def test_load_from_raw_config(self) -> None:
        raw = {
            "computers": [
                {
                    "id": "desktop",
                    "label": "Main Desktop",
                    "base_url": "http://127.0.0.1:7777",
                    "auth_token_ref": "PC_DESKTOP_TOKEN",
                    "enabled": True,
                    "allowlists": {"commands": ["whoami", "dir"]},
                }
            ]
        }
        config = load_computers_config(raw)
        assert len(config.computers) == 1
        assert config.computers[0].id == "desktop"

    def test_load_missing_section_returns_empty(self) -> None:
        config = load_computers_config({})
        assert config.computers == []

    def test_load_non_list_section_returns_empty(self) -> None:
        config = load_computers_config({"computers": "bad"})
        assert config.computers == []

    def test_load_with_timeouts(self) -> None:
        raw = {
            "computers": [
                {
                    "id": "srv",
                    "base_url": "http://192.168.1.10:7777",
                    "auth_token_ref": "PC_SRV_TOKEN",
                    "connect_timeout": 3.0,
                    "read_timeout": 60.0,
                }
            ]
        }
        config = load_computers_config(raw)
        c = config.computers[0]
        assert c.connect_timeout == 3.0
        assert c.read_timeout == 60.0


# ===========================================================================
# Service-level tests (no real network)
# ===========================================================================


class TestComputerServiceList:
    def test_list_enabled_only_by_default(self) -> None:
        svc = _make_service(
            [
                _make_config(computer_id="a", enabled=True),
                _make_config(computer_id="b", enabled=False),
            ]
        )
        items = svc.list_computers()
        assert len(items) == 1
        assert items[0].id == "a"

    def test_list_all_includes_disabled(self) -> None:
        svc = _make_service(
            [
                _make_config(computer_id="a", enabled=True),
                _make_config(computer_id="b", enabled=False),
            ]
        )
        items = svc.list_computers(include_disabled=True)
        assert len(items) == 2

    def test_list_returns_computer_info(self) -> None:
        svc = _make_service([_make_config()])
        items = svc.list_computers()
        assert isinstance(items[0], ComputerInfo)
        assert items[0].base_url == "http://127.0.0.1:7777"


class TestAllowlistEnforcement:
    """Allowlist must be checked *before* any network call is made."""

    def test_allowlisted_command_passes(self) -> None:
        # The credential returns a valid token but the URL points nowhere;
        # the allowlist check happens before the network call.
        svc = _make_service([_make_config(commands=["whoami"])])
        # We only verify that AllowlistDeniedError is NOT raised for allowed commands.
        # The actual network call will fail (no real server), but that is a
        # different exception (ConnectionError or similar), not AllowlistDeniedError.
        try:
            svc.run("desktop", "whoami")
        except AllowlistDeniedError:
            pytest.fail("AllowlistDeniedError raised for an allowed command")
        except Exception:
            # Connection error is expected — no real server at 127.0.0.1:7777
            pass

    def test_disallowed_command_raises_before_network(self) -> None:
        svc = _make_service([_make_config(commands=["whoami"])])
        # Ensure allowlist deny happens before any client creation/network attempt.
        svc._make_client = MagicMock()  # type: ignore[method-assign]

        with pytest.raises(AllowlistDeniedError, match="del"):
            svc.run("desktop", "del")

        svc._make_client.assert_not_called()  # type: ignore[union-attr]

    def test_empty_allowlist_blocks_everything(self) -> None:
        svc = _make_service([_make_config(commands=[])])
        with pytest.raises(AllowlistDeniedError):
            svc.run("desktop", "whoami")


class TestComputerServiceErrors:
    def test_unknown_computer_raises(self) -> None:
        svc = _make_service([_make_config()])
        with pytest.raises(ComputerNotFoundError):
            svc.status("nonexistent")

    def test_disabled_computer_raises_on_status(self) -> None:
        svc = _make_service([_make_config(enabled=False)])
        with pytest.raises(ComputerDisabledError):
            svc.status("desktop")

    def test_disabled_computer_raises_on_run(self) -> None:
        svc = _make_service([_make_config(enabled=False)])
        with pytest.raises(ComputerDisabledError):
            svc.run("desktop", "whoami")

    def test_missing_token_raises(self) -> None:
        svc = _make_service([_make_config()], token=None)
        with pytest.raises(MissingTokenError, match="auth_token_ref"):
            svc.status("desktop")

    def test_missing_token_raises_on_run(self) -> None:
        svc = _make_service([_make_config()], token=None)
        with pytest.raises(MissingTokenError):
            svc.run("desktop", "whoami")

    def test_computer_not_found_error_message(self) -> None:
        svc = _make_service([])
        with pytest.raises(ComputerNotFoundError, match="ghost"):
            svc.status("ghost")


# ===========================================================================
# Client-level tests against the fake HTTP server
# ===========================================================================


class TestAgentClientStatus:
    def test_status_success(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response(
            "/status",
            200,
            {"hostname": "DESKTOP-123", "os": "Windows 10", "user": "alice", "time": "2026-02-23"},
        )

        client = AgentClient(base_url, token="secret", computer_id="test")
        result = client.status()

        assert result.ok is True
        assert result.hostname == "DESKTOP-123"
        assert result.os == "Windows 10"
        assert result.user == "alice"

    def test_status_sends_auth_header(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/status", 200, {"hostname": "H", "os": "W", "user": "u", "time": "t"})

        client = AgentClient(base_url, token="my-secret-token", computer_id="test")
        client.status()

        captured = _FakeAgentHandler.captured_headers
        assert AUTH_HEADER in captured or AUTH_HEADER.lower() in {
            k.lower() for k in captured
        }, f"Auth header not found in: {list(captured.keys())}"
        # Find case-insensitively
        token_value = next(v for k, v in captured.items() if k.lower() == AUTH_HEADER.lower())
        assert token_value == "my-secret-token"

    def test_status_failure_returns_ok_false(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/status", 500, {"error": "internal error"})

        client = AgentClient(base_url, token="tok", computer_id="test")
        result = client.status()

        assert result.ok is False
        assert result.error is not None

    def test_status_timeout_returns_ok_false(self) -> None:
        # Use an address that immediately refuses connections
        client = AgentClient(
            "http://127.0.0.1:19999",
            token="tok",
            computer_id="test",
            connect_timeout=0.1,
            read_timeout=0.1,
        )
        result = client.status()
        assert result.ok is False
        assert result.error is not None


class TestAgentClientRun:
    def test_run_success(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response(
            "/run",
            200,
            {"exit_code": 0, "stdout": "alice\n", "stderr": ""},
        )

        client = AgentClient(base_url, token="secret", computer_id="test")
        result = client.run("whoami")

        assert result.ok is True
        assert result.exit_code == 0
        assert "alice" in result.stdout

    def test_run_sends_auth_header(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/run", 200, {"exit_code": 0, "stdout": "", "stderr": ""})

        client = AgentClient(base_url, token="run-token-xyz", computer_id="test")
        client.run("whoami")

        captured = _FakeAgentHandler.captured_headers
        token_value = next(v for k, v in captured.items() if k.lower() == AUTH_HEADER.lower())
        assert token_value == "run-token-xyz"

    def test_run_sends_correct_payload(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/run", 200, {"exit_code": 0, "stdout": "", "stderr": ""})

        client = AgentClient(base_url, token="tok", computer_id="test")
        client.run("ipconfig", args=["/all"], cwd="C:\\")

        payload = json.loads(_FakeAgentHandler.captured_body)
        assert payload["command"] == "ipconfig"
        assert payload["args"] == ["/all"]
        assert payload["cwd"] == "C:\\"

    def test_run_nonzero_exit_code(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/run", 200, {"exit_code": 1, "stdout": "", "stderr": "not found"})

        client = AgentClient(base_url, token="tok", computer_id="test")
        result = client.run("badcmd")

        assert result.ok is False
        assert result.exit_code == 1
        assert result.stderr == "not found"

    def test_run_connection_failure(self) -> None:
        client = AgentClient(
            "http://127.0.0.1:19999",
            token="tok",
            computer_id="test",
            connect_timeout=0.1,
            read_timeout=0.1,
        )
        result = client.run("whoami")
        assert result.ok is False
        assert result.error is not None


class TestAgentClientHealth:
    def test_health_ok(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/health", 200, {"status": "ok"})

        client = AgentClient(base_url, token="tok", computer_id="test")
        result = client.health()

        assert result.ok is True

    def test_health_degraded(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/health", 200, {"status": "degraded"})

        client = AgentClient(base_url, token="tok", computer_id="test")
        result = client.health()

        # "degraded" != "ok" so ok=False
        assert result.ok is False

    def test_health_sends_auth_header(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/health", 200, {"status": "ok"})

        client = AgentClient(base_url, token="health-tok", computer_id="test")
        client.health()

        captured = _FakeAgentHandler.captured_headers
        token_value = next(v for k, v in captured.items() if k.lower() == AUTH_HEADER.lower())
        assert token_value == "health-tok"


# ===========================================================================
# Service via fake server (integration-style, still offline)
# ===========================================================================


class TestComputerServiceWithFakeServer:
    def test_status_via_service(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response(
            "/status",
            200,
            {"hostname": "WIN-PC", "os": "Windows 11", "user": "bob", "time": "T"},
        )

        cfg = _make_config(base_url=base_url)
        svc = _make_service([cfg])
        result = svc.status("desktop")

        assert result.ok is True
        assert result.hostname == "WIN-PC"

    def test_run_via_service_allowlisted(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        set_response = fake_agent["set_response"]
        set_response("/run", 200, {"exit_code": 0, "stdout": "DESKTOP-PC\\alice\n", "stderr": ""})

        cfg = _make_config(base_url=base_url, commands=["whoami"])
        svc = _make_service([cfg])
        result = svc.run("desktop", "whoami")

        assert result.ok is True
        assert "alice" in result.stdout

    def test_run_via_service_disallowed_never_hits_server(self, fake_agent: dict) -> None:
        base_url: str = fake_agent["base_url"]
        # Don't configure any response; if the server is called it returns 404.
        cfg = _make_config(base_url=base_url, commands=["whoami"])
        svc = _make_service([cfg])

        with pytest.raises(AllowlistDeniedError):
            svc.run("desktop", "rm")


class TestPcCliSafetyGuard:
    """Tests for the rex pc run safety guards (policy + approvals + --yes)."""

    def _make_service(self, *, allowlist_ok: bool = True) -> MagicMock:
        """Build a mock ComputerService whose allowlist check is controllable."""
        service = MagicMock()
        service.get_command_allowed.return_value = (
            allowlist_ok,
            ["whoami", "dir", "ipconfig"],
        )
        service.run.return_value = type(
            "RunResult",
            (),
            {"stdout": "", "stderr": "", "ok": True, "error": None, "exit_code": 0},
        )()
        return service

    def test_run_requires_approval_before_execution(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Without a prior approval, rex pc run refuses and shows the approval ID."""
        import tempfile

        approval_dir = tempfile.mkdtemp()
        from pathlib import Path

        monkeypatch.setattr(
            "rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR",
            Path(approval_dir),
        )
        service = self._make_service()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=True, user=None
        )
        code = cmd_pc(args)

        assert code == 1
        service.run.assert_not_called()
        out = capsys.readouterr().out
        assert "Approval required" in out
        assert "rex approvals --approve" in out

    def test_run_without_yes_still_refused_after_approval(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: pytest.TempPathFactory,
    ) -> None:
        """Even with an approved approval, --yes is still required."""
        import tempfile
        from pathlib import Path

        from rex.computers.pc_run_policy import PC_RUN_WORKFLOW_ID, _command_step_id
        from rex.workflow import WorkflowApproval

        approval_dir = Path(tempfile.mkdtemp())
        monkeypatch.setattr("rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR", approval_dir)

        # Pre-create an approved approval
        step_id = _command_step_id("desktop", "whoami", [])
        approval = WorkflowApproval(
            workflow_id=PC_RUN_WORKFLOW_ID,
            step_id=step_id,
            status="approved",
            requested_by="cli",
            step_description="test",
            tool_call_summary="{}",
        )
        approval.save(approval_dir)

        service = self._make_service()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=False, user=None
        )
        code = cmd_pc(args)

        assert code == 1
        service.run.assert_not_called()
        out = capsys.readouterr().out
        assert "without explicit confirmation" in out
        assert "--yes" in out

    def test_run_with_approved_approval_and_yes_calls_service(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With an approved approval and --yes, service.run is called."""
        import tempfile
        from pathlib import Path

        from rex.computers.pc_run_policy import PC_RUN_WORKFLOW_ID, _command_step_id
        from rex.workflow import WorkflowApproval

        approval_dir = Path(tempfile.mkdtemp())
        monkeypatch.setattr("rex.computers.pc_run_policy.DEFAULT_APPROVAL_DIR", approval_dir)

        # Pre-create an approved approval
        step_id = _command_step_id("desktop", "whoami", [])
        approval = WorkflowApproval(
            workflow_id=PC_RUN_WORKFLOW_ID,
            step_id=step_id,
            status="approved",
            requested_by="cli",
            step_description="test",
            tool_call_summary="{}",
        )
        approval.save(approval_dir)

        service = self._make_service()
        monkeypatch.setattr("rex.computers.service.get_computer_service", lambda: service)

        args = argparse.Namespace(
            pc_command="run", id="desktop", cmd=["whoami"], yes=True, user=None
        )
        code = cmd_pc(args)

        assert code == 0
        service.run.assert_called_once_with("desktop", "whoami", args=[])
