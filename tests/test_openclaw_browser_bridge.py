"""Tests for rex.openclaw.browser_bridge — US-P4-022 and US-P4-023.

US-P4-022 acceptance criteria:
  - BrowserBridge exists and is importable
  - Satisfies BrowserAutomationProtocol structural check
  - execute_script, list_sessions, list_screenshots work
  - register() returns None when openclaw not installed

US-P4-023 acceptance criteria:
  - execute_script runs navigate + screenshot round-trip via run_browser_script
  - list_sessions returns session list
  - list_screenshots returns screenshot list
  - headless parameter is forwarded
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rex.openclaw.browser_bridge import OPENCLAW_AVAILABLE, BrowserBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_bridge(tmp_path: Path | None = None) -> BrowserBridge:
    """Return a BrowserBridge backed by a fresh isolated tmp directory."""
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    return BrowserBridge(storage_path=tmp_path)


# ---------------------------------------------------------------------------
# US-P4-022: Instantiation and protocol conformance
# ---------------------------------------------------------------------------


class TestBrowserBridgeInstantiation:
    def test_import(self):
        from rex.openclaw import browser_bridge  # noqa: F401

    def test_no_args(self):
        """BrowserBridge() with no args constructs without error."""
        bridge = BrowserBridge()
        assert bridge is not None

    def test_storage_path_arg(self, tmp_path):
        """BrowserBridge accepts storage_path and stores it."""
        bridge = BrowserBridge(storage_path=tmp_path)
        assert bridge.storage_path == tmp_path

    def test_openclaw_available_is_bool(self):
        assert isinstance(OPENCLAW_AVAILABLE, bool)

    def test_satisfies_protocol(self, tmp_path):
        from rex.contracts.browser import BrowserAutomationProtocol

        bridge = _fresh_bridge(tmp_path)
        assert isinstance(bridge, BrowserAutomationProtocol)


# ---------------------------------------------------------------------------
# US-P4-022: execute_script, list_sessions, list_screenshots
# ---------------------------------------------------------------------------


class TestBridgeMethods:
    @pytest.mark.asyncio
    async def test_execute_script_calls_run_browser_script(self, tmp_path):
        """execute_script loads JSON and calls run_browser_script."""
        script = {"steps": [{"action": "navigate", "params": {"url": "https://example.com"}}]}
        script_file = tmp_path / "script.json"
        script_file.write_text(json.dumps(script))

        fake_results = [{"step": 0, "action": "navigate", "status": "success"}]

        bridge = _fresh_bridge(tmp_path)
        with patch(
            "rex.openclaw.browser_bridge.run_browser_script", AsyncMock(return_value=fake_results)
        ) as mock_run:
            result = await bridge.execute_script(str(script_file), headless=True)

        mock_run.assert_awaited_once_with(script["steps"], True, None)
        assert result == fake_results

    @pytest.mark.asyncio
    async def test_execute_script_headless_false_forwarded(self, tmp_path):
        """headless=False is forwarded to run_browser_script."""
        script = {"steps": []}
        script_file = tmp_path / "s.json"
        script_file.write_text(json.dumps(script))

        bridge = _fresh_bridge(tmp_path)
        with patch(
            "rex.openclaw.browser_bridge.run_browser_script", AsyncMock(return_value=[])
        ) as mock_run:
            await bridge.execute_script(str(script_file), headless=False)

        args, kwargs = mock_run.call_args
        assert args[1] is False

    @pytest.mark.asyncio
    async def test_execute_script_passes_session_name(self, tmp_path):
        """session_name from JSON is forwarded to run_browser_script."""
        script = {"steps": [], "session_name": "my_session"}
        script_file = tmp_path / "s.json"
        script_file.write_text(json.dumps(script))

        bridge = _fresh_bridge(tmp_path)
        with patch(
            "rex.openclaw.browser_bridge.run_browser_script", AsyncMock(return_value=[])
        ) as mock_run:
            await bridge.execute_script(str(script_file))

        args, _ = mock_run.call_args
        assert args[2] == "my_session"

    def test_list_sessions_empty(self, tmp_path):
        """list_sessions returns [] when storage path has no subdirs."""
        bridge = _fresh_bridge(tmp_path)
        assert bridge.list_sessions() == []

    def test_list_sessions_with_dirs(self, tmp_path):
        """list_sessions returns subdirectory names."""
        (tmp_path / "session_abc").mkdir()
        (tmp_path / "session_xyz").mkdir()
        bridge = _fresh_bridge(tmp_path)
        sessions = bridge.list_sessions()
        assert set(sessions) == {"session_abc", "session_xyz"}

    def test_list_screenshots_empty(self, tmp_path):
        """list_screenshots returns [] when no screenshots exist."""
        bridge = _fresh_bridge(tmp_path)
        assert bridge.list_screenshots() == []

    def test_list_screenshots_with_files(self, tmp_path):
        """list_screenshots returns png/jpg filenames."""
        ss_path = tmp_path / "screenshots"
        ss_path.mkdir()
        (ss_path / "a.png").write_text("x")
        (ss_path / "b.jpg").write_text("x")
        (ss_path / "c.txt").write_text("x")  # should be excluded
        bridge = _fresh_bridge(tmp_path)
        screenshots = bridge.list_screenshots()
        assert set(screenshots) == {"a.png", "b.jpg"}


# ---------------------------------------------------------------------------
# US-P4-023: Simple browser task round-trip (navigate + screenshot)
# ---------------------------------------------------------------------------


class TestSimpleBrowserTask:
    """Verify the full flow for a navigate+screenshot script."""

    @pytest.mark.asyncio
    async def test_navigate_and_screenshot_script(self, tmp_path):
        """execute_script with navigate+screenshot steps returns step results."""
        steps = [
            {"action": "navigate", "params": {"url": "https://example.com"}},
            {"action": "screenshot", "params": {"filename": "shot.png"}},
        ]
        script_file = tmp_path / "nav_shot.json"
        script_file.write_text(json.dumps({"steps": steps}))

        expected = [
            {
                "step": 0,
                "action": "navigate",
                "status": "success",
                "url": "https://example.com",
                "title": "Example",
            },
            {"step": 1, "action": "screenshot", "status": "success", "path": "/data/shot.png"},
        ]
        bridge = _fresh_bridge(tmp_path)
        with patch(
            "rex.openclaw.browser_bridge.run_browser_script", AsyncMock(return_value=expected)
        ):
            results = await bridge.execute_script(str(script_file), headless=True)

        assert len(results) == 2
        assert results[0]["action"] == "navigate"
        assert results[0]["status"] == "success"
        assert results[1]["action"] == "screenshot"
        assert results[1]["status"] == "success"

    @pytest.mark.asyncio
    async def test_empty_script_returns_empty_list(self, tmp_path):
        """execute_script with no steps returns empty list."""
        script_file = tmp_path / "empty.json"
        script_file.write_text(json.dumps({"steps": []}))

        bridge = _fresh_bridge(tmp_path)
        with patch("rex.openclaw.browser_bridge.run_browser_script", AsyncMock(return_value=[])):
            results = await bridge.execute_script(str(script_file))

        assert results == []


# ---------------------------------------------------------------------------
# US-P4-024: Authenticated browser task (login flow through bridge)
# ---------------------------------------------------------------------------


class TestAuthenticatedBrowserTask:
    """Verify login flow and document credential bridge gap."""

    @pytest.mark.asyncio
    async def test_login_script_passed_to_run_browser_script(self, tmp_path):
        """A script with a login step is passed to run_browser_script."""
        steps = [
            {
                "action": "login",
                "params": {"url": "https://example.com/login", "credential_name": "site"},
            }
        ]
        script_file = tmp_path / "login.json"
        script_file.write_text(json.dumps({"steps": steps}))

        login_result = [
            {
                "step": 0,
                "action": "login",
                "status": "success",
                "url": "https://example.com/dashboard",
                "title": "Dashboard",
            }
        ]

        bridge = _fresh_bridge(tmp_path)
        with patch(
            "rex.openclaw.browser_bridge.run_browser_script", AsyncMock(return_value=login_result)
        ):
            results = await bridge.execute_script(str(script_file), headless=True)

        assert results[0]["action"] == "login"
        assert results[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_login_failure_result_propagated(self, tmp_path):
        """Login failure result is propagated from run_browser_script to caller."""
        steps = [
            {
                "action": "login",
                "params": {"url": "https://example.com/login", "credential_name": "bad"},
            }
        ]
        script_file = tmp_path / "login_fail.json"
        script_file.write_text(json.dumps({"steps": steps}))

        fail_result = [
            {"step": 0, "action": "login", "status": "error", "error": "Credentials not found"}
        ]

        bridge = _fresh_bridge(tmp_path)
        with patch(
            "rex.openclaw.browser_bridge.run_browser_script", AsyncMock(return_value=fail_result)
        ):
            results = await bridge.execute_script(str(script_file))

        assert results[0]["status"] == "error"
        assert "Credentials" in results[0]["error"]


# ---------------------------------------------------------------------------
# US-P4-022: register() stub
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_none_without_openclaw(self, tmp_path):
        bridge = _fresh_bridge(tmp_path)
        if not OPENCLAW_AVAILABLE:
            assert bridge.register() is None

    def test_register_accepts_agent_arg(self, tmp_path):
        bridge = _fresh_bridge(tmp_path)
        agent = MagicMock()
        if not OPENCLAW_AVAILABLE:
            assert bridge.register(agent=agent) is None
