"""Tests for rex.openclaw.browser_bridge — US-P4-022 and US-P4-023.

US-P4-022 acceptance criteria:
  - BrowserBridge exists and is importable
  - Satisfies BrowserAutomationProtocol structural check
  - Delegates every method to the underlying BrowserAutomationService
  - register() returns None when openclaw not installed

US-P4-023 acceptance criteria:
  - execute_script delegates to service.execute_script (navigate + screenshot round-trip)
  - list_sessions delegates and returns session list
  - list_screenshots delegates and returns screenshot list
  - headless parameter is forwarded
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rex.browser_automation import BrowserAutomationService
from rex.openclaw.browser_bridge import OPENCLAW_AVAILABLE, BrowserBridge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service() -> BrowserAutomationService:
    """Return a BrowserAutomationService backed by a tmp directory."""
    import tempfile
    from pathlib import Path

    d = tempfile.mkdtemp()
    return BrowserAutomationService(storage_path=Path(d))


def _fresh_bridge() -> tuple[BrowserBridge, BrowserAutomationService]:
    """Return a BrowserBridge backed by a fresh isolated service."""
    service = _make_service()
    bridge = BrowserBridge(service=service)
    return bridge, service


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

    def test_explicit_service_arg(self):
        """BrowserBridge accepts an explicit service and stores it."""
        service = _make_service()
        bridge = BrowserBridge(service=service)
        assert bridge._service is service

    def test_openclaw_available_is_bool(self):
        assert isinstance(OPENCLAW_AVAILABLE, bool)

    def test_satisfies_protocol(self):
        from rex.contracts.browser import BrowserAutomationProtocol

        bridge = BrowserBridge(service=_make_service())
        assert isinstance(bridge, BrowserAutomationProtocol)


# ---------------------------------------------------------------------------
# US-P4-022: Delegation to underlying BrowserAutomationService
# ---------------------------------------------------------------------------


class TestDelegation:
    def setup_method(self):
        self.service = _make_service()
        self.bridge = BrowserBridge(service=self.service)

    @pytest.mark.asyncio
    async def test_execute_script_delegates(self):
        """execute_script delegates to service.execute_script."""
        fake_results = [{"step": 1, "action": "navigate", "status": "success", "url": "https://example.com"}]
        self.service.execute_script = AsyncMock(return_value=fake_results)

        result = await self.bridge.execute_script("/path/to/script.json", headless=True)

        self.service.execute_script.assert_awaited_once_with("/path/to/script.json", headless=True)
        assert result == fake_results

    @pytest.mark.asyncio
    async def test_execute_script_headless_false_forwarded(self):
        """headless=False is forwarded to service.execute_script."""
        self.service.execute_script = AsyncMock(return_value=[])

        await self.bridge.execute_script("/p/s.json", headless=False)

        _, kwargs = self.service.execute_script.call_args
        assert kwargs["headless"] is False

    def test_list_sessions_delegates(self):
        """list_sessions delegates to service.list_sessions."""
        self.service.list_sessions = MagicMock(return_value=["session_a", "session_b"])

        result = self.bridge.list_sessions()

        self.service.list_sessions.assert_called_once()
        assert result == ["session_a", "session_b"]

    def test_list_screenshots_delegates(self):
        """list_screenshots delegates to service.list_screenshots."""
        self.service.list_screenshots = MagicMock(return_value=["shot1.png", "shot2.png"])

        result = self.bridge.list_screenshots()

        self.service.list_screenshots.assert_called_once()
        assert result == ["shot1.png", "shot2.png"]


# ---------------------------------------------------------------------------
# US-P4-023: Simple browser task round-trip (navigate + screenshot)
# ---------------------------------------------------------------------------


class TestSimpleBrowserTask:
    """Verify the full delegation chain for a navigate+screenshot script."""

    def setup_method(self):
        self.bridge, self.service = _fresh_bridge()

    @pytest.mark.asyncio
    async def test_navigate_and_screenshot_script(self):
        """execute_script with navigate+screenshot steps returns step results."""
        expected_results = [
            {"step": 1, "action": "navigate", "status": "success", "url": "https://example.com", "title": "Example"},
            {"step": 2, "action": "screenshot", "status": "success", "path": "/data/shot.png"},
        ]
        self.service.execute_script = AsyncMock(return_value=expected_results)

        results = await self.bridge.execute_script("/path/to/nav_shot.json", headless=True)

        assert len(results) == 2
        assert results[0]["action"] == "navigate"
        assert results[0]["status"] == "success"
        assert results[1]["action"] == "screenshot"
        assert results[1]["status"] == "success"

    @pytest.mark.asyncio
    async def test_empty_script_returns_empty_list(self):
        """execute_script with no steps returns empty list."""
        self.service.execute_script = AsyncMock(return_value=[])

        results = await self.bridge.execute_script("/path/to/empty.json")
        assert results == []

    def test_list_sessions_empty_when_none(self):
        """list_sessions returns [] when no sessions exist."""
        result = self.bridge.list_sessions()
        assert isinstance(result, list)

    def test_list_screenshots_empty_when_none(self):
        """list_screenshots returns [] when no screenshots exist."""
        result = self.bridge.list_screenshots()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_bridge_and_service_share_same_state(self):
        """Operations via bridge are reflected in the underlying service."""
        fake = [{"step": 1, "action": "navigate", "status": "success"}]
        self.service.execute_script = AsyncMock(return_value=fake)

        # Call via bridge
        result = await self.bridge.execute_script("/p.json")
        # Verify it went through the same service object
        self.service.execute_script.assert_awaited_once()
        assert result == fake


# ---------------------------------------------------------------------------
# US-P4-024: Authenticated browser task (login flow through bridge)
# ---------------------------------------------------------------------------


class TestAuthenticatedBrowserTask:
    """Verify login flow delegation and document credential bridge gap.

    The BrowserBridge delegates execute_script to BrowserAutomationService,
    which in turn calls BrowserSession.login() when a script contains a login
    step.  The login step uses Rex's credential manager internally.

    Gap documented (US-P4-021): BrowserSession.login() reaches directly into
    Rex's credential manager (rex.credentials).  OpenClaw does not have a
    credential bridge yet.  During migration, login flows continue to go
    through Rex's credential manager unchanged.
    """

    def setup_method(self):
        self.bridge, self.service = _fresh_bridge()

    @pytest.mark.asyncio
    async def test_login_script_delegated_to_service(self):
        """A script with a login step is passed to service.execute_script."""
        login_result = [
            {"step": 1, "action": "login", "status": "success", "url": "https://example.com/dashboard", "title": "Dashboard"},
        ]
        self.service.execute_script = AsyncMock(return_value=login_result)

        results = await self.bridge.execute_script("/path/to/login_script.json", headless=True)

        self.service.execute_script.assert_awaited_once_with(
            "/path/to/login_script.json", headless=True
        )
        assert results[0]["action"] == "login"
        assert results[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_multi_step_auth_flow(self):
        """navigate → login → screenshot flow is forwarded intact."""
        auth_flow = [
            {"step": 1, "action": "navigate", "status": "success"},
            {"step": 2, "action": "login", "status": "success"},
            {"step": 3, "action": "screenshot", "status": "success"},
        ]
        self.service.execute_script = AsyncMock(return_value=auth_flow)

        results = await self.bridge.execute_script("/path/auth.json")

        assert len(results) == 3
        actions = [r["action"] for r in results]
        assert actions == ["navigate", "login", "screenshot"]

    @pytest.mark.asyncio
    async def test_login_failure_result_propagated(self):
        """Login failure result is propagated from service to caller."""
        fail_result = [
            {"step": 1, "action": "login", "status": "error", "error": "Credentials not found"},
        ]
        self.service.execute_script = AsyncMock(return_value=fail_result)

        results = await self.bridge.execute_script("/path/to/login_script.json")

        assert results[0]["status"] == "error"
        assert "Credentials" in results[0]["error"]


# ---------------------------------------------------------------------------
# US-P4-022: register() stub
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_returns_none_without_openclaw(self):
        bridge = BrowserBridge(service=_make_service())
        if not OPENCLAW_AVAILABLE:
            assert bridge.register() is None

    def test_register_accepts_agent_arg(self):
        bridge = BrowserBridge(service=_make_service())
        agent = MagicMock()
        if not OPENCLAW_AVAILABLE:
            assert bridge.register(agent=agent) is None
