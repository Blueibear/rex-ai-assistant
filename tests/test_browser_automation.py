"""Tests for browser automation module (now in rex.openclaw.browser_core)."""

from unittest.mock import AsyncMock, patch

import pytest

from rex.openclaw.browser_bridge import (
    BrowserBridge,
    get_browser_service,
    reset_browser_service,
)
from rex.openclaw.browser_core import BrowserSession


class TestBrowserSession:
    """Tests for BrowserSession class."""

    @pytest.mark.asyncio
    async def test_browser_session_init(self):
        """Test BrowserSession initialization."""
        session = BrowserSession(headless=True, session_name="test_session")

        assert session.headless is True
        assert session.session_name == "test_session"
        assert session._playwright is None
        assert session._browser is None

    @pytest.mark.asyncio
    async def test_browser_session_context_manager(self):
        """Test BrowserSession as async context manager."""
        with patch("rex.openclaw.browser_core.async_playwright") as mock_playwright:
            mock_playwright_instance = AsyncMock()
            mock_playwright.return_value.start = AsyncMock(return_value=mock_playwright_instance)

            mock_browser = AsyncMock()
            mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
            mock_playwright_instance.chromium.launch_persistent_context = AsyncMock(
                return_value=mock_browser
            )

            mock_context = AsyncMock()
            mock_browser.new_context = AsyncMock(return_value=mock_context)

            mock_page = AsyncMock()
            mock_context.new_page = AsyncMock(return_value=mock_page)

            async with BrowserSession() as session:
                assert session is not None


class TestBrowserAutomationService:
    """Tests for BrowserBridge (replaces BrowserAutomationService)."""

    def test_service_init(self):
        """Test bridge initialization."""
        bridge = BrowserBridge()
        assert bridge.storage_path.name == "browser_sessions"

    def test_list_sessions_empty(self):
        """Test listing sessions when none exist."""
        bridge = BrowserBridge()
        sessions = bridge.list_sessions()
        assert isinstance(sessions, list)

    def test_list_screenshots_empty(self):
        """Test listing screenshots when none exist."""
        bridge = BrowserBridge()
        screenshots = bridge.list_screenshots()
        assert isinstance(screenshots, list)


class TestBrowserServiceSingleton:
    """Tests for browser service singleton."""

    def test_get_browser_service(self):
        """Test getting browser service singleton."""
        reset_browser_service()
        service1 = get_browser_service()
        service2 = get_browser_service()
        assert service1 is service2

    def test_reset_browser_service(self):
        """Test resetting browser service."""
        service1 = get_browser_service()
        reset_browser_service()
        service2 = get_browser_service()
        assert service1 is not service2
