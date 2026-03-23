"""
US-039: Browser automation acceptance tests.

Acceptance criteria:
- browser launches
- navigation works
- page actions executed
- Typecheck passes
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_playwright(tmp_path: Path):
    """Build a full playwright mock hierarchy for testing."""
    mock_page = AsyncMock()
    mock_page.url = "https://example.com"
    mock_page.title = AsyncMock(return_value="Example Domain")
    mock_page.goto = AsyncMock()
    mock_page.click = AsyncMock()
    mock_page.fill = AsyncMock()
    mock_page.screenshot = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()
    mock_page.content = AsyncMock(return_value="<html>test</html>")
    mock_page.evaluate = AsyncMock(return_value="body text")

    mock_context = AsyncMock()
    mock_context.new_page = AsyncMock(return_value=mock_page)
    mock_context.pages = []
    mock_context.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_context = AsyncMock(return_value=mock_context)
    mock_browser.close = AsyncMock()

    mock_playwright_instance = AsyncMock()
    mock_playwright_instance.chromium.launch = AsyncMock(return_value=mock_browser)
    mock_playwright_instance.chromium.launch_persistent_context = AsyncMock(
        return_value=mock_context
    )
    mock_playwright_instance.stop = AsyncMock()

    mock_playwright_ctx = MagicMock()
    mock_playwright_ctx.return_value.start = AsyncMock(return_value=mock_playwright_instance)

    return mock_playwright_ctx, mock_playwright_instance, mock_browser, mock_context, mock_page


# ---------------------------------------------------------------------------
# AC1: browser launches
# ---------------------------------------------------------------------------


class TestBrowserLaunches:
    """Browser launches acceptance criterion."""

    @pytest.mark.asyncio
    async def test_launch_creates_page(self, tmp_path):
        """BrowserSession.launch() creates a usable page."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            assert session._page is mock_page
            await session.close()

    @pytest.mark.asyncio
    async def test_context_manager_launch_and_close(self, tmp_path):
        """BrowserSession context manager launches and closes the browser."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, mock_inst, mock_browser, _, _ = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            async with BrowserSession(headless=True, storage_path=tmp_path) as session:
                assert session._page is not None
            # After exiting, close was called on the context
            mock_inst.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_launch_raises_when_playwright_unavailable(self, tmp_path):
        """launch() raises RuntimeError when playwright is not installed."""
        from rex.openclaw.browser_core import BrowserSession

        with patch("rex.openclaw.browser_core.async_playwright", None):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            with pytest.raises(RuntimeError, match="Playwright not installed"):
                await session.launch()

    @pytest.mark.asyncio
    async def test_launch_headless_flag_passed(self, tmp_path):
        """launch() passes headless flag to chromium.launch()."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, mock_inst, _, _, _ = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=False, storage_path=tmp_path)
            await session.launch()
            mock_inst.chromium.launch.assert_called_once_with(headless=False)
            await session.close()


# ---------------------------------------------------------------------------
# AC2: navigation works
# ---------------------------------------------------------------------------


class TestNavigationWorks:
    """Navigation works acceptance criterion."""

    @pytest.mark.asyncio
    async def test_navigate_returns_success(self, tmp_path):
        """navigate() returns status=success with title and url."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            result = await session.navigate("https://example.com")

        assert result["status"] == "success"
        assert result["title"] == "Example Domain"
        assert result["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_navigate_calls_page_goto(self, tmp_path):
        """navigate() calls page.goto with the provided URL."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            await session.navigate("https://example.com", wait_for="load")

        mock_page.goto.assert_called_once_with("https://example.com", wait_until="load")

    @pytest.mark.asyncio
    async def test_navigate_error_raises_runtime_error(self, tmp_path):
        """navigate() raises RuntimeError when page.goto fails."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)
        mock_page.goto.side_effect = Exception("net::ERR_NAME_NOT_RESOLVED")

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            with pytest.raises(RuntimeError, match="Navigation failed"):
                await session.navigate("https://bad-url-xyz.invalid")

    @pytest.mark.asyncio
    async def test_navigate_without_launch_raises(self, tmp_path):
        """navigate() raises RuntimeError if browser not launched."""
        from rex.openclaw.browser_core import BrowserSession

        session = BrowserSession(headless=True, storage_path=tmp_path)
        with pytest.raises(RuntimeError, match="Browser not launched"):
            await session.navigate("https://example.com")


# ---------------------------------------------------------------------------
# AC3: page actions executed
# ---------------------------------------------------------------------------


class TestPageActionsExecuted:
    """Page actions executed acceptance criterion."""

    @pytest.mark.asyncio
    async def test_click_action(self, tmp_path):
        """click() calls page.click and returns success."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            result = await session.click("#submit-btn")

        assert result["status"] == "success"
        assert result["selector"] == "#submit-btn"
        mock_page.click.assert_called_once()

    @pytest.mark.asyncio
    async def test_type_text_action(self, tmp_path):
        """type_text() calls page.fill and returns success."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            result = await session.type_text("#search", "hello world")

        assert result["status"] == "success"
        mock_page.fill.assert_called_once_with("#search", "hello world", timeout=5000)

    @pytest.mark.asyncio
    async def test_screenshot_action(self, tmp_path):
        """screenshot() calls page.screenshot and returns the file path."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            result = await session.screenshot(filename="test.png")

        assert result["status"] == "success"
        assert result["path"].endswith("test.png")
        mock_page.screenshot.assert_called_once()

    @pytest.mark.asyncio
    async def test_wait_action(self, tmp_path):
        """wait() suspends for the specified duration."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, _ = _make_mock_playwright(tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            with patch("rex.openclaw.browser_core.asyncio.sleep", AsyncMock()) as mock_sleep:
                result = await session.wait(500)

        assert result["status"] == "success"
        assert result["waited_ms"] == 500
        mock_sleep.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_run_browser_script_executes_steps(self, tmp_path):
        """run_browser_script() executes multiple steps in sequence."""
        from rex.openclaw.browser_core import BrowserSession, run_browser_script

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)

        steps = [
            {"action": "navigate", "params": {"url": "https://example.com"}},
            {"action": "click", "params": {"selector": "#btn"}},
            {"action": "screenshot", "params": {"filename": "out.png"}},
        ]

        original_init = BrowserSession.__init__

        def patched_init(self, headless=True, session_name=None, storage_path=None):
            original_init(self, headless=headless, session_name=session_name, storage_path=tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            with patch("rex.openclaw.browser_core.BrowserSession.__init__", patched_init):
                results = await run_browser_script(steps, headless=True, check_policy=False)

        assert len(results) == 3
        assert results[0]["status"] == "success"
        assert results[0]["action"] == "navigate"
        assert results[1]["status"] == "success"
        assert results[1]["action"] == "click"
        assert results[2]["status"] == "success"
        assert results[2]["action"] == "screenshot"

    @pytest.mark.asyncio
    async def test_run_browser_script_handles_unknown_action(self, tmp_path):
        """run_browser_script() returns error status for unknown actions."""
        from rex.openclaw.browser_core import BrowserSession, run_browser_script

        mock_pw, _, _, _, _ = _make_mock_playwright(tmp_path)

        steps = [{"action": "explode", "params": {}}]

        original_init = BrowserSession.__init__

        def patched_init(self, headless=True, session_name=None, storage_path=None):
            original_init(self, headless=headless, session_name=session_name, storage_path=tmp_path)

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            with patch("rex.openclaw.browser_core.BrowserSession.__init__", patched_init):
                results = await run_browser_script(steps, headless=True, check_policy=False)

        assert results[0]["status"] == "error"
        assert "Unknown action" in results[0]["error"]

    @pytest.mark.asyncio
    async def test_click_error_raises_runtime_error(self, tmp_path):
        """click() raises RuntimeError when the element is not found."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)
        mock_page.click.side_effect = Exception("Element not found")

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            with pytest.raises(RuntimeError, match="Click failed"):
                await session.click("#missing")

    @pytest.mark.asyncio
    async def test_type_error_raises_runtime_error(self, tmp_path):
        """type_text() raises RuntimeError when the element is not found."""
        from rex.openclaw.browser_core import BrowserSession

        mock_pw, _, _, _, mock_page = _make_mock_playwright(tmp_path)
        mock_page.fill.side_effect = Exception("Element not found")

        with patch("rex.openclaw.browser_core.async_playwright", mock_pw):
            session = BrowserSession(headless=True, storage_path=tmp_path)
            await session.launch()
            with pytest.raises(RuntimeError, match="Type failed"):
                await session.type_text("#missing", "text")
