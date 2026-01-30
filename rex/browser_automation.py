"""
Browser automation service using Playwright.

Provides web browser automation capabilities with:
- Browser session management
- Navigation, clicking, typing
- Screenshot capture
- Login helpers with credential integration
- Policy-gated execution
- Workflow integration
"""

import asyncio
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal, Optional

from rex.audit import LogEntry, get_audit_logger
from rex.contracts.core import ToolCall
from rex.credentials import get_credential_manager
from rex.policy_engine import get_policy_engine

async_playwright = None
if find_spec("playwright") is not None:
    from playwright.async_api import async_playwright as _async_playwright

    async_playwright = _async_playwright


@dataclass
class BrowserAction:
    """A single browser action step."""

    action: Literal["navigate", "click", "type", "wait", "screenshot", "login"]
    params: dict[str, Any]


class BrowserSession:
    """
    Manages a browser automation session using Playwright.

    Features:
    - Headless or headed browser launch
    - Persistent context support
    - Navigation, clicking, typing, screenshots
    - Login helpers with credential manager
    - Network idle waiting
    """

    def __init__(
        self,
        headless: bool = True,
        session_name: Optional[str] = None,
        storage_path: Optional[Path] = None,
    ):
        """
        Initialize a browser session.

        Args:
            headless: Run browser in headless mode
            session_name: Name for persistent session (loads from data/browser_sessions/)
            storage_path: Base storage path for sessions and screenshots
        """
        self.headless = headless
        self.session_name = session_name or f"session_{uuid.uuid4().hex[:8]}"
        self.storage_path = storage_path or Path("data/browser_sessions")
        self.screenshot_path = self.storage_path / "screenshots"
        self.screenshot_path.mkdir(parents=True, exist_ok=True)

        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._audit_logger = get_audit_logger()
        self._credential_manager = get_credential_manager()

    async def __aenter__(self):
        """Context manager entry - launch browser."""
        await self.launch()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close browser."""
        await self.close()

    async def launch(self) -> None:
        """Launch the browser and create a page."""
        if async_playwright is None:
            raise RuntimeError(
                "Playwright not installed. Install with: pip install playwright && "
                "playwright install chromium"
            )

        self._playwright = await async_playwright().start()

        # Check for persistent context
        context_path = self.storage_path / self.session_name

        if context_path.exists():
            # Load persistent context
            self._context = await self._playwright.chromium.launch_persistent_context(
                str(context_path),
                headless=self.headless,
            )
            self._page = (
                self._context.pages[0] if self._context.pages else await self._context.new_page()
            )
        else:
            # Create new browser and context
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context()
            self._page = await self._context.new_page()

    async def close(self) -> None:
        """Close the browser session."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def navigate(self, url: str, wait_for: str = "networkidle") -> dict[str, Any]:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to
            wait_for: Wait condition ("load", "domcontentloaded", "networkidle")

        Returns:
            Dictionary with status, title, and URL
        """
        if not self._page:
            raise RuntimeError("Browser not launched. Call launch() first.")

        start_time = datetime.now()
        action_id = str(uuid.uuid4())

        try:
            await self._page.goto(url, wait_until=wait_for)
            title = await self._page.title()
            current_url = self._page.url

            result = {
                "status": "success",
                "title": title,
                "url": current_url,
            }

            # Audit log
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="browser_navigate",
                    tool_call_args={"url": url},
                    policy_decision="allowed",
                    tool_result=result,
                    error=None,
                    duration_ms=duration_ms,
                )
            )

            return result

        except Exception as e:
            error_msg = f"Navigation failed: {str(e)}"
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self._audit_logger.log(
                LogEntry(
                    action_id=action_id,
                    tool="browser_navigate",
                    tool_call_args={"url": url},
                    policy_decision="allowed",
                    tool_result=None,
                    error=error_msg,
                    duration_ms=duration_ms,
                )
            )
            raise RuntimeError(error_msg) from e

    async def click(self, selector: str, timeout: int = 5000) -> dict[str, Any]:
        """
        Click an element by CSS selector.

        Args:
            selector: CSS selector for the element
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with status
        """
        if not self._page:
            raise RuntimeError("Browser not launched.")

        try:
            await self._page.click(selector, timeout=timeout)
            return {"status": "success", "selector": selector}
        except Exception as e:
            raise RuntimeError(f"Click failed on '{selector}': {str(e)}") from e

    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> dict[str, Any]:
        """
        Type text into an element.

        Args:
            selector: CSS selector for the element
            text: Text to type
            timeout: Timeout in milliseconds

        Returns:
            Dictionary with status
        """
        if not self._page:
            raise RuntimeError("Browser not launched.")

        try:
            await self._page.fill(selector, text, timeout=timeout)
            return {"status": "success", "selector": selector}
        except Exception as e:
            raise RuntimeError(f"Type failed on '{selector}': {str(e)}") from e

    async def wait(self, milliseconds: int) -> dict[str, Any]:
        """Wait for a specified time."""
        await asyncio.sleep(milliseconds / 1000)
        return {"status": "success", "waited_ms": milliseconds}

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> dict[str, Any]:
        """Wait for an element to appear."""
        if not self._page:
            raise RuntimeError("Browser not launched.")

        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            return {"status": "success", "selector": selector}
        except Exception as e:
            raise RuntimeError(f"Wait failed for '{selector}': {str(e)}") from e

    async def screenshot(
        self,
        filename: Optional[str] = None,
        full_page: bool = False,
    ) -> dict[str, Any]:
        """
        Take a screenshot of the current page.

        Args:
            filename: Optional filename (auto-generated if not provided)
            full_page: Capture full scrollable page

        Returns:
            Dictionary with path to screenshot
        """
        if not self._page:
            raise RuntimeError("Browser not launched.")

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"

        screenshot_file = self.screenshot_path / filename

        try:
            await self._page.screenshot(path=str(screenshot_file), full_page=full_page)
            return {
                "status": "success",
                "path": str(screenshot_file),
                "full_page": full_page,
            }
        except Exception as e:
            raise RuntimeError(f"Screenshot failed: {str(e)}") from e

    async def login(
        self,
        url: str,
        credential_name: str,
        username_selector: str = 'input[type="text"], input[name="username"]',
        password_selector: str = 'input[type="password"]',
        submit_selector: str = 'button[type="submit"], input[type="submit"]',
    ) -> dict[str, Any]:
        """
        Login to a site using stored credentials.

        Args:
            url: Login page URL
            credential_name: Name of credential in credential manager (e.g., "github")
            username_selector: CSS selector for username field
            password_selector: CSS selector for password field
            submit_selector: CSS selector for submit button

        Returns:
            Dictionary with login status
        """
        # Get credentials
        username = self._credential_manager.get_token(f"{credential_name}.username")
        password = self._credential_manager.get_token(f"{credential_name}.password")

        if not username or not password:
            raise ValueError(
                f"Credentials not found for '{credential_name}'. "
                f"Set {credential_name}.username and {credential_name}.password in credential manager."
            )

        # Navigate to login page
        await self.navigate(url)

        # Fill login form
        await self.type_text(username_selector, username)
        await self.type_text(password_selector, password)
        await self.click(submit_selector)

        # Wait for navigation
        await self._page.wait_for_load_state("networkidle")

        return {
            "status": "success",
            "url": self._page.url,
            "title": await self._page.title(),
        }

    async def download_file(self, url: str, dest_path: str) -> dict[str, Any]:
        """
        Download a file from a URL.

        Args:
            url: URL of the file to download
            dest_path: Destination path for the downloaded file

        Returns:
            Dictionary with download status and path
        """
        if not self._page:
            raise RuntimeError("Browser not launched.")

        try:
            # Start waiting for download
            async with self._page.expect_download() as download_info:
                await self._page.goto(url)

            download = await download_info.value

            # Save to destination
            await download.save_as(dest_path)

            return {
                "status": "success",
                "path": dest_path,
                "suggested_filename": download.suggested_filename,
            }
        except Exception as e:
            raise RuntimeError(f"Download failed: {str(e)}") from e

    async def get_content(self) -> str:
        """Get the current page content (HTML)."""
        if not self._page:
            raise RuntimeError("Browser not launched.")
        return await self._page.content()

    async def get_text(self, selector: Optional[str] = None) -> str:
        """
        Get text content from page or element.

        Args:
            selector: Optional CSS selector (if None, returns body text)

        Returns:
            Text content
        """
        if not self._page:
            raise RuntimeError("Browser not launched.")

        if selector:
            element = await self._page.query_selector(selector)
            if element:
                return await element.text_content()
            raise ValueError(f"Element not found: {selector}")
        else:
            return await self._page.evaluate("() => document.body.innerText")


async def run_browser_script(
    script_steps: list[dict[str, Any]],
    headless: bool = True,
    session_name: Optional[str] = None,
    check_policy: bool = True,
) -> list[dict[str, Any]]:
    """
    Run a browser automation script with multiple steps.

    Each step should have an 'action' field and 'params' dict:
    - {"action": "navigate", "params": {"url": "https://example.com"}}
    - {"action": "click", "params": {"selector": "#button"}}
    - {"action": "type", "params": {"selector": "#input", "text": "hello"}}
    - {"action": "wait", "params": {"milliseconds": 1000}}
    - {"action": "screenshot", "params": {"filename": "result.png"}}
    - {"action": "login", "params": {"url": "...", "credential_name": "site"}}

    Args:
        script_steps: List of action dictionaries
        headless: Run in headless mode
        session_name: Optional session name for persistence
        check_policy: Check policy engine before each step

    Returns:
        List of results for each step
    """
    policy_engine = get_policy_engine()
    results = []

    async with BrowserSession(headless=headless, session_name=session_name) as session:
        for i, step in enumerate(script_steps):
            action = step.get("action")
            params = step.get("params", {})

            # Policy check
            if check_policy:
                tool_call = ToolCall(
                    tool=f"browser_{action}",
                    args=params,
                    requested_by="user",
                    created_at=datetime.now(),
                )
                decision = policy_engine.decide(tool_call, metadata={})

                if not decision.allowed:
                    results.append(
                        {
                            "step": i,
                            "action": action,
                            "status": "denied",
                            "reason": decision.reason,
                        }
                    )
                    continue

                if decision.requires_approval:
                    # In a real scenario, would wait for approval
                    # For now, we'll skip or fail
                    results.append(
                        {
                            "step": i,
                            "action": action,
                            "status": "requires_approval",
                            "reason": "Policy requires approval for this action",
                        }
                    )
                    continue

            # Execute action
            try:
                if action == "navigate":
                    result = await session.navigate(**params)
                elif action == "click":
                    result = await session.click(**params)
                elif action == "type":
                    result = await session.type_text(**params)
                elif action == "wait":
                    result = await session.wait(**params)
                elif action == "screenshot":
                    result = await session.screenshot(**params)
                elif action == "login":
                    result = await session.login(**params)
                elif action == "download":
                    result = await session.download_file(**params)
                else:
                    result = {"status": "error", "error": f"Unknown action: {action}"}

                results.append(
                    {
                        "step": i,
                        "action": action,
                        **result,
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "step": i,
                        "action": action,
                        "status": "error",
                        "error": str(e),
                    }
                )

    return results


# Synchronous wrapper for CLI usage
def run_browser_script_sync(
    script_steps: list[dict[str, Any]],
    headless: bool = True,
    session_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for run_browser_script."""
    return asyncio.run(run_browser_script(script_steps, headless, session_name))


# Singleton instance (optional, for service integration)
_browser_service: Optional["BrowserAutomationService"] = None


class BrowserAutomationService:
    """
    Service wrapper for browser automation.

    Provides higher-level API for integration with Rex assistant.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("data/browser_sessions")
        self.storage_path.mkdir(parents=True, exist_ok=True)

    async def execute_script(
        self,
        script_path: str,
        headless: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Execute a browser automation script from a JSON file.

        Args:
            script_path: Path to JSON file with script steps
            headless: Run in headless mode

        Returns:
            List of step results
        """
        with open(script_path) as f:
            script_data = json.load(f)

        steps = script_data.get("steps", [])
        session_name = script_data.get("session_name")

        return await run_browser_script(steps, headless, session_name)

    def list_sessions(self) -> list[str]:
        """List available browser sessions."""
        if not self.storage_path.exists():
            return []

        return [
            d.name for d in self.storage_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

    def list_screenshots(self) -> list[str]:
        """List captured screenshots."""
        screenshot_path = self.storage_path / "screenshots"
        if not screenshot_path.exists():
            return []

        return [
            f.name
            for f in screenshot_path.iterdir()
            if f.is_file() and f.suffix in [".png", ".jpg", ".jpeg"]
        ]


def get_browser_service() -> BrowserAutomationService:
    """Get or create the global browser automation service."""
    global _browser_service
    if _browser_service is None:
        _browser_service = BrowserAutomationService()
    return _browser_service


def set_browser_service(service: BrowserAutomationService) -> None:
    """Set the global browser automation service."""
    global _browser_service
    _browser_service = service


def reset_browser_service() -> None:
    """Reset the global browser automation service (for testing)."""
    global _browser_service
    _browser_service = None
