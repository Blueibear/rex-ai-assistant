"""Core browser automation primitives — extracted from rex.browser_automation.

This module contains the low-level browser session and script-runner that
:class:`~rex.openclaw.browser_bridge.BrowserBridge` builds on.  It has no
dependency on the retired ``rex.browser_automation`` module.
"""

from __future__ import annotations

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

        context_path = self.storage_path / self.session_name

        if context_path.exists():
            self._context = await self._playwright.chromium.launch_persistent_context(
                str(context_path),
                headless=self.headless,
            )
            self._page = (
                self._context.pages[0] if self._context.pages else await self._context.new_page()
            )
        else:
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
        """Navigate to a URL."""
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
        """Click an element by CSS selector."""
        if not self._page:
            raise RuntimeError("Browser not launched.")

        try:
            await self._page.click(selector, timeout=timeout)
            return {"status": "success", "selector": selector}
        except Exception as e:
            raise RuntimeError(f"Click failed on '{selector}': {str(e)}") from e

    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> dict[str, Any]:
        """Type text into an element."""
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
        """Take a screenshot of the current page."""
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
        """Login to a site using stored credentials."""
        username = self._credential_manager.get_token(f"{credential_name}.username")
        password = self._credential_manager.get_token(f"{credential_name}.password")

        if not username or not password:
            raise ValueError(
                f"Credentials not found for '{credential_name}'. "
                f"Set {credential_name}.username and {credential_name}.password in credential manager."
            )

        await self.navigate(url)
        await self.type_text(username_selector, username)
        await self.type_text(password_selector, password)
        await self.click(submit_selector)

        await self._page.wait_for_load_state("networkidle")
        return {
            "status": "success",
            "url": self._page.url,
            "title": await self._page.title(),
        }

    async def download_file(self, url: str, dest_path: str) -> dict[str, Any]:
        """Download a file from a URL."""
        if not self._page:
            raise RuntimeError("Browser not launched.")

        try:
            async with self._page.expect_download() as download_info:
                await self._page.goto(url)

            download = await download_info.value
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
        """Get text content from page or element."""
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
    """Run a browser automation script with multiple steps."""
    policy_engine = get_policy_engine()
    results = []

    async with BrowserSession(headless=headless, session_name=session_name) as session:
        for i, step in enumerate(script_steps):
            action = step.get("action")
            params = step.get("params", {})

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
                    results.append(
                        {
                            "step": i,
                            "action": action,
                            "status": "requires_approval",
                            "reason": "Policy requires approval for this action",
                        }
                    )
                    continue

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


def run_browser_script_sync(
    script_steps: list[dict[str, Any]],
    headless: bool = True,
    session_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Synchronous wrapper for run_browser_script."""
    return asyncio.run(run_browser_script(script_steps, headless, session_name))
