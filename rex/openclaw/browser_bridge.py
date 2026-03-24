"""OpenClaw browser bridge — US-P4-022.

Implements :class:`~rex.contracts.browser.BrowserAutomationProtocol` using
:mod:`rex.openclaw.browser_core` directly.  Does NOT depend on the retired
``rex.browser_automation`` module.

Typical usage::

    from rex.openclaw.browser_bridge import BrowserBridge

    bridge = BrowserBridge()

    # Execute a browser script
    results = await bridge.execute_script("/path/to/script.json", headless=True)

    # List sessions
    sessions = bridge.list_sessions()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from rex.openclaw.browser_core import BrowserSession, run_browser_script  # noqa: F401

logger = logging.getLogger(__name__)

_bridge_singleton: BrowserBridge | None = None


def get_browser_service() -> BrowserBridge:
    """Return the global BrowserBridge singleton."""
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = BrowserBridge()
    return _bridge_singleton


def set_browser_service(service: BrowserBridge) -> None:
    """Override the global BrowserBridge singleton (for testing)."""
    global _bridge_singleton
    _bridge_singleton = service


def reset_browser_service() -> None:
    """Reset the global BrowserBridge singleton (for testing)."""
    global _bridge_singleton
    _bridge_singleton = None


class BrowserBridge:
    """Adapter that presents Rex's browser automation as an OpenClaw provider.

    Implements :class:`~rex.contracts.browser.BrowserAutomationProtocol`
    using :mod:`rex.openclaw.browser_core` directly — no dependency on the
    retired ``rex.browser_automation`` module.

    Args:
        storage_path: Base storage path for sessions and screenshots.
            Defaults to ``data/browser_sessions``.
    """

    def __init__(self, storage_path: Path | None = None) -> None:
        self.storage_path = storage_path or Path("data/browser_sessions")
        self.storage_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BrowserAutomationProtocol implementation
    # ------------------------------------------------------------------

    async def execute_script(
        self,
        script_path: str,
        headless: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute a browser automation script loaded from a JSON file.

        Args:
            script_path: Path to a JSON file containing ``steps`` and optional
                ``session_name`` fields.
            headless: When ``True``, run the browser without a visible window.

        Returns:
            List of per-step result dicts (each has ``"step"``, ``"action"``,
            ``"status"``, and action-specific keys).
        """
        with open(script_path) as f:
            script_data = json.load(f)

        steps = script_data.get("steps", [])
        session_name = script_data.get("session_name")

        return await run_browser_script(steps, headless, session_name)

    def list_sessions(self) -> list[str]:
        """Return the names of all persisted browser sessions."""
        if not self.storage_path.exists():
            return []
        return [
            d.name for d in self.storage_path.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

    def list_screenshots(self) -> list[str]:
        """Return the filenames of all captured screenshots."""
        screenshot_path = self.storage_path / "screenshots"
        if not screenshot_path.exists():
            return []
        return [
            f.name
            for f in screenshot_path.iterdir()
            if f.is_file() and f.suffix in [".png", ".jpg", ".jpeg"]
        ]

