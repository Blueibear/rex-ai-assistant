"""OpenClaw browser bridge — US-P4-022.

Implements :class:`~rex.contracts.browser.BrowserAutomationProtocol` by
delegating to Rex's existing :class:`~rex.browser_automation.BrowserAutomationService`
singleton.

This bridge is the first step in routing browser automation operations through
OpenClaw.  It presents the ``BrowserAutomationProtocol`` interface so that
callers do not need to import ``rex.browser_automation`` directly and can be
swapped once the full OpenClaw browser-control API is confirmed.

When the ``openclaw`` package is not installed, :meth:`register` logs a
warning and returns ``None``.  All other methods work without OpenClaw
installed because they delegate to the existing Rex browser automation service.

Typical usage::

    from rex.openclaw.browser_bridge import BrowserBridge

    bridge = BrowserBridge()

    # Execute a browser script
    results = await bridge.execute_script("/path/to/script.json", headless=True)

    # List sessions
    sessions = bridge.list_sessions()
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

from rex.browser_automation import (
    BrowserAutomationService as _BrowserAutomationService,
)
from rex.browser_automation import get_browser_service as _get_browser_service

logger = logging.getLogger(__name__)

OPENCLAW_AVAILABLE: bool = find_spec("openclaw") is not None

if OPENCLAW_AVAILABLE:  # pragma: no cover
    import openclaw as _openclaw  # type: ignore[import-not-found]
else:
    _openclaw = None  # type: ignore[assignment]


class BrowserBridge:
    """Adapter that presents Rex's browser automation service as an OpenClaw provider.

    Implements :class:`~rex.contracts.browser.BrowserAutomationProtocol` by
    delegating all operations to an underlying
    :class:`~rex.browser_automation.BrowserAutomationService` instance.

    When no ``service`` is supplied the global Rex singleton (via
    :func:`~rex.browser_automation.get_browser_service`) is used.

    When ``openclaw`` is installed, :meth:`register` registers the bridge
    as the browser provider so that OpenClaw routes browser tasks through Rex
    (stub — filled in once the OpenClaw browser-provider API is confirmed).

    Args:
        service: Optional explicit :class:`~rex.browser_automation.BrowserAutomationService`
            instance.  Defaults to the global Rex browser service singleton.
    """

    def __init__(self, service: _BrowserAutomationService | None = None) -> None:
        self._service: _BrowserAutomationService = (
            service if service is not None else _get_browser_service()
        )

    # ------------------------------------------------------------------
    # BrowserAutomationProtocol implementation
    # ------------------------------------------------------------------

    async def execute_script(
        self,
        script_path: str,
        headless: bool = True,
    ) -> list[dict[str, Any]]:
        """Execute a browser automation script loaded from a JSON file.

        Delegates to :meth:`~rex.browser_automation.BrowserAutomationService.execute_script`.

        Args:
            script_path: Path to a JSON file containing ``steps`` and optional
                ``session_name`` fields.
            headless: When ``True``, run the browser without a visible window.

        Returns:
            List of per-step result dicts (each has ``"step"``, ``"action"``,
            ``"status"``, and action-specific keys).
        """
        return await self._service.execute_script(script_path, headless=headless)

    def list_sessions(self) -> list[str]:
        """Return the names of all persisted browser sessions.

        Delegates to :meth:`~rex.browser_automation.BrowserAutomationService.list_sessions`.
        """
        return self._service.list_sessions()

    def list_screenshots(self) -> list[str]:
        """Return the filenames of all captured screenshots.

        Delegates to :meth:`~rex.browser_automation.BrowserAutomationService.list_screenshots`.
        """
        return self._service.list_screenshots()

    # ------------------------------------------------------------------
    # OpenClaw registration
    # ------------------------------------------------------------------

    def register(self, agent: Any = None) -> Any:
        """Register this bridge as the OpenClaw browser provider.

        When ``openclaw`` is installed, this method registers the bridge so
        that OpenClaw routes browser tasks through Rex's browser automation
        service.  When OpenClaw is absent, logs a warning and returns ``None``.

        .. note::
            The exact OpenClaw browser-provider registration call is a stub
            (see PRD §8.5 — *"Confirm OpenClaw's browser registration mechanism"*).
            Replace the ``# TODO`` below once the API is confirmed.

        Args:
            agent: Optional OpenClaw agent handle.

        Returns:
            The registration handle from OpenClaw, or ``None``.
        """
        if not OPENCLAW_AVAILABLE:
            logger.warning(
                "openclaw package not installed — BrowserBridge not registered as browser provider"
            )
            return None

        # TODO: replace with real OpenClaw browser provider registration once API is confirmed.
        # Expected shape (to be verified):
        #   handle = _openclaw.register_browser_provider(
        #       provider=self,
        #       agent=agent,
        #   )
        #   return handle
        logger.warning(
            "OpenClaw browser provider registration stub — update once API is confirmed (PRD §8.5)"
        )
        return None
