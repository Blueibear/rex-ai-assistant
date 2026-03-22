"""Protocol defining the browser automation interface for Rex.

This contract captures the public API of ``rex.browser_automation`` so that an
OpenClaw-backed adapter can be substituted transparently.

Two protocols are defined:

- ``BrowserSessionProtocol`` — the async session API (navigate, click, type, …)
- ``BrowserAutomationProtocol`` — the service-level API used by Rex integrations
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class BrowserSessionProtocol(Protocol):
    """Structural protocol for a single browser automation session.

    Covers the full public API of ``rex.browser_automation.BrowserSession``:

    - ``launch`` / ``close`` — lifecycle management
    - ``navigate`` / ``click`` / ``type_text`` / ``wait`` / ``wait_for_selector`` — interaction
    - ``screenshot`` — capture
    - ``login`` — credential-backed login helper
    - ``download_file`` — file retrieval
    - ``get_content`` / ``get_text`` — content extraction

    All methods are *async*; implementations must be awaitable.
    """

    async def launch(self) -> None:
        """Launch the browser and create an initial page."""
        ...

    async def close(self) -> None:
        """Close the browser session and release all resources."""
        ...

    async def navigate(self, url: str, wait_for: str = "networkidle") -> dict[str, Any]:
        """Navigate to *url* and wait for the page to settle.

        Args:
            url: The URL to navigate to.
            wait_for: Wait condition — ``"load"``, ``"domcontentloaded"``, or
                ``"networkidle"``.

        Returns:
            Dict with ``"status"``, ``"title"``, and ``"url"`` keys.
        """
        ...

    async def click(self, selector: str, timeout: int = 5000) -> dict[str, Any]:
        """Click an element identified by CSS *selector*.

        Args:
            selector: CSS selector for the target element.
            timeout: Maximum wait time in milliseconds.

        Returns:
            Dict with ``"status"`` and ``"selector"`` keys.
        """
        ...

    async def type_text(self, selector: str, text: str, timeout: int = 5000) -> dict[str, Any]:
        """Fill *selector* element with *text*.

        Args:
            selector: CSS selector for the input element.
            text: Text to type/fill.
            timeout: Maximum wait time in milliseconds.

        Returns:
            Dict with ``"status"`` and ``"selector"`` keys.
        """
        ...

    async def wait(self, milliseconds: int) -> dict[str, Any]:
        """Pause execution for *milliseconds*.

        Returns:
            Dict with ``"status"`` and ``"waited_ms"`` keys.
        """
        ...

    async def wait_for_selector(self, selector: str, timeout: int = 5000) -> dict[str, Any]:
        """Wait until *selector* element appears in the DOM.

        Args:
            selector: CSS selector to wait for.
            timeout: Maximum wait time in milliseconds.

        Returns:
            Dict with ``"status"`` and ``"selector"`` keys.
        """
        ...

    async def screenshot(
        self,
        filename: Optional[str] = None,
        full_page: bool = False,
    ) -> dict[str, Any]:
        """Capture a screenshot of the current page.

        Args:
            filename: Output filename (auto-generated when ``None``).
            full_page: When ``True``, capture the full scrollable page.

        Returns:
            Dict with ``"status"``, ``"path"``, and ``"full_page"`` keys.
        """
        ...

    async def login(
        self,
        url: str,
        credential_name: str,
        username_selector: str = 'input[type="text"], input[name="username"]',
        password_selector: str = 'input[type="password"]',
        submit_selector: str = 'button[type="submit"], input[type="submit"]',
    ) -> dict[str, Any]:
        """Log in to a site using credentials from the credential manager.

        Args:
            url: Login page URL.
            credential_name: Key used to look up credentials (e.g., ``"github"``).
            username_selector: CSS selector for the username input.
            password_selector: CSS selector for the password input.
            submit_selector: CSS selector for the submit button.

        Returns:
            Dict with ``"status"``, ``"url"``, and ``"title"`` keys.
        """
        ...

    async def download_file(self, url: str, dest_path: str) -> dict[str, Any]:
        """Download a file from *url* to *dest_path*.

        Args:
            url: URL of the file to download.
            dest_path: Filesystem path where the file should be saved.

        Returns:
            Dict with ``"status"``, ``"path"``, and ``"suggested_filename"`` keys.
        """
        ...

    async def get_content(self) -> str:
        """Return the current page's full HTML content."""
        ...

    async def get_text(self, selector: Optional[str] = None) -> str:
        """Return text content from the page or from a specific element.

        Args:
            selector: CSS selector.  When ``None``, returns ``document.body.innerText``.

        Returns:
            Plain-text content.
        """
        ...


@runtime_checkable
class BrowserAutomationProtocol(Protocol):
    """Structural protocol for the Rex browser automation service.

    Covers the public API of ``rex.browser_automation.BrowserAutomationService``:

    - ``execute_script`` — run a multi-step script from a JSON file
    - ``list_sessions`` — enumerate persisted browser sessions
    - ``list_screenshots`` — enumerate captured screenshots
    """

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
        ...

    def list_sessions(self) -> list[str]:
        """Return the names of all persisted browser sessions."""
        ...

    def list_screenshots(self) -> list[str]:
        """Return the filenames of all captured screenshots."""
        ...
