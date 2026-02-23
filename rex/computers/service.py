"""High-level computer service used by the CLI.

This module bridges config + credentials + the HTTP client into a single,
easy-to-use facade for CLI commands.

Usage example::

    svc = get_computer_service()
    result = svc.status("desktop")
    print(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from rex.computers.client import AgentClient, HealthResult, RunResult, StatusResult
from rex.computers.config import ComputerConfig, ComputersConfig, load_computers_config

logger = logging.getLogger(__name__)


class ComputerNotFoundError(Exception):
    """Raised when a requested computer ID is not in the config."""


class ComputerDisabledError(Exception):
    """Raised when a requested computer is disabled."""


class AllowlistDeniedError(Exception):
    """Raised when a command is not on the client-side allowlist."""


class MissingTokenError(Exception):
    """Raised when the auth token for a computer is not configured."""


@dataclass
class ComputerInfo:
    """Summary information about a configured computer."""

    id: str
    label: str
    base_url: str
    enabled: bool
    allowed_commands: list[str]


class ComputerService:
    """Facade over config, credentials, and the agent HTTP client.

    Parameters
    ----------
    computers_config:
        Parsed :class:`~rex.computers.config.ComputersConfig`.
    credential_manager:
        A :class:`~rex.credentials.CredentialManager` instance used to
        resolve auth tokens from ``auth_token_ref`` values.
    """

    def __init__(
        self,
        computers_config: ComputersConfig,
        credential_manager: Any,
    ) -> None:
        self._config = computers_config
        self._creds = credential_manager

    # ------------------------------------------------------------------
    # List / inspect
    # ------------------------------------------------------------------

    def list_computers(self, *, include_disabled: bool = False) -> list[ComputerInfo]:
        """Return a list of configured computers.

        Args:
            include_disabled: When ``True``, include disabled entries too.

        Returns:
            List of :class:`ComputerInfo` summaries.
        """
        computers = self._config.list_all() if include_disabled else self._config.list_enabled()
        return [
            ComputerInfo(
                id=c.id,
                label=c.label,
                base_url=c.base_url,
                enabled=c.enabled,
                allowed_commands=list(c.allowlists.commands),
            )
            for c in computers
        ]

    # ------------------------------------------------------------------
    # Remote operations
    # ------------------------------------------------------------------

    def health(self, computer_id: str) -> HealthResult:
        """Call ``GET /health`` on the specified computer.

        Args:
            computer_id: The ``id`` field from config.

        Returns:
            :class:`~rex.computers.client.HealthResult`.

        Raises:
            :class:`ComputerNotFoundError`: If the computer ID is unknown.
            :class:`ComputerDisabledError`: If the computer is disabled.
            :class:`MissingTokenError`: If the auth token is not configured.
        """
        client = self._make_client(computer_id)
        return client.health()

    def status(self, computer_id: str) -> StatusResult:
        """Call ``GET /status`` on the specified computer.

        Args:
            computer_id: The ``id`` field from config.

        Returns:
            :class:`~rex.computers.client.StatusResult`.

        Raises:
            :class:`ComputerNotFoundError`: If the computer ID is unknown.
            :class:`ComputerDisabledError`: If the computer is disabled.
            :class:`MissingTokenError`: If the auth token is not configured.
        """
        client = self._make_client(computer_id)
        return client.status()

    def run(
        self,
        computer_id: str,
        command: str,
        args: list[str] | None = None,
        cwd: str | None = None,
    ) -> RunResult:
        """Execute a command on the specified computer via the agent API.

        The command is checked against the client-side allowlist **before** any
        network call is made.

        Args:
            computer_id: The ``id`` field from config.
            command: Command name to run (e.g. ``"whoami"``).
            args: Optional list of arguments.
            cwd: Optional working directory on the remote host.

        Returns:
            :class:`~rex.computers.client.RunResult`.

        Raises:
            :class:`ComputerNotFoundError`: If the computer ID is unknown.
            :class:`ComputerDisabledError`: If the computer is disabled.
            :class:`MissingTokenError`: If the auth token is not configured.
            :class:`AllowlistDeniedError`: If the command is not allowlisted.
        """
        cfg = self._resolve_computer(computer_id)
        if not cfg.is_command_allowed(command):
            allowed = ", ".join(cfg.allowlists.commands) or "(none)"
            raise AllowlistDeniedError(
                f"Command {command!r} is not on the allowlist for computer {computer_id!r}. "
                f"Allowed commands: {allowed}"
            )
        client = self._make_client(computer_id, cfg=cfg)
        return client.run(command, args=args, cwd=cwd)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_computer(self, computer_id: str) -> ComputerConfig:
        """Resolve and validate a computer config entry.

        Raises:
            :class:`ComputerNotFoundError`: unknown ID.
            :class:`ComputerDisabledError`: computer is disabled.
        """
        cfg = self._config.get_computer(computer_id)
        if cfg is None:
            raise ComputerNotFoundError(
                f"No computer with id {computer_id!r} found in config. "
                "Run 'rex pc list --all' to see all configured computers."
            )
        if not cfg.enabled:
            raise ComputerDisabledError(
                f"Computer {computer_id!r} is disabled. "
                "Enable it in config/rex_config.json to use it."
            )
        return cfg

    def _resolve_token(self, cfg: ComputerConfig) -> str:
        """Resolve the auth token via CredentialManager.

        Raises:
            :class:`MissingTokenError`: if the token is not configured.
        """
        token = self._creds.get_token(cfg.auth_token_ref)
        if not token:
            raise MissingTokenError(
                f"Auth token for computer {cfg.id!r} is not configured. "
                f"Set the environment variable referenced by auth_token_ref={cfg.auth_token_ref!r} "
                "in your .env file or config/credentials.json."
            )
        return token

    def _make_client(self, computer_id: str, *, cfg: ComputerConfig | None = None) -> AgentClient:
        """Build an :class:`AgentClient` for the given computer.

        Args:
            computer_id: Computer ID.
            cfg: Pre-resolved config (avoids duplicate lookup).

        Returns:
            Configured :class:`AgentClient`.

        Raises:
            :class:`ComputerNotFoundError`, :class:`ComputerDisabledError`,
            :class:`MissingTokenError`.
        """
        if cfg is None:
            cfg = self._resolve_computer(computer_id)
        token = self._resolve_token(cfg)
        return AgentClient(
            base_url=cfg.base_url,
            token=token,
            connect_timeout=cfg.connect_timeout,
            read_timeout=cfg.read_timeout,
            computer_id=cfg.id,
        )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy)
# ---------------------------------------------------------------------------

_service: ComputerService | None = None


def get_computer_service() -> ComputerService:
    """Return the module-level :class:`ComputerService` singleton.

    Config is loaded from ``rex_config.json`` and credentials from the
    global :func:`~rex.credentials.get_credential_manager`.
    """
    global _service  # noqa: PLW0603
    if _service is None:
        from rex.config_manager import load_config
        from rex.credentials import get_credential_manager

        raw = load_config()
        computers_config = load_computers_config(raw)
        _service = ComputerService(
            computers_config=computers_config,
            credential_manager=get_credential_manager(),
        )
    return _service


__all__ = [
    "AllowlistDeniedError",
    "ComputerDisabledError",
    "ComputerInfo",
    "ComputerNotFoundError",
    "ComputerService",
    "MissingTokenError",
    "get_computer_service",
]
