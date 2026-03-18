"""Pydantic v2 config models for the ``computers[]`` section of rex_config.json.

Example config fragment::

    {
      "computers": [
        {
          "id": "desktop",
          "label": "Main Desktop",
          "base_url": "http://127.0.0.1:7777",
          "auth_token_ref": "PC_DESKTOP_TOKEN",
          "enabled": true,
          "allowlists": {
            "commands": ["whoami", "dir", "ipconfig", "systeminfo"]
          }
        }
      ]
    }

Security notes
--------------
- ``auth_token_ref`` is a CredentialManager lookup key.  The actual token is
  **never** stored in config; it must live in ``.env`` or
  ``config/credentials.json``.
- ``allowlists.commands`` is enforced **client-side** before any remote call.
- ``base_url`` must be an http(s) URL.  The default expectation is localhost,
  but this is not hardcoded; any valid URL is accepted.
- ``enabled=false`` hides the computer from normal list output.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class ComputerAllowlists(BaseModel):
    """Allowlisted operations for a remote computer."""

    model_config = ConfigDict(extra="forbid")

    commands: list[str] = Field(
        default_factory=list,
        description=(
            "List of command names (no arguments) that are permitted for remote execution. "
            "Checked client-side before any network call."
        ),
    )


class ComputerConfig(BaseModel):
    """Configuration for a single remote Windows computer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique computer identifier (e.g. 'desktop')")
    label: str = Field(default="", description="Human-friendly label")
    base_url: str = Field(
        ...,
        description=(
            "Base URL of the agent API on the remote computer "
            "(e.g. 'http://127.0.0.1:7777'). Must be http or https."
        ),
    )
    auth_token_ref: str = Field(
        ...,
        description=(
            "CredentialManager lookup key for the bearer token used to authenticate "
            "with the agent API.  The token itself must be stored in .env or "
            "config/credentials.json — never in this config file."
        ),
    )
    enabled: bool = Field(
        default=True,
        description=(
            "Whether this computer is active.  Disabled computers are hidden from "
            "normal list output and cannot be targeted by CLI commands."
        ),
    )
    allowlists: ComputerAllowlists = Field(
        default_factory=ComputerAllowlists,
        description="Client-side allowlists for permitted operations.",
    )
    connect_timeout: float = Field(
        default=5.0,
        description="Connection timeout in seconds for HTTP requests to the agent.",
    )
    read_timeout: float = Field(
        default=30.0,
        description="Read timeout in seconds for HTTP requests to the agent.",
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"base_url must use http or https scheme, got: {parsed.scheme!r}")
        if not parsed.netloc:
            raise ValueError("base_url must include a host (netloc)")
        return v.rstrip("/")

    def is_command_allowed(self, command: str) -> bool:
        """Return True if *command* appears in the client-side command allowlist.

        Only the base command name is checked (not arguments).  The caller is
        responsible for splitting the command from its arguments before calling
        this method.
        """
        return command in self.allowlists.commands


class ComputersConfig(BaseModel):
    """Top-level computers configuration holding all remote computer entries."""

    model_config = ConfigDict(extra="forbid")

    computers: list[ComputerConfig] = Field(
        default_factory=list,
        description="List of configured remote computers.",
    )

    def get_computer(self, computer_id: str) -> ComputerConfig | None:
        """Return the config for *computer_id*, or ``None`` if not found."""
        for computer in self.computers:
            if computer.id == computer_id:
                return computer
        return None

    def list_enabled(self) -> list[ComputerConfig]:
        """Return only enabled computers."""
        return [c for c in self.computers if c.enabled]

    def list_all(self) -> list[ComputerConfig]:
        """Return all computers including disabled ones."""
        return list(self.computers)


def load_computers_config(raw_config: dict[str, Any]) -> ComputersConfig:
    """Parse the ``computers`` section from the merged runtime config.

    Args:
        raw_config: The full runtime config dict (e.g. from ``load_config()``).

    Returns:
        A :class:`ComputersConfig` model.  If the ``computers`` key is absent or
        not a list, an empty config with no computers is returned.
    """
    computers_section = raw_config.get("computers")
    if not computers_section or not isinstance(computers_section, list):
        return ComputersConfig()
    return cast(ComputersConfig, ComputersConfig.model_validate({"computers": computers_section}))


__all__ = [
    "ComputerAllowlists",
    "ComputerConfig",
    "ComputersConfig",
    "load_computers_config",
]
