"""Multi-account messaging configuration models (Pydantic v2).

Config lives in ``config/rex_config.json`` under the ``messaging`` key.
Secrets (Twilio auth tokens) are **never** stored in config; instead each
account carries a ``credential_ref`` that maps to a key in the
``CredentialManager`` (environment variable or ``config/credentials.json``).

Example config fragment::

    {
      "messaging": {
        "backend": "twilio",
        "default_account_id": "primary",
        "accounts": [
          {
            "id": "primary",
            "label": "Main Twilio",
            "from_number": "+15551234567",
            "credential_ref": "twilio:primary"
          }
        ]
      }
    }
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class MessagingAccountConfig(BaseModel):
    """Configuration for a single messaging/SMS account."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique account identifier (e.g. 'primary')")
    label: str = Field(default="", description="Human-friendly label")
    from_number: str = Field(
        ..., description="Sender phone number in E.164 format (e.g. '+15551234567')"
    )
    credential_ref: str = Field(
        ...,
        description=(
            "Key used to look up credentials via CredentialManager. "
            "The credential token is expected to be in 'account_sid:auth_token' format."
        ),
    )


class MessagingConfig(BaseModel):
    """Top-level messaging configuration with multi-account support."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["stub", "twilio"] = Field(
        default="stub",
        description="SMS backend to use: 'stub' (offline/mock) or 'twilio' (real delivery)",
    )
    default_account_id: str | None = Field(
        default=None,
        description="Account ID used when no explicit account is specified",
    )
    accounts: list[MessagingAccountConfig] = Field(
        default_factory=list,
        description="List of configured messaging accounts",
    )

    def get_account(self, account_id: str | None = None) -> MessagingAccountConfig | None:
        """Resolve an account by ID using the routing precedence.

        Routing order:
        1. *account_id* argument (explicit selection).
        2. ``default_account_id`` from config.
        3. First account in list (deterministic fallback).
        """
        if not self.accounts:
            return None

        if account_id:
            for acct in self.accounts:
                if acct.id == account_id:
                    return acct
            logger.warning("Requested messaging account '%s' not found", account_id)
            return None

        if self.default_account_id:
            for acct in self.accounts:
                if acct.id == self.default_account_id:
                    return acct
            logger.warning(
                "Default messaging account '%s' not found; falling back",
                self.default_account_id,
            )

        return self.accounts[0]

    def list_account_ids(self) -> list[str]:
        """Return all configured account IDs."""
        return [acct.id for acct in self.accounts]


def load_messaging_config(raw_config: dict[str, Any]) -> MessagingConfig:
    """Parse the ``messaging`` section from the merged runtime config.

    Args:
        raw_config: The full runtime config dict.

    Returns:
        A ``MessagingConfig`` model.  If the ``messaging`` key is absent or
        empty the model defaults to stub mode with no accounts.
    """
    messaging_section = raw_config.get("messaging")
    if not messaging_section or not isinstance(messaging_section, dict):
        return MessagingConfig()
    return MessagingConfig.model_validate(messaging_section)


__all__ = [
    "MessagingAccountConfig",
    "MessagingConfig",
    "load_messaging_config",
]
