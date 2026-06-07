"""Multi-account email configuration models (Pydantic v2).

Config lives in ``config/rex_config.json`` under the ``email`` key.
Secrets (passwords, app-passwords) are **never** stored in config; instead
each account carries a ``credential_ref`` that maps to a key in the
``CredentialManager`` (environment variable or ``config/credentials.json``).

Example config fragment::

    {
      "email": {
        "default_account_id": "personal",
        "accounts": [
          {
            "id": "personal",
            "label": "Personal Gmail",
            "address": "you@gmail.com",
            "imap": {"host": "imap.gmail.com", "port": 993, "ssl": true},
            "smtp": {"host": "smtp.gmail.com", "port": 587, "starttls": true},
            "credential_ref": "email:personal"
          }
        ]
      }
    }
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


class ImapServerConfig(BaseModel):
    """IMAP server settings for one account."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(..., description="IMAP server hostname")
    port: int = Field(default=993, description="IMAP server port")
    ssl: bool = Field(default=True, description="Use IMAP4-SSL")


class SmtpServerConfig(BaseModel):
    """SMTP server settings for one account."""

    model_config = ConfigDict(extra="forbid")

    host: str = Field(..., description="SMTP server hostname")
    port: int = Field(default=587, description="SMTP server port")
    starttls: bool = Field(default=True, description="Use STARTTLS (else SMTPS)")


class EmailAccountConfig(BaseModel):
    """Configuration for a single email account."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Unique account identifier (e.g. 'personal')")
    label: str = Field(default="", description="Human-friendly label")
    address: str = Field(..., description="Email address for this account")
    imap: ImapServerConfig = Field(..., description="IMAP server settings")
    smtp: SmtpServerConfig = Field(..., description="SMTP server settings")
    credential_ref: str = Field(
        ...,
        description=(
            "Key used to look up username/password via CredentialManager. "
            "The credential token is expected to be in 'username:password' format."
        ),
    )


class EmailConfig(BaseModel):
    """Top-level email configuration with multi-account support."""

    model_config = ConfigDict(extra="forbid")

    default_account_id: str | None = Field(
        default=None,
        description="Account ID used when no explicit account is specified",
    )
    accounts: list[EmailAccountConfig] = Field(
        default_factory=list,
        description="List of configured email accounts",
    )

    def get_account(self, account_id: str | None = None) -> EmailAccountConfig | None:
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
            logger.warning("Requested email account '%s' not found", account_id)
            return None

        if self.default_account_id:
            for acct in self.accounts:
                if acct.id == self.default_account_id:
                    return acct
            logger.warning(
                "Default email account '%s' not found; falling back",
                self.default_account_id,
            )

        return self.accounts[0]

    def list_account_ids(self) -> list[str]:
        """Return all configured account IDs."""
        return [acct.id for acct in self.accounts]


def load_email_config(raw_config: dict[str, Any]) -> EmailConfig:
    """Parse the ``email`` section from the merged runtime config.

    Args:
        raw_config: The full runtime config dict.

    Returns:
        An ``EmailConfig`` model.  If the ``email`` key is absent or empty
        the model defaults to an empty account list (stub mode).
    """
    email_section = raw_config.get("email")
    if not email_section or not isinstance(email_section, dict):
        return EmailConfig()
    return EmailConfig.model_validate(email_section)


__all__ = [
    "EmailAccountConfig",
    "EmailConfig",
    "ImapServerConfig",
    "SmtpServerConfig",
    "load_email_config",
]
