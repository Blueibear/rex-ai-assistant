"""Canonical per-user email account authorization and routing (issue #303).

This module is the single source of truth for deciding **which email accounts
a user may touch** and **which account serves a request**.  Every real email
surface (service, CLI, Electron bridge, OpenClaw tools, scheduler, notification
channel) must route account selection through :class:`EmailAccountResolver`.

Ownership model
---------------
- ``email.accounts`` in ``config/rex_config.json`` is the canonical non-secret
  connection-metadata list (host/port/address/``credential_ref``).
- ``users.{user_id}.email_accounts`` is the authoritative authorization map
  assigning account IDs to users (parsed into
  ``AppConfig.user_email_accounts``).
- ``users.{user_id}.default_email_account_id`` selects that user's default
  account.  It must reference an account the user owns; a foreign or unknown
  default is ignored (fail closed to the user's own fallback).
- Legacy ``email.accounts`` entries not assigned to any user belong **only**
  to the distinct ``default`` profile.  They are never shared with or
  silently reassigned to named users.
- Legacy ``email.default_account_id`` applies only to the ``default`` profile.

Routing order (always within the requesting user's authorized set):

1. Explicit requested account ID (after ownership validation).
2. The user's configured default account.
3. Legacy global default — only for the explicit ``default`` profile.
4. The first account assigned to the user (deterministic, config order).
5. Otherwise: no account (callers fail closed / report not-configured).

Unauthorized and nonexistent accounts are indistinguishable to callers:
both raise :class:`EmailAccountAccessError` with the same generic message.
Credential lookup happens only *after* ownership validation and only with the
authorized account definition's own ``credential_ref``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rex.identity import validate_user_id

logger = logging.getLogger(__name__)

#: Runtime config file backing the authorization map.
_CONFIG_PATH = Path("config/rex_config.json")


def config_stamp() -> int | None:
    """Modification stamp of the runtime config file.

    Used by long-lived services to invalidate cached resolvers so that
    revoking or reassigning a user's email accounts takes effect without a
    process restart.
    """
    try:
        return _CONFIG_PATH.stat().st_mtime_ns
    except OSError:
        return None


#: The distinct legacy single-user profile.  Legacy global accounts and the
#: legacy global default belong to this profile only.
DEFAULT_PROFILE = "default"

#: Generic message for unauthorized-or-nonexistent accounts.  Must not vary
#: with account existence so callers cannot enumerate other users' accounts.
ACCOUNT_UNAVAILABLE_MSG = "Email account {account_id!r} is not available for user {user_id!r}"

_IDENTITY_REQUIRED_MSG = (
    "Email operations require an explicit valid user identity; "
    "refusing to fall back to a default or shared account"
)


class EmailAccountAccessError(PermissionError):
    """The requested account is not available to the requesting user.

    Raised both for accounts owned by another user and for accounts that do
    not exist, with an identical message shape, so the two cases cannot be
    distinguished by the caller.
    """


class EmailIdentityError(PermissionError):
    """A real email operation was attempted without a valid user identity."""


def require_user_id(user_id: object) -> str:
    """Validate *user_id* for email operations, failing closed.

    Returns:
        The validated user ID.

    Raises:
        EmailIdentityError: If *user_id* is missing, blank, malformed, or
            traversal-style.  Raised before any account or credential lookup.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise EmailIdentityError(_IDENTITY_REQUIRED_MSG)
    try:
        return validate_user_id(user_id)
    except ValueError as exc:
        raise EmailIdentityError(_IDENTITY_REQUIRED_MSG) from exc


class EmailAccountResolver:
    """Authorization and routing for per-user email accounts.

    Args:
        email_config:  Parsed ``EmailConfig`` (canonical account definitions).
        user_accounts: ``{user_id: [UserEmailAccount, ...]}`` authorization map.
        user_defaults: ``{user_id: account_id}`` per-user default selection.
    """

    def __init__(
        self,
        email_config: Any,
        user_accounts: dict[str, list[Any]] | None = None,
        user_defaults: dict[str, str] | None = None,
    ) -> None:
        self._email_config = email_config
        self._user_accounts: dict[str, list[Any]] = dict(user_accounts or {})
        self._user_defaults: dict[str, str] = dict(user_defaults or {})

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_raw_config(cls, raw_config: dict[str, Any] | None) -> EmailAccountResolver:
        """Build a resolver from the merged runtime config dict."""
        from rex.config import _parse_user_default_email_accounts, _parse_user_email_accounts
        from rex.email_backends.account_config import EmailConfig

        raw = raw_config or {}
        email_section = raw.get("email")
        if not isinstance(email_section, dict):
            email_section = {}
        # Only pass the keys EmailConfig knows about; the email section may
        # also carry unrelated keys such as ``provider``.
        try:
            email_config = EmailConfig.model_validate(
                {
                    "default_account_id": email_section.get("default_account_id"),
                    "accounts": email_section.get("accounts") or [],
                }
            )
        except Exception as exc:
            logger.warning("Invalid email account configuration ignored: %s", exc)
            email_config = EmailConfig()

        users_block = raw.get("users")
        user_accounts = _parse_user_email_accounts(users_block, email_section.get("accounts") or [])
        user_defaults = _parse_user_default_email_accounts(users_block)
        return cls(email_config, user_accounts, user_defaults)

    @classmethod
    def load(cls) -> EmailAccountResolver:
        """Build a resolver from ``config/rex_config.json``."""
        from rex.config_manager import load_config

        try:
            raw = load_config()
        except Exception as exc:
            logger.warning("Failed to load config for email account resolution: %s", exc)
            raw = {}
        return cls.from_raw_config(raw)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def email_config(self) -> Any:
        return self._email_config

    def has_configured_accounts(self) -> bool:
        """True when any real account definition exists."""
        return bool(getattr(self._email_config, "accounts", None))

    def get_account_definition(self, account_id: str) -> Any | None:
        """Return the canonical account definition for *account_id*, if any."""
        for acct in getattr(self._email_config, "accounts", []) or []:
            if acct.id == account_id:
                return acct
        return None

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def entries_for_user(self, user_id: str) -> list[Any]:
        """Return the ``UserEmailAccount`` entries authorized for *user_id*.

        For the ``default`` profile this includes legacy account definitions
        that no user has claimed.  Named users get only explicit assignments.
        """
        validated = require_user_id(user_id)
        entries = list(self._user_accounts.get(validated, []))
        if validated == DEFAULT_PROFILE:
            assigned = {
                entry.account_id
                for user_entries in self._user_accounts.values()
                for entry in user_entries
            }
            have = {entry.account_id for entry in entries}
            from rex.config import UserEmailAccount

            for acct in getattr(self._email_config, "accounts", []) or []:
                if acct.id not in assigned and acct.id not in have:
                    entries.append(
                        UserEmailAccount(
                            account_id=acct.id,
                            display_name=acct.label or acct.address,
                            backend="imap",
                            credentials_key=acct.credential_ref,
                        )
                    )
        return entries

    def account_ids_for_user(self, user_id: str) -> list[str]:
        """Return account IDs authorized for *user_id* (deterministic order)."""
        return [entry.account_id for entry in self.entries_for_user(user_id)]

    def configured_user_ids(self) -> list[str]:
        """Return the users with at least one authorized account.

        Only valid user IDs are returned (invalid persisted IDs are skipped,
        fail closed).  The ``default`` profile is included when legacy
        unassigned accounts exist.  Order is deterministic.
        """
        ids: list[str] = []
        for user_id in self._user_accounts:
            try:
                validated = require_user_id(user_id)
            except EmailIdentityError:
                logger.warning("Skipping email accounts assigned to an invalid user ID")
                continue
            if self._user_accounts.get(validated) and validated not in ids:
                ids.append(validated)
        if DEFAULT_PROFILE not in ids and self.entries_for_user(DEFAULT_PROFILE):
            ids.append(DEFAULT_PROFILE)
        return ids

    def is_account_authorized(self, user_id: str, account_id: str) -> bool:
        return account_id in self.account_ids_for_user(user_id)

    def check_account_access(self, user_id: str, account_id: str) -> None:
        """Raise :class:`EmailAccountAccessError` unless *user_id* owns *account_id*."""
        validated = require_user_id(user_id)
        if not self.is_account_authorized(validated, account_id):
            raise EmailAccountAccessError(
                ACCOUNT_UNAVAILABLE_MSG.format(account_id=account_id, user_id=validated)
            )

    def accounts_for_user(self, user_id: str) -> list[Any]:
        """Return the canonical account definitions authorized for *user_id*."""
        definitions = []
        for entry in self.entries_for_user(user_id):
            definition = self.get_account_definition(entry.account_id)
            if definition is not None:
                definitions.append(definition)
        return definitions

    def provider_entry_for_user(self, user_id: str) -> Any | None:
        """Return the user's first OAuth-provider account entry, if any.

        Used by the GUI inbox surfaces: a ``gmail``/``outlook`` backend entry
        carries its own ``credentials_key`` and does not reference an
        ``email.accounts`` IMAP definition.
        """
        for entry in self.entries_for_user(user_id):
            if getattr(entry, "backend", "") in ("gmail", "outlook"):
                return entry
        return None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def default_account_id_for_user(self, user_id: str) -> str | None:
        """Return the user's effective default account ID, or ``None``.

        A configured default that the user does not own is ignored (fail
        closed) rather than granting access or aborting.
        """
        validated = require_user_id(user_id)
        owned = self.account_ids_for_user(validated)
        configured = self._user_defaults.get(validated)
        if configured:
            if configured in owned:
                return configured
            logger.warning(
                "Ignoring default email account for user %r: not owned by that user",
                validated,
            )
        if validated == DEFAULT_PROFILE:
            legacy = getattr(self._email_config, "default_account_id", None)
            if legacy and legacy in owned:
                return str(legacy)
        return None

    def resolve_account_id(self, user_id: str, account_id: str | None = None) -> str | None:
        """Resolve the account that serves a request for *user_id*.

        Returns:
            The resolved account ID, or ``None`` when the user has no
            authorized accounts (callers fail closed or report
            not-configured).

        Raises:
            EmailIdentityError: On missing/invalid identity.
            EmailAccountAccessError: When *account_id* is explicitly requested
                but is not available to this user (unauthorized or
                nonexistent — indistinguishable).
        """
        validated = require_user_id(user_id)
        owned = self.account_ids_for_user(validated)

        if account_id:
            if account_id not in owned:
                raise EmailAccountAccessError(
                    ACCOUNT_UNAVAILABLE_MSG.format(account_id=account_id, user_id=validated)
                )
            return account_id

        default_id = self.default_account_id_for_user(validated)
        if default_id:
            return default_id

        # Deterministic fallback: first authorized account that has a real
        # definition; else the first authorized entry (stub-mode setups).
        for owned_id in owned:
            if self.get_account_definition(owned_id) is not None:
                return owned_id
        return owned[0] if owned else None

    def resolve_account(self, user_id: str, account_id: str | None = None) -> Any | None:
        """Resolve the canonical account definition serving a request.

        Same contract as :meth:`resolve_account_id`, but returns the account
        definition (or ``None`` if the resolved account has no definition,
        e.g. stub-only setups).
        """
        resolved_id = self.resolve_account_id(user_id, account_id)
        if resolved_id is None:
            return None
        return self.get_account_definition(resolved_id)


__all__ = [
    "ACCOUNT_UNAVAILABLE_MSG",
    "DEFAULT_PROFILE",
    "EmailAccountAccessError",
    "EmailAccountResolver",
    "EmailIdentityError",
    "require_user_id",
]
