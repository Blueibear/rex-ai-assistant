"""Canonical per-user calendar account authorization and routing (issue #303).

This module is the single source of truth for deciding **which calendar
accounts a user may touch** and **which account serves a request**.  Every
real calendar surface (service, CLI, GUI/API routes, Electron bridge,
OpenClaw tools, local tool executor, scheduler) must route account selection
through :class:`CalendarAccountResolver`.

Ownership model
---------------
- ``calendar.accounts`` in ``config/rex_config.json`` is the canonical
  non-secret account-definition list (provider, ICS source,
  ``credential_ref`` — an environment-variable *name*, never a secret value).
- ``users.{user_id}.calendar_accounts`` is the authoritative authorization
  map assigning account IDs to users.
- ``users.{user_id}.default_calendar_account_id`` selects that user's
  default account.  It must reference an account the user owns; a foreign or
  unknown default is ignored (fail closed to the user's own fallback).
- ``calendar.accounts`` entries not assigned to any user belong **only** to
  the distinct ``default`` profile.  They are never shared with or silently
  reassigned to named users.
- The legacy global calendar configuration (``calendar.backend``/
  ``calendar.ics`` and ``calendar.provider`` with its global environment
  token) is synthesized into reserved legacy accounts that belong only to
  the ``default`` profile.  Named users can never use them — not even via an
  explicit assignment — so legacy global credentials are usable only by the
  explicit ``default`` profile.
- Legacy ``calendar.default_account_id`` applies only to the ``default``
  profile.

Routing order (always within the requesting user's authorized set):

1. Explicit requested account ID (after ownership validation).
2. The user's configured default account.
3. Legacy global default — only for the explicit ``default`` profile.
4. The first account assigned to the user (deterministic, config order).
5. Otherwise: no account (callers fail closed / report not-configured).

Unauthorized and nonexistent accounts are indistinguishable to callers:
both raise :class:`CalendarAccountAccessError` with the same generic
message.  Credential lookup happens only *after* ownership validation and
only with the authorized account definition's own ``credential_ref``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rex.identity import validate_user_id

logger = logging.getLogger(__name__)

#: Runtime config file backing the authorization map.
_CONFIG_PATH = Path("config/rex_config.json")


def config_stamp() -> int | None:
    """Modification stamp of the runtime config file.

    Used by long-lived services to invalidate cached resolvers so that
    revoking or reassigning a user's calendar accounts takes effect without
    a process restart.
    """
    try:
        return _CONFIG_PATH.stat().st_mtime_ns
    except OSError:
        return None


#: The distinct legacy single-user profile.  Legacy global calendar
#: configuration and the legacy global default belong to this profile only.
DEFAULT_PROFILE = "default"

#: Reserved IDs for accounts synthesized from the legacy global calendar
#: configuration.  These are never available to named users.
LEGACY_ICS_ACCOUNT_ID = "legacy-ics"
LEGACY_PROVIDER_ACCOUNT_PREFIX = "legacy-provider-"
_LEGACY_GOOGLE_TOKEN_ENV = "GOOGLE_CALENDAR_ACCESS_TOKEN"

#: Generic message for unauthorized-or-nonexistent accounts.  Must not vary
#: with account existence so callers cannot enumerate other users' accounts.
ACCOUNT_UNAVAILABLE_MSG = "Calendar account {account_id!r} is not available for user {user_id!r}"

_IDENTITY_REQUIRED_MSG = (
    "Calendar operations require an explicit valid user identity; "
    "refusing to fall back to a default or shared account"
)


class CalendarAccountAccessError(PermissionError):
    """The requested account is not available to the requesting user.

    Raised both for accounts owned by another user and for accounts that do
    not exist, with an identical message shape, so the two cases cannot be
    distinguished by the caller.
    """


class CalendarIdentityError(PermissionError):
    """A real calendar operation was attempted without a valid user identity."""


def require_user_id(user_id: object) -> str:
    """Validate *user_id* for calendar operations, failing closed.

    Returns:
        The validated user ID.

    Raises:
        CalendarIdentityError: If *user_id* is missing, blank, malformed, or
            traversal-style.  Raised before any account or credential lookup.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise CalendarIdentityError(_IDENTITY_REQUIRED_MSG)
    try:
        return validate_user_id(user_id)
    except ValueError as exc:
        raise CalendarIdentityError(_IDENTITY_REQUIRED_MSG) from exc


@dataclass(frozen=True)
class CalendarAccountDefinition:
    """Canonical non-secret calendar account definition.

    ``credential_ref`` names the logical credential-vault entry for this
    account's token. It is a *reference*, never the
    secret itself, and must not be echoed to CLI output, API responses,
    events, or error messages.
    """

    id: str
    label: str = ""
    provider: str = "ics"  # "ics" | "google" | "outlook" | "stub"
    credential_ref: str = ""
    ics_source: str = ""
    ics_url_timeout: int = 15
    legacy: bool = False  # synthesized from legacy global config


def _parse_account_definition(raw: dict[str, Any]) -> CalendarAccountDefinition | None:
    account_id = str(raw.get("id") or "").strip()
    if not account_id:
        return None
    ics_raw = raw.get("ics")
    ics_cfg = ics_raw if isinstance(ics_raw, dict) else {}
    try:
        url_timeout = int(ics_cfg.get("url_timeout", 15))
    except (TypeError, ValueError):
        url_timeout = 15
    return CalendarAccountDefinition(
        id=account_id,
        label=str(raw.get("label") or ""),
        provider=str(raw.get("provider") or "ics").strip().lower(),
        credential_ref=str(raw.get("credential_ref") or ""),
        ics_source=str(ics_cfg.get("source") or raw.get("ics_source") or ""),
        ics_url_timeout=url_timeout,
        legacy=False,
    )


def _synthesize_legacy_accounts(
    calendar_section: dict[str, Any],
) -> list[CalendarAccountDefinition]:
    """Build reserved legacy accounts from the pre-#303 global configuration.

    These represent the single-user global calendar setup (``backend: ics``
    and/or ``provider: google``) and belong only to the ``default`` profile.
    """
    legacy: list[CalendarAccountDefinition] = []

    backend = str(calendar_section.get("backend") or "").strip().lower()
    ics_raw = calendar_section.get("ics")
    ics_cfg = ics_raw if isinstance(ics_raw, dict) else {}
    source = str(ics_cfg.get("source") or "")
    if backend == "ics" and source:
        try:
            url_timeout = int(ics_cfg.get("url_timeout", 15))
        except (TypeError, ValueError):
            url_timeout = 15
        legacy.append(
            CalendarAccountDefinition(
                id=LEGACY_ICS_ACCOUNT_ID,
                label="Legacy ICS calendar",
                provider="ics",
                ics_source=source,
                ics_url_timeout=url_timeout,
                legacy=True,
            )
        )

    provider = str(calendar_section.get("provider") or "").strip().lower()
    if provider == "gmail":
        provider = "google"
    if provider in ("google", "outlook"):
        legacy.append(
            CalendarAccountDefinition(
                id=f"{LEGACY_PROVIDER_ACCOUNT_PREFIX}{provider}",
                label=f"Legacy {provider} calendar",
                provider=provider,
                credential_ref=_LEGACY_GOOGLE_TOKEN_ENV if provider == "google" else "",
                legacy=True,
            )
        )

    return legacy


def _parse_user_assignments(users_block: object) -> dict[str, list[str]]:
    """Parse ``users.{user_id}.calendar_accounts`` into ``{user_id: [account_id]}``.

    Accepts entries as ``{"account_id": "..."}`` dicts or bare strings.
    """
    result: dict[str, list[str]] = {}
    if not isinstance(users_block, dict):
        return result
    for user_id, user_data in users_block.items():
        if not isinstance(user_data, dict):
            continue
        raw_accounts = user_data.get("calendar_accounts", [])
        if not isinstance(raw_accounts, list):
            continue
        parsed: list[str] = []
        for entry in raw_accounts:
            if isinstance(entry, str) and entry.strip():
                parsed.append(entry.strip())
            elif isinstance(entry, dict):
                account_id = str(entry.get("account_id") or "").strip()
                if account_id:
                    parsed.append(account_id)
        if parsed:
            result[str(user_id)] = parsed
    return result


def _parse_user_defaults(users_block: object) -> dict[str, str]:
    """Parse ``users.{user_id}.default_calendar_account_id`` into a per-user map."""
    result: dict[str, str] = {}
    if not isinstance(users_block, dict):
        return result
    for user_id, user_data in users_block.items():
        if not isinstance(user_data, dict):
            continue
        default_id = user_data.get("default_calendar_account_id")
        if isinstance(default_id, str) and default_id.strip():
            result[str(user_id)] = default_id.strip()
    return result


class CalendarAccountResolver:
    """Authorization and routing for per-user calendar accounts.

    Args:
        accounts:        Canonical explicit account definitions.
        legacy_accounts: Accounts synthesized from the legacy global config
                         (``default`` profile only).
        user_accounts:   ``{user_id: [account_id, ...]}`` authorization map.
        user_defaults:   ``{user_id: account_id}`` per-user default selection.
        legacy_default_account_id: Legacy global ``calendar.default_account_id``
                         (``default`` profile only).
    """

    def __init__(
        self,
        accounts: list[CalendarAccountDefinition] | None = None,
        legacy_accounts: list[CalendarAccountDefinition] | None = None,
        user_accounts: dict[str, list[str]] | None = None,
        user_defaults: dict[str, str] | None = None,
        legacy_default_account_id: str | None = None,
    ) -> None:
        self._accounts: list[CalendarAccountDefinition] = list(accounts or [])
        self._legacy_accounts: list[CalendarAccountDefinition] = list(legacy_accounts or [])
        self._user_accounts: dict[str, list[str]] = dict(user_accounts or {})
        self._user_defaults: dict[str, str] = dict(user_defaults or {})
        self._legacy_default_account_id = legacy_default_account_id

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_raw_config(cls, raw_config: dict[str, Any] | None) -> CalendarAccountResolver:
        """Build a resolver from the merged runtime config dict."""
        raw = raw_config or {}
        calendar_section = raw.get("calendar")
        if not isinstance(calendar_section, dict):
            calendar_section = {}

        accounts: list[CalendarAccountDefinition] = []
        accounts_raw = calendar_section.get("accounts")
        if isinstance(accounts_raw, list):
            for item in accounts_raw:
                if not isinstance(item, dict):
                    continue
                definition = _parse_account_definition(item)
                if definition is None:
                    logger.warning("Skipping malformed calendar account entry (missing id)")
                    continue
                accounts.append(definition)

        legacy_accounts = _synthesize_legacy_accounts(calendar_section)

        users_block = raw.get("users")
        user_accounts = _parse_user_assignments(users_block)
        user_defaults = _parse_user_defaults(users_block)

        legacy_default = calendar_section.get("default_account_id")
        legacy_default_id = (
            legacy_default.strip()
            if isinstance(legacy_default, str) and legacy_default.strip()
            else None
        )

        return cls(
            accounts=accounts,
            legacy_accounts=legacy_accounts,
            user_accounts=user_accounts,
            user_defaults=user_defaults,
            legacy_default_account_id=legacy_default_id,
        )

    @classmethod
    def load(cls) -> CalendarAccountResolver:
        """Build a resolver from ``config/rex_config.json``."""
        from rex.config_manager import load_config

        try:
            raw = load_config()
        except Exception as exc:
            logger.warning("Failed to load config for calendar account resolution: %s", exc)
            raw = {}
        return cls.from_raw_config(raw)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def has_configured_accounts(self) -> bool:
        """True when any real account definition exists (explicit or legacy)."""
        return bool(self._accounts or self._legacy_accounts)

    def get_account_definition(self, account_id: str) -> CalendarAccountDefinition | None:
        """Return the canonical account definition for *account_id*, if any.

        This is an internal lookup used *after* ownership authorization;
        surfaces must never expose foreign definitions obtained here.
        """
        for acct in self._accounts:
            if acct.id == account_id:
                return acct
        for acct in self._legacy_accounts:
            if acct.id == account_id:
                return acct
        return None

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    def account_ids_for_user(self, user_id: str) -> list[str]:
        """Return the account IDs authorized for *user_id* (deterministic order).

        Named users get only their explicit assignments, restricted to
        accounts that actually exist in ``calendar.accounts``.  Legacy
        synthesized accounts are excluded for named users even when a config
        entry tries to assign one (legacy global credentials are usable only
        by the ``default`` profile).

        The ``default`` profile additionally receives every explicit account
        no user has claimed, plus the legacy synthesized accounts.
        """
        validated = require_user_id(user_id)
        explicit_ids = [acct.id for acct in self._accounts]
        legacy_ids = [acct.id for acct in self._legacy_accounts]

        owned: list[str] = []
        for account_id in self._user_accounts.get(validated, []):
            if account_id in legacy_ids:
                logger.warning(
                    "Ignoring assignment of legacy calendar account to user %r: "
                    "legacy global calendar configuration is restricted to the "
                    "'default' profile",
                    validated,
                )
                continue
            if account_id in explicit_ids and account_id not in owned:
                owned.append(account_id)

        if validated == DEFAULT_PROFILE:
            assigned = {account_id for ids in self._user_accounts.values() for account_id in ids}
            for account_id in explicit_ids:
                if account_id not in assigned and account_id not in owned:
                    owned.append(account_id)
            for account_id in legacy_ids:
                if account_id not in owned:
                    owned.append(account_id)
        return owned

    def accounts_for_user(self, user_id: str) -> list[CalendarAccountDefinition]:
        """Return the canonical account definitions authorized for *user_id*."""
        definitions = []
        for account_id in self.account_ids_for_user(user_id):
            definition = self.get_account_definition(account_id)
            if definition is not None:
                definitions.append(definition)
        return definitions

    def configured_user_ids(self) -> list[str]:
        """Return the users with at least one authorized account.

        Only valid user IDs are returned (invalid persisted IDs are skipped,
        fail closed).  The ``default`` profile is included when unassigned or
        legacy accounts exist.  Order is deterministic.
        """
        ids: list[str] = []
        for user_id in self._user_accounts:
            try:
                validated = require_user_id(user_id)
            except CalendarIdentityError:
                logger.warning("Skipping calendar accounts assigned to an invalid user ID")
                continue
            if validated not in ids and self.account_ids_for_user(validated):
                ids.append(validated)
        if DEFAULT_PROFILE not in ids and self.account_ids_for_user(DEFAULT_PROFILE):
            ids.append(DEFAULT_PROFILE)
        return ids

    def is_account_authorized(self, user_id: str, account_id: str) -> bool:
        return account_id in self.account_ids_for_user(user_id)

    def check_account_access(self, user_id: str, account_id: str) -> None:
        """Raise :class:`CalendarAccountAccessError` unless *user_id* owns *account_id*."""
        validated = require_user_id(user_id)
        if not self.is_account_authorized(validated, account_id):
            raise CalendarAccountAccessError(
                ACCOUNT_UNAVAILABLE_MSG.format(account_id=account_id, user_id=validated)
            )

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
                "Ignoring default calendar account for user %r: not owned by that user",
                validated,
            )
        if validated == DEFAULT_PROFILE:
            legacy = self._legacy_default_account_id
            if legacy and legacy in owned:
                return legacy
        return None

    def resolve_account_id(self, user_id: str, account_id: str | None = None) -> str | None:
        """Resolve the account that serves a request for *user_id*.

        Returns:
            The resolved account ID, or ``None`` when the user has no
            authorized accounts (callers fail closed or report
            not-configured).

        Raises:
            CalendarIdentityError: On missing/invalid identity.
            CalendarAccountAccessError: When *account_id* is explicitly
                requested but is not available to this user (unauthorized or
                nonexistent — indistinguishable).
        """
        validated = require_user_id(user_id)
        owned = self.account_ids_for_user(validated)

        if account_id:
            if account_id not in owned:
                raise CalendarAccountAccessError(
                    ACCOUNT_UNAVAILABLE_MSG.format(account_id=account_id, user_id=validated)
                )
            return account_id

        default_id = self.default_account_id_for_user(validated)
        if default_id:
            return default_id

        return owned[0] if owned else None

    def resolve_account(
        self, user_id: str, account_id: str | None = None
    ) -> CalendarAccountDefinition | None:
        """Resolve the canonical account definition serving a request.

        Same contract as :meth:`resolve_account_id`, but returns the account
        definition.
        """
        resolved_id = self.resolve_account_id(user_id, account_id)
        if resolved_id is None:
            return None
        return self.get_account_definition(resolved_id)

    def provider_account_for_user(self, user_id: str) -> CalendarAccountDefinition | None:
        """Return the user's first OAuth-provider (google/outlook) account.

        Used by the GUI/bridge calendar surfaces, which speak the provider
        API rather than ICS.  Ownership is enforced by
        :meth:`accounts_for_user`.
        """
        for definition in self.accounts_for_user(user_id):
            if definition.provider in ("google", "outlook"):
                return definition
        return None


__all__ = [
    "ACCOUNT_UNAVAILABLE_MSG",
    "DEFAULT_PROFILE",
    "LEGACY_ICS_ACCOUNT_ID",
    "LEGACY_PROVIDER_ACCOUNT_PREFIX",
    "CalendarAccountAccessError",
    "CalendarAccountDefinition",
    "CalendarAccountResolver",
    "CalendarIdentityError",
    "config_stamp",
    "require_user_id",
]
