"""Unit tests for rex.calendar_accounts (issue #303).

Covers the canonical per-user calendar account authorization and routing
model: identity validation, ownership checks, generic unauthorized errors,
per-user defaults, deterministic routing, legacy-account policy, and owner
enumeration.
"""

from __future__ import annotations

import pytest

from rex.calendar_accounts import (
    ACCOUNT_UNAVAILABLE_MSG,
    DEFAULT_PROFILE,
    LEGACY_ICS_ACCOUNT_ID,
    CalendarAccountAccessError,
    CalendarAccountResolver,
    CalendarIdentityError,
    require_user_id,
)

ALICE = "alice"
BOB = "bob"


def _two_user_raw_config() -> dict:
    return {
        "calendar": {
            "accounts": [
                {
                    "id": "alice-cal",
                    "label": "Alice calendar",
                    "provider": "ics",
                    "ics": {"source": "https://example.com/alice.ics"},
                },
                {
                    "id": "bob-cal",
                    "label": "Bob calendar",
                    "provider": "google",
                    "credential_ref": "GOOGLE_CALENDAR_TOKEN_BOB",
                },
            ],
        },
        "users": {
            ALICE: {"calendar_accounts": [{"account_id": "alice-cal"}]},
            BOB: {"calendar_accounts": [{"account_id": "bob-cal"}]},
        },
    }


def _legacy_only_raw_config() -> dict:
    return {
        "calendar": {
            "backend": "ics",
            "ics": {"source": "https://example.com/legacy.ics", "url_timeout": 10},
            "provider": "google",
        },
    }


# ---------------------------------------------------------------------------
# Identity validation
# ---------------------------------------------------------------------------


class TestRequireUserId:
    @pytest.mark.parametrize(
        "bad",
        [None, "", "   ", 42, b"alice", "../../etc/passwd", "..", "a/b", "a\\b", "user id"],
    )
    def test_invalid_identities_fail_closed(self, bad):
        with pytest.raises(CalendarIdentityError):
            require_user_id(bad)

    def test_valid_identity_returned(self):
        assert require_user_id("alice") == "alice"
        assert require_user_id("default") == "default"

    def test_error_is_permission_error(self):
        with pytest.raises(PermissionError):
            require_user_id(None)


# ---------------------------------------------------------------------------
# Ownership and enumeration
# ---------------------------------------------------------------------------


class TestOwnership:
    def test_users_see_only_their_own_accounts(self):
        resolver = CalendarAccountResolver.from_raw_config(_two_user_raw_config())
        assert resolver.account_ids_for_user(ALICE) == ["alice-cal"]
        assert resolver.account_ids_for_user(BOB) == ["bob-cal"]

    def test_unassigned_named_user_has_no_accounts(self):
        resolver = CalendarAccountResolver.from_raw_config(_two_user_raw_config())
        assert resolver.account_ids_for_user("charlie") == []

    def test_foreign_explicit_account_raises_generic_error(self):
        resolver = CalendarAccountResolver.from_raw_config(_two_user_raw_config())
        with pytest.raises(CalendarAccountAccessError) as exc:
            resolver.resolve_account_id(ALICE, "bob-cal")
        assert str(exc.value) == ACCOUNT_UNAVAILABLE_MSG.format(account_id="bob-cal", user_id=ALICE)

    def test_nonexistent_account_indistinguishable_from_foreign(self):
        resolver = CalendarAccountResolver.from_raw_config(_two_user_raw_config())
        with pytest.raises(CalendarAccountAccessError) as foreign:
            resolver.resolve_account_id(ALICE, "bob-cal")
        with pytest.raises(CalendarAccountAccessError) as missing:
            resolver.resolve_account_id(ALICE, "no-such-account")
        # Same message shape: only the requested ID varies.
        assert str(foreign.value).replace("bob-cal", "X") == str(missing.value).replace(
            "no-such-account", "X"
        )

    def test_check_account_access_requires_valid_identity(self):
        resolver = CalendarAccountResolver.from_raw_config(_two_user_raw_config())
        with pytest.raises(CalendarIdentityError):
            resolver.check_account_access("../evil", "alice-cal")

    def test_configured_user_ids_are_deterministic_and_valid(self):
        raw = _two_user_raw_config()
        raw["users"]["../bad"] = {"calendar_accounts": [{"account_id": "alice-cal"}]}
        resolver = CalendarAccountResolver.from_raw_config(raw)
        assert resolver.configured_user_ids() == [ALICE, BOB]

    def test_unassigned_accounts_belong_to_default_profile(self):
        raw = _two_user_raw_config()
        raw["calendar"]["accounts"].append(
            {"id": "orphan-cal", "provider": "ics", "ics": {"source": "x.ics"}}
        )
        resolver = CalendarAccountResolver.from_raw_config(raw)
        assert "orphan-cal" in resolver.account_ids_for_user(DEFAULT_PROFILE)
        assert "orphan-cal" not in resolver.account_ids_for_user(ALICE)
        assert "orphan-cal" not in resolver.account_ids_for_user(BOB)


# ---------------------------------------------------------------------------
# Legacy policy
# ---------------------------------------------------------------------------


class TestLegacyPolicy:
    def test_legacy_accounts_belong_only_to_default(self):
        resolver = CalendarAccountResolver.from_raw_config(_legacy_only_raw_config())
        default_ids = resolver.account_ids_for_user(DEFAULT_PROFILE)
        assert LEGACY_ICS_ACCOUNT_ID in default_ids
        assert resolver.account_ids_for_user(ALICE) == []

    def test_named_user_cannot_claim_legacy_account_even_explicitly(self):
        raw = _legacy_only_raw_config()
        raw["users"] = {ALICE: {"calendar_accounts": [{"account_id": LEGACY_ICS_ACCOUNT_ID}]}}
        resolver = CalendarAccountResolver.from_raw_config(raw)
        assert resolver.account_ids_for_user(ALICE) == []
        with pytest.raises(CalendarAccountAccessError):
            resolver.resolve_account_id(ALICE, LEGACY_ICS_ACCOUNT_ID)

    def test_legacy_provider_account_synthesized_for_default(self):
        resolver = CalendarAccountResolver.from_raw_config(_legacy_only_raw_config())
        providers = {a.provider for a in resolver.accounts_for_user(DEFAULT_PROFILE)}
        assert providers == {"ics", "google"}

    def test_empty_ics_source_is_not_configured(self):
        resolver = CalendarAccountResolver.from_raw_config(
            {"calendar": {"backend": "ics", "ics": {"source": ""}}}
        )
        assert not resolver.has_configured_accounts()

    def test_stub_backend_is_not_configured(self):
        resolver = CalendarAccountResolver.from_raw_config({"calendar": {"backend": "stub"}})
        assert not resolver.has_configured_accounts()

    def test_legacy_global_default_applies_only_to_default_profile(self):
        raw = _two_user_raw_config()
        raw["calendar"]["default_account_id"] = "alice-cal"
        # alice-cal is assigned to alice, so the default profile does not own
        # it and the legacy default must be ignored even for `default`.
        resolver = CalendarAccountResolver.from_raw_config(raw)
        assert resolver.default_account_id_for_user(DEFAULT_PROFILE) is None
        # Named users are unaffected by the legacy global default.
        assert resolver.default_account_id_for_user(BOB) is None
        assert resolver.resolve_account_id(BOB) == "bob-cal"


# ---------------------------------------------------------------------------
# Per-user defaults and routing
# ---------------------------------------------------------------------------


class TestDefaultsAndRouting:
    def test_per_user_defaults_are_independent(self):
        raw = _two_user_raw_config()
        raw["users"][ALICE]["calendar_accounts"].append({"account_id": "alice-cal-2"})
        raw["calendar"]["accounts"].append(
            {"id": "alice-cal-2", "provider": "ics", "ics": {"source": "y.ics"}}
        )
        raw["users"][ALICE]["default_calendar_account_id"] = "alice-cal-2"
        raw["users"][BOB]["default_calendar_account_id"] = "bob-cal"
        resolver = CalendarAccountResolver.from_raw_config(raw)
        assert resolver.resolve_account_id(ALICE) == "alice-cal-2"
        assert resolver.resolve_account_id(BOB) == "bob-cal"

    def test_foreign_default_is_ignored_fail_closed(self):
        raw = _two_user_raw_config()
        raw["users"][ALICE]["default_calendar_account_id"] = "bob-cal"
        resolver = CalendarAccountResolver.from_raw_config(raw)
        # Falls back to Alice's own first account, never to Bob's.
        assert resolver.resolve_account_id(ALICE) == "alice-cal"

    def test_first_assigned_account_fallback_is_deterministic(self):
        raw = _two_user_raw_config()
        raw["calendar"]["accounts"].append(
            {"id": "alice-cal-2", "provider": "ics", "ics": {"source": "y.ics"}}
        )
        raw["users"][ALICE]["calendar_accounts"].append({"account_id": "alice-cal-2"})
        resolver = CalendarAccountResolver.from_raw_config(raw)
        assert resolver.resolve_account_id(ALICE) == "alice-cal"

    def test_no_accounts_resolves_to_none(self):
        resolver = CalendarAccountResolver.from_raw_config(_two_user_raw_config())
        assert resolver.resolve_account_id("charlie") is None

    def test_resolution_requires_identity(self):
        resolver = CalendarAccountResolver.from_raw_config(_two_user_raw_config())
        with pytest.raises(CalendarIdentityError):
            resolver.resolve_account_id(None)  # type: ignore[arg-type]
        with pytest.raises(CalendarIdentityError):
            resolver.resolve_account_id("")

    def test_provider_account_for_user_respects_ownership(self):
        resolver = CalendarAccountResolver.from_raw_config(_two_user_raw_config())
        assert resolver.provider_account_for_user(ALICE) is None  # ics only
        bob_account = resolver.provider_account_for_user(BOB)
        assert bob_account is not None and bob_account.id == "bob-cal"
        assert resolver.provider_account_for_user("charlie") is None
