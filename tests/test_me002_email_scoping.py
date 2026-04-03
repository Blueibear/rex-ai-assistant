"""Tests for US-ME-002: Email service — route operations to requesting user's accounts only."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_account(account_id: str, owner: str) -> Any:
    """Return a minimal UserEmailAccount-like object."""
    from rex.config import UserEmailAccount

    return UserEmailAccount(
        account_id=account_id,
        display_name=f"{owner}/{account_id}",
        backend="imap",
        credentials_key=f"KEY_{account_id.upper()}",
    )


def _make_service(user_email_accounts: dict) -> Any:
    """Return an EmailService with injected per-user account map."""
    from rex.email_service import EmailService

    return EmailService(user_email_accounts=user_email_accounts)


# ---------------------------------------------------------------------------
# get_accounts()
# ---------------------------------------------------------------------------


def test_get_accounts_returns_owned_accounts():
    """get_accounts(user_id) returns only that user's accounts."""
    alice_account = _make_account("work", "alice")
    bob_account = _make_account("personal", "bob")

    svc = _make_service({"alice": [alice_account], "bob": [bob_account]})

    alice_accounts = svc.get_accounts("alice")
    assert len(alice_accounts) == 1
    assert alice_accounts[0].account_id == "work"


def test_get_accounts_does_not_return_other_users_accounts():
    """get_accounts('alice') never returns Bob's accounts."""
    alice_account = _make_account("work", "alice")
    bob_account = _make_account("personal", "bob")

    svc = _make_service({"alice": [alice_account], "bob": [bob_account]})

    alice_accounts = svc.get_accounts("alice")
    ids = {a.account_id for a in alice_accounts}
    assert "personal" not in ids  # Bob's account


def test_get_accounts_returns_empty_for_unknown_user():
    """get_accounts returns [] for a user with no configured accounts."""
    svc = _make_service({"alice": [_make_account("work", "alice")]})
    assert svc.get_accounts("charlie") == []


def test_get_accounts_multiple_accounts_for_user():
    """A user with two accounts gets both back."""
    svc = _make_service(
        {
            "alice": [
                _make_account("work", "alice"),
                _make_account("personal", "alice"),
            ]
        }
    )
    accounts = svc.get_accounts("alice")
    assert len(accounts) == 2
    account_ids = {a.account_id for a in accounts}
    assert account_ids == {"work", "personal"}


# ---------------------------------------------------------------------------
# _check_account_access() — PermissionError on cross-user access
# ---------------------------------------------------------------------------


def test_check_account_access_allows_own_account():
    """No exception raised when user accesses their own account."""
    svc = _make_service({"alice": [_make_account("work", "alice")]})
    svc._check_account_access("alice", "work")  # should not raise


def test_check_account_access_raises_for_foreign_account():
    """PermissionError raised when user tries to access another user's account."""
    svc = _make_service(
        {
            "alice": [_make_account("work", "alice")],
            "bob": [_make_account("personal", "bob")],
        }
    )
    import pytest

    with pytest.raises(PermissionError, match="alice"):
        svc._check_account_access("alice", "personal")  # Bob's account


def test_check_account_access_no_op_when_no_scoping():
    """When user_email_accounts is empty (no scoping), all access is allowed."""
    from rex.email_service import EmailService

    svc = EmailService()  # no user_email_accounts
    svc._check_account_access("alice", "any-account")  # should not raise


# ---------------------------------------------------------------------------
# send() enforces account ownership
# ---------------------------------------------------------------------------


def test_send_raises_permission_error_on_foreign_account():
    """send() raises PermissionError when user_id + account_id cross-reference."""
    svc = _make_service(
        {
            "alice": [_make_account("work", "alice")],
            "bob": [_make_account("personal", "bob")],
        }
    )
    import pytest

    with pytest.raises(PermissionError):
        svc.send(
            to="someone@example.com",
            subject="test",
            body="hello",
            user_id="alice",
            account_id="personal",  # Bob's account
        )


def test_send_allows_own_account():
    """send() succeeds when user_id owns the account_id."""
    svc = _make_service({"alice": [_make_account("work", "alice")]})
    result = svc.send(
        to="bob@example.com",
        subject="Hi",
        body="Hello",
        user_id="alice",
        account_id="work",
    )
    assert result.get("ok") is True


def test_send_without_user_id_skips_ownership_check():
    """send() without user_id does not enforce ownership (backwards compat)."""
    svc = _make_service({"alice": [_make_account("work", "alice")]})
    # No user_id — should not raise even with an account_id that belongs to alice
    result = svc.send(to="x@y.com", subject="s", body="b", account_id="work")
    assert result.get("ok") is True


# ---------------------------------------------------------------------------
# Integration: two users, verify isolation
# ---------------------------------------------------------------------------


def test_integration_two_users_see_only_own_accounts():
    """Integration: alice and bob each have one account; neither sees the other's."""
    svc = _make_service(
        {
            "alice": [_make_account("alice-work", "alice")],
            "bob": [_make_account("bob-work", "bob")],
        }
    )

    alice_accounts = svc.get_accounts("alice")
    bob_accounts = svc.get_accounts("bob")

    assert len(alice_accounts) == 1
    assert alice_accounts[0].account_id == "alice-work"

    assert len(bob_accounts) == 1
    assert bob_accounts[0].account_id == "bob-work"

    # Cross-access must fail
    import pytest

    with pytest.raises(PermissionError):
        svc._check_account_access("alice", "bob-work")

    with pytest.raises(PermissionError):
        svc._check_account_access("bob", "alice-work")
