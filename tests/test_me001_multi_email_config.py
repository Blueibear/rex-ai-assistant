"""Tests for US-ME-001: Multi-email config schema — per-user account list."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# UserEmailAccount dataclass
# ---------------------------------------------------------------------------


def test_user_email_account_fields():
    """UserEmailAccount has account_id, display_name, backend, credentials_key."""
    import dataclasses

    from rex.config import UserEmailAccount

    fields = {f.name for f in dataclasses.fields(UserEmailAccount)}
    assert {"account_id", "display_name", "backend", "credentials_key"}.issubset(fields)


def test_user_email_account_defaults():
    """UserEmailAccount has sensible defaults."""
    from rex.config import UserEmailAccount

    ua = UserEmailAccount(account_id="work")
    assert ua.display_name == ""
    assert ua.backend == "imap"
    assert ua.credentials_key == ""


# ---------------------------------------------------------------------------
# AppConfig — user_email_accounts field
# ---------------------------------------------------------------------------


def test_appconfig_has_user_email_accounts_field():
    """AppConfig has user_email_accounts defaulting to empty dict."""
    import dataclasses

    from rex.config import AppConfig

    fields = {f.name for f in dataclasses.fields(AppConfig)}
    assert "user_email_accounts" in fields

    # Default factory should produce empty dict
    for f in dataclasses.fields(AppConfig):
        if f.name == "user_email_accounts":
            default = f.default_factory()  # type: ignore[misc]
            assert default == {}


# ---------------------------------------------------------------------------
# build_app_config — new users.{user_id}.email_accounts format
# ---------------------------------------------------------------------------


def test_build_app_config_parses_users_email_accounts():
    """build_app_config populates user_email_accounts from users block."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
            "users": {
                "alice": {
                    "email_accounts": [
                        {
                            "account_id": "work",
                            "display_name": "Alice Work",
                            "backend": "imap",
                            "credentials_key": "EMAIL_ALICE_WORK",
                        },
                        {
                            "account_id": "personal",
                            "display_name": "Alice Personal",
                            "backend": "gmail",
                            "credentials_key": "EMAIL_ALICE_PERSONAL",
                        },
                    ]
                }
            },
        }
    )
    assert "alice" in cfg.user_email_accounts
    accounts = cfg.user_email_accounts["alice"]
    assert len(accounts) == 2
    assert accounts[0].account_id == "work"
    assert accounts[0].display_name == "Alice Work"
    assert accounts[0].backend == "imap"
    assert accounts[0].credentials_key == "EMAIL_ALICE_WORK"
    assert accounts[1].account_id == "personal"
    assert accounts[1].backend == "gmail"


def test_build_app_config_multiple_users():
    """Multiple users each get their own account list."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
            "users": {
                "alice": {
                    "email_accounts": [
                        {"account_id": "a1", "backend": "imap", "credentials_key": "K1"}
                    ]
                },
                "bob": {
                    "email_accounts": [
                        {"account_id": "b1", "backend": "gmail", "credentials_key": "K2"}
                    ]
                },
            },
        }
    )
    assert len(cfg.user_email_accounts["alice"]) == 1
    assert len(cfg.user_email_accounts["bob"]) == 1
    assert cfg.user_email_accounts["alice"][0].account_id == "a1"
    assert cfg.user_email_accounts["bob"][0].account_id == "b1"


def test_build_app_config_empty_when_no_users():
    """user_email_accounts is empty dict when users block is absent."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {"models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"}}
    )
    assert cfg.user_email_accounts == {}


def test_build_app_config_skips_malformed_entry():
    """Entries without account_id are silently skipped."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
            "users": {
                "alice": {
                    "email_accounts": [
                        {"display_name": "no id here"},  # missing account_id
                        {"account_id": "valid", "credentials_key": "K"},
                    ]
                }
            },
        }
    )
    assert len(cfg.user_email_accounts["alice"]) == 1
    assert cfg.user_email_accounts["alice"][0].account_id == "valid"


# ---------------------------------------------------------------------------
# Migration shim — old email.accounts → default user
# ---------------------------------------------------------------------------


def test_migration_shim_promotes_legacy_accounts():
    """Old email.accounts list is migrated to user_email_accounts['default']."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
            "email": {
                "accounts": [
                    {
                        "id": "primary",
                        "address": "user@example.com",
                        "imap_host": "imap.example.com",
                        "credential_ref": "EMAIL_PRIMARY",
                    }
                ]
            },
        }
    )
    assert "default" in cfg.user_email_accounts
    migrated = cfg.user_email_accounts["default"]
    assert len(migrated) == 1
    assert migrated[0].account_id == "primary"
    assert migrated[0].credentials_key == "EMAIL_PRIMARY"
    assert migrated[0].backend == "imap"


def test_migration_shim_not_applied_when_new_format_present():
    """Legacy migration shim is skipped when new users block is present."""
    from rex.config import build_app_config

    cfg = build_app_config(
        {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
            "users": {
                "alice": {"email_accounts": [{"account_id": "work", "credentials_key": "K"}]}
            },
            "email": {
                "accounts": [
                    {
                        "id": "legacy",
                        "address": "old@example.com",
                        "imap_host": "imap.example.com",
                    }
                ]
            },
        }
    )
    # New format took priority; "default" user from legacy should NOT appear
    assert "default" not in cfg.user_email_accounts
    assert "alice" in cfg.user_email_accounts


# ---------------------------------------------------------------------------
# rex_config.example.json shows two accounts for one user
# ---------------------------------------------------------------------------


def test_example_config_has_users_with_two_email_accounts():
    """rex_config.example.json contains users.alice with 2 email accounts."""
    import json
    from pathlib import Path

    example = Path("config/rex_config.example.json")
    if not example.exists():
        example = Path(__file__).parent.parent / "config" / "rex_config.example.json"

    data = json.loads(example.read_text(encoding="utf-8"))
    users = data.get("users", {})
    assert users, "users block missing from rex_config.example.json"

    # Find any user with at least 2 email accounts
    found = False
    for user_data in users.values():
        accounts = user_data.get("email_accounts", [])
        if len(accounts) >= 2:
            found = True
            break
    assert found, "No user in example config has 2+ email accounts"
