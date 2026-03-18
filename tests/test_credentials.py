"""Tests for the credential manager module."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from rex.credentials import (
    Credential,
    CredentialManager,
    CredentialRefreshError,
    get_credential_manager,
    mask_token,
    set_credential_manager,
)


class TestMaskToken:
    """Tests for the mask_token function."""

    def test_mask_token_normal(self):
        """Test masking a normal token."""
        assert mask_token("abcd1234efgh5678") == "abcd...5678"

    def test_mask_token_short(self):
        """Test masking a short token (less than 8 chars)."""
        assert mask_token("short") == "*****"

    def test_mask_token_empty(self):
        """Test masking an empty token."""
        assert mask_token("") == "[empty]"

    def test_mask_token_none(self):
        """Test masking None."""
        assert mask_token(None) == "[empty]"

    def test_mask_token_custom_visible(self):
        """Test masking with custom visible characters."""
        assert mask_token("abcdefghijklmnop", visible_chars=2) == "ab...op"


class TestCredential:
    """Tests for the Credential dataclass."""

    def test_credential_creation(self):
        """Test creating a credential."""
        cred = Credential(
            name="test",
            token="secret123",
            expires_at=None,
            scopes=["read", "write"],
            source="env",
        )
        assert cred.name == "test"
        assert cred.token == "secret123"
        assert cred.expires_at is None
        assert cred.scopes == ["read", "write"]
        assert cred.source == "env"

    def test_credential_not_expired_when_no_expiry(self):
        """Test that credential without expiry is not expired."""
        cred = Credential(name="test", token="secret")
        assert not cred.is_expired()

    def test_credential_not_expired_when_future(self):
        """Test that credential with future expiry is not expired."""
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        cred = Credential(name="test", token="secret", expires_at=future)
        assert not cred.is_expired()

    def test_credential_expired_when_past(self):
        """Test that credential with past expiry is expired."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        cred = Credential(name="test", token="secret", expires_at=past)
        assert cred.is_expired()

    def test_credential_repr_masks_token(self):
        """Test that repr masks the token."""
        cred = Credential(name="test", token="supersecrettoken123")
        repr_str = repr(cred)
        assert "supersecrettoken123" not in repr_str
        # Token is masked with first 4 and last 4 chars: "supe...n123"
        assert "supe..." in repr_str
        assert "n123" in repr_str


class TestCredentialManager:
    """Tests for the CredentialManager class."""

    def test_load_from_env_with_prefix(self):
        """Test loading credentials from environment with REX_ prefix."""
        with patch.dict(os.environ, {"REX_EMAIL_TOKEN": "env_token_123"}):
            manager = CredentialManager()
            token = manager.get_token("email")
            assert token == "env_token_123"

    def test_load_from_env_without_prefix(self):
        """Test loading credentials from environment without prefix."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai_key_123"}, clear=False):
            manager = CredentialManager()
            token = manager.get_token("openai")
            assert token == "openai_key_123"

    def test_load_from_config_file(self):
        """Test loading credentials from JSON config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "credentials.json"
            config_data = {"credentials": {"test_service": "config_token_456"}}
            config_path.write_text(json.dumps(config_data))

            manager = CredentialManager(config_path=config_path)
            token = manager.get_token("test_service")
            assert token == "config_token_456"

    def test_load_from_config_file_with_metadata(self):
        """Test loading credentials with full metadata from config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "credentials.json"
            future = datetime.now(timezone.utc) + timedelta(days=1)
            config_data = {
                "credentials": {
                    "full_service": {
                        "token": "full_token_789",
                        "expires_at": future.isoformat(),
                        "scopes": ["read", "write"],
                    }
                }
            }
            config_path.write_text(json.dumps(config_data))

            manager = CredentialManager(config_path=config_path)
            cred = manager.get_credential("full_service")
            assert cred is not None
            assert cred.token == "full_token_789"
            assert cred.scopes == ["read", "write"]
            assert not cred.is_expired()

    def test_config_overrides_env(self):
        """Test that config file overrides environment variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "credentials.json"
            config_data = {"credentials": {"email": "config_email_token"}}
            config_path.write_text(json.dumps(config_data))

            with patch.dict(os.environ, {"REX_EMAIL_TOKEN": "env_email_token"}):
                manager = CredentialManager(config_path=config_path)
                token = manager.get_token("email")
                assert token == "config_email_token"

    def test_get_token_returns_none_when_not_found(self):
        """Test that get_token returns None for unknown services."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        assert manager.get_token("unknown_service") is None

    def test_set_token_at_runtime(self):
        """Test setting a token at runtime."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        manager.set_token("new_service", "runtime_token")
        assert manager.get_token("new_service") == "runtime_token"

    def test_set_token_with_expiry(self):
        """Test setting a token with expiry."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        future = datetime.now(timezone.utc) + timedelta(hours=1)
        manager.set_token("expiring_service", "expiring_token", expires_at=future)

        cred = manager.get_credential("expiring_service")
        assert cred is not None
        assert cred.expires_at == future
        assert not cred.is_expired()

    def test_has_token_returns_true_when_valid(self):
        """Test has_token returns True for valid token."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        manager.set_token("valid_service", "valid_token")
        assert manager.has_token("valid_service")

    def test_has_token_returns_false_when_missing(self):
        """Test has_token returns False for missing token."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        assert not manager.has_token("missing_service")

    def test_has_token_returns_false_when_expired(self):
        """Test has_token returns False for expired token."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        manager.set_token("expired_service", "expired_token", expires_at=past)
        assert not manager.has_token("expired_service")

    def test_list_services(self):
        """Test listing all services with credentials."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        manager.set_token("service_a", "token_a")
        manager.set_token("service_b", "token_b")

        services = manager.list_services()
        assert "service_a" in services
        assert "service_b" in services

    def test_reload_preserves_runtime_credentials(self):
        """Test that reload preserves runtime credentials."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        manager.set_token("runtime_service", "runtime_token")

        manager.reload()

        assert manager.get_token("runtime_service") == "runtime_token"

    def test_refresh_token_without_handler_raises_error(self):
        """Test that refresh_token raises error without handler."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        manager.set_token("no_refresh_service", "original_token")

        with pytest.raises(CredentialRefreshError) as exc_info:
            manager.refresh_token("no_refresh_service")

        assert "no_refresh_service" in str(exc_info.value)
        assert "not implemented" in str(exc_info.value).lower()

    def test_refresh_token_with_handler(self):
        """Test that refresh_token works with registered handler."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        manager.set_token("refreshable", "old_token")

        def refresh_handler(current: str) -> str:
            return "new_token_from_refresh"

        manager.register_refresh_handler("refreshable", refresh_handler)
        new_token = manager.refresh_token("refreshable")

        assert new_token == "new_token_from_refresh"
        assert manager.get_token("refreshable") == "new_token_from_refresh"

    def test_add_credential_mapping(self):
        """Test adding custom credential mapping."""
        with patch.dict(os.environ, {"REX_CUSTOM_VAR": "custom_value"}):
            manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
            manager.add_credential_mapping("custom_service", "CUSTOM_VAR")

            # Force reload to pick up new mapping
            manager.reload()

            token = manager.get_token("custom_service")
            assert token == "custom_value"

    def test_get_credential_info_masks_token(self):
        """Test that get_credential_info masks the token."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        manager.set_token("info_service", "supersecrettoken123")

        info = manager.get_credential_info("info_service")
        assert info is not None
        assert "supersecrettoken123" not in info["token_preview"]
        assert "..." in info["token_preview"]

    def test_get_credential_info_returns_none_for_unknown(self):
        """Test that get_credential_info returns None for unknown service."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        assert manager.get_credential_info("unknown") is None

    def test_expired_token_triggers_auto_refresh(self):
        """Test that expired token triggers auto-refresh when handler exists."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        manager.set_token("auto_refresh", "expired_token", expires_at=past)

        def refresh_handler(current: str) -> str:
            return "refreshed_token"

        manager.register_refresh_handler("auto_refresh", refresh_handler)

        # get_token with auto_refresh=True should refresh
        token = manager.get_token("auto_refresh", auto_refresh=True)
        assert token == "refreshed_token"

    def test_expired_token_without_handler_returns_none(self):
        """Test that expired token without handler returns None."""
        manager = CredentialManager(config_path=Path("/nonexistent/path.json"))
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        manager.set_token("no_handler", "expired_token", expires_at=past)

        # get_token with auto_refresh=True but no handler returns None
        token = manager.get_token("no_handler", auto_refresh=True)
        assert token is None


class TestGlobalCredentialManager:
    """Tests for global credential manager functions."""

    def test_get_credential_manager_returns_singleton(self):
        """Test that get_credential_manager returns singleton."""
        # Reset global state
        set_credential_manager(None)  # type: ignore

        manager1 = get_credential_manager()
        manager2 = get_credential_manager()

        assert manager1 is manager2

    def test_set_credential_manager_replaces_singleton(self):
        """Test that set_credential_manager replaces the singleton."""
        custom_manager = CredentialManager(config_path=Path("/custom/path.json"))
        set_credential_manager(custom_manager)

        assert get_credential_manager() is custom_manager

        # Reset for other tests
        set_credential_manager(None)  # type: ignore


class TestCredentialManagerEdgeCases:
    """Edge case tests for CredentialManager."""

    def test_invalid_config_json(self):
        """Test handling of invalid JSON in config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "credentials.json"
            config_path.write_text("invalid json {{{")

            manager = CredentialManager(config_path=config_path)
            # Should not raise, just skip loading
            assert manager.list_services() == []

    def test_config_with_non_dict_root(self):
        """Test handling of non-dict root in config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "credentials.json"
            config_path.write_text('["not", "a", "dict"]')

            manager = CredentialManager(config_path=config_path)
            assert manager.list_services() == []

    def test_custom_env_prefix(self):
        """Test custom environment variable prefix."""
        with patch.dict(os.environ, {"CUSTOM_EMAIL_TOKEN": "custom_prefix_token"}):
            manager = CredentialManager(
                env_prefix="CUSTOM_",
                config_path=Path("/nonexistent/path.json"),
            )
            token = manager.get_token("email")
            assert token == "custom_prefix_token"

    def test_lazy_loading(self):
        """Test that credentials are loaded lazily."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "credentials.json"
            # Don't create the file yet

            manager = CredentialManager(config_path=config_path)
            assert not manager._loaded

            # Create the file after manager creation
            config_data = {"credentials": {"lazy_service": "lazy_token"}}
            config_path.write_text(json.dumps(config_data))

            # Now access should trigger loading
            token = manager.get_token("lazy_service")
            assert token == "lazy_token"
            assert manager._loaded
