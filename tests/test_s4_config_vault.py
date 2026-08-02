"""Fail-closed AppConfig credential authority tests for S4."""

from __future__ import annotations

import os

import pytest

from rex.config import ConfigurationError, build_app_config, load_config
from rex.credential_vault import InMemoryCredentialVault, VaultUnavailableError


def _base_config() -> dict:
    return {
        "models": {
            "llm_provider": "transformers",
            "llm_model": "sshleifer/tiny-gpt2",
        }
    }


def _with_reference(logical_name: str, ref: str, **metadata) -> dict:
    config = _base_config()
    config["credential_refs"] = {
        "household": {
            logical_name: {
                "ref": ref,
                **metadata,
            }
        }
    }
    return config


def test_app_config_resolves_an_exact_contextual_vault_reference(monkeypatch):
    monkeypatch.delenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    vault = InMemoryCredentialVault()
    ref = "cred_" + "O" * 32
    vault.set_secret(ref, "vault-value", integration="openai", account=None, slot="api_key")
    monkeypatch.setattr("rex.credential_vault.get_credential_vault", lambda **_kwargs: vault)

    config = _with_reference(
        "OPENAI_API_KEY",
        ref,
        integration="openai",
        account=None,
        slot="api_key",
    )
    assert build_app_config(config).openai_api_key == "vault-value"


def test_configured_vault_reference_wins_over_legacy_environment_value(monkeypatch):
    """A configured vault reference is authoritative; legacy mode is fallback only, never an override."""
    monkeypatch.setenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "operator-value")
    vault = InMemoryCredentialVault()
    ref = "cred_" + "O" * 32
    vault.set_secret(ref, "vault-value", integration="openai", account=None, slot="api_key")
    monkeypatch.setattr("rex.credential_vault.get_credential_vault", lambda **_kwargs: vault)
    config = _with_reference(
        "OPENAI_API_KEY",
        ref,
        integration="openai",
        account=None,
        slot="api_key",
    )

    assert build_app_config(config).openai_api_key == "vault-value"


def test_legacy_environment_value_is_used_only_when_no_vault_reference_is_configured(monkeypatch):
    monkeypatch.setenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "operator-value")
    monkeypatch.setattr(
        "rex.credential_vault.get_credential_vault",
        lambda **_kwargs: InMemoryCredentialVault(),
    )

    assert build_app_config(_base_config()).openai_api_key == "operator-value"


def test_environment_is_ignored_without_explicit_legacy_mode(monkeypatch):
    monkeypatch.delenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "plaintext-value")
    monkeypatch.setattr(
        "rex.credential_vault.get_credential_vault",
        lambda **_kwargs: InMemoryCredentialVault(),
    )

    assert build_app_config(_base_config()).openai_api_key is None


def test_vault_unavailable_does_not_authorize_environment_fallback(monkeypatch):
    monkeypatch.delenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", raising=False)
    monkeypatch.setenv("HA_TOKEN", "plaintext-value")

    def unavailable(**_kwargs):
        raise VaultUnavailableError("vault unavailable")

    monkeypatch.setattr("rex.credential_vault.get_credential_vault", unavailable)
    assert build_app_config(_base_config()).ha_token is None


def test_swapped_or_unexpected_reference_metadata_fails_closed(monkeypatch):
    monkeypatch.delenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", raising=False)
    ref = "cred_" + "X" * 32
    swapped = _with_reference(
        "OPENAI_API_KEY",
        ref,
        integration="email",
        account="other",
        slot="password",
    )
    with pytest.raises(ConfigurationError):
        build_app_config(swapped)

    unexpected = _with_reference(
        "OPENAI_API_KEY",
        ref,
        integration="openai",
        account=None,
        slot="api_key",
        attacker_controlled=True,
    )
    with pytest.raises(ConfigurationError):
        build_app_config(unexpected)


@pytest.mark.parametrize("registry", [[], {"household": []}])
def test_malformed_reference_registry_fails_closed(monkeypatch, registry):
    monkeypatch.delenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", raising=False)
    with pytest.raises(ConfigurationError):
        build_app_config({"credential_refs": registry})


def test_nonopaque_reference_is_rejected_before_vault_lookup(monkeypatch):
    monkeypatch.delenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", raising=False)
    looked_up = False

    def unexpected_lookup(**_kwargs):
        nonlocal looked_up
        looked_up = True
        raise AssertionError("invalid references must fail before vault lookup")

    monkeypatch.setattr("rex.credential_vault.get_credential_vault", unexpected_lookup)
    config = _with_reference(
        "OPENAI_API_KEY",
        "OPENAI_API_KEY",
        integration="openai",
        account=None,
        slot="api_key",
    )
    with pytest.raises(ConfigurationError):
        build_app_config(config)
    assert looked_up is False


def test_persisted_dotenv_is_not_loaded_without_explicit_legacy_mode(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=persisted-value\n", encoding="utf-8")
    monkeypatch.delenv("REX_ALLOW_PLAINTEXT_CREDENTIAL_FALLBACK", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    load_config(env_path=env_path, reload=True, json_config=_base_config())
    assert "OPENAI_API_KEY" not in os.environ
