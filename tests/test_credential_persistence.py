"""Transactional household credential persistence tests."""

from __future__ import annotations

import json

import pytest

from rex.credential_persistence import persist_household_secrets
from rex.credential_vault import InMemoryCredentialVault


def _patch_vault(monkeypatch, vault):
    monkeypatch.setattr("rex.credential_persistence.get_credential_vault", lambda **_kwargs: vault)


def test_blank_values_update_nonsecret_config_without_constructing_vault(tmp_path, monkeypatch):
    config_path = tmp_path / "rex_config.json"

    def unexpected(**_kwargs):
        raise AssertionError("blank credentials must not construct a vault")

    monkeypatch.setattr("rex.credential_persistence.get_credential_vault", unexpected)
    result = persist_household_secrets(
        {"HA_TOKEN": "   "},
        config_path=config_path,
        update_config=lambda config: config.update({"home_assistant": {"base_url": "http://ha"}}),
    )
    assert result == {}
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "home_assistant": {"base_url": "http://ha"}
    }


def test_secret_is_verified_and_only_contextual_reference_is_persisted(tmp_path, monkeypatch):
    config_path = tmp_path / "rex_config.json"
    vault = InMemoryCredentialVault()
    _patch_vault(monkeypatch, vault)
    marker = "secret-marker"

    refs = persist_household_secrets({"OPENAI_API_KEY": marker}, config_path=config_path)

    ref = refs["OPENAI_API_KEY"]
    raw = config_path.read_text(encoding="utf-8")
    assert marker not in raw
    assert json.loads(raw)["credential_refs"]["household"]["OPENAI_API_KEY"] == {
        "ref": ref,
        "integration": "openai",
        "account": None,
        "slot": "api_key",
    }
    assert vault.get_secret(ref, integration="openai", account=None, slot="api_key") == marker


def test_config_failure_removes_staged_secret_and_preserves_original(tmp_path, monkeypatch):
    config_path = tmp_path / "rex_config.json"
    original = '{"existing": true}\n'
    config_path.write_text(original, encoding="utf-8")
    vault = InMemoryCredentialVault()
    _patch_vault(monkeypatch, vault)

    def fail_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("rex.config_manager.save_config", fail_save)
    with pytest.raises(OSError, match="disk full"):
        persist_household_secrets({"OPENAI_API_KEY": "secret-marker"}, config_path=config_path)
    assert config_path.read_text(encoding="utf-8") == original
    assert vault.list_entries() == []


def test_vault_readback_failure_removes_the_staged_secret(tmp_path, monkeypatch):
    config_path = tmp_path / "rex_config.json"

    class LyingVault(InMemoryCredentialVault):
        def get_secret(self, *args, **kwargs):
            return "different"

    vault = LyingVault()
    _patch_vault(monkeypatch, vault)
    with pytest.raises(RuntimeError, match="readback"):
        persist_household_secrets({"OPENAI_API_KEY": "secret-marker"}, config_path=config_path)
    assert not config_path.exists()
    assert vault.list_entries() == []


def test_reference_readback_failure_restores_registry_and_removes_staged_secret(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "rex_config.json"
    original = '{"existing": true}\n'
    config_path.write_text(original, encoding="utf-8")
    vault = InMemoryCredentialVault()
    _patch_vault(monkeypatch, vault)
    real_strict = __import__(
        "rex.credential_persistence", fromlist=["_strict_config"]
    )._strict_config
    reads = 0

    def corrupt_readback(path):
        nonlocal reads
        reads += 1
        if reads == 2:
            return {}
        return real_strict(path)

    monkeypatch.setattr("rex.credential_persistence._strict_config", corrupt_readback)
    with pytest.raises(RuntimeError, match="readback"):
        persist_household_secrets({"OPENAI_API_KEY": "secret-marker"}, config_path=config_path)
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"existing": True}
    assert vault.list_entries() == []


def test_second_secret_failure_removes_first_staged_entry_and_writes_no_config(
    tmp_path, monkeypatch
):
    config_path = tmp_path / "rex_config.json"

    class SecondWriteFailsVault(InMemoryCredentialVault):
        writes = 0

        def set_secret(self, *args, **kwargs):
            self.writes += 1
            if self.writes == 2:
                raise RuntimeError("vault write failed")
            return super().set_secret(*args, **kwargs)

    vault = SecondWriteFailsVault()
    _patch_vault(monkeypatch, vault)

    with pytest.raises(RuntimeError, match="vault write failed"):
        persist_household_secrets(
            {
                "OPENAI_API_KEY": "first-marker",
                "HA_TOKEN": "second-marker",
            },
            config_path=config_path,
        )

    assert not config_path.exists()
    assert vault.list_entries() == []
