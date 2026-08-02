"""Adversarial tests for the one-time S4 plaintext credential migration."""

from __future__ import annotations

import json

import pytest

from rex.credential_vault import InMemoryCredentialVault, VaultUnavailableError
from scripts import migrate_credentials_to_vault as migration
from scripts.migrate_credentials_to_vault import MigrationError, discover_candidates, main, migrate


@pytest.fixture
def paths(tmp_path):
    return tmp_path / ".env", tmp_path / "credentials.json", tmp_path / "rex_config.json"


def _patch_vault(monkeypatch, vault):
    monkeypatch.setattr("rex.credential_vault.get_credential_vault", lambda **_kwargs: vault)


def _run(paths, **overrides):
    env_path, credentials_path, config_path = paths
    arguments = {
        "env_path": env_path,
        "credentials_json_path": credentials_path,
        "config_path": config_path,
        "scope": "household",
        "owner": "household",
        "apply": True,
    }
    arguments.update(overrides)
    return migrate(**arguments)


def _stored_record(config_path, logical_name="OPENAI_API_KEY"):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config["credential_refs"]["household"][logical_name]


class TestDiscoveryAndDryRun:
    def test_dotenv_semantics_are_used_without_exposing_values(self, paths):
        env_path, credentials_path, _config_path = paths
        marker = "secret with spaces"
        env_path.write_text(f'OPENAI_API_KEY="{marker}"\nOTHER=1\n', encoding="utf-8")
        candidates = discover_candidates(env_path=env_path, credentials_json_path=credentials_path)
        assert [(candidate.logical_name, candidate.value) for candidate in candidates] == [
            ("OPENAI_API_KEY", marker)
        ]
        assert marker not in repr([candidate.logical_name for candidate in candidates])

    def test_duplicate_secret_keys_fail_closed(self, paths):
        env_path, credentials_path, _config_path = paths
        env_path.write_text("OPENAI_API_KEY=first\nOPENAI_API_KEY=second\n", encoding="utf-8")
        with pytest.raises(MigrationError, match="duplicate"):
            discover_candidates(env_path=env_path, credentials_json_path=credentials_path)

    def test_dry_run_performs_no_write_or_vault_construction(self, paths, monkeypatch):
        env_path, _credentials_path, config_path = paths
        original = "OPENAI_API_KEY=secret-marker\n"
        env_path.write_text(original, encoding="utf-8")

        def unexpected(**_kwargs):
            raise AssertionError("dry-run constructed the vault")

        monkeypatch.setattr("rex.credential_vault.get_credential_vault", unexpected)
        results = _run(paths, apply=False)
        assert results[0].status == "planned"
        assert env_path.read_text(encoding="utf-8") == original
        assert not config_path.exists()
        assert not list(env_path.parent.glob("*.bak"))

    @pytest.mark.parametrize(
        ("scope", "owner"),
        [("household", "alice"), ("user", "household"), ("user", "../alice")],
    )
    def test_scope_and_owner_must_be_explicitly_valid(self, paths, scope, owner):
        with pytest.raises(MigrationError):
            _run(paths, scope=scope, owner=owner, apply=False)


class TestApply:
    def test_write_readback_registry_readback_then_atomic_sanitize(self, paths, monkeypatch):
        env_path, _credentials_path, config_path = paths
        marker = "unique-secret-marker"
        env_path.write_text(f"OPENAI_API_KEY={marker}\nUNRELATED=keep\n", encoding="utf-8")
        vault = InMemoryCredentialVault()
        _patch_vault(monkeypatch, vault)

        results = _run(paths)
        assert results[0].status == "migrated"
        record = _stored_record(config_path)
        assert record == {
            "ref": record["ref"],
            "integration": "openai",
            "account": None,
            "slot": "api_key",
            "migrated_from": "env",
        }
        assert record["ref"].startswith("cred_")
        assert (
            vault.get_secret(record["ref"], integration="openai", account=None, slot="api_key")
            == marker
        )
        assert env_path.read_text(encoding="utf-8") == "UNRELATED=keep\n"
        assert marker not in config_path.read_text(encoding="utf-8")
        assert not list(env_path.parent.glob("*.bak"))
        assert not list(env_path.parent.glob("*.tmp-*"))

    def test_identical_destination_is_idempotent(self, paths, monkeypatch):
        env_path, _credentials_path, _config_path = paths
        source = "OPENAI_API_KEY=secret-marker\n"
        env_path.write_text(source, encoding="utf-8")
        vault = InMemoryCredentialVault()
        _patch_vault(monkeypatch, vault)
        _run(paths)

        env_path.write_text(source, encoding="utf-8")
        results = _run(paths)
        assert results[0].status == "already_migrated"
        assert env_path.read_text(encoding="utf-8") == ""
        assert len(vault.list_entries()) == 1

    def test_different_destination_conflicts_and_leaves_source_untouched(self, paths, monkeypatch):
        env_path, _credentials_path, config_path = paths
        source = "OPENAI_API_KEY=new-secret\n"
        env_path.write_text(source, encoding="utf-8")
        vault = InMemoryCredentialVault()
        ref = "cred_" + "C" * 32
        vault.set_secret(ref, "old-secret", integration="openai", account=None, slot="api_key")
        config = {
            "credential_refs": {
                "household": {
                    "OPENAI_API_KEY": {
                        "ref": ref,
                        "integration": "openai",
                        "account": None,
                        "slot": "api_key",
                    }
                }
            }
        }
        config_path.write_text(json.dumps(config), encoding="utf-8")
        original_config = config_path.read_text(encoding="utf-8")
        _patch_vault(monkeypatch, vault)

        results = _run(paths)
        assert results[0].status == "conflict"
        assert env_path.read_text(encoding="utf-8") == source
        assert config_path.read_text(encoding="utf-8") == original_config
        assert (
            vault.get_secret(ref, integration="openai", account=None, slot="api_key")
            == "old-secret"
        )

    def test_vault_verification_failure_rolls_back_without_plaintext_artifact(
        self, paths, monkeypatch
    ):
        env_path, _credentials_path, config_path = paths
        source = "OPENAI_API_KEY=secret-marker\n"
        env_path.write_text(source, encoding="utf-8")

        class LyingVault(InMemoryCredentialVault):
            def get_secret(self, *args, **kwargs):
                return "different"

        vault = LyingVault()
        _patch_vault(monkeypatch, vault)
        with pytest.raises(MigrationError, match="verification"):
            _run(paths)
        assert env_path.read_text(encoding="utf-8") == source
        assert not config_path.exists()
        assert vault.list_entries() == []
        assert not list(env_path.parent.glob("*.bak"))

    def test_sanitize_failure_restores_registry_and_removes_staged_entry(self, paths, monkeypatch):
        env_path, _credentials_path, config_path = paths
        source = "OPENAI_API_KEY=secret-marker\n"
        env_path.write_text(source, encoding="utf-8")
        config_path.write_text('{"existing": true}\n', encoding="utf-8")
        vault = InMemoryCredentialVault()
        _patch_vault(monkeypatch, vault)
        monkeypatch.setattr(
            migration, "_sanitize_env", lambda *_args: (_ for _ in ()).throw(OSError("denied"))
        )

        with pytest.raises(MigrationError):
            _run(paths)
        assert env_path.read_text(encoding="utf-8") == source
        assert config_path.read_text(encoding="utf-8") == '{"existing": true}\n'
        assert vault.list_entries() == []

    def test_failed_rollback_delete_is_encrypted_and_journaled_for_restart(
        self, paths, monkeypatch
    ):
        env_path, _credentials_path, config_path = paths
        marker = "rollback-secret-marker"
        env_path.write_text(f"OPENAI_API_KEY={marker}\n", encoding="utf-8")

        class DeferredDeleteVault(InMemoryCredentialVault):
            allow_delete = False

            def delete_secret(self, *args, **kwargs):
                if not self.allow_delete:
                    raise OSError("temporarily unavailable")
                return super().delete_secret(*args, **kwargs)

        vault = DeferredDeleteVault()
        _patch_vault(monkeypatch, vault)
        real_sanitize = migration._sanitize_env
        monkeypatch.setattr(
            migration,
            "_sanitize_env",
            lambda *_args: (_ for _ in ()).throw(OSError("denied")),
        )

        with pytest.raises(MigrationError):
            _run(paths)
        journal_path = config_path.with_name("credential_migration_recovery.json")
        journal_text = journal_path.read_text(encoding="utf-8")
        assert marker not in journal_text
        assert len(vault.list_entries()) == 1
        assert env_path.read_text(encoding="utf-8") == f"OPENAI_API_KEY={marker}\n"

        vault.allow_delete = True
        monkeypatch.setattr(migration, "_sanitize_env", real_sanitize)
        results = _run(paths)
        assert results[0].status == "migrated"
        assert not journal_path.exists()
        assert len(vault.list_entries()) == 1
        assert env_path.read_text(encoding="utf-8") == ""

    def test_credentials_json_is_sanitized_without_touching_noncredential_data(
        self, paths, monkeypatch
    ):
        _env_path, credentials_path, config_path = paths
        credentials_path.write_text(
            json.dumps({"credentials": {"openai": "json-secret"}, "keep": "metadata"}),
            encoding="utf-8",
        )
        vault = InMemoryCredentialVault()
        _patch_vault(monkeypatch, vault)
        _run(paths)
        remaining = json.loads(credentials_path.read_text(encoding="utf-8"))
        assert remaining == {"credentials": {}, "keep": "metadata"}
        assert "json-secret" not in config_path.read_text(encoding="utf-8")


class TestCli:
    def test_cli_output_contains_no_secret_or_exception_detail(self, paths, monkeypatch, capsys):
        env_path, credentials_path, config_path = paths
        marker = "cli-secret-marker"
        env_path.write_text(f"OPENAI_API_KEY={marker}\n", encoding="utf-8")

        def unavailable(**_kwargs):
            raise VaultUnavailableError(f"failure mentions {marker}")

        monkeypatch.setattr("rex.credential_vault.get_credential_vault", unavailable)
        code = main(
            [
                "--env-path",
                str(env_path),
                "--credentials-json-path",
                str(credentials_path),
                "--config-path",
                str(config_path),
                "--scope",
                "household",
                "--owner",
                "household",
                "--apply",
                "--json",
            ]
        )
        output = capsys.readouterr().out
        assert code == 1
        assert marker not in output
        assert "VaultUnavailableError" not in output
        assert json.loads(output) == [
            {"logical_name": "migration", "source": "operation", "status": "failed", "detail": None}
        ]
