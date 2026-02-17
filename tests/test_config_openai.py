import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from rex.config import load_config


def test_openai_provider_allows_none_llm_model():
    json_config = {
        "models": {
            "llm_provider": "openai",
            "llm_model": None,
        },
        "openai": {
            "model": "hermes-3-llama-3.1-8b",
            "base_url": "http://127.0.0.1:1234/v1",
        },
    }

    config = load_config(json_config=json_config, reload=True)

    assert config.llm_provider == "openai"
    assert config.llm_model is None
    assert config.openai_model == "hermes-3-llama-3.1-8b"


# ---------------------------------------------------------------------------
# Migration command tests
# ---------------------------------------------------------------------------


class TestMigrateLegacyEnv:
    """Tests for rex-config migrate-legacy-env command."""

    def test_migration_writes_config_file(self, tmp_path: Path):
        """Migration creates a config file with the migrated value."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://localhost:1234/v1\n")

        config_file = tmp_path / "rex_config.json"

        from rex.config_manager import migrate_legacy_env_to_config

        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://localhost:1234/v1"}):
            notes = migrate_legacy_env_to_config(
                env_path=str(env_file),
                config_path=str(config_file),
            )

        assert config_file.exists(), "Config file should be created by migration"
        config_data = json.loads(config_file.read_text())
        assert config_data["openai"]["base_url"] == "http://localhost:1234/v1"
        assert any("Migrated" in n for n in notes)

    def test_migration_does_not_overwrite_existing_value(self, tmp_path: Path):
        """Migration skips config keys that already have non-default values."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://new-url/v1\n")

        config_file = tmp_path / "rex_config.json"
        # Write config with an existing custom value
        existing = {"openai": {"model": None, "base_url": "http://existing/v1"}}
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(json.dumps(existing))

        from rex.config_manager import migrate_legacy_env_to_config

        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://new-url/v1"}):
            notes = migrate_legacy_env_to_config(
                env_path=str(env_file),
                config_path=str(config_file),
            )

        config_data = json.loads(config_file.read_text())
        # Existing value must be preserved
        assert config_data["openai"]["base_url"] == "http://existing/v1"

    def test_migration_sets_openai_base_url_key(self, tmp_path: Path):
        """Migration writes the openai.base_url key when OPENAI_BASE_URL is present."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://custom-endpoint/v1\n")

        config_file = tmp_path / "rex_config.json"

        from rex.config_manager import migrate_legacy_env_to_config

        with patch.dict(os.environ, {"OPENAI_BASE_URL": "http://custom-endpoint/v1"}):
            migrate_legacy_env_to_config(
                env_path=str(env_file),
                config_path=str(config_file),
            )

        config_data = json.loads(config_file.read_text())
        assert "openai" in config_data
        assert "base_url" in config_data["openai"]
        assert config_data["openai"]["base_url"] == "http://custom-endpoint/v1"

    def test_migration_subprocess_isolation(self, tmp_path: Path):
        """Running the migration via subprocess produces output and creates the config file."""
        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://test-api/v1\n")

        config_file = tmp_path / "rex_config.json"

        repo_root = Path(__file__).resolve().parent.parent
        env = os.environ.copy()
        env["OPENAI_BASE_URL"] = "http://test-api/v1"
        env["PYTHONPATH"] = str(repo_root)

        result = subprocess.run(
            [
                sys.executable, "-c",
                "from rex.config_manager import migrate_legacy_env_to_config; "
                f"notes = migrate_legacy_env_to_config(env_path={str(env_file)!r}, config_path={str(config_file)!r}); "
                "print('\\n'.join(notes))",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            cwd=str(tmp_path),
        )

        assert result.returncode == 0, f"Migration failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        assert config_file.exists()
        config_data = json.loads(config_file.read_text())
        assert config_data["openai"]["base_url"] == "http://test-api/v1"
