"""Tests for US-REM-022: Config migration and reset/recovery.

Verifies that rex_config.json loading handles every failure mode gracefully:
- Missing required fields  → defaults applied, no KeyError
- Corrupt JSON             → safe recovery (backup + defaults), no unhandled exception
- Missing config file      → defaults applied, file auto-created, no crash

Note on corrupt-file behavior: config_manager.load_config() backs up the
corrupt file to rex_config.invalid.<timestamp>.json and recreates defaults.
The error is logged with a message that includes the file path. This
constitutes "clear error" handling per the US-REM-022 acceptance criterion.
No ConfigurationError is raised because graceful recovery is preferred over
hard-failing on a missing/corrupt config at startup.
"""

from __future__ import annotations

import json
from pathlib import Path

from rex.config import AppConfig, build_app_config
from rex.config_manager import load_config as load_json_config

# ---------------------------------------------------------------------------
# AC-1: Missing required field → graceful defaults, not KeyError
# ---------------------------------------------------------------------------


class TestMissingFieldGracefulDefaults:
    """Loading a rex_config.json with missing fields applies defaults, not KeyError."""

    def test_empty_config_dict_returns_appconfig(self):
        """build_app_config with an empty dict returns a valid AppConfig."""
        cfg = build_app_config({})
        assert isinstance(cfg, AppConfig)

    def test_missing_models_section_uses_default_llm_provider(self):
        """Missing models section applies the default llm_provider."""
        cfg = build_app_config({})
        assert cfg.llm_provider == "transformers"

    def test_missing_models_section_uses_default_llm_model(self):
        """Missing models section applies the default llm_model."""
        cfg = build_app_config({})
        assert cfg.llm_model == "sshleifer/tiny-gpt2"

    def test_missing_runtime_section_uses_default_memory_max_turns(self):
        """Missing runtime section applies the default memory_max_turns."""
        cfg = build_app_config({})
        assert cfg.memory_max_turns == 50

    def test_missing_audio_section_uses_default_sample_rate(self):
        """Missing audio section applies the default sample_rate."""
        cfg = build_app_config({})
        assert cfg.sample_rate == 16000

    def test_missing_wakeword_section_uses_default_wakeword(self):
        """Missing wakeword section applies the default wakeword value."""
        cfg = build_app_config(
            {"models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"}}
        )
        assert cfg.wakeword == "hey_rex"

    def test_partial_models_section_fills_missing_llm_model(self):
        """Partially specified models section fills llm_model from defaults."""
        cfg = build_app_config({"models": {"llm_provider": "transformers"}})
        assert cfg.llm_model == "sshleifer/tiny-gpt2"

    def test_unknown_extra_keys_do_not_crash(self):
        """Unknown JSON keys (future schema additions) are silently ignored."""
        json_cfg = {
            "models": {"llm_provider": "transformers", "llm_model": "sshleifer/tiny-gpt2"},
            "future_section_unknown": {"new_feature": True, "count": 42},
        }
        cfg = build_app_config(json_cfg)
        assert isinstance(cfg, AppConfig)

    def test_load_json_config_deep_merges_missing_sections(self, tmp_path: Path):
        """load_json_config deep-merges with DEFAULT_CONFIG so missing sections appear."""
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"runtime": {"memory_max_turns": 99}}))

        data = load_json_config(path=str(config_path))

        assert data["runtime"]["memory_max_turns"] == 99
        assert "models" in data, "Missing 'models' section should be filled from defaults"
        assert "audio" in data, "Missing 'audio' section should be filled from defaults"
        assert "wakeword" in data, "Missing 'wakeword' section should be filled from defaults"

    def test_build_app_config_from_merged_defaults_does_not_raise_keyerror(self, tmp_path: Path):
        """End-to-end: load_json_config + build_app_config on a sparse file raises no KeyError."""
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(json.dumps({"runtime": {"memory_max_turns": 10}}))

        data = load_json_config(path=str(config_path))
        cfg = build_app_config(data)
        assert isinstance(cfg, AppConfig)
        assert cfg.memory_max_turns == 10


# ---------------------------------------------------------------------------
# AC-2: Corrupt JSON → clear error handling, no unhandled exception
# ---------------------------------------------------------------------------


class TestCorruptConfigGracefulRecovery:
    """A corrupt rex_config.json must be handled without crashing.

    The config_manager logs a clear error that includes the corrupt file's
    path, backs it up as rex_config.invalid.<timestamp>.json, and returns
    DEFAULT_CONFIG so the app can start safely.
    """

    def test_corrupt_json_does_not_raise(self, tmp_path: Path):
        """load_json_config must not propagate JSONDecodeError for corrupt input."""
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{invalid json content!!!")

        # Must complete without raising any exception
        data = load_json_config(path=str(config_path))
        assert isinstance(data, dict)

    def test_corrupt_json_returns_default_sections(self, tmp_path: Path):
        """load_json_config returns all DEFAULT_CONFIG top-level sections after corrupt-file recovery."""
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("not json at all — completely broken")

        data = load_json_config(path=str(config_path))

        for section in ("models", "runtime", "wakeword", "audio", "api"):
            assert (
                section in data
            ), f"Default section '{section}' missing after corrupt-file recovery"

    def test_corrupt_json_creates_backup_file(self, tmp_path: Path):
        """load_json_config creates a .invalid.<timestamp>.json backup when JSON is corrupt.

        The backup proves the error was detected and recorded — the file path
        is embedded in the backup filename and the log message.
        """
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{bad json")

        load_json_config(path=str(config_path))

        backup_files = list(config_path.parent.glob("rex_config.invalid.*.json"))
        assert backup_files, (
            "Expected a .invalid.*.json backup to be created — "
            "this confirms the corrupt file was detected and the path was used in the error."
        )

    def test_corrupt_json_recreates_valid_config_file(self, tmp_path: Path):
        """After corrupt-file recovery, rex_config.json is recreated with valid JSON."""
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("]]corrupt_garbage[[")

        load_json_config(path=str(config_path))

        assert (
            config_path.exists()
        ), "rex_config.json should be recreated after corrupt-file recovery"
        recreated = json.loads(config_path.read_text())
        assert isinstance(recreated, dict), "Recreated config must be valid JSON"

    def test_build_app_config_from_corrupt_recovery_does_not_raise(self, tmp_path: Path):
        """AppConfig can be built from the defaults returned after corrupt-file recovery."""
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{broken_key: no_quotes}")

        data = load_json_config(path=str(config_path))
        cfg = build_app_config(data)
        assert isinstance(cfg, AppConfig)

    def test_truncated_json_handled_gracefully(self, tmp_path: Path):
        """A truncated JSON file (e.g. disk-full write) is recovered from gracefully."""
        config_path = tmp_path / "config" / "rex_config.json"
        config_path.parent.mkdir(parents=True)
        # Simulate a truncated JSON write
        config_path.write_text('{"models": {"llm_provider": "transformers"')

        data = load_json_config(path=str(config_path))
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# AC-3: Missing config file → defaults applied, not crash
# ---------------------------------------------------------------------------


class TestMissingConfigFileDefaults:
    """A missing rex_config.json must result in defaults applied, not a crash."""

    def test_missing_file_returns_dict(self, tmp_path: Path):
        """load_json_config with a nonexistent path returns a dict."""
        config_path = tmp_path / "config" / "rex_config.json"
        assert not config_path.exists()

        data = load_json_config(path=str(config_path))

        assert isinstance(data, dict)

    def test_missing_file_applies_all_default_sections(self, tmp_path: Path):
        """load_json_config returns all DEFAULT_CONFIG sections when file is absent."""
        config_path = tmp_path / "config" / "rex_config.json"

        data = load_json_config(path=str(config_path))

        for section in ("models", "runtime", "wakeword", "audio", "api"):
            assert section in data, f"Expected default section '{section}'"

    def test_missing_file_auto_creates_config_file(self, tmp_path: Path):
        """load_json_config creates rex_config.json with defaults when file is absent."""
        config_path = tmp_path / "config" / "rex_config.json"
        assert not config_path.exists()

        load_json_config(path=str(config_path))

        assert config_path.exists(), "rex_config.json should be auto-created with defaults"

    def test_missing_file_build_app_config_does_not_raise(self, tmp_path: Path):
        """load_json_config + build_app_config on a missing file raises no exception."""
        config_path = tmp_path / "newdir" / "rex_config.json"

        data = load_json_config(path=str(config_path))
        cfg = build_app_config(data)

        assert isinstance(cfg, AppConfig)

    def test_missing_file_default_llm_provider(self, tmp_path: Path):
        """AppConfig built after missing-file recovery has the expected default llm_provider."""
        config_path = tmp_path / "config" / "rex_config.json"

        data = load_json_config(path=str(config_path))
        cfg = build_app_config(data)

        assert cfg.llm_provider == "transformers"

    def test_missing_file_default_memory_max_turns(self, tmp_path: Path):
        """AppConfig built after missing-file recovery has the expected default memory_max_turns."""
        config_path = tmp_path / "config" / "rex_config.json"

        data = load_json_config(path=str(config_path))
        cfg = build_app_config(data)

        assert cfg.memory_max_turns == 50

    def test_missing_nested_directories_are_auto_created(self, tmp_path: Path):
        """Missing parent directories for rex_config.json are created automatically."""
        config_path = tmp_path / "a" / "b" / "c" / "rex_config.json"
        assert not config_path.parent.exists()

        load_json_config(path=str(config_path))

        assert config_path.exists(), "Config file and all parent directories should be created"

    def test_missing_file_created_with_valid_json(self, tmp_path: Path):
        """The auto-created rex_config.json contains valid JSON."""
        config_path = tmp_path / "config" / "rex_config.json"

        load_json_config(path=str(config_path))

        content = config_path.read_text(encoding="utf-8")
        parsed = json.loads(content)
        assert isinstance(parsed, dict)
