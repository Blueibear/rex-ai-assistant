"""Tests for US-306: Unify wake-word config into a single 'wakeword' section.

Acceptance criteria:
  - build_app_config reads wakeword settings from 'wakeword.*' keys
  - Legacy 'wake_word' key is auto-migrated to 'wakeword' with a deprecation log
  - Migration does not overwrite keys already present in 'wakeword'
  - Config with only 'wakeword' section works correctly (no migration needed)
  - rex_config.json has no 'wake_word' key
  - rex_config.schema.json uses 'wakeword' as the canonical key
"""

from __future__ import annotations

import json
import logging
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent


class TestBuildAppConfigReadsFromWakeword:
    def test_reads_threshold_from_wakeword_section(self):
        from rex.config import build_app_config

        cfg = build_app_config(
            {
                "audio": {"sample_rate": 16000},
                "models": {},
                "runtime": {},
                "wakeword": {"threshold": 0.7},
            }
        )
        assert cfg.wakeword_threshold == 0.7

    def test_reads_backend_from_wakeword_section(self):
        from rex.config import build_app_config

        cfg = build_app_config(
            {
                "audio": {"sample_rate": 16000},
                "models": {},
                "runtime": {},
                "wakeword": {"backend": "custom_onnx"},
            }
        )
        assert cfg.wakeword_backend == "custom_onnx"

    def test_reads_wakeword_phrase_from_wakeword_section(self):
        from rex.config import build_app_config

        cfg = build_app_config(
            {
                "audio": {"sample_rate": 16000},
                "models": {},
                "runtime": {},
                "wakeword": {"wakeword": "hey rex"},
            }
        )
        assert cfg.wakeword == "hey rex"


class TestLegacyWakeWordMigration:
    def test_wake_word_key_is_migrated_to_wakeword(self):
        """Legacy wake_word config is auto-migrated without error."""
        from rex.config import build_app_config

        cfg = build_app_config(
            {
                "audio": {"sample_rate": 16000},
                "models": {},
                "runtime": {},
                "wake_word": {"threshold": 0.3, "backend": "custom_onnx"},
            }
        )
        assert cfg.wakeword_threshold == 0.3
        assert cfg.wakeword_backend == "custom_onnx"

    def test_migration_does_not_overwrite_existing_wakeword_values(self):
        """Values already in 'wakeword' take precedence over 'wake_word'."""
        from rex.config import build_app_config

        cfg = build_app_config(
            {
                "audio": {"sample_rate": 16000},
                "models": {},
                "runtime": {},
                "wake_word": {"threshold": 0.9},
                "wakeword": {"threshold": 0.2},
            }
        )
        assert cfg.wakeword_threshold == 0.2

    def test_migration_logs_deprecation_warning(self, caplog):
        """A deprecation warning is logged when wake_word is encountered."""
        from rex.config import build_app_config

        with caplog.at_level(logging.WARNING, logger="rex.config"):
            build_app_config(
                {
                    "audio": {"sample_rate": 16000},
                    "models": {},
                    "runtime": {},
                    "wake_word": {"threshold": 0.3},
                }
            )
        assert any(
            "wake_word" in record.message and "deprecated" in record.message
            for record in caplog.records
        )


class TestConfigFilesUseCanonicalKey:
    def test_rex_config_json_has_no_wake_word_key(self):
        """config/rex_config.json must not contain the legacy 'wake_word' key."""
        config_path = REPO_ROOT / "config" / "rex_config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "wake_word" not in data, (
            "config/rex_config.json still contains the legacy 'wake_word' key — "
            "rename it to 'wakeword'"
        )

    def test_rex_config_json_has_wakeword_key(self):
        """config/rex_config.json must contain the canonical 'wakeword' key."""
        config_path = REPO_ROOT / "config" / "rex_config.json"
        data = json.loads(config_path.read_text(encoding="utf-8"))
        assert "wakeword" in data

    def test_schema_uses_wakeword_key(self):
        """Schema must define 'wakeword', not 'wake_word'."""
        schema_path = REPO_ROOT / "config" / "rex_config.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        props = schema.get("properties", {})
        assert "wakeword" in props, "Schema must have 'wakeword' property"
        assert "wake_word" not in props, "Schema must not have legacy 'wake_word' property"

    def test_schema_required_uses_wakeword(self):
        """Schema required list must reference 'wakeword' not 'wake_word'."""
        schema_path = REPO_ROOT / "config" / "rex_config.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        required = schema.get("required", [])
        assert "wakeword" in required
        assert "wake_word" not in required
