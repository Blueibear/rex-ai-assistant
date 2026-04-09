"""Tests for US-307: Remove legacy REX_WAKEWORD_THRESHOLD env var.

Acceptance criteria:
  - REX_WAKEWORD_THRESHOLD is not in ENV_TO_CONFIG_MAPPING (no runtime effect)
  - Setting REX_WAKEWORD_THRESHOLD produces no startup warning
  - Threshold is configured via wakeword.threshold in rex_config.json only
  - settings_schema no longer defines REX_WAKEWORD_THRESHOLD
"""

from __future__ import annotations

import os
from unittest.mock import patch


class TestWakewordThresholdEnvRemoved:
    def test_not_in_env_to_config_mapping(self):
        """REX_WAKEWORD_THRESHOLD must not be in ENV_TO_CONFIG_MAPPING."""
        from rex.config_manager import ENV_TO_CONFIG_MAPPING

        assert "REX_WAKEWORD_THRESHOLD" not in ENV_TO_CONFIG_MAPPING

    def test_setting_env_var_produces_no_legacy_warning(self):
        """Even if set, REX_WAKEWORD_THRESHOLD triggers no legacy-env warning."""
        from rex.config_manager import get_legacy_env_warnings

        with patch.dict(os.environ, {"REX_WAKEWORD_THRESHOLD": "0.3"}):
            warnings = get_legacy_env_warnings()

        matching = [w for w in warnings if "REX_WAKEWORD_THRESHOLD" in w]
        assert not matching, f"Unexpected warning for removed env var: {matching}"

    def test_settings_schema_does_not_define_threshold_var(self):
        """settings_schema must not expose REX_WAKEWORD_THRESHOLD as a SettingDef."""
        from utils.settings_schema import AUTHORITATIVE_SETTINGS

        keys = [s.key for s in AUTHORITATIVE_SETTINGS]
        assert "REX_WAKEWORD_THRESHOLD" not in keys

    def test_threshold_still_readable_from_json_config(self):
        """wakeword.threshold in rex_config.json is still the config path."""
        from rex.config import build_app_config

        cfg = build_app_config(
            {
                "audio": {"sample_rate": 16000},
                "models": {},
                "runtime": {},
                "wakeword": {"threshold": 0.25},
            }
        )
        assert cfg.wakeword_threshold == 0.25
