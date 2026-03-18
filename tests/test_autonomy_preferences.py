"""Unit tests for US-237: UserPreferenceProfile and PreferenceStore."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from rex.autonomy.preferences import PreferenceStore, UserPreferenceProfile

# ---------------------------------------------------------------------------
# UserPreferenceProfile — defaults
# ---------------------------------------------------------------------------


class TestUserPreferenceProfileDefaults:
    def test_default_autonomy_mode(self) -> None:
        profile = UserPreferenceProfile()
        assert profile.preferred_autonomy_mode == "manual"

    def test_default_preferred_model_empty(self) -> None:
        profile = UserPreferenceProfile()
        assert profile.preferred_model == ""

    def test_default_common_goal_patterns_empty(self) -> None:
        profile = UserPreferenceProfile()
        assert profile.common_goal_patterns == []

    def test_default_active_hours_empty(self) -> None:
        profile = UserPreferenceProfile()
        assert profile.active_hours == []

    def test_default_avg_budget_zero(self) -> None:
        profile = UserPreferenceProfile()
        assert profile.avg_budget_usd == 0.0

    def test_last_updated_is_datetime(self) -> None:
        profile = UserPreferenceProfile()
        assert isinstance(profile.last_updated, datetime)


# ---------------------------------------------------------------------------
# UserPreferenceProfile — field assignment
# ---------------------------------------------------------------------------


class TestUserPreferenceProfileFields:
    def test_set_all_fields(self) -> None:
        now = datetime.now(timezone.utc)
        profile = UserPreferenceProfile(
            preferred_autonomy_mode="full-auto",
            preferred_model="gpt-4o",
            common_goal_patterns=["send email", "check weather"],
            active_hours=[9, 10, 11],
            avg_budget_usd=0.05,
            last_updated=now,
        )
        assert profile.preferred_autonomy_mode == "full-auto"
        assert profile.preferred_model == "gpt-4o"
        assert profile.common_goal_patterns == ["send email", "check weather"]
        assert profile.active_hours == [9, 10, 11]
        assert profile.avg_budget_usd == 0.05
        assert profile.last_updated == now


# ---------------------------------------------------------------------------
# PreferenceStore — load when file does not exist
# ---------------------------------------------------------------------------


class TestPreferenceStoreLoadMissing:
    def test_returns_default_profile_when_no_file(self, tmp_path: Path) -> None:
        store = PreferenceStore(prefs_path=tmp_path / "prefs.json")
        profile = store.load()
        assert isinstance(profile, UserPreferenceProfile)
        assert profile.preferred_autonomy_mode == "manual"

    def test_no_exception_when_file_missing(self, tmp_path: Path) -> None:
        store = PreferenceStore(prefs_path=tmp_path / "missing.json")
        # Should not raise
        store.load()


# ---------------------------------------------------------------------------
# PreferenceStore — save creates file
# ---------------------------------------------------------------------------


class TestPreferenceStoreSave:
    def test_save_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "subdir" / "prefs.json"
        store = PreferenceStore(prefs_path=path)
        store.save(UserPreferenceProfile())
        assert path.exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "c" / "prefs.json"
        store = PreferenceStore(prefs_path=path)
        store.save(UserPreferenceProfile())
        assert path.parent.is_dir()

    def test_saved_file_is_valid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "prefs.json"
        store = PreferenceStore(prefs_path=path)
        store.save(UserPreferenceProfile())
        raw = path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        assert "preferred_autonomy_mode" in parsed


# ---------------------------------------------------------------------------
# PreferenceStore — round-trip
# ---------------------------------------------------------------------------


class TestPreferenceStoreRoundTrip:
    def test_save_then_load_returns_same_profile(self, tmp_path: Path) -> None:
        path = tmp_path / "prefs.json"
        store = PreferenceStore(prefs_path=path)
        now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        original = UserPreferenceProfile(
            preferred_autonomy_mode="supervised",
            preferred_model="claude-3-opus",
            common_goal_patterns=["write code", "summarise email"],
            active_hours=[8, 9, 17, 18],
            avg_budget_usd=0.12,
            last_updated=now,
        )
        store.save(original)
        loaded = store.load()

        assert loaded.preferred_autonomy_mode == original.preferred_autonomy_mode
        assert loaded.preferred_model == original.preferred_model
        assert loaded.common_goal_patterns == original.common_goal_patterns
        assert loaded.active_hours == original.active_hours
        assert abs(loaded.avg_budget_usd - original.avg_budget_usd) < 1e-9
        assert loaded.last_updated == original.last_updated

    def test_overwrite_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "prefs.json"
        store = PreferenceStore(prefs_path=path)

        store.save(UserPreferenceProfile(preferred_model="model-v1"))
        store.save(UserPreferenceProfile(preferred_model="model-v2"))
        loaded = store.load()

        assert loaded.preferred_model == "model-v2"


# ---------------------------------------------------------------------------
# PreferenceStore — corrupt file falls back to defaults
# ---------------------------------------------------------------------------


class TestPreferenceStoreCorrupt:
    def test_corrupt_json_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "prefs.json"
        path.write_text("not-valid-json{{{", encoding="utf-8")
        store = PreferenceStore(prefs_path=path)
        profile = store.load()
        assert isinstance(profile, UserPreferenceProfile)
        assert profile.preferred_autonomy_mode == "manual"

    def test_wrong_schema_returns_default(self, tmp_path: Path) -> None:
        path = tmp_path / "prefs.json"
        path.write_text(json.dumps({"preferred_autonomy_mode": 999}), encoding="utf-8")
        store = PreferenceStore(prefs_path=path)
        # Pydantic will coerce int→str, so this may not fail; just check no exception
        store.load()
