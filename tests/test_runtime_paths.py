from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import rex.config as app_config
import rex.config_manager as config_manager
import rex.runtime_paths as runtime_paths_module
from rex.runtime_paths import (
    config_path,
    data_dir,
    env_path,
    household_data_dir,
    memory_dir,
    profiles_dir,
    runtime_root,
    user_data_dir,
    users_data_dir,
)


def test_runtime_path_override_is_independent_of_cwd(monkeypatch, tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setenv("ASKREX_RUNTIME_DIR", str(runtime))
    monkeypatch.chdir(unrelated)

    assert runtime_root() == runtime.resolve()
    assert config_path() == (runtime / "config" / "rex_config.json").resolve()
    assert env_path() == (runtime / ".env").resolve()
    assert profiles_dir() == (runtime / "profiles").resolve()
    assert data_dir() == (runtime / "data").resolve()
    assert memory_dir() == (runtime / "Memory").resolve()


def test_config_manager_default_never_writes_under_process_cwd(monkeypatch, tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setenv("ASKREX_RUNTIME_DIR", str(runtime))
    monkeypatch.chdir(unrelated)

    loaded = config_manager.load_config()

    assert loaded["active_profile"] == "default"
    assert (runtime / "config" / "rex_config.json").is_file()
    assert not (unrelated / "config").exists()


def test_profile_merge_runs_without_legacy_environment_warnings(
    monkeypatch, tmp_path: Path
) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    (profiles / "default.json").write_text(
        json.dumps(
            {
                "profile_version": 1,
                "name": "default",
                "description": "test profile",
                "capabilities": ["test-capability"],
                "overrides": {"models": {"llm_model": "profile-selected-model"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(app_config, "get_legacy_env_warnings", lambda: [])
    app_config._cached_config = None

    resolved = app_config.load_config(
        reload=True,
        json_config={"active_profile": "default", "profiles_dir": str(profiles)},
    )

    assert resolved.llm_model == "profile-selected-model"
    assert resolved.capabilities == ["test-capability"]


def test_config_load_from_foreign_cwd_writes_only_to_runtime_root(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["ASKREX_RUNTIME_DIR"] = str(runtime)
    env["PYTHONPATH"] = str(repo)
    env.pop("OPENAI_API_KEY", None)
    (runtime).mkdir(parents=True)
    (runtime / ".env").write_text("OPENAI_API_KEY=runtime-secret\n", encoding="utf-8")
    for legacy_key in ("REX_ACTIVE_USER", "REX_USER_ID"):
        env.pop(legacy_key, None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rex.config import load_config; c=load_config(reload=True); "
            "print(c.active_profile); print(c.openai_api_key)",
        ],
        cwd=unrelated,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["default", "runtime-secret"]
    assert (runtime / "config" / "rex_config.json").is_file()
    assert (runtime / "profiles" / "default.json").is_file()
    assert not (unrelated / "config").exists()
    assert not (unrelated / "profiles").exists()


def test_legacy_data_override_remains_exact_household_root(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "legacy-data"
    monkeypatch.setenv("REX_DATA_DIR", str(root))
    monkeypatch.delenv("ASKREX_HOUSEHOLD_DATA_DIR", raising=False)
    monkeypatch.delenv("ASKREX_USERS_DATA_DIR", raising=False)

    assert data_dir() == root.resolve()
    assert household_data_dir() == root.resolve()
    assert users_data_dir() == (root / "users").resolve()
    assert user_data_dir("james") == (root / "users" / "james").resolve()


def test_explicit_private_and_household_roots_override_legacy_data(
    monkeypatch, tmp_path: Path
) -> None:
    legacy = tmp_path / "legacy"
    household = tmp_path / "shared"
    users = tmp_path / "private"
    monkeypatch.setenv("REX_DATA_DIR", str(legacy))
    monkeypatch.setenv("ASKREX_HOUSEHOLD_DATA_DIR", str(household))
    monkeypatch.setenv("ASKREX_USERS_DATA_DIR", str(users))

    assert household_data_dir() == household.resolve()
    assert users_data_dir() == users.resolve()
    assert user_data_dir("cole") == (users / "cole").resolve()


def test_windows_os_users_receive_distinct_runtime_roots(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ASKREX_RUNTIME_DIR", raising=False)
    monkeypatch.setattr(runtime_paths_module, "source_checkout_root", lambda start=None: None)
    monkeypatch.setattr(runtime_paths_module.sys, "platform", "win32")

    first = tmp_path / "windows-user-one"
    second = tmp_path / "windows-user-two"
    monkeypatch.setenv("LOCALAPPDATA", str(first))
    first_root = runtime_root()
    monkeypatch.setenv("LOCALAPPDATA", str(second))
    second_root = runtime_root()

    assert first_root == (first / "AskRex").resolve()
    assert second_root == (second / "AskRex").resolve()
    assert first_root != second_root
