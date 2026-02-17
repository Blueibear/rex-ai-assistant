"""Tests for rex-config CLI (show, legacy flags, migrate-legacy-env dry-run).

Covers:
  HIGH-02 — rex-config show / --show / --reload print to stdout.
  HIGH-03 — migrate-legacy-env --dry-run uses .env data (not process env),
             and produces the same planned action set as the real migration.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

# Resolve a Python interpreter that can import rex (has all required deps).
# The pytest runner may use an isolated venv without pydantic, so prefer the
# interpreter that was used to *install* the package when it differs.
def _find_rex_python() -> str:
    """Return the path to a Python interpreter that can import rex.config."""
    candidates = [sys.executable, shutil.which("python3") or "", "/usr/local/bin/python3"]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            r = subprocess.run(
                [candidate, "-c", "import rex.config"],
                capture_output=True, timeout=10,
            )
            if r.returncode == 0:
                return candidate
        except (OSError, subprocess.TimeoutExpired):
            continue
    # Fallback — let the caller surface the error
    return sys.executable

_REX_PYTHON = _find_rex_python()


def _run_rex_config(args: List[str], *, cwd: Path, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    """Run 'python -m rex.config <args>' from the given working directory."""
    env = {k: v for k, v in os.environ.items()}
    env["PYTHONPATH"] = str(REPO_ROOT)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [_REX_PYTHON, "-m", "rex.config", *args],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
        cwd=str(cwd),
    )


_DEFAULT_PROFILE = json.dumps({
    "profile_version": 1,
    "name": "default",
    "description": "Test profile",
    "capabilities": [],
    "overrides": {},
})

_MINIMAL_CONFIG = json.dumps({
    "models": {
        "llm_provider": "transformers",
        "llm_model": "sshleifer/tiny-gpt2",
    }
})


def _setup_work_dir(tmp_path: Path) -> Path:
    """Populate *tmp_path* with the minimum filesystem structure that allows
    ``python -m rex.config`` to run without errors:
      - config/rex_config.json  (valid minimal JSON config)
      - profiles/default.json   (minimal profile so profile loading succeeds)
      - .env                    (empty, avoids inheriting repo secrets)
    Returns *tmp_path* for convenience.
    """
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "rex_config.json").write_text(_MINIMAL_CONFIG)

    (tmp_path / "profiles").mkdir(parents=True, exist_ok=True)
    (tmp_path / "profiles" / "default.json").write_text(_DEFAULT_PROFILE)

    (tmp_path / ".env").write_text("")
    return tmp_path


def _minimal_config_dir(tmp_path: Path) -> Path:
    """Alias for _setup_work_dir kept for readability in show-command tests."""
    return _setup_work_dir(tmp_path)


def _extract_action_tuples(notes: List[str]) -> List[Tuple[str, str, str]]:
    """Parse 'Migrated KEY -> path = value' or '[dry-run] Would migrate KEY -> path = VALUE'
    notes into (env_key, config_path, raw_value_str) tuples for comparison."""
    tuples = []
    pattern = re.compile(
        r"(?:\[dry-run\] Would migrate|Migrated)\s+(\S+)\s+->\s+(\S+)\s+=\s+(.*)"
    )
    for note in notes:
        m = pattern.match(note.strip())
        if m:
            tuples.append((m.group(1), m.group(2), m.group(3).strip()))
    return tuples


# ---------------------------------------------------------------------------
# HIGH-02: rex-config show / --show / --reload print config to stdout
# ---------------------------------------------------------------------------


class TestRexConfigShow:
    """rex-config show must print JSON config to stdout."""

    def test_show_subcommand_stdout(self, tmp_path: Path):
        """rex-config show produces JSON on stdout containing known top-level keys."""
        work_dir = _minimal_config_dir(tmp_path)
        result = _run_rex_config(["show"], cwd=work_dir)
        assert result.returncode == 0, (
            f"rex-config show failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        # Must produce parseable JSON
        data = json.loads(result.stdout)
        # These keys come from AppConfig.to_dict() / asdict()
        for key in ("openai_model", "openai_base_url", "llm_provider", "wakeword"):
            assert key in data, f"Expected key '{key}' in show output, got keys: {sorted(data)}"

    def test_show_subcommand_stdout_not_empty(self, tmp_path: Path):
        """rex-config show stdout must be non-empty."""
        work_dir = _minimal_config_dir(tmp_path)
        result = _run_rex_config(["show"], cwd=work_dir)
        assert result.returncode == 0
        assert result.stdout.strip(), "stdout should not be empty"

    def test_show_subcommand_no_config_key_on_stderr(self, tmp_path: Path):
        """Config key=value output must NOT appear only in stderr (must be on stdout)."""
        work_dir = _minimal_config_dir(tmp_path)
        result = _run_rex_config(["show"], cwd=work_dir)
        assert result.returncode == 0
        # The main output should be valid JSON on stdout
        assert result.stdout.strip().startswith("{"), (
            f"Expected JSON on stdout, got: {result.stdout[:200]!r}"
        )


class TestRexConfigLegacyFlagShow:
    """rex-config --show (legacy flag) must produce the same stdout as the subcommand."""

    def test_legacy_show_flag_stdout(self, tmp_path: Path):
        """rex-config --show produces JSON on stdout."""
        work_dir = _minimal_config_dir(tmp_path)
        result = _run_rex_config(["--show"], cwd=work_dir)
        assert result.returncode == 0, (
            f"rex-config --show failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        data = json.loads(result.stdout)
        assert "openai_base_url" in data
        assert "openai_model" in data

    def test_legacy_show_same_keys_as_subcommand(self, tmp_path: Path):
        """rex-config --show and rex-config show must output the same set of keys."""
        work_dir = _minimal_config_dir(tmp_path)
        show_result = _run_rex_config(["show"], cwd=work_dir)
        flag_result = _run_rex_config(["--show"], cwd=work_dir)
        assert show_result.returncode == 0
        assert flag_result.returncode == 0
        show_keys = set(json.loads(show_result.stdout).keys())
        flag_keys = set(json.loads(flag_result.stdout).keys())
        assert show_keys == flag_keys, (
            f"Key mismatch between 'show' and '--show':\n"
            f"  only in 'show': {show_keys - flag_keys}\n"
            f"  only in '--show': {flag_keys - show_keys}"
        )


class TestRexConfigLegacyFlagReload:
    """rex-config --reload (legacy flag) must also show config on stdout."""

    def test_legacy_reload_flag_stdout(self, tmp_path: Path):
        """rex-config --reload produces JSON on stdout."""
        work_dir = _minimal_config_dir(tmp_path)
        result = _run_rex_config(["--reload"], cwd=work_dir)
        assert result.returncode == 0, (
            f"rex-config --reload failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        data = json.loads(result.stdout)
        assert "openai_base_url" in data
        assert "openai_model" in data

    def test_legacy_reload_same_keys_as_show(self, tmp_path: Path):
        """rex-config --reload and rex-config show must output the same set of keys."""
        work_dir = _minimal_config_dir(tmp_path)
        show_result = _run_rex_config(["show"], cwd=work_dir)
        reload_result = _run_rex_config(["--reload"], cwd=work_dir)
        assert show_result.returncode == 0
        assert reload_result.returncode == 0
        show_keys = set(json.loads(show_result.stdout).keys())
        reload_keys = set(json.loads(reload_result.stdout).keys())
        assert show_keys == reload_keys


# ---------------------------------------------------------------------------
# HIGH-03: migrate-legacy-env --dry-run uses .env, not process env
# ---------------------------------------------------------------------------


class TestDryRunUsesEnvFile:
    """--dry-run must read from the .env file, not os.environ."""

    def test_dry_run_sees_env_file_variable_not_in_process_env(self, tmp_path: Path):
        """Dry-run reports OPENAI_BASE_URL migration even when the process env
        does NOT contain OPENAI_BASE_URL."""
        work_dir = _setup_work_dir(tmp_path)
        # Overwrite the empty .env with OPENAI_BASE_URL
        env_file = work_dir / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://test-dry-run.local/v1\n")
        # Point the CLI at an isolated config file (not the default config/ location)
        config_file = work_dir / "dry_run_config.json"

        # Ensure OPENAI_BASE_URL is absent from the subprocess environment
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_BASE_URL"}
        env["PYTHONPATH"] = str(REPO_ROOT)

        result = subprocess.run(
            [
                _REX_PYTHON, "-m", "rex.config",
                "migrate-legacy-env",
                "--dry-run",
                "--env-path", str(env_file),
                "--config-path", str(config_file),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=str(work_dir),
        )

        assert result.returncode == 0, (
            f"migrate-legacy-env --dry-run failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        combined = result.stdout + result.stderr
        assert "openai.base_url" in combined, (
            f"Expected 'openai.base_url' in dry-run output.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    def test_dry_run_does_not_write_config_file(self, tmp_path: Path):
        """--dry-run must not create or modify the config file."""
        work_dir = _setup_work_dir(tmp_path)
        env_file = work_dir / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://no-write.local/v1\n")
        config_file = work_dir / "should_not_exist.json"

        env = {k: v for k, v in os.environ.items() if k != "OPENAI_BASE_URL"}
        env["PYTHONPATH"] = str(REPO_ROOT)

        result = subprocess.run(
            [
                _REX_PYTHON, "-m", "rex.config",
                "migrate-legacy-env",
                "--dry-run",
                "--env-path", str(env_file),
                "--config-path", str(config_file),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=str(work_dir),
        )

        assert result.returncode == 0
        assert not config_file.exists(), (
            "--dry-run must not write the config file"
        )


# ---------------------------------------------------------------------------
# HIGH-03: dry-run and non-dry-run produce the same action identity set
# ---------------------------------------------------------------------------


class TestDryRunVsRealMigrationConsistency:
    """dry-run and non-dry-run must agree on which keys would be/were migrated."""

    def test_same_action_identities(self, tmp_path: Path):
        """Dry-run planned actions match real migration actions for env_key and config path."""
        from rex.config_manager import migrate_legacy_env_to_config

        env_file = tmp_path / ".env"
        env_file.write_text(
            "OPENAI_BASE_URL=http://consistency-test.local/v1\n"
            "REX_LLM_PROVIDER=openai\n"
        )

        dry_config = tmp_path / "dry_config.json"
        real_config = tmp_path / "real_config.json"

        dry_notes = migrate_legacy_env_to_config(
            env_path=str(env_file),
            config_path=str(dry_config),
            dry_run=True,
        )
        real_notes = migrate_legacy_env_to_config(
            env_path=str(env_file),
            config_path=str(real_config),
            dry_run=False,
        )

        dry_actions = _extract_action_tuples(dry_notes)
        real_actions = _extract_action_tuples(real_notes)

        assert dry_actions, "dry-run should report at least one planned action"
        assert real_actions, "real migration should report at least one action"

        # Compare (env_key, config_path) pairs — values may differ in repr vs str
        dry_pairs = {(k, p) for k, p, _ in dry_actions}
        real_pairs = {(k, p) for k, p, _ in real_actions}
        assert dry_pairs == real_pairs, (
            f"dry-run and real migration disagree on which keys to migrate:\n"
            f"  dry-run only: {dry_pairs - real_pairs}\n"
            f"  real only:    {real_pairs - dry_pairs}"
        )

    def test_dry_run_values_match_real_migration_values(self, tmp_path: Path):
        """For each migrated key, dry-run and real migration must agree on the parsed value."""
        import ast
        from rex.config_manager import migrate_legacy_env_to_config

        env_file = tmp_path / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://value-check.local/v1\n")

        dry_config = tmp_path / "dry_config.json"
        real_config = tmp_path / "real_config.json"

        dry_notes = migrate_legacy_env_to_config(
            env_path=str(env_file), config_path=str(dry_config), dry_run=True,
        )
        real_notes = migrate_legacy_env_to_config(
            env_path=str(env_file), config_path=str(real_config), dry_run=False,
        )

        dry_by_key = {(k, p): v for k, p, v in _extract_action_tuples(dry_notes)}
        real_by_key = {(k, p): v for k, p, v in _extract_action_tuples(real_notes)}

        for key_pair, dry_repr in dry_by_key.items():
            real_str = real_by_key.get(key_pair)
            assert real_str is not None, f"Real migration missing key {key_pair}"
            # dry-run stores repr (e.g. "'http://...'"), real stores str value.
            # Eval the repr to compare.
            try:
                dry_val = ast.literal_eval(dry_repr)
            except (ValueError, SyntaxError):
                dry_val = dry_repr
            assert str(dry_val) == real_str, (
                f"Value mismatch for {key_pair}:\n"
                f"  dry-run: {dry_repr!r} -> {dry_val!r}\n"
                f"  real:    {real_str!r}"
            )

    def test_dry_run_subprocess_consistency(self, tmp_path: Path):
        """Via subprocess: dry-run output mentions openai.base_url; real migration writes it."""
        work_dir = _setup_work_dir(tmp_path)
        env_file = work_dir / ".env"
        env_file.write_text("OPENAI_BASE_URL=http://subproc-consistency.local/v1\n")
        dry_config = work_dir / "dry_cfg.json"
        real_config = work_dir / "real_cfg.json"

        base_env = {k: v for k, v in os.environ.items() if k != "OPENAI_BASE_URL"}
        base_env["PYTHONPATH"] = str(REPO_ROOT)

        dry_result = subprocess.run(
            [
                _REX_PYTHON, "-m", "rex.config",
                "migrate-legacy-env", "--dry-run",
                "--env-path", str(env_file),
                "--config-path", str(dry_config),
            ],
            capture_output=True, text=True, timeout=60, env=base_env, cwd=str(work_dir),
        )
        real_result = subprocess.run(
            [
                _REX_PYTHON, "-m", "rex.config",
                "migrate-legacy-env",
                "--env-path", str(env_file),
                "--config-path", str(real_config),
            ],
            capture_output=True, text=True, timeout=60, env=base_env, cwd=str(work_dir),
        )

        assert dry_result.returncode == 0
        assert real_result.returncode == 0

        assert "openai.base_url" in dry_result.stdout, (
            f"Expected 'openai.base_url' in dry-run stdout.\n{dry_result.stdout}"
        )
        # Real migration should have written the config file
        assert real_config.exists(), "Real migration must create config file"
        written = json.loads(real_config.read_text())
        assert written.get("openai", {}).get("base_url") == "http://subproc-consistency.local/v1"


# ---------------------------------------------------------------------------
# Discoverability sanity check
# ---------------------------------------------------------------------------


class TestRexConfigDiscoverability:
    """rex-config help output mentions key subcommands."""

    def test_help_mentions_show(self, tmp_path: Path):
        """rex-config --help must mention the show subcommand."""
        work_dir = _setup_work_dir(tmp_path)
        result = _run_rex_config(["--help"], cwd=work_dir)
        assert result.returncode == 0
        assert "show" in result.stdout.lower()

    def test_help_mentions_migrate(self, tmp_path: Path):
        """rex-config --help must mention the migrate-legacy-env subcommand."""
        work_dir = _setup_work_dir(tmp_path)
        result = _run_rex_config(["--help"], cwd=work_dir)
        assert result.returncode == 0
        assert "migrate" in result.stdout.lower()
