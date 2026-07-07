"""Release metadata consistency tests."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject_version() -> str:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    return str(pyproject["project"]["version"])


def _manifest_version() -> str:
    manifest = json.loads(
        (_REPO_ROOT / ".release-please-manifest.json").read_text(encoding="utf-8")
    )
    return str(manifest["."])


def test_release_manifest_matches_python_package_version() -> None:
    assert _manifest_version() == _pyproject_version()


def test_current_release_version_is_not_initial_placeholder() -> None:
    assert _pyproject_version() != "0.1.0"
