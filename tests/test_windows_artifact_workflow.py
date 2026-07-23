from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_windows_artifact_workflow_exercises_installed_release() -> None:
    workflow = (ROOT / ".github/workflows/windows-electron-artifact.yml").read_text(
        encoding="utf-8"
    )
    for required in (
        "runs-on: windows-latest",
        'python-version: "3.11.9"',
        'node-version: "20.19.1"',
        "npm ci",
        "python -m build --wheel",
        "npm run dist",
        "verify_electron_package_contents.py",
        "test_installed_electron_artifact.ps1",
    ):
        assert required in workflow


def test_release_automation_is_gated_by_windows_artifact() -> None:
    workflow = (ROOT / ".github/workflows/release-please.yml").read_text(encoding="utf-8")
    assert "uses: ./.github/workflows/windows-electron-artifact.yml" in workflow
    assert "needs: windows-electron-artifact" in workflow


def test_installed_artifact_harness_retries_identity_with_diagnostics() -> None:
    harness = (ROOT / "scripts/test_installed_electron_artifact.ps1").read_text(encoding="utf-8")
    assert "$attempt -le 3" in harness
    assert "failed after 3 attempts" in harness
    assert "$identityOutput -join ' '" in harness
