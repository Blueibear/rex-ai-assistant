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
        "Validate PowerShell artifact scripts",
        "npm ci",
        "python -m build --wheel",
        "npm run dist",
        "verify_electron_package_contents.py",
        "test_installed_electron_artifact.ps1",
        "askrex-windows-smoke-diagnostics",
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
    assert "$identityResult.Stdout.Trim()" in harness
    assert "$identityResult.Stderr.Trim()" in harness
    assert "Write-SmokeDiagnostics 'failure'" in harness


def test_installed_artifact_harness_uses_utf8_file_backed_stdin() -> None:
    harness = (ROOT / "scripts/test_installed_electron_artifact.ps1").read_text(encoding="utf-8")
    assert "System.Text.UTF8Encoding($false)" in harness
    assert "System.IO.File]::WriteAllText" in harness
    assert "RedirectStandardInput = $stdinPath" in harness
    assert "Start-Process @startProcessArguments" in harness
    assert "StandardInputEncoding" not in harness
    assert "StandardInput.BaseStream" not in harness


def test_installed_artifact_harness_uses_clean_reinstall_lifecycle() -> None:
    harness = (ROOT / "scripts/test_installed_electron_artifact.ps1").read_text(encoding="utf-8")
    phases = (
        "initial-install",
        "first-uninstall",
        "reinstall",
        "final-uninstall",
    )
    positions = [harness.index(phase) for phase in phases]
    assert positions == sorted(positions)
    assert "function Invoke-Uninstaller" in harness
    assert "function Assert-Uninstalled" in harness


def test_installed_artifact_harness_pins_build_python_before_path_isolation() -> None:
    harness = (ROOT / "scripts/test_installed_electron_artifact.ps1").read_text(encoding="utf-8")
    resolve_position = harness.index("$buildPythonPath = (Get-Command $BuildPython")
    isolate_position = harness.index("$env:PATH = Join-Path $env:SystemRoot 'System32'")
    assert resolve_position < isolate_position
    assert harness.count("& $buildPythonPath") == 2


def test_installed_artifact_harness_does_not_turn_cleanup_locks_into_product_failure() -> None:
    harness = (ROOT / "scripts/test_installed_electron_artifact.ps1").read_text(encoding="utf-8")
    assert "function Remove-SmokeTestRoot" in harness
    assert "$attempt -le 10" in harness
    assert "Write-Warning \"Could not fully remove temporary smoke directory" in harness
    assert "Remove-SmokeTestRoot $testRoot" in harness
