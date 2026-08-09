from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLAUDE = ROOT / "CLAUDE.md"


def _claude() -> str:
    return CLAUDE.read_text(encoding="utf-8")


def test_claude_root_python_inventory_matches_filesystem() -> None:
    text = _claude()
    root_files = sorted(path.name for path in ROOT.glob("*.py"))
    assert len(root_files) == 27
    assert "### Root-level `.py` files (27 total)" in text
    assert "9 active root-level" not in text
    for name in root_files:
        assert f"- {name}" in text


def test_claude_console_scripts_match_pyproject() -> None:
    text = _claude()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert len(scripts) == 6
    for name, target in scripts.items():
        assert f"- {name} -> {target}" in text


def test_claude_voice_openclaw_and_docker_truth_matches_current_contract() -> None:
    text = _claude()
    assert "The source CLI defaults to Hold-to-Talk/manual activation" in text
    assert "`--mode wake-word` is explicitly selected" in text
    assert "both OpenClaw flags default to `False`" in text
    assert "OpenClaw is experimental/off by default" in text
    assert "valid HTTP(S) gateway URL plus `OPENCLAW_GATEWAY_TOKEN`" in text
    assert "`/healthz` may establish reachability only" in text
    assert "connection/auth/429/5xx failures fall back locally with a structured warning" in text
    assert "A 403 remains a hard policy denial" in text
    assert "Docker is developer/operator-only" in text
    assert "python -m rex doctor --healthcheck" in text


def test_claude_install_and_primary_gui_truth() -> None:
    text = _claude()
    assert "Developer/operator source install: `pip install .`" in text
    assert "End-user install path: packaged Windows Electron installer" in text
    assert "React + Electron under `gui/` is the primary packaged interface" in text
    assert "`rex.gui_app` is a developer-only Flask API/dashboard" in text


def test_claude_current_document_references_exist() -> None:
    expected = [
        "docs/BRANDING.md",
        "SURFACE-CLASSIFICATION.md",
        "docs/claude/COMMANDS_AND_ENTRYPOINTS.md",
        "docs/claude/CONFIG_AND_SECURITY.md",
        "INTEGRATIONS_STATUS.md",
        "docs/claude/TESTING_AND_QUALITY.md",
        "CONTRIBUTING.md",
        "archived/ARCHIVED.md",
        "docs/voice_pipeline.md",
        "docs/testing/SKIPPED-TESTS-INVENTORY.md",
        "INSTALL.md",
        "README.md",
        "docs/mobile/MOBILE_API_SETUP_WINDOWS.md",
        "docs/planning/source-of-truth/REX_Unified_Build_Spec_UPDATED.md",
        "docs/planning/source-of-truth/REX_ACTIVE_CHECKLIST.md",
        "docs/planning/TEAM_LEAD_OPERATING_RULES.md",
        "docs/mobile/DEVICE_PAIRING.md",
        "docs/mobile/STRONG_AUTH.md",
    ]
    text = _claude()
    for path in expected:
        assert path in text
        assert (ROOT / path).exists(), path
