from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "docs" / "AUDIT-CROSS-DOC.md"

EXPECTED_SCRIPTS = {
    "rex": "rex.cli:main",
    "rex-config": "rex.config:cli",
    "rex-speak-api": "rex_speak_api:main",
    "rex-agent": "rex.computers.agent_server:main",
    "rex-gui": "rex.gui_app:main",
    "rex-tool-server": "rex.openclaw.tool_server:main",
}


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_cross_doc_audit_exists_and_records_console_scripts() -> None:
    audit = AUDIT.read_text(encoding="utf-8")
    pyproject = tomllib.loads(_text("pyproject.toml"))
    assert pyproject["project"]["scripts"] == EXPECTED_SCRIPTS
    for name, target in EXPECTED_SCRIPTS.items():
        assert f"`{name}` -> `{target}`" in audit


def test_cross_doc_audit_records_exact_root_python_inventory() -> None:
    audit = AUDIT.read_text(encoding="utf-8")
    root_files = sorted(path.name for path in ROOT.glob("*.py"))
    assert len(root_files) == 27
    assert "Root-level `.py` count: `27`" in audit
    for name in root_files:
        assert f"`{name}`" in audit


def test_cross_doc_audit_records_voice_openclaw_docker_and_ha_code_truth() -> None:
    audit = AUDIT.read_text(encoding="utf-8")
    rex_loop = _text("rex_loop.py")
    config = _text("rex/config.py")
    dockerfile = _text("Dockerfile")
    ha = _text("rex/ha/mutation_service.py")

    assert 'default="hold-to-talk"' in rex_loop
    assert "Source voice default: `hold-to-talk`; `wake-word` is explicit beta opt-in." in audit
    assert "use_openclaw_tools: bool = False" in config
    assert "use_openclaw_voice_backend: bool = False" in config
    assert (
        "OpenClaw defaults: tools `false`, voice backend `false`; enabled mode requires URL + token."
        in audit
    )
    assert "CMD python -m rex doctor --healthcheck" in dockerfile
    assert (
        "Docker tier: `developer-only`; healthcheck is `python -m rex doctor --healthcheck`."
        in audit
    )
    for status in ("VERIFIED", "ATTEMPTED_UNVERIFIED", "CONFIRMATION_REQUIRED", "DENIED", "FAILED"):
        assert status in ha
    assert "HA mutation success is `verified` only after independent state observation" in audit


def test_current_docs_use_same_cross_doc_claims() -> None:
    install = _text("INSTALL.md")
    running = _text("RUNNING.md")
    claude = _text("CLAUDE.md")

    assert "**Shippable installer; source command is development-only**" in install
    assert (
        "**Developer-only** - defaults to Hold-to-Talk; `--mode wake-word` is beta opt-in"
        in install
    )
    assert "**Experimental** - off by default; requires explicit gateway configuration" in install
    assert "Developer-only source voice loop; defaults to Hold-to-Talk" in running
    assert "Experimental, off by default" in running
    assert "Developer/operator source install: `pip install .`" in claude
    assert "`use_openclaw_tools` defaults to `False`" in claude
    assert (
        "legacy `use_openclaw_voice_backend` flag also defaults to `False` but is ignored" in claude
    )
    assert "connection/auth/429/5xx failures fall back locally with a structured warning" in claude


def test_cross_doc_audit_local_links_exist() -> None:
    audit = AUDIT.read_text(encoding="utf-8")
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", audit):
        if "://" in target or target.startswith("#"):
            continue
        path = target.split("#", 1)[0]
        assert (AUDIT.parent / path).resolve().exists(), target


def test_claude_requires_verified_checks_before_merge() -> None:
    claude = _text("CLAUDE.md")
    audit = AUDIT.read_text(encoding="utf-8")
    contributing = _text("CONTRIBUTING.md")
    assert "`master` is not branch-protected" in claude
    assert "Do not use `gh pr merge --auto`" in claude
    assert "all required GitHub checks are green on the exact PR head" in claude
    assert "GitHub reports `master` is not branch-protected" in audit
    assert (
        "Do not merge a PR until every required GitHub check is green on the exact head"
        in contributing
    )
