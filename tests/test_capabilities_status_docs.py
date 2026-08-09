from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_has_linked_capabilities_status_table() -> None:
    readme = _text("README.md")
    assert "## Capabilities & Status" in readme
    required_rows = [
        ("Electron desktop GUI", "`shippable`", "docs/UI_SURFACES.md"),
        ("CLI (`rex`)", "`shippable`", "docs/ARCHITECTURE.md"),
        (
            "Source voice loop / wake word",
            "`developer-only` / beta wake word",
            "docs/voice_pipeline.md",
        ),
        ("Flask/API (`rex-gui`)", "`developer-only`", "docs/UI_SURFACES.md"),
        ("Config CLI (`rex-config`)", "`developer-only`", "docs/configuration.md"),
        ("TTS API (`rex-speak-api`)", "`developer-only`", "docs/api.md"),
        ("Windows agent (`rex-agent`)", "`developer-only`", "docs/computers.md"),
        ("OpenClaw tool server / gateway", "`experimental`", "docs/openclaw-migration-status.md"),
        (
            "Mobile API backend",
            "`developer-only` backend; mobile pre-release",
            "docs/mobile/MOBILE_API_SETUP_WINDOWS.md",
        ),
        ("Docker image", "`developer-only`", "docs/docker.md"),
        ("Home Assistant", "Supported, credential-gated", "docs/home_assistant.md"),
        ("Email", "Partial", "docs/email.md"),
        ("Calendar", "Partial / read-only", "docs/calendar.md"),
        ("SMS / Phone", "Experimental", "docs/messaging.md"),
        ("Legacy Tkinter / Shopping PWA", "`archived`", "docs/UI_SURFACES.md"),
    ]
    for name, status, link in required_rows:
        row = next((line for line in readme.splitlines() if line.startswith(f"| {name} |")), "")
        assert row, name
        assert status in row, row
        assert f"]({link}" in row, row
        assert (ROOT / link.split("#", 1)[0]).exists(), link


def test_ui_surface_summary_uses_canonical_classifications() -> None:
    ui = _text("docs/UI_SURFACES.md")
    expected = {
        "CLI (text chat)": "**Shippable**",
        "Voice loop": "**Developer-only**",
        "Electron desktop GUI": "**Shippable**",
        "Python/Flask local API and experimental web dashboard": "**Developer-only**",
        "Shopping PWA": "**Archived**",
        "TTS API": "**Developer-only**",
        "OpenClaw tool server": "**Experimental**",
        "Windows computer agent": "**Developer-only**",
        "Flask proxy": "**Deprecated**",
        "Tkinter window (`gui.py`)": "**Archived**",
    }
    for surface, status in expected.items():
        row = next((line for line in ui.splitlines() if line.startswith(f"| {surface} |")), "")
        assert row, surface
        assert status in row, row


def test_openclaw_reachability_claims_are_consistent() -> None:
    readme = _text("README.md")
    integrations = _text("INTEGRATIONS_STATUS.md")
    migration = _text("docs/openclaw-migration-status.md")

    assert "health check can prove gateway reachability" in readme
    assert "health check can establish `reachable`" in integrations
    assert "authentication and tool capability" in integrations
    assert "authentication and tool capability are not yet proven" in migration


def test_canonical_docs_cross_link_each_other() -> None:
    readme = _text("README.md")
    assert "[SURFACE-CLASSIFICATION.md](SURFACE-CLASSIFICATION.md)" in readme
    assert "[INTEGRATIONS_STATUS.md](INTEGRATIONS_STATUS.md)" in readme
    assert "[docs/UI_SURFACES.md](docs/UI_SURFACES.md)" in readme
