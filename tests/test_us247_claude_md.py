from pathlib import Path


def test_claude_md_documents_canonical_gui_without_legacy_terms() -> None:
    text = Path("CLAUDE.md").read_text(encoding="utf-8")

    assert "GUI: React + Electron under `gui/` is the primary packaged interface." in text
    assert "`rex.gui_app` is a developer-only Flask API/dashboard" in text
    assert "- rex-gui -> rex.gui_app:main" in text
    assert "rex-gui" in text
    assert "run_gui" not in text
    assert "tkinter" not in text
    assert "Archived Tkinter files are unsupported." in text
