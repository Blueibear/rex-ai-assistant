from pathlib import Path


def test_claude_md_documents_canonical_gui_without_legacy_terms() -> None:
    text = Path("CLAUDE.md").read_text(encoding="utf-8")

    assert (
        "GUI: Web dashboard via `rex.gui_app` (React + Flask). `gui.py` is deprecated."
        in text
    )
    assert "- rex-gui -> rex.gui_app:main" in text
    assert "rex-gui" in text
    assert "run_gui" not in text
    assert "tkinter" not in text
    assert "Tkinter" not in text
