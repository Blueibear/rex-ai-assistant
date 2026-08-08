from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPRECATION_HEADER = "\n".join(
    [
        "# DEPRECATED: Use `askrex-gui` (web dashboard) instead.",
        "# This Tkinter launcher will be removed in the next major release.",
        "# See docs/UI_SURFACES.md for the canonical GUI entry point.",
    ]
)


def test_ui_surfaces_doc_exists_with_expected_rows() -> None:
    text = (ROOT / "docs" / "UI_SURFACES.md").read_text(encoding="utf-8")

    assert (
        "| CLI (text chat) | `rex` | **Shippable** | Core text interface; canonical CLI entry point |"
        in text
    )
    assert "| Voice loop | `python rex_loop.py` | **Developer-only** |" in text
    assert (
        "| Electron desktop GUI | Installed AskRex app or `cd gui && npm.cmd run dev` | **Shippable** |"
        in text
    )
    assert (
        "| Python/Flask local API and experimental web dashboard | `rex-gui` | **Developer-only** |"
        in text
    )
    assert "| Shopping PWA | served by `rex` or `rex-gui` | **Archived** |" in text
    assert "| TTS API | `rex-speak-api` | **Developer-only** |" in text
    assert "| OpenClaw tool server | `rex-tool-server` | **Experimental** |" in text
    assert "| Windows computer agent | `rex-agent` | **Developer-only** |" in text
    assert (
        "| Tkinter window (`gui.py`) | `python archived/tkinter_gui/run_gui.py` | **Archived** |"
        in text
    )


def test_readme_points_to_electron_as_canonical_gui() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "current primary GUI" in text
    assert "`rex-gui`" in text
    assert "legacy Tkinter launcher" in text
    assert "run_gui.py" not in text


def test_legacy_tkinter_launchers_are_archived_with_deprecation_header() -> None:
    for relative_path in ("archived/tkinter_gui/run_gui.py", "archived/tkinter_gui/gui.py"):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert text.startswith(DEPRECATION_HEADER)


def test_startup_docs_do_not_reference_run_gui_py() -> None:
    for relative_path in ("README.md", "INSTALL.md"):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "run_gui.py" not in text
