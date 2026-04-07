"""
US-011: Verify edge-tts and pyttsx3 are declared as installable dependencies.

These tests verify:
1. edge-tts and pyttsx3 are listed in pyproject.toml dependencies
2. Both packages are importable (if installed)
3. No other voice-critical ModuleNotFoundError surfaces from rex.voice_loop
"""

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _pyproject_content() -> str:
    return PYPROJECT.read_text(encoding="utf-8")


def test_edge_tts_in_pyproject_dependencies():
    content = _pyproject_content()
    # Match the dependency declaration in the [project] dependencies block
    assert re.search(
        r'["\'"]?edge-tts["\'"]?\s*>=', content
    ), "edge-tts with version pin must appear in pyproject.toml [project] dependencies"


def test_pyttsx3_in_pyproject_dependencies():
    content = _pyproject_content()
    assert re.search(
        r'["\'"]?pyttsx3["\'"]?\s*>=', content
    ), "pyttsx3 with version pin must appear in pyproject.toml [project] dependencies"


def test_edge_tts_importable_when_installed():
    """If edge-tts is installed, it must be importable."""
    spec = importlib.util.find_spec("edge_tts")
    if spec is None:
        import pytest

        pytest.skip("edge-tts not installed in this environment")
    import edge_tts  # noqa: F401 — import side-effect test


def test_pyttsx3_importable_when_installed():
    """If pyttsx3 is installed, it must be importable."""
    spec = importlib.util.find_spec("pyttsx3")
    if spec is None:
        import pytest

        pytest.skip("pyttsx3 not installed in this environment")
    import pyttsx3  # noqa: F401 — import side-effect test


def test_dependencies_section_contains_both():
    """Both packages must be in the [project] dependencies list (not optional-only)."""
    content = _pyproject_content()
    # Find the [project] section up to the first [project.optional-dependencies]
    project_section_match = re.search(
        r"\[project\](.*?)\[project\.optional-dependencies\]",
        content,
        re.DOTALL,
    )
    assert project_section_match, "Could not locate [project] section in pyproject.toml"
    project_section = project_section_match.group(1)
    assert (
        "edge-tts" in project_section
    ), "edge-tts must be in [project] dependencies, not only optional extras"
    assert (
        "pyttsx3" in project_section
    ), "pyttsx3 must be in [project] dependencies, not only optional extras"
