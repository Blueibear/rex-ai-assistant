"""Tests for US-144: Getting Started section with exactly 5 steps."""

import re
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def _read_readme() -> str:
    return README.read_text(encoding="utf-8")


def _extract_getting_started(text: str) -> str:
    """Extract everything from ## Getting Started to the next ## heading."""
    match = re.search(r"## Getting Started\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    assert match, "Could not find ## Getting Started section in README"
    return match.group(1)


def test_getting_started_section_exists():
    text = _read_readme()
    assert "## Getting Started" in text


def test_getting_started_is_early_in_file():
    text = _read_readme()
    toc_pos = text.find("## Table of Contents")
    gs_pos = text.find("## Getting Started")
    assert toc_pos != -1, "Table of Contents missing"
    assert gs_pos != -1, "Getting Started missing"
    assert gs_pos > toc_pos, "Getting Started should come after Table of Contents"


def test_getting_started_has_clone_step():
    section = _extract_getting_started(_read_readme())
    assert "git clone" in section, "Getting Started must include git clone command"


def test_getting_started_has_install_script_step():
    section = _extract_getting_started(_read_readme())
    assert (
        "install.sh" in section or "install.ps1" in section
    ), "Getting Started must reference install script"


def test_getting_started_has_lm_studio_step():
    section = _extract_getting_started(_read_readme())
    assert (
        "LM Studio" in section or "lmstudio" in section.lower()
    ), "Getting Started must include LM Studio configuration step"


def test_getting_started_has_run_rex_electron_step():
    section = _extract_getting_started(_read_readme())
    assert re.search(r"`rex`|rex\b", section), "Getting Started must include step to run Rex"
    assert "Electron" in section, "Getting Started must point users to the Electron app"


def test_getting_started_has_verify_step():
    section = _extract_getting_started(_read_readme())
    assert (
        "verify" in section.lower() or "doctor" in section.lower() or "ready" in section.lower()
    ), "Getting Started must include a verification step"


def test_getting_started_has_exactly_five_numbered_steps():
    section = _extract_getting_started(_read_readme())
    steps = re.findall(r"^\d+\.", section, re.MULTILINE)
    assert (
        len(steps) == 5
    ), f"Getting Started must have exactly 5 numbered steps, found {len(steps)}"


def test_getting_started_no_more_than_five_steps():
    section = _extract_getting_started(_read_readme())
    steps = re.findall(r"^\d+\.", section, re.MULTILINE)
    assert len(steps) <= 5, f"Getting Started must not exceed 5 steps, found {len(steps)}"


def test_clone_step_has_cd_command():
    section = _extract_getting_started(_read_readme())
    assert (
        "cd askrex-assistant" in section
        or "cd rex-ai-assistant" in section
        or "cd AskRex-Assistant" in section
    ), "Clone step must include cd into the directory"


def test_install_step_has_exact_bash_command():
    section = _extract_getting_started(_read_readme())
    assert "bash install.sh" in section, "Install step must show exact bash command"


def test_install_step_has_exact_powershell_command():
    section = _extract_getting_started(_read_readme())
    assert (
        r".\install.ps1" in section or ".\\install.ps1" in section
    ), "Install step must show exact PowerShell command"


def test_lm_studio_step_has_url():
    section = _extract_getting_started(_read_readme())
    assert (
        "localhost:1234" in section
    ), "LM Studio step must include the local server URL localhost:1234"


def test_verify_step_has_doctor_command():
    section = _extract_getting_started(_read_readme())
    assert (
        "doctor.py" in section or "rex doctor" in section
    ), "Verify step must include the doctor command"


def test_no_external_links_required_in_getting_started():
    """Each step must be actionable without reading another section."""
    section = _extract_getting_started(_read_readme())
    # The advanced install footnote/link is OK, but steps should not say
    # "see X section" as a prerequisite
    lines = section.splitlines()
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+\.", stripped):
            assert (
                "see " not in stripped.lower() or "docs/" in stripped.lower()
            ), f"Step line must not redirect to another section: {stripped}"


def test_getting_started_steps_cover_all_required_topics():
    section = _extract_getting_started(_read_readme())
    required = ["clone", "install", "LM Studio", "rex", "verify"]
    for topic in required:
        assert topic.lower() in section.lower(), f"Getting Started must cover topic: {topic}"


def test_readme_getting_started_link_in_toc():
    text = _read_readme()
    assert (
        "[Getting Started](#getting-started)" in text
    ), "README Table of Contents must link to Getting Started"
