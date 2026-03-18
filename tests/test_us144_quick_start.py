"""Tests for US-144: Quick Start section with exactly 5 steps."""

import re
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def _read_readme() -> str:
    return README.read_text(encoding="utf-8")


def _extract_quick_start(text: str) -> str:
    """Extract everything from ## Quick Start to the next ## heading."""
    match = re.search(r"## Quick Start\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    assert match, "Could not find ## Quick Start section in README"
    return match.group(1)


def test_quick_start_section_exists():
    text = _read_readme()
    assert "## Quick Start" in text


def test_quick_start_is_early_in_file():
    text = _read_readme()
    toc_pos = text.find("## Table of Contents")
    qs_pos = text.find("## Quick Start")
    assert toc_pos != -1, "Table of Contents missing"
    assert qs_pos != -1, "Quick Start missing"
    assert qs_pos > toc_pos, "Quick Start should come after Table of Contents"


def test_quick_start_has_clone_step():
    qs = _extract_quick_start(_read_readme())
    assert "git clone" in qs, "Quick Start must include git clone command"


def test_quick_start_has_install_script_step():
    qs = _extract_quick_start(_read_readme())
    assert "install.sh" in qs or "install.ps1" in qs, "Quick Start must reference install script"


def test_quick_start_has_lm_studio_step():
    qs = _extract_quick_start(_read_readme())
    assert (
        "LM Studio" in qs or "lmstudio" in qs.lower()
    ), "Quick Start must include LM Studio configuration step"


def test_quick_start_has_run_rex_step():
    qs = _extract_quick_start(_read_readme())
    assert re.search(r"`rex`|rex\b", qs), "Quick Start must include step to run rex"


def test_quick_start_has_verify_step():
    qs = _extract_quick_start(_read_readme())
    assert (
        "verify" in qs.lower() or "doctor" in qs.lower() or "ready" in qs.lower()
    ), "Quick Start must include a verification step"


def test_quick_start_has_exactly_five_numbered_steps():
    qs = _extract_quick_start(_read_readme())
    steps = re.findall(r"^\d+\.", qs, re.MULTILINE)
    assert len(steps) == 5, f"Quick Start must have exactly 5 numbered steps, found {len(steps)}"


def test_quick_start_no_more_than_five_steps():
    qs = _extract_quick_start(_read_readme())
    steps = re.findall(r"^\d+\.", qs, re.MULTILINE)
    assert len(steps) <= 5, f"Quick Start must not exceed 5 steps, found {len(steps)}"


def test_clone_step_has_cd_command():
    qs = _extract_quick_start(_read_readme())
    assert "cd rex-ai-assistant" in qs, "Clone step must include cd into the directory"


def test_install_step_has_exact_bash_command():
    qs = _extract_quick_start(_read_readme())
    assert "bash install.sh" in qs, "Install step must show exact bash command"


def test_install_step_has_exact_powershell_command():
    qs = _extract_quick_start(_read_readme())
    assert (
        r".\install.ps1" in qs or ".\\install.ps1" in qs
    ), "Install step must show exact PowerShell command"


def test_lm_studio_step_has_url():
    qs = _extract_quick_start(_read_readme())
    assert "localhost:1234" in qs, "LM Studio step must include the local server URL localhost:1234"


def test_verify_step_has_doctor_command():
    qs = _extract_quick_start(_read_readme())
    assert "doctor.py" in qs or "rex doctor" in qs, "Verify step must include the doctor command"


def test_no_external_links_required_in_quick_start():
    """Each step must be actionable without reading another section."""
    qs = _extract_quick_start(_read_readme())
    # The advanced install footnote/link is OK, but steps should not say
    # "see X section" as a prerequisite
    lines = qs.splitlines()
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+\.", stripped):
            assert (
                "see " not in stripped.lower() or "docs/" in stripped.lower()
            ), f"Step line must not redirect to another section: {stripped}"


def test_quick_start_steps_cover_all_required_topics():
    qs = _extract_quick_start(_read_readme())
    required = ["clone", "install", "LM Studio", "rex", "verify"]
    for topic in required:
        assert topic.lower() in qs.lower(), f"Quick Start must cover topic: {topic}"


def test_readme_quick_start_link_in_toc():
    text = _read_readme()
    assert (
        "[Quick Start]" in text or "[quick-start]" in text.lower()
    ), "README Table of Contents must link to Quick Start"
