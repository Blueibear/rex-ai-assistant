"""
US-146: Add visual structure to the README (badges, section dividers, call-outs)

Acceptance criteria:
  - repo status badges added (CI status, Python version, license)
  - each major section clearly headed with a level-2 heading
  - important warnings or prerequisites use a blockquote or note callout, not inline text
  - README renders correctly on GitHub (no broken markdown)
  - Typecheck passes
"""

import re
from pathlib import Path

README = Path(__file__).parent.parent / "README.md"


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC1: repo status badges added (CI status, Python version, license)
# ---------------------------------------------------------------------------


def test_ci_badge_present():
    text = _readme_text()
    assert "ci.yml/badge.svg" in text, "CI status badge not found"


def test_python_version_badge_present():
    text = _readme_text()
    assert "python" in text.lower() and "badge" in text.lower(), "Python version badge not found"
    # more specific check
    assert (
        "python-3.9" in text
        or "python%203.9" in text
        or "python-3.9%2B" in text
        or "python-3.11" in text
        or "python-3.11%2B" in text
    ), "Python version badge must reference a supported Python version"


def test_license_badge_present():
    text = _readme_text()
    # Must have a license badge (shields.io or similar)
    assert "license" in text.lower(), "License badge/text not found"
    assert "MIT" in text, "License badge must reference MIT"
    # Specifically a badge image for license
    assert re.search(
        r"badge.*license|license.*badge", text, re.IGNORECASE
    ), "License badge (img with 'license' or 'badge') not found"


def test_badges_appear_near_top():
    """Badges should appear within the first 15 lines."""
    lines = _readme_text().splitlines()
    badge_line = None
    for i, line in enumerate(lines[:15]):
        if "badge.svg" in line or "shields.io/badge" in line:
            badge_line = i
            break
    assert badge_line is not None, "Badges not found within first 15 lines of README"


# ---------------------------------------------------------------------------
# AC2: each major section clearly headed with a level-2 heading
# ---------------------------------------------------------------------------

EXPECTED_SECTIONS = [
    "Quick Start",
    "Features",
    "Requirements",
    "Development",
    "Contributing",
    "License",
]


def test_major_sections_use_h2_headings():
    text = _readme_text()
    h2_headings = re.findall(r"^## (.+)", text, re.MULTILINE)
    heading_text = " ".join(h2_headings).lower()
    for section in EXPECTED_SECTIONS:
        assert (
            section.lower() in heading_text
        ), f"Expected level-2 heading for '{section}' not found"


def test_no_major_section_uses_h1():
    """Only the title should be h1; major sections must be h2 or deeper."""
    lines = _readme_text().splitlines()
    h1_lines = [ln for ln in lines if re.match(r"^# [^#]", ln)]
    # Only the title line should be h1
    assert (
        len(h1_lines) == 1
    ), f"Expected exactly 1 h1 heading (the title), found {len(h1_lines)}: {h1_lines}"


def test_sections_in_table_of_contents():
    """TOC entries should link to level-2 headings."""
    text = _readme_text()
    toc_links = re.findall(r"\[.+?\]\(#.+?\)", text)
    assert (
        len(toc_links) >= 5
    ), f"Table of Contents should have at least 5 links, found {len(toc_links)}"


# ---------------------------------------------------------------------------
# AC3: important warnings/prerequisites use a blockquote or note callout
# ---------------------------------------------------------------------------


def test_windows_note_is_blockquote():
    """The Windows simpleaudio note must be in a blockquote (starts with '>'), not inline text."""
    lines = _readme_text().splitlines()
    windows_note_line = None
    for line in lines:
        if "simpleaudio" in line and "Windows" in line:
            windows_note_line = line
            break
    assert windows_note_line is not None, "Windows simpleaudio note not found in README"
    assert windows_note_line.strip().startswith(
        ">"
    ), f"Windows note must be a blockquote (start with '>'), got: {windows_note_line!r}"


def test_advanced_install_note_is_blockquote():
    """The Advanced / Developer Install note must be in a blockquote."""
    lines = _readme_text().splitlines()
    adv_line = None
    for line in lines:
        if "Advanced" in line and ("Install" in line or "Developer" in line):
            adv_line = line
            break
    assert adv_line is not None, "Advanced/Developer Install note not found"
    assert adv_line.strip().startswith(
        ">"
    ), f"Advanced install note must be a blockquote, got: {adv_line!r}"


def test_no_bare_bold_note_warnings():
    """No important note should be just **Note ...**: in a plain paragraph (not blockquote)."""
    lines = _readme_text().splitlines()
    for line in lines:
        stripped = line.strip()
        # A line that starts with **Note but NOT as a blockquote is a violation
        if re.match(r"^\*\*Note", stripped) and not stripped.startswith(">"):
            raise AssertionError(
                f"Found bare **Note** warning not in a blockquote: {line!r}\n"
                "Convert to '> **Note ...**: ...' format."
            )


# ---------------------------------------------------------------------------
# AC4: README renders correctly on GitHub (basic markdown validity)
# ---------------------------------------------------------------------------


def test_no_unclosed_code_fences():
    """Count of ``` delimiters must be even (every opened fence is closed)."""
    text = _readme_text()
    # Count standalone ``` occurrences (lines that start/end a fence)
    fence_lines = [ln for ln in text.splitlines() if re.match(r"^```", ln.strip())]
    assert (
        len(fence_lines) % 2 == 0
    ), f"Odd number of ``` fence markers ({len(fence_lines)}) — unclosed code block"


def test_no_broken_links_in_toc():
    """TOC anchor links should have matching headings."""
    text = _readme_text()
    toc_anchors = re.findall(r"\(#([^)]+)\)", text)
    headings = re.findall(r"^#{1,6} (.+)", text, re.MULTILINE)

    def to_anchor(heading: str) -> str:
        # Match GitHub's anchor algorithm:
        # 1. lowercase
        # 2. remove chars that are not letters, digits, spaces, or hyphens
        # 3. replace each space with a hyphen (NOT collapsing multiple spaces)
        anchor = heading.lower()
        anchor = re.sub(r"[^a-z0-9 -]", "", anchor)
        anchor = anchor.replace(" ", "-")
        return anchor

    heading_anchors = {to_anchor(h) for h in headings}
    for anchor in toc_anchors:
        assert anchor in heading_anchors, f"TOC link '#{anchor}' does not match any heading anchor"


def test_readme_has_content():
    """README must be non-empty and have substantial content."""
    text = _readme_text()
    assert len(text) > 500, "README is too short to be valid"


def test_title_is_present():
    text = _readme_text()
    first_line = text.strip().splitlines()[0]
    assert first_line.startswith(
        "# "
    ), f"README must start with a level-1 title, got: {first_line!r}"
