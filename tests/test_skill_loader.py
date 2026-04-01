"""Unit tests for rex.skills.loader.load_skills_from_directory."""

from __future__ import annotations

from pathlib import Path

from rex.skills.loader import load_skills_from_directory
from rex.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry(tmp_path: Path) -> SkillRegistry:
    return SkillRegistry(skills_path=tmp_path / "skills.json")


def _write_skill(skills_dir: Path, filename: str, content: str) -> Path:
    skills_dir.mkdir(parents=True, exist_ok=True)
    path = skills_dir / filename
    path.write_text(content, encoding="utf-8")
    return path


_VALID_SKILL = """\
SKILL_METADATA = {
    "name": "greet",
    "description": "Say hello",
    "triggers": ["hello", "hi there"],
}

def run(transcript):
    return "Hello!"
"""

_NO_METADATA = """\
def run(transcript):
    return "no metadata here"
"""

_MISSING_TRIGGERS = """\
SKILL_METADATA = {
    "name": "broken",
    "description": "missing triggers",
}

def run(transcript):
    return "oops"
"""

_EMPTY_NAME = """\
SKILL_METADATA = {
    "name": "",
    "description": "empty name",
    "triggers": ["foo"],
}
"""

_EMPTY_TRIGGERS = """\
SKILL_METADATA = {
    "name": "notrigger",
    "description": "no triggers",
    "triggers": [],
}
"""

_SYNTAX_ERROR = """\
SKILL_METADATA = {
    "name": "bad",
    "description": "syntax error below",
    "triggers": ["x"],
}

def run(:
    pass
"""


# ---------------------------------------------------------------------------
# Valid script discovered and registered
# ---------------------------------------------------------------------------


def test_valid_script_is_registered(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "greet.py", _VALID_SKILL)
    reg = _registry(tmp_path)

    result = load_skills_from_directory(reg, skills_dir)

    assert len(result) == 1
    assert result[0].name == "greet"
    assert result[0].trigger_patterns == ["hello", "hi there"]
    assert result[0].handler == str(skills_dir / "greet.py")


def test_metadata_extracted_correctly(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "greet.py", _VALID_SKILL)
    reg = _registry(tmp_path)
    load_skills_from_directory(reg, skills_dir)

    skill = reg.list_skills()[0]
    assert skill.name == "greet"
    assert skill.description == "Say hello"
    assert "hello" in skill.trigger_patterns
    assert skill.enabled is True


# ---------------------------------------------------------------------------
# Invalid / missing metadata skipped
# ---------------------------------------------------------------------------


def test_no_metadata_skipped(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "bare.py", _NO_METADATA)
    reg = _registry(tmp_path)

    result = load_skills_from_directory(reg, skills_dir)

    assert result == []
    assert reg.list_skills() == []


def test_missing_triggers_key_skipped(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "missing.py", _MISSING_TRIGGERS)
    reg = _registry(tmp_path)

    result = load_skills_from_directory(reg, skills_dir)

    assert result == []


def test_empty_name_skipped(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "emptyname.py", _EMPTY_NAME)
    reg = _registry(tmp_path)
    assert load_skills_from_directory(reg, skills_dir) == []


def test_empty_triggers_skipped(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "notrigger.py", _EMPTY_TRIGGERS)
    reg = _registry(tmp_path)
    assert load_skills_from_directory(reg, skills_dir) == []


def test_syntax_error_skipped(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "bad.py", _SYNTAX_ERROR)
    reg = _registry(tmp_path)
    assert load_skills_from_directory(reg, skills_dir) == []


# ---------------------------------------------------------------------------
# Mixed directory: valid and invalid coexist
# ---------------------------------------------------------------------------


def test_mixed_directory_only_valid_registered(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "good.py", _VALID_SKILL)
    _write_skill(skills_dir, "bad.py", _NO_METADATA)
    reg = _registry(tmp_path)

    result = load_skills_from_directory(reg, skills_dir)

    assert len(result) == 1
    assert result[0].name == "greet"


# ---------------------------------------------------------------------------
# Idempotency — repeated scan does not double-register
# ---------------------------------------------------------------------------


def test_repeated_scan_is_idempotent(tmp_path):
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "greet.py", _VALID_SKILL)
    reg = _registry(tmp_path)

    load_skills_from_directory(reg, skills_dir)
    load_skills_from_directory(reg, skills_dir)

    assert len(reg.list_skills()) == 1


# ---------------------------------------------------------------------------
# Missing directory
# ---------------------------------------------------------------------------


def test_missing_directory_returns_empty(tmp_path):
    reg = _registry(tmp_path)
    result = load_skills_from_directory(reg, tmp_path / "nonexistent")
    assert result == []


# ---------------------------------------------------------------------------
# Example skill file is valid
# ---------------------------------------------------------------------------


def test_example_weather_skill_is_valid(tmp_path):
    """The bundled example_weather_skill.py must pass the loader."""
    example = Path("plugins/skills/example_weather_skill.py")
    if not example.exists():
        import pytest

        pytest.skip("example_weather_skill.py not found")

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    import shutil

    shutil.copy(example, skills_dir / example.name)

    reg = _registry(tmp_path)
    result = load_skills_from_directory(reg, skills_dir)

    assert len(result) == 1
    assert result[0].name == "weather"
