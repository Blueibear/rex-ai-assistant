"""Unit tests for rex.skills.registry.SkillRegistry."""

from __future__ import annotations

import json

from rex.skills.registry import Skill, SkillRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry(tmp_path) -> SkillRegistry:
    return SkillRegistry(skills_path=tmp_path / "skills.json")


def _add_skill(reg: SkillRegistry, name: str = "test-skill") -> Skill:
    return reg.register(
        name=name,
        description="A test skill",
        trigger_patterns=["test", "demo"],
        handler="plugins.skills.test_skill:run",
    )


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------


def test_register_creates_skill(tmp_path):
    reg = _registry(tmp_path)
    skill = _add_skill(reg)
    assert skill.id
    assert skill.name == "test-skill"
    assert skill.enabled is True
    assert "test" in skill.trigger_patterns


def test_register_raises_on_empty_name(tmp_path):
    reg = _registry(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="name"):
        reg.register(
            name="  ",
            description="x",
            trigger_patterns=["x"],
            handler="x",
        )


def test_register_raises_on_no_triggers(tmp_path):
    reg = _registry(tmp_path)
    import pytest

    with pytest.raises(ValueError, match="trigger"):
        reg.register(
            name="foo",
            description="x",
            trigger_patterns=[],
            handler="x",
        )


# ---------------------------------------------------------------------------
# list_skills()
# ---------------------------------------------------------------------------


def test_list_skills_returns_all(tmp_path):
    reg = _registry(tmp_path)
    _add_skill(reg, "s1")
    _add_skill(reg, "s2")
    assert len(reg.list_skills()) == 2


def test_list_skills_include_disabled_false(tmp_path):
    reg = _registry(tmp_path)
    s1 = _add_skill(reg, "s1")
    _add_skill(reg, "s2")
    reg.disable(s1.id)
    active = reg.list_skills(include_disabled=False)
    assert len(active) == 1
    assert active[0].name == "s2"


# ---------------------------------------------------------------------------
# enable() / disable()
# ---------------------------------------------------------------------------


def test_enable_disable_round_trip(tmp_path):
    reg = _registry(tmp_path)
    skill = _add_skill(reg)
    assert skill.enabled is True

    reg.disable(skill.id)
    assert reg.get(skill.id).enabled is False

    reg.enable(skill.id)
    assert reg.get(skill.id).enabled is True


def test_enable_unknown_id_returns_false(tmp_path):
    reg = _registry(tmp_path)
    assert reg.enable("no-such-id") is False


def test_disable_unknown_id_returns_false(tmp_path):
    reg = _registry(tmp_path)
    assert reg.disable("no-such-id") is False


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------


def test_delete_removes_skill(tmp_path):
    reg = _registry(tmp_path)
    skill = _add_skill(reg)
    assert reg.delete(skill.id) is True
    assert reg.get(skill.id) is None
    assert len(reg.list_skills()) == 0


def test_delete_unknown_id_returns_false(tmp_path):
    reg = _registry(tmp_path)
    assert reg.delete("no-such-id") is False


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_skills_persisted_to_json(tmp_path):
    reg = _registry(tmp_path)
    skill = _add_skill(reg)

    # Read the file directly.
    data = json.loads((tmp_path / "skills.json").read_text())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == skill.id
    assert data[0]["name"] == "test-skill"


def test_skills_loaded_from_existing_file(tmp_path):
    # Write a skills file, then open a new registry instance.
    reg1 = _registry(tmp_path)
    skill = _add_skill(reg1)

    reg2 = _registry(tmp_path)
    loaded = reg2.get(skill.id)
    assert loaded is not None
    assert loaded.name == skill.name
    assert loaded.trigger_patterns == skill.trigger_patterns


def test_delete_persisted(tmp_path):
    reg1 = _registry(tmp_path)
    skill = _add_skill(reg1)
    reg1.delete(skill.id)

    reg2 = _registry(tmp_path)
    assert len(reg2.list_skills()) == 0


def test_missing_file_gives_empty_registry(tmp_path):
    reg = _registry(tmp_path)
    assert reg.list_skills() == []
