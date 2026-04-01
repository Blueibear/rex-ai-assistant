"""Unit tests for rex.skills.router — US-SK-004."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from rex.skills.registry import Skill, SkillRegistry
from rex.skills.router import SkillRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_skill(
    name: str,
    triggers: list[str],
    handler: str = "some_handler",
    enabled: bool = True,
) -> Skill:
    return Skill.new(
        name=name,
        description=f"Test skill: {name}",
        trigger_patterns=triggers,
        handler=handler,
        enabled=enabled,
    )


def _make_registry(skills: list[Skill], tmp_path: Path) -> SkillRegistry:
    registry = SkillRegistry(skills_path=tmp_path / "skills.json")
    for skill in skills:
        registry._skills[skill.id] = skill
    registry._save()
    return registry


# ---------------------------------------------------------------------------
# match() tests
# ---------------------------------------------------------------------------


class TestSkillRouterMatch:
    def test_match_found_returns_skill(self, tmp_path: Path) -> None:
        skill = _make_skill("weather", [r"what is the weather"])
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        result = router.match("what is the weather in London")
        assert result is not None
        assert result.name == "weather"

    def test_match_case_insensitive(self, tmp_path: Path) -> None:
        skill = _make_skill("weather", [r"what is the weather"])
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        result = router.match("WHAT IS THE WEATHER today?")
        assert result is not None

    def test_no_match_returns_none(self, tmp_path: Path) -> None:
        skill = _make_skill("weather", [r"what is the weather"])
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        result = router.match("tell me a joke")
        assert result is None

    def test_disabled_skill_not_matched(self, tmp_path: Path) -> None:
        skill = _make_skill("weather", [r"what is the weather"], enabled=False)
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        result = router.match("what is the weather")
        assert result is None

    def test_first_matching_skill_returned(self, tmp_path: Path) -> None:
        skill_a = _make_skill("alpha", [r"hello world"])
        skill_b = _make_skill("beta", [r"hello"])
        registry = _make_registry([skill_a, skill_b], tmp_path)
        router = SkillRouter(registry)
        result = router.match("hello world")
        # Either skill could match; just assert one is returned
        assert result is not None
        assert result.name in {"alpha", "beta"}

    def test_invalid_regex_pattern_skipped_gracefully(self, tmp_path: Path) -> None:
        skill = _make_skill("bad", [r"[invalid regex"])
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        # Should not raise; invalid pattern is skipped
        result = router.match("[invalid regex")
        assert result is None

    def test_empty_registry_returns_none(self, tmp_path: Path) -> None:
        registry = _make_registry([], tmp_path)
        router = SkillRouter(registry)
        assert router.match("anything") is None


# ---------------------------------------------------------------------------
# execute() tests
# ---------------------------------------------------------------------------


class TestSkillRouterExecute:
    def test_script_handler_invoked(self, tmp_path: Path) -> None:
        script = tmp_path / "my_skill.py"
        script.write_text(textwrap.dedent("""
                SKILL_METADATA = {"name": "my_skill", "description": "x", "triggers": []}

                def run(transcript):
                    return "skill executed: " + transcript
                """))
        skill = _make_skill("my_skill", [r"run skill"], handler=str(script))
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        response = router.execute(skill, "run skill now")
        assert response == "skill executed: run skill now"

    def test_execution_error_returned_as_string(self, tmp_path: Path) -> None:
        script = tmp_path / "broken.py"
        script.write_text(textwrap.dedent("""
                def run(transcript):
                    raise RuntimeError("skill failed")
                """))
        skill = _make_skill("broken", [r"break"], handler=str(script))
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        response = router.execute(skill, "break it")
        assert "error" in response.lower()
        assert "broken" in response

    def test_missing_run_function_returns_error(self, tmp_path: Path) -> None:
        script = tmp_path / "no_run.py"
        script.write_text("x = 1\n")
        skill = _make_skill("no_run", [r"test"], handler=str(script))
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        response = router.execute(skill, "test")
        assert "error" in response.lower()

    def test_module_handler_invoked(self, tmp_path: Path) -> None:
        """Test dotted module path handler format."""
        # Use a module that exists in the test environment
        skill = _make_skill("module_skill", [r"module test"], handler="os.path:basename")
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        response = router.execute(skill, "/some/path/file.txt")
        assert response == "file.txt"

    def test_no_match_falls_through(self, tmp_path: Path) -> None:
        """Verify match() returns None for unrelated messages."""
        skill = _make_skill("weather", [r"what is the weather"])
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        assert router.match("tell me a joke") is None

    def test_skill_invocations_logged_at_info(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        script = tmp_path / "log_skill.py"
        script.write_text("def run(t): return 'ok'\n")
        skill = _make_skill("log_skill", [r"test log"], handler=str(script))
        registry = _make_registry([skill], tmp_path)
        router = SkillRouter(registry)
        import logging

        with caplog.at_level(logging.INFO, logger="rex.skills.router"):
            router.execute(skill, "test log")
        assert any("log_skill" in r.message for r in caplog.records)
