"""Tests for rex.skills.trainer — SkillTrainer and detect_skill_creation_intent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from rex.skills.trainer import SkillTrainer, detect_skill_creation_intent

# ---------------------------------------------------------------------------
# detect_skill_creation_intent
# ---------------------------------------------------------------------------


class TestDetectSkillCreationIntent:
    def test_teach_yourself_matches(self):
        result = detect_skill_creation_intent("teach yourself to check the battery")
        assert result == "check the battery"

    def test_learn_how_to_matches(self):
        result = detect_skill_creation_intent("learn how to send reminders")
        assert result == "send reminders"

    def test_add_a_skill_that_matches(self):
        result = detect_skill_creation_intent("add a skill that plays music")
        assert result == "plays music"

    def test_create_a_skill_for_matches(self):
        result = detect_skill_creation_intent("create a skill for weather reports")
        assert result == "weather reports"

    def test_remember_how_to_matches(self):
        result = detect_skill_creation_intent("remember how to greet users")
        assert result == "greet users"

    def test_make_a_skill_to_matches(self):
        result = detect_skill_creation_intent("make a skill to control lights")
        assert result == "control lights"

    def test_trailing_punctuation_stripped(self):
        result = detect_skill_creation_intent("teach yourself to check the battery.")
        assert result == "check the battery"

    def test_case_insensitive(self):
        result = detect_skill_creation_intent("TEACH YOURSELF TO check the battery")
        assert result == "check the battery"

    def test_non_training_message_returns_none(self):
        assert detect_skill_creation_intent("what is the weather today?") is None

    def test_empty_message_returns_none(self):
        assert detect_skill_creation_intent("") is None

    def test_unrelated_message_returns_none(self):
        assert detect_skill_creation_intent("turn on the living room lights") is None


# ---------------------------------------------------------------------------
# SkillTrainer.handle_if_training_request
# ---------------------------------------------------------------------------


class TestSkillTrainerHandleIfTrainingRequest:
    def test_non_training_message_returns_none(self, tmp_path):
        trainer = SkillTrainer(skills_dir=tmp_path)
        registry = MagicMock()
        result = trainer.handle_if_training_request("what time is it?", registry)
        assert result is None
        registry.register.assert_not_called()

    def test_training_message_creates_skill_file(self, tmp_path):
        trainer = SkillTrainer(skills_dir=tmp_path)
        registry = MagicMock()
        registry.register.return_value = MagicMock(name="check the battery", id="abc123")

        result = trainer.handle_if_training_request("teach yourself to check the battery", registry)

        assert result is not None
        assert "check the battery" in result.lower() or "learned" in result.lower()
        skill_files = list(tmp_path.glob("*.py"))
        assert len(skill_files) == 1

    def test_training_message_registers_skill(self, tmp_path):
        trainer = SkillTrainer(skills_dir=tmp_path)
        registry = MagicMock()
        registry.register.return_value = MagicMock(name="send reminders", id="xyz789")

        trainer.handle_if_training_request("learn how to send reminders", registry)

        registry.register.assert_called_once()
        call_kwargs = registry.register.call_args.kwargs
        assert "send reminders" in call_kwargs["name"]

    def test_generated_script_contains_no_todo_implement(self, tmp_path):
        trainer = SkillTrainer(skills_dir=tmp_path)
        registry = MagicMock()
        registry.register.return_value = MagicMock(name="play music", id="m123")

        trainer.handle_if_training_request("make a skill to play music", registry)

        skill_files = list(tmp_path.glob("*.py"))
        assert skill_files, "Expected a generated skill file"
        content = skill_files[0].read_text(encoding="utf-8")
        assert "TODO: implement" not in content

    def test_generated_script_has_honest_return(self, tmp_path):
        trainer = SkillTrainer(skills_dir=tmp_path)
        registry = MagicMock()
        registry.register.return_value = MagicMock(name="play music", id="m123")

        trainer.handle_if_training_request("make a skill to play music", registry)

        skill_files = list(tmp_path.glob("*.py"))
        assert skill_files
        content = skill_files[0].read_text(encoding="utf-8")
        assert "not yet fully implemented" in content

    def test_write_failure_returns_error_message(self, tmp_path):
        trainer = SkillTrainer(skils_dir=tmp_path / "nonexistent" / "nested")
        registry = MagicMock()

        with patch.object(Path, "mkdir", side_effect=PermissionError("no write")):
            result = trainer.handle_if_training_request("teach yourself to do something", registry)

        assert result is not None
        assert (
            "couldn't" in result.lower() or "failed" in result.lower() or "tried" in result.lower()
        )
        registry.register.assert_not_called()

    def test_register_failure_returns_partial_error_message(self, tmp_path):
        trainer = SkillTrainer(skils_dir=tmp_path)
        registry = MagicMock()
        registry.register.side_effect = RuntimeError("registry full")

        result = trainer.handle_if_training_request("teach yourself to do something", registry)

        assert result is not None
        assert (
            "couldn't" in result.lower()
            or "failed" in result.lower()
            or "created" in result.lower()
        )
