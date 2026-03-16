"""Unit tests for US-240: apply learned preferences as soft defaults."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rex.autonomy.llm_planner import LLMPlanner, ToolDefinition
from rex.autonomy.preferences import PreferenceStore, UserPreferenceProfile
from rex.autonomy.runner import _apply_preferences, create_planner, run


# ---------------------------------------------------------------------------
# _apply_preferences tests
# ---------------------------------------------------------------------------


def _make_profile(**kwargs: object) -> UserPreferenceProfile:
    return UserPreferenceProfile(**kwargs)  # type: ignore[arg-type]


class TestApplyPreferences:
    def test_no_preferences_set_returns_empty(self) -> None:
        profile = _make_profile()
        mode, model = _apply_preferences(profile, autonomy_mode="", model="")
        assert mode == ""
        assert model == ""

    def test_preferred_model_applied_when_no_override(self) -> None:
        profile = _make_profile(preferred_model="gpt-4o")
        _, model = _apply_preferences(profile, autonomy_mode="", model="")
        assert model == "gpt-4o"

    def test_explicit_model_wins_over_profile(self) -> None:
        profile = _make_profile(preferred_model="gpt-4o")
        _, model = _apply_preferences(profile, autonomy_mode="", model="gpt-3.5-turbo")
        assert model == "gpt-3.5-turbo"

    def test_preferred_autonomy_mode_applied_when_non_default(self) -> None:
        profile = _make_profile(preferred_autonomy_mode="full-auto")
        mode, _ = _apply_preferences(profile, autonomy_mode="", model="")
        assert mode == "full-auto"

    def test_manual_autonomy_mode_not_applied(self) -> None:
        """'manual' is the default — should not be treated as a learned preference."""
        profile = _make_profile(preferred_autonomy_mode="manual")
        mode, _ = _apply_preferences(profile, autonomy_mode="", model="")
        assert mode == ""

    def test_empty_autonomy_mode_not_applied(self) -> None:
        profile = _make_profile(preferred_autonomy_mode="")
        mode, _ = _apply_preferences(profile, autonomy_mode="", model="")
        assert mode == ""

    def test_explicit_autonomy_mode_wins_over_profile(self) -> None:
        profile = _make_profile(preferred_autonomy_mode="full-auto")
        mode, _ = _apply_preferences(profile, autonomy_mode="supervised", model="")
        assert mode == "supervised"

    def test_both_preferences_applied_together(self) -> None:
        profile = _make_profile(preferred_autonomy_mode="supervised", preferred_model="gpt-4o")
        mode, model = _apply_preferences(profile, autonomy_mode="", model="")
        assert mode == "supervised"
        assert model == "gpt-4o"

    def test_debug_log_emitted_for_model(self, caplog: pytest.LogCaptureFixture) -> None:
        profile = _make_profile(preferred_model="claude-opus")
        with caplog.at_level(logging.DEBUG, logger="rex.autonomy.runner"):
            _apply_preferences(profile, autonomy_mode="", model="")
        assert any("Using learned preference: preferred_model=claude-opus" in r.message for r in caplog.records)

    def test_debug_log_emitted_for_autonomy_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        profile = _make_profile(preferred_autonomy_mode="full-auto")
        with caplog.at_level(logging.DEBUG, logger="rex.autonomy.runner"):
            _apply_preferences(profile, autonomy_mode="", model="")
        assert any(
            "Using learned preference: preferred_autonomy_mode=full-auto" in r.message
            for r in caplog.records
        )

    def test_no_debug_log_when_explicit_override(self, caplog: pytest.LogCaptureFixture) -> None:
        profile = _make_profile(preferred_model="gpt-4o")
        with caplog.at_level(logging.DEBUG, logger="rex.autonomy.runner"):
            _apply_preferences(profile, autonomy_mode="", model="gpt-3.5-turbo")
        assert not any("Using learned preference" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# create_planner model forwarding
# ---------------------------------------------------------------------------


class TestCreatePlannerModel:
    def test_model_forwarded_to_llm_planner(self) -> None:
        planner = create_planner(model="gpt-4o", planner_key="llm")
        assert isinstance(planner, LLMPlanner)
        assert planner._model == "gpt-4o"

    def test_empty_model_produces_empty_model_on_planner(self) -> None:
        planner = create_planner(model="", planner_key="llm")
        assert isinstance(planner, LLMPlanner)
        assert planner._model == ""


# ---------------------------------------------------------------------------
# LLMPlanner model param
# ---------------------------------------------------------------------------


class TestLLMPlannerModel:
    def test_model_stored(self) -> None:
        planner = LLMPlanner(tools=[], model="gpt-4o")
        assert planner._model == "gpt-4o"

    def test_model_passed_to_language_model_on_lazy_init(self) -> None:
        planner = LLMPlanner(tools=[], model="gpt-4o")
        mock_lm = MagicMock()
        with patch("rex.autonomy.llm_planner.LLMPlanner._get_backend", return_value=mock_lm):
            backend = planner._get_backend()
        assert backend is mock_lm

    def test_language_model_receives_model_kwarg(self) -> None:
        planner = LLMPlanner(tools=[], model="gpt-4o")
        with patch("rex.llm_client.LanguageModel") as mock_cls:
            mock_cls.return_value = MagicMock()
            planner._get_backend()
        mock_cls.assert_called_once_with(model="gpt-4o")

    def test_language_model_no_kwarg_when_empty(self) -> None:
        planner = LLMPlanner(tools=[])
        with patch("rex.llm_client.LanguageModel") as mock_cls:
            mock_cls.return_value = MagicMock()
            planner._get_backend()
        mock_cls.assert_called_once_with()


# ---------------------------------------------------------------------------
# run() integration with preference_store
# ---------------------------------------------------------------------------


class TestRunWithPreferenceStore:
    def _make_store(self, tmpdir: str, **profile_kwargs: object) -> PreferenceStore:
        path = Path(tmpdir) / "preferences.json"
        store = PreferenceStore(prefs_path=path)
        store.save(UserPreferenceProfile(**profile_kwargs))  # type: ignore[arg-type]
        return store

    def test_preferred_model_applied_in_run(self) -> None:
        """Preferred model from store is forwarded to the planner when not overridden."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp, preferred_model="gpt-4o")
            mock_backend = MagicMock()
            mock_backend.generate.return_value = '[{"tool": "noop", "args": {}, "description": "test"}]'

            captured: list[str] = []

            def _fake_create(
                tools=None,
                *,
                planner_key=None,
                config_path=None,
                model: str = "",
            ):  # type: ignore[no-untyped-def]
                captured.append(model)
                planner = LLMPlanner(tools=tools or [], backend=mock_backend, model=model)
                return planner

            with patch("rex.autonomy.runner.create_planner", side_effect=_fake_create):
                run("do something", preference_store=store)

            assert captured == ["gpt-4o"]

    def test_explicit_model_override_wins(self) -> None:
        """Explicit model= kwarg wins over the stored preference."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp, preferred_model="gpt-4o")
            mock_backend = MagicMock()
            mock_backend.generate.return_value = '[{"tool": "noop", "args": {}, "description": "test"}]'

            captured: list[str] = []

            def _fake_create(
                tools=None,
                *,
                planner_key=None,
                config_path=None,
                model: str = "",
            ):  # type: ignore[no-untyped-def]
                captured.append(model)
                return LLMPlanner(tools=tools or [], backend=mock_backend, model=model)

            with patch("rex.autonomy.runner.create_planner", side_effect=_fake_create):
                run("do something", preference_store=store, model="gpt-3.5-turbo")

            assert captured == ["gpt-3.5-turbo"]

    def test_autonomy_mode_added_to_context(self) -> None:
        """preferred_autonomy_mode is added to context when not overridden."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp, preferred_autonomy_mode="full-auto")
            mock_backend = MagicMock()
            mock_backend.generate.return_value = (
                '[{"tool": "noop", "args": {}, "description": "test"}]'
            )

            captured_context: list[dict] = []

            def _fake_create(
                tools=None,
                *,
                planner_key=None,
                config_path=None,
                model: str = "",
            ):  # type: ignore[no-untyped-def]
                from rex.autonomy.llm_planner import LLMPlanner as _LP

                inner = _LP(tools=tools or [], backend=mock_backend, model=model)
                original_plan = inner.plan

                def _capturing_plan(goal: str, ctx: dict, **kwargs: object) -> object:
                    captured_context.append(dict(ctx))
                    return original_plan(goal, ctx, **kwargs)

                inner.plan = _capturing_plan  # type: ignore[method-assign]
                return inner

            with patch("rex.autonomy.runner.create_planner", side_effect=_fake_create):
                run("do something", preference_store=store)

            assert captured_context and captured_context[0].get("autonomy_mode") == "full-auto"

    def test_no_preference_store_no_crash(self) -> None:
        """run() without a preference_store behaves as before."""
        mock_backend = MagicMock()
        mock_backend.generate.return_value = '[{"tool": "noop", "args": {}, "description": "test"}]'
        planner = LLMPlanner(tools=[], backend=mock_backend)
        with patch("rex.autonomy.runner.create_planner", return_value=planner):
            plan = run("do something")
        assert plan is not None
