"""Tests for US-080: Email triage rules engine.

Acceptance criteria verified:
- [x] triage rules stored in config file (JSON or YAML)
- [x] rules support matching on: sender address, subject pattern, body keyword
- [x] rules evaluated in declared priority order; first match wins
- [x] adding or modifying a rule takes effect without restarting or modifying source code
- [x] Typecheck passes (enforced by mypy in CI)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from rex.email_backends.base import EmailEnvelope
from rex.email_backends.triage_rules import (
    TriageRule,
    TriageRulesEngine,
    load_rules_from_file,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 11, 9, 0, 0, tzinfo=timezone.utc)


def _env(
    *,
    sender: str = "sender@example.com",
    subject: str = "Hello",
    snippet: str = "",
) -> EmailEnvelope:
    return EmailEnvelope(
        message_id="test-id",
        from_addr=sender,
        subject=subject,
        snippet=snippet,
        received_at=_NOW,
        to_addrs=["me@example.com"],
        labels=[],
    )


def _write_rules(path: Path, rules: list[dict]) -> None:  # type: ignore[type-arg]
    path.write_text(json.dumps({"rules": rules}), encoding="utf-8")


# ---------------------------------------------------------------------------
# AC1 — rules stored in a JSON config file
# ---------------------------------------------------------------------------


class TestRulesStoredInFile:
    def test_engine_loads_rules_from_json_file(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "triage_rules.json"
        _write_rules(
            rules_file,
            [{"name": "test-urgent", "category": "urgent", "match": {"subject": "fire"}}],
        )
        engine = TriageRulesEngine(rules_file)
        assert len(engine.rules) == 1
        assert engine.rules[0].name == "test-urgent"

    def test_engine_handles_missing_file_gracefully(self, tmp_path: Path) -> None:
        engine = TriageRulesEngine(tmp_path / "nonexistent.json")
        assert engine.rules == []

    def test_load_rules_from_file_helper(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        _write_rules(
            rules_file,
            [{"category": "newsletter", "match": {"sender": "newsletter@"}}],
        )
        rules = load_rules_from_file(rules_file)
        assert len(rules) == 1
        assert rules[0].category == "newsletter"

    def test_invalid_category_raises_value_error(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "bad.json"
        _write_rules(
            rules_file,
            [{"category": "spam", "match": {"subject": "win"}}],
        )
        with pytest.raises(ValueError, match="invalid category"):
            load_rules_from_file(rules_file)

    def test_bundled_config_file_is_valid(self) -> None:
        """The config/triage_rules.json shipped with the repo is valid JSON."""
        repo_rules = Path(__file__).parent.parent / "config" / "triage_rules.json"
        assert repo_rules.exists(), "config/triage_rules.json must exist"
        rules = load_rules_from_file(repo_rules)
        assert len(rules) >= 1

    def test_triage_rule_dataclass_fields(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "r.json"
        _write_rules(
            rules_file,
            [
                {
                    "name": "my-rule",
                    "category": "fyi",
                    "match": {
                        "sender": "bot@",
                        "subject": "update",
                        "body": "see details",
                    },
                }
            ],
        )
        rules = load_rules_from_file(rules_file)
        r = rules[0]
        assert isinstance(r, TriageRule)
        assert r.name == "my-rule"
        assert r.category == "fyi"
        assert r.sender_pattern == "bot@"
        assert r.subject_pattern == "update"
        assert r.body_pattern == "see details"


# ---------------------------------------------------------------------------
# AC2 — rules match on sender, subject, body keyword
# ---------------------------------------------------------------------------


class TestMatchFields:
    def _engine(self, rules: list[dict], tmp_path: Path) -> TriageRulesEngine:  # type: ignore[type-arg]
        p = tmp_path / "rules.json"
        _write_rules(p, rules)
        return TriageRulesEngine(p)

    def test_match_by_sender(self, tmp_path: Path) -> None:
        engine = self._engine([{"category": "urgent", "match": {"sender": "alerts@"}}], tmp_path)
        assert engine.categorize(_env(sender="alerts@example.com")) == "urgent"
        assert engine.categorize(_env(sender="normal@example.com")) is None

    def test_match_by_subject(self, tmp_path: Path) -> None:
        engine = self._engine(
            [{"category": "action_required", "match": {"subject": "invoice"}}],
            tmp_path,
        )
        assert engine.categorize(_env(subject="Invoice #123 due")) == "action_required"
        assert engine.categorize(_env(subject="Hello world")) is None

    def test_match_by_body_keyword(self, tmp_path: Path) -> None:
        engine = self._engine(
            [{"category": "urgent", "match": {"body": "join the incident bridge"}}],
            tmp_path,
        )
        env_match = _env(snippet="Please join the incident bridge now.")
        env_no_match = _env(snippet="Have a nice day.")
        assert engine.categorize(env_match) == "urgent"
        assert engine.categorize(env_no_match) is None

    def test_all_three_fields_must_match(self, tmp_path: Path) -> None:
        engine = self._engine(
            [
                {
                    "category": "urgent",
                    "match": {
                        "sender": "ops@",
                        "subject": "outage",
                        "body": "down",
                    },
                }
            ],
            tmp_path,
        )
        # Only sender matches
        assert engine.categorize(_env(sender="ops@example.com")) is None
        # Sender + subject match but body missing
        assert engine.categorize(_env(sender="ops@example.com", subject="Outage alert")) is None
        # All three match
        assert (
            engine.categorize(
                _env(
                    sender="ops@example.com",
                    subject="Outage alert",
                    snippet="Service is down",
                )
            )
            == "urgent"
        )

    def test_matching_is_case_insensitive(self, tmp_path: Path) -> None:
        engine = self._engine(
            [{"category": "newsletter", "match": {"subject": "newsletter"}}], tmp_path
        )
        assert engine.categorize(_env(subject="NEWSLETTER: March Edition")) == "newsletter"
        assert engine.categorize(_env(subject="Newsletter update")) == "newsletter"

    def test_match_field_is_a_regex(self, tmp_path: Path) -> None:
        engine = self._engine(
            [{"category": "fyi", "match": {"subject": r"\bpr\s+(merged|closed)\b"}}],
            tmp_path,
        )
        assert engine.categorize(_env(subject="Your PR merged")) == "fyi"
        assert engine.categorize(_env(subject="Your PR closed")) == "fyi"
        assert engine.categorize(_env(subject="PR open")) is None

    def test_empty_match_dict_matches_everything(self, tmp_path: Path) -> None:
        engine = self._engine([{"category": "fyi", "match": {}}], tmp_path)
        assert engine.categorize(_env(subject="Anything at all")) == "fyi"

    def test_categorize_returns_none_when_no_rule_fires(self, tmp_path: Path) -> None:
        engine = self._engine([{"category": "urgent", "match": {"subject": "fire"}}], tmp_path)
        assert engine.categorize(_env(subject="Normal email")) is None


# ---------------------------------------------------------------------------
# AC3 — rules evaluated in declared priority order; first match wins
# ---------------------------------------------------------------------------


class TestPriorityOrder:
    def test_first_matching_rule_wins(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        _write_rules(
            rules_file,
            [
                {"name": "rule-a", "category": "urgent", "match": {"subject": "outage"}},
                {"name": "rule-b", "category": "fyi", "match": {"subject": "outage"}},
            ],
        )
        engine = TriageRulesEngine(rules_file)
        # Both rules would match, but rule-a comes first → urgent
        assert engine.categorize(_env(subject="Production outage")) == "urgent"

    def test_second_rule_fires_when_first_does_not_match(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        _write_rules(
            rules_file,
            [
                {"name": "r1", "category": "urgent", "match": {"subject": "fire"}},
                {"name": "r2", "category": "newsletter", "match": {"sender": "news@"}},
            ],
        )
        engine = TriageRulesEngine(rules_file)
        env = _env(sender="news@example.com", subject="Weekly update")
        assert engine.categorize(env) == "newsletter"

    def test_later_rules_not_evaluated_after_first_match(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        # First rule: catch-all (empty match) → fyi
        # Second rule: would match anything → urgent (should never fire)
        _write_rules(
            rules_file,
            [
                {"name": "catch-all", "category": "fyi", "match": {}},
                {"name": "never", "category": "urgent", "match": {}},
            ],
        )
        engine = TriageRulesEngine(rules_file)
        assert engine.categorize(_env(subject="Anything")) == "fyi"

    def test_rules_loaded_in_file_order(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        _write_rules(
            rules_file,
            [
                {"name": "first", "category": "urgent", "match": {"subject": "alpha"}},
                {"name": "second", "category": "newsletter", "match": {"subject": "beta"}},
                {"name": "third", "category": "action_required", "match": {"subject": "gamma"}},
            ],
        )
        engine = TriageRulesEngine(rules_file)
        names = [r.name for r in engine.rules]
        assert names == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# AC4 — changes take effect without restart (auto-reload on mtime change)
# ---------------------------------------------------------------------------


class TestAutoReload:
    def test_reload_method_picks_up_new_rules(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        _write_rules(
            rules_file,
            [{"category": "fyi", "match": {"subject": "hello"}}],
        )
        engine = TriageRulesEngine(rules_file)
        assert engine.categorize(_env(subject="hello")) == "fyi"

        # Modify the file to change the category
        _write_rules(
            rules_file,
            [{"category": "urgent", "match": {"subject": "hello"}}],
        )
        engine.reload()
        assert engine.categorize(_env(subject="hello")) == "urgent"

    def test_auto_reload_on_mtime_change(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        _write_rules(
            rules_file,
            [{"category": "fyi", "match": {"subject": "newsletter"}}],
        )
        engine = TriageRulesEngine(rules_file)
        assert engine.categorize(_env(subject="newsletter")) == "fyi"

        # Overwrite file and bump mtime explicitly
        _write_rules(
            rules_file,
            [{"category": "newsletter", "match": {"subject": "newsletter"}}],
        )
        # Force mtime to differ by setting it 1 second ahead
        stat = rules_file.stat()
        new_mtime = stat.st_mtime + 1
        import os

        os.utime(rules_file, (new_mtime, new_mtime))

        # Next categorize call should auto-reload
        result = engine.categorize(_env(subject="newsletter"))
        assert result == "newsletter"

    def test_adding_rule_to_empty_engine_takes_effect(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps({"rules": []}), encoding="utf-8")
        engine = TriageRulesEngine(rules_file)
        assert engine.categorize(_env(subject="invoice")) is None

        # Add a rule
        _write_rules(
            rules_file,
            [{"category": "action_required", "match": {"subject": "invoice"}}],
        )
        engine.reload()
        assert engine.categorize(_env(subject="Invoice #99")) == "action_required"

    def test_removing_rule_takes_effect_after_reload(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        _write_rules(
            rules_file,
            [{"category": "urgent", "match": {"subject": "fire"}}],
        )
        engine = TriageRulesEngine(rules_file)
        assert engine.categorize(_env(subject="fire")) == "urgent"

        # Remove all rules
        rules_file.write_text(json.dumps({"rules": []}), encoding="utf-8")
        engine.reload()
        assert engine.categorize(_env(subject="fire")) is None

    def test_categorize_with_fallback_uses_builtin_when_no_rule_fires(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        rules_file.write_text(json.dumps({"rules": []}), encoding="utf-8")
        engine = TriageRulesEngine(rules_file)
        # Built-in categorize would assign "urgent" for CRITICAL subject
        result = engine.categorize_with_fallback(_env(subject="CRITICAL: server down"))
        assert result == "urgent"

    def test_categorize_with_fallback_rule_overrides_builtin(self, tmp_path: Path) -> None:
        rules_file = tmp_path / "rules.json"
        # Override: force "CRITICAL" emails to fyi (unusual but valid config)
        _write_rules(
            rules_file,
            [{"category": "fyi", "match": {"subject": "CRITICAL"}}],
        )
        engine = TriageRulesEngine(rules_file)
        # Rule fires → fyi, even though built-in would say urgent
        result = engine.categorize_with_fallback(_env(subject="CRITICAL: server down"))
        assert result == "fyi"
