"""Unit tests for rex.integrations.triage_engine — EmailTriageEngine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from rex.integrations.models import EmailMessage
from rex.integrations.triage_engine import EmailTriageEngine, TriageBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(response: str) -> TriageBackend:
    """Return a mock TriageBackend that generates *response*."""
    mock = MagicMock(spec=TriageBackend)
    mock.generate.return_value = response
    return mock  # type: ignore[return-value]


def _make_message(
    id: str = "msg-001",
    subject: str = "Test subject",
    sender: str = "sender@example.com",
    body_text: str = "Hello world",
) -> EmailMessage:
    return EmailMessage(
        id=id,
        thread_id="thread-" + id,
        subject=subject,
        sender=sender,
        recipients=["me@example.com"],
        body_text=body_text,
        date=datetime.now(timezone.utc),
        is_read=False,
        labels=["INBOX"],
    )


# ---------------------------------------------------------------------------
# triage()
# ---------------------------------------------------------------------------


class TestTriage:
    def test_triage_sets_priority_field(self) -> None:
        backend = _make_backend(json.dumps({"priority": "high", "category": "action-required"}))
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        result = engine.triage([msg])
        assert len(result) == 1
        assert result[0].priority == "high"

    def test_triage_returns_new_copy_not_same_object(self) -> None:
        backend = _make_backend(json.dumps({"priority": "low", "category": "newsletter"}))
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        result = engine.triage([msg])
        assert result[0] is not msg

    def test_triage_all_priority_levels(self) -> None:
        for level in ("low", "medium", "high", "critical"):
            backend = _make_backend(json.dumps({"priority": level, "category": "other"}))
            engine = EmailTriageEngine(backend=backend)
            msg = _make_message(id=level)
            result = engine.triage([msg])
            assert result[0].priority == level

    def test_triage_multiple_messages(self) -> None:
        responses = [
            json.dumps({"priority": "high", "category": "action-required"}),
            json.dumps({"priority": "low", "category": "newsletter"}),
        ]
        call_count = 0

        class MultiBackend:
            def generate(self, messages: list[dict[str, str]], *, max_tokens: int = 128) -> str:
                nonlocal call_count
                r = responses[call_count]
                call_count += 1
                return r

        engine = EmailTriageEngine(backend=MultiBackend())
        msgs = [_make_message(id="a"), _make_message(id="b")]
        result = engine.triage(msgs)
        assert result[0].priority == "high"
        assert result[1].priority == "low"


# ---------------------------------------------------------------------------
# triage_with_categories()
# ---------------------------------------------------------------------------


class TestTriageWithCategories:
    def test_returns_category_tuple(self) -> None:
        backend = _make_backend(json.dumps({"priority": "medium", "category": "receipt"}))
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        pairs = engine.triage_with_categories([msg])
        assert len(pairs) == 1
        updated_msg, category = pairs[0]
        assert updated_msg.priority == "medium"
        assert category == "receipt"

    def test_all_valid_categories_accepted(self) -> None:
        valid = [
            "action-required", "newsletter", "receipt", "personal",
            "notification", "social", "promotion", "spam", "update", "other",
        ]
        for cat in valid:
            backend = _make_backend(json.dumps({"priority": "low", "category": cat}))
            engine = EmailTriageEngine(backend=backend)
            msg = _make_message(id=cat)
            pairs = engine.triage_with_categories([msg])
            assert pairs[0][1] == cat


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------


class TestCache:
    def test_cache_prevents_re_scoring(self) -> None:
        backend = _make_backend(json.dumps({"priority": "high", "category": "action-required"}))
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        engine.triage([msg])
        engine.triage([msg])
        # Backend should only have been called once
        assert backend.generate.call_count == 1  # type: ignore[attr-defined]

    def test_clear_cache_forces_re_score(self) -> None:
        backend = _make_backend(json.dumps({"priority": "high", "category": "action-required"}))
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        engine.triage([msg])
        engine.clear_cache()
        engine.triage([msg])
        assert backend.generate.call_count == 2  # type: ignore[attr-defined]

    def test_different_ids_scored_separately(self) -> None:
        backend = _make_backend(json.dumps({"priority": "high", "category": "action-required"}))
        engine = EmailTriageEngine(backend=backend)
        msgs = [_make_message(id="x1"), _make_message(id="x2")]
        engine.triage(msgs)
        assert backend.generate.call_count == 2  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fallback / error handling
# ---------------------------------------------------------------------------


class TestFallback:
    def test_invalid_json_falls_back_to_low_other(self) -> None:
        backend = _make_backend("not json at all")
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        result = engine.triage([msg])
        assert result[0].priority == "low"

    def test_markdown_fenced_json_is_parsed(self) -> None:
        fenced = "```json\n{\"priority\": \"critical\", \"category\": \"action-required\"}\n```"
        backend = _make_backend(fenced)
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        result = engine.triage([msg])
        assert result[0].priority == "critical"

    def test_invalid_priority_falls_back_to_low(self) -> None:
        backend = _make_backend(json.dumps({"priority": "urgent", "category": "other"}))
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        result = engine.triage([msg])
        assert result[0].priority == "low"

    def test_invalid_category_falls_back_to_other(self) -> None:
        backend = _make_backend(json.dumps({"priority": "high", "category": "unknown-tag"}))
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        pairs = engine.triage_with_categories([msg])
        assert pairs[0][1] == "other"

    def test_backend_exception_falls_back_to_low_other(self) -> None:
        mock = MagicMock(spec=TriageBackend)
        mock.generate.side_effect = RuntimeError("LLM unavailable")
        engine = EmailTriageEngine(backend=mock)  # type: ignore[arg-type]
        msg = _make_message()
        result = engine.triage([msg])
        assert result[0].priority == "low"

    def test_non_dict_json_falls_back_to_low_other(self) -> None:
        backend = _make_backend(json.dumps(["priority", "high"]))
        engine = EmailTriageEngine(backend=backend)
        msg = _make_message()
        result = engine.triage([msg])
        assert result[0].priority == "low"
