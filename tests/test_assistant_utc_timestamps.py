from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from rex.assistant import Assistant


class CapturingHistoryStore:
    def __init__(self) -> None:
        self.timestamps = []

    def save_turn(self, user_id, role, content, timestamp) -> None:
        self.timestamps.append(timestamp)


def test_record_completion_persists_timezone_aware_utc_timestamp() -> None:
    assistant = Assistant.__new__(Assistant)
    assistant._user_id = "james"
    assistant._history_store = CapturingHistoryStore()
    assistant._histories = {"james": []}
    assistant._history_limit = 10
    assistant._log_turn = lambda *args, **kwargs: None

    Assistant._record_completion(assistant, "hello", "hi", user_id="james")

    assert len(assistant._history_store.timestamps) == 2
    for timestamp in assistant._history_store.timestamps:
        assert timestamp.tzinfo is not None
        assert timestamp.utcoffset() == timedelta(0)


def test_assistant_source_has_no_naive_utcnow_calls() -> None:
    source = Path("rex/assistant.py").read_text(encoding="utf-8")
    assert "datetime.utcnow(" not in source
