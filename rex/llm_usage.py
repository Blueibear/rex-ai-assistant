"""LLM usage tracking — append-only JSON log with 10 MB rotation."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from rex.logging_utils import get_logger

logger = get_logger(__name__)

_DEFAULT_PATH = Path("data/llm_usage.json")
_MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def _usage_path() -> Path:
    return Path(os.environ.get("REX_LLM_USAGE_PATH", str(_DEFAULT_PATH)))


def record_usage(
    *,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    timestamp: str | None = None,
) -> None:
    """Append one usage record to the usage log file.

    Args:
        model: Model name (e.g. ``"llama3"``).
        prompt_tokens: Number of prompt/input tokens.
        completion_tokens: Number of completion/output tokens.
        timestamp: ISO-8601 timestamp; defaults to current UTC time.
    """
    path = _usage_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    entry: dict[str, Any] = {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "timestamp": timestamp or datetime.now(UTC).isoformat(),
    }

    # Rotate if file exceeds 10 MB
    if path.exists() and path.stat().st_size >= _MAX_SIZE_BYTES:
        rotated = path.with_suffix(".json.1")
        path.replace(rotated)
        logger.info("LLM usage log rotated to %s", rotated)

    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError as exc:
        logger.warning("Failed to write LLM usage record: %s", exc)


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    """Return all usage records from the log file (empty list if missing)."""
    target = path or _usage_path()
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError as exc:
        logger.warning("Failed to read LLM usage log: %s", exc)
    return records


def summarise(path: Path | None = None) -> dict[str, Any]:
    """Return a usage summary dict.

    Returns::

        {
            "total_requests": int,
            "total_tokens": int,
            "by_model": {
                "<model>": {"requests": int, "prompt_tokens": int,
                             "completion_tokens": int, "total_tokens": int},
                ...
            },
        }
    """
    records = load_records(path)

    total_requests = len(records)
    total_prompt = 0
    total_completion = 0
    by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    )

    for rec in records:
        prompt = int(rec.get("prompt_tokens", 0))
        completion = int(rec.get("completion_tokens", 0))
        model = str(rec.get("model", "unknown"))
        total_prompt += prompt
        total_completion += completion
        by_model[model]["requests"] += 1
        by_model[model]["prompt_tokens"] += prompt
        by_model[model]["completion_tokens"] += completion
        by_model[model]["total_tokens"] += prompt + completion

    return {
        "total_requests": total_requests,
        "total_tokens": total_prompt + total_completion,
        "by_model": dict(by_model),
    }


# ---------------------------------------------------------------------------
# Cloud vs local classification
# ---------------------------------------------------------------------------

_CLOUD_PREFIXES = (
    "gpt-",
    "text-",
    "o1",
    "o3",
    "babbage",
    "davinci",
    "curie",
    "ada",
    "claude-",
    "gemini-",
    "mistral-",
)


def _is_cloud_model(model: str) -> bool:
    """Return True if *model* is a cloud-hosted model (OpenAI, Anthropic, Google, etc.)."""
    if not model:
        return False
    lower = model.lower()
    for prefix in _CLOUD_PREFIXES:
        if lower.startswith(prefix):
            return True
    return False


def _empty_bucket() -> dict[str, int]:
    return {"requests": 0, "tokens": 0}


def _period_start(now: datetime, period: str) -> datetime:
    if period == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if period == "week":
        return (now - timedelta(days=now.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    # month
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def usage_api_summary(path: Path | None = None) -> dict[str, Any]:
    """Return usage split by local/cloud for today, this week, and this month.

    Returns::

        {
            "local": {"requests": int, "tokens": int},
            "cloud": {"requests": int, "tokens": int},
            "by_period": {
                "today": {
                    "local": {"requests": int, "tokens": int},
                    "cloud": {"requests": int, "tokens": int},
                },
                "week": {...},
                "month": {...},
            },
        }
    """
    records = load_records(path)
    now = datetime.now(UTC)

    totals: dict[str, dict[str, int]] = {
        "local": _empty_bucket(),
        "cloud": _empty_bucket(),
    }
    by_period: dict[str, dict[str, dict[str, int]]] = {
        "today": {"local": _empty_bucket(), "cloud": _empty_bucket()},
        "week": {"local": _empty_bucket(), "cloud": _empty_bucket()},
        "month": {"local": _empty_bucket(), "cloud": _empty_bucket()},
    }
    period_starts = {p: _period_start(now, p) for p in ("today", "week", "month")}

    for rec in records:
        model = str(rec.get("model", "unknown"))
        bucket = "cloud" if _is_cloud_model(model) else "local"
        tokens = int(rec.get("prompt_tokens", 0)) + int(rec.get("completion_tokens", 0))

        totals[bucket]["requests"] += 1
        totals[bucket]["tokens"] += tokens

        ts_raw = rec.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(str(ts_raw))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            ts = None

        if ts is not None:
            for period, start in period_starts.items():
                if ts >= start:
                    by_period[period][bucket]["requests"] += 1
                    by_period[period][bucket]["tokens"] += tokens

    return {
        "local": totals["local"],
        "cloud": totals["cloud"],
        "by_period": by_period,
    }
