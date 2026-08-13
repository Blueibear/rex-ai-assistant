"""Conservative validation for terminal assistant output."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

MODEL_FAILURE_MESSAGE = (
    "I couldn't produce a reliable response from the selected model. " "Please try again."
)
_MAX_OUTPUT_CHARS = 120_000
_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
_REPEAT_TOKEN_MIN_COUNT = 200
_REPEAT_TOKEN_DOMINANCE = 0.85
_REPEAT_LINE_MIN_COUNT = 20
_REPEAT_LINE_DOMINANCE = 0.85
_PROVIDER_ERROR_PATTERNS = (
    re.compile(
        r'^\s*error\s+code\s*:\s*(?:4\d\d|5\d\d)\b.*["\']error["\']\s*:',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"^\s*(?:openai|anthropic|ollama|provider|upstream)\s+(?:api\s+)?error\s*:",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*\[Ollama:\s+(?:connection failed\b|model\b.+\bnot found\b|unexpected error:).+\]\s*$",
        re.IGNORECASE | re.DOTALL,
    ),
)


@dataclass(frozen=True, slots=True)
class OutputValidation:
    valid: bool
    reason: str | None = None


def validate_model_output(text: str) -> OutputValidation:
    """Reject only unmistakable provider/model failure shapes."""
    stripped = text.strip()
    if not stripped:
        return OutputValidation(False, "empty_output")
    if stripped.casefold() == "(silence)":
        return OutputValidation(False, "empty_model_content")
    if len(text) > _MAX_OUTPUT_CHARS:
        return OutputValidation(False, "output_length_spike")
    if any(pattern.search(text) for pattern in _PROVIDER_ERROR_PATTERNS):
        return OutputValidation(False, "provider_error_payload")
    if re.search(r"(.)\1{511,}", text, re.DOTALL):
        return OutputValidation(False, "repeated_character_flood")

    if len(text) >= 2_000:
        alnum_ratio = sum(char.isalnum() for char in text) / len(text)
        if alnum_ratio < 0.05:
            return OutputValidation(False, "non_language_symbol_flood")

    tokens = [token.casefold() for token in _TOKEN_RE.findall(text)]
    if len(tokens) >= _REPEAT_TOKEN_MIN_COUNT:
        _token, count = Counter(tokens).most_common(1)[0]
        if count / len(tokens) >= _REPEAT_TOKEN_DOMINANCE:
            return OutputValidation(False, "repeated_token_flood")

    lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
    if len(lines) >= _REPEAT_LINE_MIN_COUNT:
        _line, count = Counter(lines).most_common(1)[0]
        if count >= 18 and count / len(lines) >= _REPEAT_LINE_DOMINANCE:
            return OutputValidation(False, "repeated_line_flood")

    return OutputValidation(True)


__all__ = ["MODEL_FAILURE_MESSAGE", "OutputValidation", "validate_model_output"]
