"""Utilities for text-to-speech handling."""

from __future__ import annotations

import re

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Common single-word abbreviations (see rex/voice_loop.py for the canonical list).
_ABBREV_WORDS: frozenset[str] = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "al",
        "st",
        "fig",
        "dept",
        "est",
        "approx",
        "cf",
        "rev",
        "gen",
        "col",
        "lt",
        "sgt",
        "capt",
        "gov",
        "sen",
        "rep",
        "no",
        "vol",
        "ave",
        "blvd",
    ]
)
_ABBREV_DOT: frozenset[str] = frozenset(["e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "u.n"])
_ABBREV_PLACEHOLDER = "\x00"


def _protect_abbreviations(text: str) -> str:
    """Replace trailing periods in known abbreviations with a placeholder.

    Original casing is preserved via a capturing group in each substitution.
    """
    protected = text
    for abbr in _ABBREV_WORDS:
        protected = re.sub(
            rf"(?<!\w)({re.escape(abbr)})\.\s",
            r"\1" + _ABBREV_PLACEHOLDER + " ",
            protected,
            flags=re.IGNORECASE,
        )
    for abbr in _ABBREV_DOT:
        protected = re.sub(
            rf"({re.escape(abbr)})\.\s",
            r"\1" + _ABBREV_PLACEHOLDER + " ",
            protected,
            flags=re.IGNORECASE,
        )
    return protected


_TOKEN_RE = re.compile(r"\S+")


def _count_tokens(text: str) -> int:
    """Approximate token count using whitespace-delimited tokens."""
    return len(_TOKEN_RE.findall(text))


def _split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Split an overlong sentence into max-token chunks."""
    tokens = _TOKEN_RE.findall(sentence)
    if not tokens:
        return []
    if len(tokens) <= max_tokens:
        return [" ".join(tokens)]
    return [" ".join(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def chunk_text_for_xtts(text: str, *, max_tokens: int = 300) -> list[str]:
    """Chunk text into XTTS-safe segments while preserving sentence boundaries.

    XTTS enforces a ~400 token limit. We chunk at 300 tokens to stay within
    the safe margin. Token counting is approximated using whitespace-delimited
    tokens to avoid pulling in a tokenizer dependency.
    """
    if not text:
        return []

    normalized = " ".join(text.strip().split())
    if not normalized:
        return []

    protected = _protect_abbreviations(normalized)
    sentences = [
        sentence.replace(_ABBREV_PLACEHOLDER, ".").strip()
        for sentence in _SENTENCE_SPLIT_RE.split(protected)
        if sentence.strip()
    ]
    if not sentences:
        return [normalized]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_chunks = _split_long_sentence(sentence, max_tokens)
        for sentence_chunk in sentence_chunks:
            sentence_tokens = _count_tokens(sentence_chunk)
            if current_tokens and current_tokens + sentence_tokens > max_tokens:
                chunks.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sentence_chunk)
            current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


__all__ = ["chunk_text_for_xtts"]
