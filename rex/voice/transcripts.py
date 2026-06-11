"""Transcript guards, text normalization, and sentence splitting helpers — extracted verbatim from ``rex/voice_loop.py`` (US-REM-028)."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from importlib.util import find_spec

from rex.voice.optional_imports import (
    _import_optional,
)


def _vl():
    """Return the ``rex.voice_loop`` facade module at call time.

    ``rex.voice_loop`` remains the single patch point for settings, lazy
    importers, audio helpers, and pipeline classes (tests monkeypatch
    ``rex.voice_loop.<name>``). Resolving through the facade at call time
    preserves that behavior without an import cycle at module load time.
    """
    import importlib

    return importlib.import_module("rex.voice_loop")


# Sentence-boundary pattern for streaming TTS sentence splitting.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# Short phrase used to pre-warm the TTS engine on startup.
_WARMUP_PHRASE = "."

# Common single-word abbreviations that should not trigger sentence boundaries.
# Matched as whole words (case-insensitive) followed by "." and whitespace.
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

# Abbreviations containing internal dots (e.g., i.e.) followed by "." and whitespace.
_ABBREV_DOT: frozenset[str] = frozenset(["e.g", "i.e", "a.m", "p.m", "u.s", "u.k", "u.n"])

# Placeholder character used to protect abbreviation periods during splitting.
_ABBREV_PLACEHOLDER = "\x00"

_LOW_VALUE_TRANSCRIPT_WORDS: frozenset[str] = frozenset(
    {
        "ah",
        "alright",
        "annably",
        "good",
        "hm",
        "hmm",
        "much",
        "nope",
        "nowey",
        "ok",
        "okay",
        "please",
        "thanks",
        "thank",
        "uh",
        "um",
        "very",
        "yeah",
        "yep",
        "yes",
        "you",
    }
)
_LOW_VALUE_TRANSCRIPT_EXACT: frozenset[str] = frozenset(
    {
        "alright",
        "good",
        "ok",
        "okay",
        "thank you",
        "thank you very much",
        "thanks",
        "thanks a lot",
        "you are welcome",
        "youre welcome",
        "you're welcome",
    }
)
_LOW_VALUE_TRANSCRIPT_PHRASES: tuple[str, ...] = (
    "thanks for watching",
    "thank you for watching",
    "if there is anything else",
    "if there's anything else",
)
_WEAK_TRANSCRIPT_WORDS: frozenset[str] = frozenset(
    {
        "again",
        "eh",
        "huh",
        "hm",
        "hmm",
        "pardon",
        "repeat",
        "sorry",
        "uh",
        "um",
        "what",
    }
)
_WEAK_TRANSCRIPT_EXACT: frozenset[str] = frozenset(
    {
        "again",
        "eh",
        "huh",
        "hm",
        "hmm",
        "pardon",
        "repeat",
        "sorry",
        "uh",
        "um",
        "what",
        "what was that",
    }
)
_WEAK_TRANSCRIPT_RETRY_PROMPT = "I only caught part of that. Please repeat the question."
_SUSPICIOUS_TRANSCRIPT_RETRY_PROMPT = "I may have misheard that. What did you need?"
_SUSPICIOUS_NEED_LEAD_WORDS: frozenset[str] = frozenset({"neutral"})
_SUSPICIOUS_NEED_TOKENS: frozenset[str] = frozenset({"kate"})
_CLARIFICATION_REPLY_MARKERS: tuple[str, ...] = (
    "could you clarify",
    "can you clarify",
    "please clarify",
    "what do you need",
    "what would you like",
    "what kind",
    "which one",
    "which",
    "can you tell me more",
    "please repeat",
)
_ACTION_TRANSCRIPT_RE = re.compile(
    r"\b(?:"
    r"alarm|battery|brightness|calendar|close|cpu|date|disk|email|"
    r"find|forecast|google|launch|light|lights|look|memory|message|"
    r"open|pause|play|power|remind|resume|run|search|send|set|skip|"
    r"sms|start|stop|temperature|text|time|timer|turn|volume|weather"
    r")\b",
    re.IGNORECASE,
)
_TRANSCRIPT_WORD_RE = re.compile(r"[a-z0-9']+")
_WAKE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:hey|hi)\s+)?(?:jarvis|rex)(?:[\s,;:.-]+|$)",
    re.IGNORECASE,
)
_MIN_WAKE_PREROLL_SOURCE_SECONDS = 0.2
_COMMAND_CAPTURE_CHUNK_SECONDS = 0.25
_COMMAND_CAPTURE_MIN_SECONDS = 3.0
_COMMAND_CAPTURE_MAX_SECONDS = 10.0
_COMMAND_CAPTURE_END_SILENCE_SECONDS = 0.9
_COMMAND_CAPTURE_RMS_THRESHOLD = 0.003
_DEFAULT_STT_INITIAL_PROMPT = (
    "The audio is an English voice command to Rex, a home assistant. "
    "It may ask for time, date, weather, recipes, reminders, smart home control, "
    "or general help."
)


def _protect_abbreviations(text: str) -> str:
    """Replace trailing periods in known abbreviations with a placeholder.

    This prevents *_SENTENCE_BOUNDARY* from treating abbreviations like
    "Dr.", "Mr.", or "e.g." as sentence-ending punctuation.  Original
    casing is preserved via a capturing group in each substitution.
    """
    protected = text
    # Single-word abbreviations: word-boundary + abbr + "." + whitespace.
    # Group 1 captures the original-cased abbreviation so it is preserved.
    for abbr in _ABBREV_WORDS:
        protected = re.sub(
            rf"(?<!\w)({re.escape(abbr)})\.\s",
            r"\1" + _ABBREV_PLACEHOLDER + " ",
            protected,
            flags=re.IGNORECASE,
        )
    # Dot-internal abbreviations: abbr + "." + whitespace
    for abbr in _ABBREV_DOT:
        protected = re.sub(
            rf"({re.escape(abbr)})\.\s",
            r"\1" + _ABBREV_PLACEHOLDER + " ",
            protected,
            flags=re.IGNORECASE,
        )
    return protected


def _normalize_transcript_for_guard(text: str) -> str:
    text = text.lower().replace("’", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return " ".join(text.split())


def _is_low_value_transcript(transcript: str) -> bool:
    """Return True for likely Whisper filler/hallucination with no user command."""
    normalized = _normalize_transcript_for_guard(transcript)
    if not normalized:
        return True

    if _ACTION_TRANSCRIPT_RE.search(normalized):
        return False

    if normalized in _LOW_VALUE_TRANSCRIPT_EXACT:
        return True

    if any(phrase in normalized for phrase in _LOW_VALUE_TRANSCRIPT_PHRASES):
        return True

    if normalized.count("thank you") >= 2:
        return True

    words = _TRANSCRIPT_WORD_RE.findall(normalized)
    if not words:
        return True

    low_value_words = sum(1 for word in words if word in _LOW_VALUE_TRANSCRIPT_WORDS)
    if len(words) <= 4 and low_value_words == len(words):
        return True

    return len(words) >= 8 and low_value_words / len(words) >= 0.45


def _is_weak_transcript_fragment(transcript: str) -> bool:
    """Return True for fragments too ambiguous to route to the assistant."""
    normalized = _normalize_transcript_for_guard(transcript)
    if not normalized:
        return True

    if _ACTION_TRANSCRIPT_RE.search(normalized):
        return False

    if normalized in _WEAK_TRANSCRIPT_EXACT:
        return True

    words = _TRANSCRIPT_WORD_RE.findall(normalized)
    if not words:
        return True

    return len(words) <= 2 and all(word in _WEAK_TRANSCRIPT_WORDS for word in words)


def _is_suspicious_voice_transcript(transcript: str) -> bool:
    """Return True for plausible-looking ASR corruption that needs confirmation."""
    normalized = _normalize_transcript_for_guard(transcript)
    if not normalized:
        return False

    words = _TRANSCRIPT_WORD_RE.findall(normalized)
    if not words:
        return False

    if len(words) <= 6 and words[0] in _SUSPICIOUS_NEED_LEAD_WORDS and "need" in words:
        return True

    return (
        len(words) <= 8
        and "need" in words
        and bool(_SUSPICIOUS_NEED_TOKENS.intersection(words))
        and "recipe" not in words
    )


def _looks_like_clarification_reply(reply: str, transcript: str) -> bool:
    normalized_reply = _normalize_transcript_for_guard(reply)
    if any(marker in normalized_reply for marker in _CLARIFICATION_REPLY_MARKERS):
        return True

    transcript_words = _TRANSCRIPT_WORD_RE.findall(_normalize_transcript_for_guard(transcript))
    return len(transcript_words) <= 3 and reply.strip().endswith("?")


def _combine_followup_transcript(first: str, followup: str) -> str:
    return f"{first.rstrip(' .?!')} {followup.lstrip()}".strip()


def _strip_wake_prefix(transcript: str) -> str:
    """Remove a wake phrase that leaked into STT from wake-frame pre-roll."""
    stripped = transcript.strip()
    return _WAKE_PREFIX_RE.sub("", stripped).strip()


def _split_into_sentences(text: str) -> list[str]:
    """Split *text* into sentence-sized chunks for streaming TTS.

    Uses NLTK ``sent_tokenize`` when available; otherwise falls back to an
    abbreviation-aware regex splitter that does not break on common titles
    (Dr., Mr.) or abbreviations (e.g., etc.).
    """
    stripped = text.strip()
    if not stripped:
        return []

    # Try NLTK sent_tokenize first (handles abbreviations natively).
    if find_spec("nltk") is not None:
        try:
            nltk = _import_optional("nltk")
            if nltk is None:
                raise ImportError("nltk is not available")

            sentences = nltk.sent_tokenize(stripped)
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # punkt tokenizer not downloaded or other NLTK error — fall through.
            pass

    # Abbreviation-aware regex fallback.
    protected = _protect_abbreviations(stripped)
    parts = _SENTENCE_BOUNDARY.split(protected)
    return [s.replace(_ABBREV_PLACEHOLDER, ".").strip() for s in parts if s.strip()]


async def _sentence_stream(text: str) -> AsyncIterator[str]:
    """Yield sentences from *text* as an async iterator."""
    for sentence in _split_into_sentences(text):
        yield sentence


def _extract_completed_sentences(buffer: str) -> tuple[list[str], str]:
    """Return completed sentences and the remaining partial buffer."""
    protected = _protect_abbreviations(buffer)
    matches = list(_SENTENCE_BOUNDARY.finditer(protected))
    if not matches:
        return [], buffer

    split_index = matches[-1].end()
    completed_text = buffer[:split_index]
    remainder = buffer[split_index:]
    return _split_into_sentences(completed_text), remainder


async def _sentence_buffer_stream(tokens: AsyncIterator[str]) -> AsyncIterator[str]:
    """Convert a token stream into sentence chunks for streaming TTS."""
    buffer = ""
    async for token in tokens:
        if not token:
            continue
        buffer += token
        sentences, buffer = _extract_completed_sentences(buffer)
        for sentence in sentences:
            yield sentence

    for sentence in _split_into_sentences(buffer):
        yield sentence
