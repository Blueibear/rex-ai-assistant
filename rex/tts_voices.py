"""TTS voice enumeration for all supported providers.

Provides list_voices(provider) -> list[dict] so the GUI can populate a
voice selector dropdown for whichever TTS backend is active.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default directory for XTTS speaker WAV files (relative to project root)
_DEFAULT_SPEAKERS_DIR = Path(__file__).resolve().parent.parent / "voices"

# Cache for edge-tts voice list (avoids repeated async network calls)
_edge_tts_cache: Optional[List[Dict]] = None


def list_voices(provider: str, *, speakers_dir: Optional[Path] = None) -> List[Dict]:
    """List available voices for the given TTS provider.

    Args:
        provider: One of 'xtts', 'edge-tts', 'pyttsx3'.
        speakers_dir: Override directory for XTTS speaker WAV files.

    Returns:
        List of dicts with keys: ``id``, ``name``, ``language``, ``gender``
        (``gender`` may be ``None``).  Returns ``[]`` if the provider
        dependencies are missing or the directory is empty.
    """
    p = provider.lower().strip()
    if p == "xtts":
        return _list_xtts_voices(speakers_dir=speakers_dir)
    if p in ("edge-tts", "edge_tts", "edgetts"):
        return _list_edge_tts_voices()
    if p == "pyttsx3":
        return _list_pyttsx3_voices()
    logger.warning("Unknown TTS provider for voice listing: %s", provider)
    return []


def _list_xtts_voices(*, speakers_dir: Optional[Path] = None) -> List[Dict]:
    """List WAV speaker files in the XTTS speakers directory."""
    directory = speakers_dir or _DEFAULT_SPEAKERS_DIR
    if not directory.is_dir():
        logger.debug("XTTS speakers directory not found: %s", directory)
        return []
    voices: List[Dict] = []
    for wav_file in sorted(directory.glob("*.wav")):
        stem = wav_file.stem
        voices.append(
            {
                "id": str(wav_file),
                "name": stem.replace("_", " ").replace("-", " ").title(),
                "language": "en",
                "gender": None,
            }
        )
    return voices


def _list_edge_tts_voices() -> List[Dict]:
    """List voices available from edge-tts (result cached for the session)."""
    global _edge_tts_cache
    if _edge_tts_cache is not None:
        return _edge_tts_cache

    try:
        import asyncio

        import edge_tts  # type: ignore[import]

        async def _fetch() -> List[Dict]:
            raw = await edge_tts.list_voices()
            return [
                {
                    "id": v.get("ShortName", ""),
                    "name": v.get("FriendlyName", v.get("ShortName", "")),
                    "language": v.get("Locale", ""),
                    "gender": v.get("Gender"),
                }
                for v in raw
            ]

        _edge_tts_cache = asyncio.run(_fetch())
        return _edge_tts_cache
    except ImportError:
        logger.debug("edge-tts not installed; no voices available")
        return []
    except Exception as exc:
        logger.warning("Failed to list edge-tts voices: %s", exc)
        return []


def _list_pyttsx3_voices() -> List[Dict]:
    """List voices from the pyttsx3 engine."""
    try:
        import pyttsx3  # type: ignore[import]

        engine = pyttsx3.init()
        raw_voices = engine.getProperty("voices") or []
        voices: List[Dict] = []
        for v in raw_voices:
            voice_id = getattr(v, "id", "") or ""
            name = getattr(v, "name", "") or voice_id
            lang = ""
            languages = getattr(v, "languages", None)
            if isinstance(languages, (list, tuple)) and languages:
                lang = str(languages[0])
            voices.append(
                {
                    "id": voice_id,
                    "name": name,
                    "language": lang,
                    "gender": None,
                }
            )
        engine.stop()
        return voices
    except ImportError:
        logger.debug("pyttsx3 not installed; no voices available")
        return []
    except Exception as exc:
        logger.warning("Failed to list pyttsx3 voices: %s", exc)
        return []


def clear_edge_tts_cache() -> None:
    """Clear the cached edge-tts voice list (useful for testing)."""
    global _edge_tts_cache
    _edge_tts_cache = None


__all__ = ["list_voices", "clear_edge_tts_cache"]
