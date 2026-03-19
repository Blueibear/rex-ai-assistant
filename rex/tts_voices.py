"""TTS voice enumeration for all supported providers.

Provides list_voices(provider) -> list[dict] so the GUI can populate a
voice selector dropdown for whichever TTS backend is active.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from pathlib import Path

logger = logging.getLogger(__name__)

# Default directory for XTTS speaker WAV files (relative to project root)
_DEFAULT_SPEAKERS_DIR = Path(__file__).resolve().parent.parent / "voices"

# Cache for edge-tts voice list (avoids repeated async network calls)
_edge_tts_cache: list[dict] | None = None


def list_voices(provider: str, *, speakers_dir: Path | None = None) -> list[dict]:
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


def _list_xtts_voices(*, speakers_dir: Path | None = None) -> list[dict]:
    """List WAV speaker files in the XTTS speakers directory."""
    directory = speakers_dir or _DEFAULT_SPEAKERS_DIR
    if not directory.is_dir():
        logger.debug("XTTS speakers directory not found: %s", directory)
        return []
    voices: list[dict] = []
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


def _list_edge_tts_voices() -> list[dict]:
    """List voices available from edge-tts (result cached for the session)."""
    global _edge_tts_cache
    if _edge_tts_cache is not None:
        return _edge_tts_cache

    try:
        import asyncio

        import edge_tts

        async def _fetch() -> list[dict]:
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


def _list_pyttsx3_voices() -> list[dict]:
    """List voices from the pyttsx3 engine."""
    try:
        import pyttsx3

        engine = pyttsx3.init()
        raw_voices = engine.getProperty("voices") or []
        voices: list[dict] = []
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


async def synthesize_sample(
    provider: str,
    voice_id: str,
    text: str = "Hello, I'm Rex.",
) -> bytes:
    """Synthesize a short audio sample for the given voice and provider.

    Args:
        provider: One of 'xtts', 'edge-tts', 'pyttsx3'.
        voice_id: Provider-specific voice identifier returned by ``list_voices``.
        text: Phrase to synthesize (max 50 characters; longer text is truncated).

    Returns:
        Raw audio bytes (WAV for xtts/pyttsx3, MP3 for edge-tts).
        Raises ``RuntimeError`` if synthesis fails or the provider is unsupported.
    """
    # Enforce 50-character limit to keep synthesis fast.
    text = text[:50]

    p = provider.lower().strip()
    if p == "xtts":
        return await _synthesize_xtts(voice_id, text)
    if p in ("edge-tts", "edge_tts", "edgetts"):
        return await _synthesize_edge_tts(voice_id, text)
    if p == "pyttsx3":
        return await _synthesize_pyttsx3(voice_id, text)
    raise RuntimeError(f"Unsupported TTS provider for sample synthesis: {provider}")


async def _synthesize_xtts(voice_id: str, text: str) -> bytes:
    """Synthesize audio using Coqui XTTS with the given speaker WAV file."""
    import asyncio
    import os
    import tempfile

    def _run() -> bytes:
        if find_spec("TTS") is None:
            raise RuntimeError("Coqui TTS is not installed")
        from rex.compat import ensure_transformers_compatibility

        ensure_transformers_compatibility()
        from TTS.api import TTS

        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmpfile = f.name
        try:
            tts.tts_to_file(
                text,
                speaker_wav=voice_id,
                language="en",
                file_path=tmpfile,
            )
            with open(tmpfile, "rb") as fh:
                return fh.read()
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


async def _synthesize_edge_tts(voice_id: str, text: str) -> bytes:
    """Synthesize audio using edge-tts (returns MP3 bytes)."""
    import os
    import tempfile

    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("edge-tts is not installed") from exc

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmpfile = f.name
    try:
        communicate = edge_tts.Communicate(text, voice_id)
        await communicate.save(tmpfile)
        with open(tmpfile, "rb") as fh:
            return fh.read()
    finally:
        if os.path.exists(tmpfile):
            os.unlink(tmpfile)


async def _synthesize_pyttsx3(voice_id: str, text: str) -> bytes:
    """Synthesize audio using pyttsx3 (returns WAV bytes)."""
    import asyncio
    import os
    import tempfile

    def _run() -> bytes:
        try:
            import pyttsx3
        except ImportError as exc:
            raise RuntimeError("pyttsx3 is not installed") from exc

        engine = pyttsx3.init()
        engine.setProperty("voice", voice_id)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmpfile = f.name
        try:
            engine.save_to_file(text, tmpfile)
            engine.runAndWait()
            engine.stop()
            with open(tmpfile, "rb") as fh:
                return fh.read()
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run)


def clear_edge_tts_cache() -> None:
    """Clear the cached edge-tts voice list (useful for testing)."""
    global _edge_tts_cache
    _edge_tts_cache = None


__all__ = ["list_voices", "synthesize_sample", "clear_edge_tts_cache"]
