"""Voice discovery utility for TTS engines.

Discovers available voices from different TTS providers:
- pyttsx3 (Windows/macOS/Linux native TTS)
- edge-tts (Microsoft Edge TTS)
- Coqui XTTS (file-based voice cloning)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

LOGGER = logging.getLogger(__name__)


@dataclass
class Voice:
    """Represents a discovered TTS voice."""
    id: str
    name: str
    provider: str
    language: Optional[str] = None
    gender: Optional[str] = None
    sample_path: Optional[Path] = None

    def display_name(self) -> str:
        """Get display name for UI."""
        if self.language:
            return f"{self.name} ({self.language}) - {self.provider}"
        return f"{self.name} - {self.provider}"


def discover_pyttsx3_voices() -> List[Voice]:
    """Discover voices from pyttsx3 (native TTS)."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        engine.stop()

        result = []
        for idx, voice in enumerate(voices):
            # Extract language if available
            lang = None
            if voice.languages and len(voice.languages) > 0:
                lang_code = voice.languages[0]
                if isinstance(lang_code, bytes):
                    lang_code = lang_code.decode('utf-8', errors='ignore')
                lang = str(lang_code)[:2] if lang_code else None

            result.append(Voice(
                id=voice.id,
                name=voice.name,
                provider='pyttsx3',
                language=lang
            ))

        LOGGER.info(f"Discovered {len(result)} pyttsx3 voices")
        return result

    except Exception as e:
        LOGGER.warning(f"Failed to discover pyttsx3 voices: {e}")
        return []


def discover_edge_tts_voices() -> List[Voice]:
    """Discover voices from edge-tts."""
    try:
        import asyncio
        import edge_tts

        async def get_voices():
            voices = await edge_tts.list_voices()
            return voices

        voices = asyncio.run(get_voices())

        result = []
        for voice in voices:
            name = voice.get('ShortName', voice.get('Name', 'Unknown'))
            lang = voice.get('Locale', '')[:2]
            gender = voice.get('Gender', '')

            result.append(Voice(
                id=name,
                name=name,
                provider='edge-tts',
                language=lang,
                gender=gender
            ))

        LOGGER.info(f"Discovered {len(result)} edge-tts voices")
        return result

    except ImportError:
        LOGGER.debug("edge-tts not installed")
        return []
    except Exception as e:
        LOGGER.warning(f"Failed to discover edge-tts voices: {e}")
        return []


def discover_xtts_voices(voices_dir: Optional[Path] = None) -> List[Voice]:
    """Discover voice reference files for XTTS.

    Args:
        voices_dir: Directory containing voice reference files.
                   Defaults to Memory/*/voice.wav

    Returns:
        List of discovered voices
    """
    result = []

    # Default voices directory
    if voices_dir is None:
        voices_dir = Path("Memory")

    if not voices_dir.exists():
        LOGGER.debug(f"Voices directory not found: {voices_dir}")
        return result

    try:
        # Look for voice.wav files in Memory subdirectories
        for user_dir in voices_dir.iterdir():
            if user_dir.is_dir():
                voice_file = user_dir / "voice.wav"
                if voice_file.exists():
                    result.append(Voice(
                        id=str(voice_file),
                        name=user_dir.name,
                        provider='xtts',
                        sample_path=voice_file
                    ))

        # Also look for standalone voice files
        for voice_file in voices_dir.glob("*.wav"):
            if voice_file.stem != "voice":  # Skip if already found above
                result.append(Voice(
                    id=str(voice_file),
                    name=voice_file.stem,
                    provider='xtts',
                    sample_path=voice_file
                ))

        LOGGER.info(f"Discovered {len(result)} XTTS voice references")
        return result

    except Exception as e:
        LOGGER.warning(f"Failed to discover XTTS voices: {e}")
        return []


def discover_all_voices(provider: Optional[str] = None) -> List[Voice]:
    """Discover voices from all available providers.

    Args:
        provider: Specific provider to query ('pyttsx3', 'edge', 'xtts'),
                 or None for all providers.

    Returns:
        List of all discovered voices
    """
    voices = []

    if provider is None or provider == 'pyttsx3':
        voices.extend(discover_pyttsx3_voices())

    if provider is None or provider == 'edge':
        voices.extend(discover_edge_tts_voices())

    if provider is None or provider == 'xtts':
        voices.extend(discover_xtts_voices())

    return voices


def get_voice_dropdown_options(provider: Optional[str] = None) -> List[str]:
    """Get list of voice names for dropdown UI.

    Args:
        provider: TTS provider name

    Returns:
        List of voice display names
    """
    voices = discover_all_voices(provider)
    return [voice.display_name() for voice in voices]
