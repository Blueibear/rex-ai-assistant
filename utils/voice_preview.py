"""Voice preview utility for TTS testing.

Provides non-blocking audio playback for voice previews in the GUI.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

LOGGER = logging.getLogger(__name__)


def play_audio_file(file_path: Path, on_complete: Optional[callable] = None) -> None:
    """Play an audio file in a background thread.

    Args:
        file_path: Path to audio file to play
        on_complete: Optional callback when playback completes
    """
    def _play():
        try:
            # Try using playsound (simple, cross-platform)
            try:
                import playsound
                playsound.playsound(str(file_path), block=True)
            except ImportError:
                # Fall back to platform-specific methods
                import sys
                if sys.platform == "win32":
                    import winsound
                    winsound.PlaySound(str(file_path), winsound.SND_FILENAME)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["afplay", str(file_path)], check=True)
                else:  # Linux
                    import subprocess
                    subprocess.run(["aplay", str(file_path)], check=True,
                                 stderr=subprocess.DEVNULL)

            if on_complete:
                on_complete()

        except Exception as e:
            LOGGER.error(f"Failed to play audio file {file_path}: {e}")

    thread = threading.Thread(target=_play, daemon=True)
    thread.start()


def generate_and_play_voice_sample(voice_id: str, provider: str, text: str = "Hello, this is a voice preview test.") -> bool:
    """Generate and play a TTS sample for the given voice.

    Args:
        voice_id: Voice identifier
        provider: TTS provider name (pyttsx3, edge, xtts)
        text: Text to speak

    Returns:
        True if playback started successfully, False otherwise
    """
    try:
        if provider == 'pyttsx3':
            return _play_pyttsx3_sample(voice_id, text)
        elif provider == 'edge' or provider == 'edge-tts':
            return _play_edge_sample(voice_id, text)
        elif provider == 'xtts':
            # XTTS voice samples are file-based
            if Path(voice_id).exists():
                play_audio_file(Path(voice_id))
                return True
            return False
        else:
            LOGGER.warning(f"Unknown TTS provider: {provider}")
            return False

    except Exception as e:
        LOGGER.error(f"Failed to generate voice sample: {e}")
        return False


def _play_pyttsx3_sample(voice_id: str, text: str) -> bool:
    """Play a sample using pyttsx3."""
    def _speak():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('voice', voice_id)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            LOGGER.error(f"pyttsx3 playback error: {e}")

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
    return True


def _play_edge_sample(voice_id: str, text: str) -> bool:
    """Play a sample using edge-tts."""
    def _speak():
        try:
            import asyncio
            import edge_tts
            import tempfile
            import os

            async def generate():
                communicate = edge_tts.Communicate(text, voice_id)
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
                    tmp_path = tmp.name

                await communicate.save(tmp_path)
                return tmp_path

            # Generate audio file
            tmp_file = asyncio.run(generate())

            # Play it
            play_audio_file(Path(tmp_file))

            # Clean up after a delay
            def cleanup():
                import time
                time.sleep(5)
                try:
                    os.unlink(tmp_file)
                except Exception:
                    pass

            threading.Thread(target=cleanup, daemon=True).start()

        except Exception as e:
            LOGGER.error(f"edge-tts playback error: {e}")

    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
    return True
