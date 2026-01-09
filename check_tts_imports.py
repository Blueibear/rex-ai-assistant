"""Check what imports TTS stream_generator.py needs from transformers."""

import sys
from pathlib import Path

try:
    import TTS
    tts_path = Path(TTS.__file__).parent
    stream_gen_file = tts_path / "tts" / "layers" / "xtts" / "stream_generator.py"

    print(f"Reading: {stream_gen_file}\n")

    # Read the file
    with open(stream_gen_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the import section - show first 60 lines to see all imports
    print("First 60 lines of stream_generator.py:")
    print("=" * 70)
    for i, line in enumerate(lines[:60], 1):
        print(f"{i:3d}: {line.rstrip()}")

except ImportError:
    print("ERROR: TTS is not installed")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
