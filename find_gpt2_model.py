"""Find where GPT2InferenceModel is defined in TTS."""

import sys
from pathlib import Path

try:
    import TTS
    tts_path = Path(TTS.__file__).parent

    # Search for GPT2InferenceModel definition
    xtts_dir = tts_path / "tts" / "layers" / "xtts"

    print(f"Searching in: {xtts_dir}\n")

    for py_file in xtts_dir.glob("*.py"):
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
            if "class GPT2InferenceModel" in content:
                print(f"Found GPT2InferenceModel in: {py_file}")

                # Show the class definition
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if "class GPT2InferenceModel" in line:
                        print(f"\nClass definition (lines {i+1}-{i+20}):")
                        for j in range(i, min(i+20, len(lines))):
                            print(f"  {j+1:3d}: {lines[j]}")
                        break

except ImportError:
    print("ERROR: TTS is not installed")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
