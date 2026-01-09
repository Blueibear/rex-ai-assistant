"""Check if the TTS patch was applied correctly."""

import sys
from pathlib import Path

try:
    import TTS
    tts_path = Path(TTS.__file__).parent
    stream_gen_file = tts_path / "tts" / "layers" / "xtts" / "stream_generator.py"

    if not stream_gen_file.exists():
        print(f"ERROR: stream_generator.py not found at {stream_gen_file}")
        sys.exit(1)

    print(f"Found stream_generator.py at:")
    print(f"  {stream_gen_file}")
    print()

    # Read the file
    with open(stream_gen_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if patched
    if "# PATCHED FOR TRANSFORMERS COMPATIBILITY" in content:
        print("✓ File shows patch marker")

        # Find the import section
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "# PATCHED FOR TRANSFORMERS COMPATIBILITY" in line:
                print(f"\nPatch section (lines {i+1}-{i+20}):")
                for j in range(i, min(i+20, len(lines))):
                    print(f"  {j+1:3d}: {lines[j]}")
                break
    else:
        print("✗ File is NOT patched (no patch marker found)")

        # Show the import section
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "from transformers import" in line:
                print(f"\nOriginal import section (lines {i+1}-{i+10}):")
                for j in range(i, min(i+10, len(lines))):
                    print(f"  {j+1:3d}: {lines[j]}")
                break

    # Check for .pyc cache
    pycache_dir = stream_gen_file.parent / "__pycache__"
    if pycache_dir.exists():
        pyc_files = list(pycache_dir.glob("stream_generator*.pyc"))
        if pyc_files:
            print(f"\n⚠️  Found {len(pyc_files)} cached .pyc file(s):")
            for pyc in pyc_files:
                print(f"  {pyc}")
            print("\nDeleting cache files...")
            for pyc in pyc_files:
                pyc.unlink()
                print(f"  Deleted: {pyc.name}")
        else:
            print("\n✓ No .pyc cache files found")
    else:
        print("\n✓ No __pycache__ directory")

except ImportError:
    print("ERROR: TTS is not installed")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
