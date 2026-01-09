"""Patch Coqui TTS stream_generator.py to fix transformers import compatibility.

This script modifies the installed TTS library file that imports BeamSearchScorer
from transformers, updating it to import from the new location.
"""

import os
import sys
from pathlib import Path


def find_tts_stream_generator():
    """Find the TTS stream_generator.py file in site-packages."""
    try:
        import TTS
        tts_path = Path(TTS.__file__).parent
        stream_gen_file = tts_path / "tts" / "layers" / "xtts" / "stream_generator.py"

        if stream_gen_file.exists():
            return stream_gen_file
        else:
            print(f"ERROR: Could not find stream_generator.py at expected location: {stream_gen_file}")
            return None
    except ImportError:
        print("ERROR: TTS is not installed")
        return None


def patch_stream_generator(file_path):
    """Patch the stream_generator.py file to fix transformers imports."""
    print(f"Patching file: {file_path}")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if "# PATCHED FOR TRANSFORMERS COMPATIBILITY" in content:
        print("✓ File is already patched!")
        return True

    # Find the problematic import line
    old_import = "from transformers import ("

    if old_import not in content:
        print(f"WARNING: Could not find expected import statement in file")
        print("The file may have already been modified or have a different structure")
        return False

    # Create the new import with try/except fallback
    # Each class is in a different module in transformers 4.38+
    new_import = """# PATCHED FOR TRANSFORMERS COMPATIBILITY
# Import each class from its actual location in transformers 4.38+
try:
    from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
    from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
    from transformers.generation.configuration_utils import GenerationConfig
    from transformers.generation.logits_process import LogitsProcessorList
    from transformers.generation.stopping_criteria import StoppingCriteriaList
    from transformers.generation.utils import GenerationMixin
    from transformers.modeling_utils import PreTrainedModel
except (ImportError, AttributeError):
    try:
        # Fallback for older transformers versions (all in generation module)
        from transformers.generation import (
            BeamSearchScorer,
            ConstrainedBeamSearchScorer,
            DisjunctiveConstraint,
            GenerationConfig,
            GenerationMixin,
            LogitsProcessorList,
            PhrasalConstraint,
            StoppingCriteriaList,
        )
        from transformers.modeling_utils import PreTrainedModel
    except (ImportError, AttributeError):
        # Final fallback to main transformers namespace (very old versions)
        from transformers import ("""

    # Replace the import
    content = content.replace(old_import, new_import)

    # Make backup
    backup_path = str(file_path) + ".backup"
    print(f"Creating backup: {backup_path}")
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Write patched content
    print(f"Writing patched file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✓ Successfully patched stream_generator.py!")
    return True


def main():
    print("=" * 70)
    print("TTS Transformers Compatibility Patcher")
    print("=" * 70)

    # Find the file
    print("\n1. Locating TTS stream_generator.py...")
    file_path = find_tts_stream_generator()

    if not file_path:
        print("\n✗ FAILED: Could not locate TTS stream_generator.py")
        print("\nMake sure TTS is installed:")
        print("  pip install TTS")
        return 1

    print(f"   Found: {file_path}")

    # Patch the file
    print("\n2. Patching file...")
    success = patch_stream_generator(file_path)

    if not success:
        print("\n✗ FAILED: Could not patch file")
        return 1

    print("\n" + "=" * 70)
    print("SUCCESS! TTS has been patched for transformers compatibility")
    print("=" * 70)
    print("\nYou can now run: python gui.py")
    print("\nTo restore the original file if needed:")
    print(f"  copy {file_path}.backup {file_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
