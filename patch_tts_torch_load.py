"""Patch Coqui TTS io.py to fix PyTorch 2.6 torch.load compatibility.

PyTorch 2.6 changed torch.load() default from weights_only=False to True.
TTS needs to load model files with custom classes, so we set weights_only=False.
"""

import sys
from pathlib import Path


def find_tts_io_file():
    """Find the TTS io.py file in site-packages."""
    try:
        import TTS
        tts_path = Path(TTS.__file__).parent
        io_file = tts_path / "utils" / "io.py"

        if io_file.exists():
            return io_file
        else:
            print(f"ERROR: Could not find io.py at expected location: {io_file}")
            return None
    except ImportError:
        print("ERROR: TTS is not installed")
        return None


def patch_io_file(file_path):
    """Patch the io.py file to add weights_only=False to torch.load()."""
    print(f"Patching file: {file_path}")

    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if "weights_only=False" in content:
        print("✓ File is already patched!")
        return True

    # Find the torch.load line
    old_line = "    return torch.load(f, map_location=map_location, **kwargs)"

    if old_line not in content:
        print(f"WARNING: Could not find expected torch.load line in file")
        print("The file may have already been modified or have a different structure")
        return False

    # Replace with patched version
    new_line = "    return torch.load(f, map_location=map_location, weights_only=False, **kwargs)"

    content = content.replace(old_line, new_line)

    # Make backup
    backup_path = str(file_path) + ".backup"
    print(f"Creating backup: {backup_path}")

    # Read original file for backup
    with open(file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()

    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(original_content)

    # Write patched content
    print(f"Writing patched file...")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("✓ Successfully patched io.py!")
    return True


def main():
    print("=" * 70)
    print("TTS PyTorch 2.6 torch.load Compatibility Patcher")
    print("=" * 70)

    # Find the file
    print("\n1. Locating TTS io.py...")
    file_path = find_tts_io_file()

    if not file_path:
        print("\n✗ FAILED: Could not locate TTS io.py")
        print("\nMake sure TTS is installed:")
        print("  pip install TTS")
        return 1

    print(f"   Found: {file_path}")

    # Patch the file
    print("\n2. Patching file...")
    success = patch_io_file(file_path)

    if not success:
        print("\n✗ FAILED: Could not patch file")
        return 1

    print("\n" + "=" * 70)
    print("SUCCESS! TTS has been patched for PyTorch 2.6 compatibility")
    print("=" * 70)
    print("\nYou can now run: python gui.py")
    print("\nTo restore the original file if needed:")
    print(f"  copy {file_path}.backup {file_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
