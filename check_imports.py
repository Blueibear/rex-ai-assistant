#!/usr/bin/env python3
"""Lightweight import check to verify module structure without loading heavy dependencies."""

import sys
from pathlib import Path

# Ensure repo is on path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

def check_module_exists(module_path: str) -> bool:
    """Check if a module file exists and has valid syntax."""
    try:
        import py_compile
        py_compile.compile(module_path, doraise=True)
        return True
    except Exception as e:
        print(f"  ✗ Syntax error in {module_path}: {e}")
        return False

def main():
    """Run import structure checks."""
    print("Rex AI Assistant - Import Structure Check")
    print("=" * 60)

    critical_modules = [
        "gui.py",
        "gui_settings_tab.py",
        "utils/env_loader.py",
        "utils/env_schema.py",
        "utils/env_writer.py",
        "utils/tooltips.py",
        "rex/voice_loop.py",
        "rex/voice_loop_optimized.py",
        "rex/mqtt_client.py",
        "rex/mqtt_audio_router.py",
        "rex/config.py",
        "rex/memory.py",
        "rex/assistant.py",
    ]

    all_ok = True
    print("\nChecking critical modules for syntax errors:")
    for module in critical_modules:
        module_path = repo_root / module
        if module_path.exists():
            if check_module_exists(str(module_path)):
                print(f"  ✓ {module}")
            else:
                all_ok = False
        else:
            print(f"  ⚠ {module} - not found")

    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All critical modules have valid syntax")
        print("\nNote: Runtime dependencies (numpy, tkinter, torch, etc.) are")
        print("checked when the application starts. Install requirements:")
        print("  pip install -r requirements.txt")
        return 0
    else:
        print("✗ Some modules have syntax errors")
        return 1

if __name__ == "__main__":
    sys.exit(main())
