#!/usr/bin/env python3
"""Scan all rex/ modules for import errors."""
import importlib
import importlib.util
import os
import sys
import traceback

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

# Set testing env var to avoid side effects
os.environ["REX_TESTING"] = "true"

SKIP_MODULES = {
    # These trigger heavy side-effect imports or hardware access
    "rex.computers.agent_server",
}


def find_modules(base_dir):
    modules = []
    for root, dirs, files in os.walk(base_dir):
        # Skip __pycache__
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in files:
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(root, fname)
            # Convert to module path
            rel = os.path.relpath(fpath, repo_root)
            module = rel.replace(os.sep, ".").removesuffix(".py")
            modules.append(module)
    return sorted(modules)


def try_import(module_name):
    try:
        importlib.import_module(module_name)
        return None
    except Exception as e:
        return (type(e).__name__, str(e), traceback.format_exc())


def main():
    rex_dir = os.path.join(repo_root, "rex")
    modules = find_modules(rex_dir)

    failures = []
    successes = []

    for mod in modules:
        if mod in SKIP_MODULES:
            print(f"  SKIP  {mod}")
            continue
        err = try_import(mod)
        if err:
            failures.append((mod, err))
            print(f"  FAIL  {mod}: {err[0]}: {err[1][:80]}")
        else:
            successes.append(mod)
            print(f"  OK    {mod}")

    print(f"\n\n=== SUMMARY ===")
    print(f"Total modules: {len(modules)}")
    print(f"OK: {len(successes)}")
    print(f"FAILED: {len(failures)}")

    if failures:
        print("\n=== FAILURES ===")
        for mod, (etype, emsg, etb) in failures:
            print(f"\n--- {mod} ---")
            print(f"  {etype}: {emsg}")
            print(etb)

    return len(failures)


if __name__ == "__main__":
    sys.exit(main())
