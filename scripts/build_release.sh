#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Cleaning build artifacts..."
rm -rf build dist *.egg-info

if [ -d ".pytest_cache" ]; then
  rm -rf .pytest_cache
fi

if [ -d ".ruff_cache" ]; then
  rm -rf .ruff_cache
fi

echo "Running tests..."
python -m pytest -q

echo "Building wheel..."
python -m pip install --upgrade build >/dev/null
python -m build

DOC_ARCHIVE="dist/rex-docs-$(date +%Y%m%d).tar.gz"

echo "Packaging docs into ${DOC_ARCHIVE}..."
tar -czf "$DOC_ARCHIVE" docs README.md INSTALL.md

echo "Release build complete."
