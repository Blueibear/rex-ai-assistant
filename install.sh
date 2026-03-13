#!/usr/bin/env bash
# install.sh — single-command Rex installer for Linux and macOS
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="${REX_PYTHON:-python3}"

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

# Verify Python is available
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    fail "Python not found. Install Python 3.10+ and ensure it is on your PATH."
fi

# Require Python 3.9+
PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) \
    || fail "Could not determine Python version."
MAJOR="${PYTHON_VERSION%%.*}"
MINOR="${PYTHON_VERSION#*.}"
if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 9 ]; }; then
    fail "Python 3.9 or newer is required (found $PYTHON_VERSION)."
fi

echo "Creating virtual environment in $VENV_DIR ..."
"$PYTHON" -m venv "$VENV_DIR" || fail "Failed to create virtual environment."

PIP="$VENV_DIR/bin/pip"
REX="$VENV_DIR/bin/rex"

echo "Upgrading pip ..."
"$PIP" install --upgrade pip setuptools wheel >/dev/null \
    || fail "Failed to upgrade pip."

echo "Installing Rex with all dependencies ..."
"$PIP" install "$REPO_DIR[full]" \
    || fail "pip install failed. Check the error above and re-run after resolving it."

echo "Verifying install ..."
if ! "$REX" --help >/dev/null 2>&1; then
    fail "Rex was installed but the 'rex' command did not respond. Check the install log above."
fi

echo ""
echo "Rex is installed. Run \`rex\` to start."
echo ""
echo "To activate the virtual environment manually:"
echo "  source $VENV_DIR/bin/activate"
