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
    fail "Python not found. Install Python 3.11 and ensure it is on your PATH."
fi

# Require Python 3.11 exactly for the supported full install path
PYTHON_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) \
    || fail "Could not determine Python version."
MAJOR="${PYTHON_VERSION%%.*}"
MINOR="${PYTHON_VERSION#*.}"
if [ "$MAJOR" -ne 3 ] || [ "$MINOR" -ne 11 ]; then
    fail "Unsupported Python $PYTHON_VERSION for the Rex full install path. Use Python 3.11. Fresh installs on Python 3.13/3.14 are known to fail in the ML/TTS dependency path."
fi

echo "Creating virtual environment in $VENV_DIR ..."
"$PYTHON" -m venv "$VENV_DIR" || fail "Failed to create virtual environment."

PIP="$VENV_DIR/bin/pip"
REX="$VENV_DIR/bin/rex"

echo "Upgrading pip ..."
"$PIP" install --upgrade pip setuptools wheel >/dev/null \
    || fail "Failed to upgrade pip."

echo "Installing Rex with the supported full dependency set ..."
"$PIP" install "$REPO_DIR[full]" \
    || fail "pip install failed. Check the error above and re-run after resolving it."

echo "Bootstrapping default config ..."
ENV_FILE="$REPO_DIR/.env"
ENV_EXAMPLE="$REPO_DIR/.env.example"
if [ ! -f "$ENV_FILE" ] && [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_FILE"
    echo "Created .env from .env.example — edit it to add your API keys."
elif [ ! -f "$ENV_FILE" ]; then
    touch "$ENV_FILE"
    echo "Created empty .env — edit it to add your API keys before running Rex."
else
    echo ".env already exists — skipping."
fi

echo "Running health check ..."
if ! "$REX" doctor 2>&1; then
    echo ""
    echo "WARNING: 'rex doctor' reported one or more issues (see above)."
    echo "Rex is installed but may need additional configuration."
    echo "Edit .env with your API keys and re-run 'rex doctor' to clear warnings."
fi

echo ""
echo "Rex is installed. Run \`rex\` to start."
echo ""
echo "To activate the virtual environment manually:"
echo "  source $VENV_DIR/bin/activate"
