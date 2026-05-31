#!/usr/bin/env bash
# tests/smoke/test_electron_package.sh
#
# Smoke test: builds the Electron package and verifies the Python bridge is
# reachable from the packaged output (not from the source tree).
#
# What this test does:
#   1. Builds the Electron package via electron-builder --dir.
#   2. Verifies all 20 registered bridge scripts exist in the packaged output.
#   3. Sends a bridge health-check request directly to a packaged bridge script
#      using the Python venv — WITHOUT the source-tree bridge/ on PYTHONPATH.
#      A valid JSON response (ok=true or ok=false) proves the bridge is reachable.
#   4. Launches the packaged Electron app and waits for the bridge validation
#      startup signal logged by validateBridges() in bridgeResolver.ts.
#      This step is best-effort unless REQUIRE_ELECTRON_SIGNAL=1.
#
# Usage:
#   bash tests/smoke/test_electron_package.sh
#
#   Linux CI with virtual display (required for headless Electron launch):
#   REQUIRE_ELECTRON_SIGNAL=1 xvfb-run bash tests/smoke/test_electron_package.sh
#
# Environment variables:
#   SKIP_BUILD=1               Skip npm ci / build / electron-builder.
#                              Requires a pre-built package in gui/dist/.
#   REQUIRE_ELECTRON_SIGNAL=1  Fail if the Electron startup signal is not
#                              received within SMOKE_TIMEOUT seconds.
#                              Default: 0 (best-effort; Python bridge check is gate).
#   SMOKE_TIMEOUT              Seconds to wait for Electron startup signal.
#                              Default: 30.
#
# Exit codes:
#   0  All required checks passed (steps 1-3 must pass; step 4 best-effort).
#   1  A required check failed: build error, missing bridge scripts,
#      bridge unreachable (no JSON response), or REQUIRE_ELECTRON_SIGNAL=1 timeout.
#
# Platform notes:
#   - Windows: Electron is a GUI-subsystem binary; stderr capture from bash may
#     not work. The Python bridge health check (step 3) is the primary gate.
#   - Linux CI: Use xvfb-run to provide a virtual display for the Electron launch.
#   - macOS: No virtual display needed for the Python check; Electron launch may
#     require permissions (codesigning) for step 4.
#
# Requirements:
#   - Node.js / npm in PATH
#   - Python venv activated (VIRTUAL_ENV set) or .venv in repo root
#   - rex package installed in the Python environment (pip install .)

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GUI_DIR="$REPO_ROOT/gui"
SKIP_BUILD="${SKIP_BUILD:-0}"
REQUIRE_ELECTRON_SIGNAL="${REQUIRE_ELECTRON_SIGNAL:-0}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-30}"

# Bridge script used for the health check.
# Sends an unknown command so no database or config reads are triggered.
HEALTH_CHECK_SCRIPT="rex_memories_bridge.py"
HEALTH_CHECK_PAYLOAD='{"command":"smoke-health-check"}'

# All bridge scripts registered in gui/src/main/bridgeResolver.ts
# that must be present in the packaged output.
REQUIRED_BRIDGES=(
  rex_tasks_bridge.py
  rex_reminders_bridge.py
  rex_shopping_list_bridge.py
  rex_speaker_bridge.py
  rex_chat_bridge.py
  rex_chat_stream_bridge.py
  rex_voices_bridge.py
  rex_voice_enrollment_bridge.py
  rex_voice_sample_bridge.py
  rex_voice_upload_bridge.py
  rex_wakeword_list_bridge.py
  rex_wakeword_train_bridge.py
  rex_wakeword_sample_bridge.py
  rex_stt_bridge.py
  rex_memories_bridge.py
  rex_file_extract_bridge.py
  rex_voice_bridge.py
  rex_calendar_bridge.py
  rex_email_bridge.py
  rex_sms_bridge.py
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { echo "[smoke] $*"; }
warn() { echo "[smoke] WARN: $*" >&2; }
fail() { echo "[smoke] FAIL: $*" >&2; exit 1; }

os_type() {
  local s
  s="$(uname -s 2>/dev/null || echo Windows)"
  case "$s" in
    Darwin*) echo macos ;;
    Linux*)  echo linux ;;
    *)       echo windows ;;
  esac
}

# ---------------------------------------------------------------------------
# Cleanup trap — kill background Electron process and temp files on exit
# ---------------------------------------------------------------------------
_ELECTRON_PID=""
_ELECTRON_LOG=""
_cleanup() {
  if [[ -n "$_ELECTRON_PID" ]]; then
    kill "$_ELECTRON_PID" 2>/dev/null || true
    wait "$_ELECTRON_PID" 2>/dev/null || true
  fi
  if [[ -n "$_ELECTRON_LOG" && -f "$_ELECTRON_LOG" ]]; then
    rm -f "$_ELECTRON_LOG"
  fi
}
trap _cleanup EXIT

# ===========================================================================
# Step 1: Build Electron package
# ===========================================================================
if [[ "$SKIP_BUILD" == "1" ]]; then
  log "Step 1: Skipping build (SKIP_BUILD=1)."
else
  log "Step 1: Building Electron package..."
  cd "$GUI_DIR"
  npm ci --prefer-offline 2>/dev/null || npm ci
  npm run build
  npx electron-builder --publish never --dir
  cd "$REPO_ROOT"
  log "Build complete."
fi

# ===========================================================================
# Step 2: Locate packaged output
# ===========================================================================
log "Step 2: Locating packaged output..."
OS="$(os_type)"
case "$OS" in
  macos)
    RESOURCES="$GUI_DIR/dist/mac/AskRex.app/Contents/Resources"
    ELECTRON_BIN="$GUI_DIR/dist/mac/AskRex.app/Contents/MacOS/AskRex"
    ;;
  linux)
    RESOURCES="$GUI_DIR/dist/linux-unpacked/resources"
    ELECTRON_BIN="$GUI_DIR/dist/linux-unpacked/askrex"
    ;;
  *)
    RESOURCES="$GUI_DIR/dist/win-unpacked/resources"
    ELECTRON_BIN="$GUI_DIR/dist/win-unpacked/AskRex.exe"
    ;;
esac

BRIDGE_DIR="$RESOURCES/bridge"
if [[ ! -d "$BRIDGE_DIR" ]]; then
  fail "Bridge directory not found: $BRIDGE_DIR. Run electron-builder first or check extraResources config."
fi
log "Bridge directory: $BRIDGE_DIR"

# ===========================================================================
# Step 3: Verify all bridge scripts present in packaged output
# ===========================================================================
log "Step 3: Verifying ${#REQUIRED_BRIDGES[@]} bridge scripts in packaged output..."
MISSING_COUNT=0
for script in "${REQUIRED_BRIDGES[@]}"; do
  if [[ ! -f "$BRIDGE_DIR/$script" ]]; then
    warn "Missing: $script"
    MISSING_COUNT=$((MISSING_COUNT + 1))
  fi
done
if [[ "$MISSING_COUNT" -gt 0 ]]; then
  fail "$MISSING_COUNT bridge script(s) missing from packaged output. Check extraResources in gui/package.json."
fi
log "All ${#REQUIRED_BRIDGES[@]} bridge scripts present."

# ===========================================================================
# Step 3b: Bridge health check — direct Python execution from packaged path
#
# Proves that:
#   (a) Python can find and execute a bridge script from process.resourcesPath
#   (b) The rex package is importable (rex.bridge_utils is the import-time dep)
#   (c) The bridge returns valid JSON
#   (d) Source-tree bridge/ directory is NOT on PYTHONPATH
#
# Uses an unknown command so no database or config is accessed.
# A {"ok": false, "error": "Unknown command..."} response is a passing result.
# ===========================================================================
log "Step 3b: Bridge health check — Python executes packaged bridge (no source-tree PYTHONPATH)..."

# Resolve Python executable: prefer the repo venv, fallback to system python.
PYTHON_EXE=""
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if [[ "$OS" == "windows" ]]; then
    PYTHON_EXE="$VIRTUAL_ENV/Scripts/python.exe"
  else
    PYTHON_EXE="$VIRTUAL_ENV/bin/python"
  fi
fi
if [[ -z "$PYTHON_EXE" || ! -x "$PYTHON_EXE" ]]; then
  if [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON_EXE="$REPO_ROOT/.venv/Scripts/python.exe"
  elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_EXE="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON_EXE="python"
  fi
fi
log "Python: $PYTHON_EXE"

BRIDGE_SCRIPT="$BRIDGE_DIR/$HEALTH_CHECK_SCRIPT"
BRIDGE_OUTPUT=""
BRIDGE_OK=0

# Clear PYTHONPATH so source-tree bridge/ is absent from the search path.
if ! BRIDGE_OUTPUT=$(echo "$HEALTH_CHECK_PAYLOAD" | \
    PYTHONPATH="" "$PYTHON_EXE" "$BRIDGE_SCRIPT" 2>/dev/null); then
  BRIDGE_OK=1
fi

if [[ -z "$BRIDGE_OUTPUT" ]]; then
  fail "Bridge script produced no output. Check that rex is installed: pip install ."
fi

if [[ "$BRIDGE_OK" -ne 0 ]]; then
  fail "Bridge script exited non-zero. Output: $BRIDGE_OUTPUT"
fi

# Validate JSON with 'ok' key.
JSON_VALID=0
echo "$BRIDGE_OUTPUT" | "$PYTHON_EXE" \
  -c "import sys, json; d = json.load(sys.stdin); sys.exit(0 if 'ok' in d else 1)" \
  2>/dev/null || JSON_VALID=1
if [[ "$JSON_VALID" -ne 0 ]]; then
  fail "Bridge did not return valid JSON with 'ok' key. Output: $BRIDGE_OUTPUT"
fi

log "Bridge health check passed. Response: $BRIDGE_OUTPUT"

# ===========================================================================
# Step 4: Electron launch — wait for bridge startup signal
#
# Launches the packaged app with ELECTRON_ENABLE_LOGGING=1 and watches stderr
# for the signal emitted by validateBridges() in bridgeResolver.ts:
#   "[bridgeResolver] All bridge scripts validated successfully."
#
# Platform notes:
#   - Linux CI: requires xvfb-run (virtual display).
#   - Windows: Electron is a GUI-subsystem binary; stderr may not be captured
#     from bash. This step is best-effort unless REQUIRE_ELECTRON_SIGNAL=1.
#   - macOS: Electron launch should work without a virtual display.
#
# Primary gate is step 3b (Python bridge health check). This step provides
# additional end-to-end confidence that the packaged app itself detects its
# bridges on startup. Set REQUIRE_ELECTRON_SIGNAL=1 in CI after xvfb is set up.
# ===========================================================================
log "Step 4: Electron launch check (timeout: ${SMOKE_TIMEOUT}s)..."
STARTUP_SIGNAL=0

if [[ ! -f "$ELECTRON_BIN" ]]; then
  warn "Electron binary not found: $ELECTRON_BIN — skipping launch check."
else
  _ELECTRON_LOG="$(mktemp)"
  ELECTRON_ENABLE_LOGGING=1 "$ELECTRON_BIN" --no-sandbox 2>"$_ELECTRON_LOG" &
  _ELECTRON_PID=$!
  log "Launched Electron (PID: $_ELECTRON_PID) — waiting for bridge validation signal..."

  ELAPSED=0
  while [[ $ELAPSED -lt $SMOKE_TIMEOUT ]]; do
    if grep -q "bridgeResolver.*All bridge scripts validated" "$_ELECTRON_LOG" 2>/dev/null; then
      STARTUP_SIGNAL=1
      break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
  done

  kill "$_ELECTRON_PID" 2>/dev/null || true
  wait "$_ELECTRON_PID" 2>/dev/null || true
  _ELECTRON_PID=""  # prevent double-kill in cleanup trap

  if [[ $STARTUP_SIGNAL -eq 1 ]]; then
    log "Electron startup signal received: bridge validation confirmed."
  else
    warn "Bridge startup signal not received within ${SMOKE_TIMEOUT}s."
    if [[ -s "$_ELECTRON_LOG" ]]; then
      warn "Electron log tail (last 5 lines):"
      tail -5 "$_ELECTRON_LOG" | sed 's/^/  /' >&2
    else
      warn "No Electron log output captured."
      warn "On Windows, GUI-subsystem binaries may not write stderr to bash."
      warn "On Linux CI, ensure xvfb-run wraps this script."
    fi
    if [[ "$REQUIRE_ELECTRON_SIGNAL" == "1" ]]; then
      fail "REQUIRE_ELECTRON_SIGNAL=1 but startup signal was not received."
    fi
    warn "Continuing — bridge health check (step 3b) already confirmed bridge is reachable."
  fi
fi

# ===========================================================================
# Done
# ===========================================================================
log "Smoke test PASSED."
log "  Steps verified: build, bridge scripts present (${#REQUIRED_BRIDGES[@]}), bridge health check."
if [[ $STARTUP_SIGNAL -eq 1 ]]; then
  log "  Electron startup signal: received."
else
  log "  Electron startup signal: not required (use REQUIRE_ELECTRON_SIGNAL=1 on Linux+xvfb)."
fi
exit 0
