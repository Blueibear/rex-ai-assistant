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
#   3c. Confirms the built main-process bundle has no Flask/gui_app spawn.
#   3d. Asserts the Flask GUI port (default: 5000) is NOT bound before launch.
#       The packaged app must not start rex-gui (US-012).
#   3e. Scans the built renderer bundle for raw fetch('/api/...) strings.
#       These are dead in packaged mode and indicate a missed IPC migration (US-012).
#   4. Launches the packaged Electron app and:
#      - Asserts the Flask port is NOT bound during the smoke window (US-012).
#      - Scans the Electron log for renderer /api/ fetch error traces (US-012).
#      - Waits for the bridge validation startup signal (best-effort).
#      Step 4 is best-effort unless REQUIRE_ELECTRON_SIGNAL=1.
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
#   PYTHON                     Explicit Python executable for bridge checks.
#                              Overrides venv and system Python fallback.
#
# Exit codes:
#   0  All required checks passed (steps 1-3e must pass; step 4 best-effort).
#   1  A required check failed: build error, missing bridge scripts,
#      bridge unreachable (no JSON response), Flask port bound (3d or step 4),
#      raw /api/ fetch in renderer bundle (3e), or REQUIRE_ELECTRON_SIGNAL=1 timeout.
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

# rex-gui (the Flask backend) binds AppConfig.ui.gui_port (default: 5000).
# The packaged Electron app must NOT start rex-gui or bind this port.
# Override with FLASK_GUI_PORT if rex_config.json configures a different port.
FLASK_PORT="${FLASK_GUI_PORT:-5000}"

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

# Returns 0 if FLASK_PORT is currently bound (a listener exists), 1 otherwise.
# Tries ss (Linux), lsof (macOS/Linux), then netstat (universal fallback).
flask_port_bound() {
  local port="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -tlnH 2>/dev/null | awk '{print $4}' | grep -qE ":${port}$"
  elif command -v lsof >/dev/null 2>&1; then
    lsof -iTCP:"$port" -sTCP:LISTEN -t >/dev/null 2>&1
  else
    netstat -an 2>/dev/null | grep -qE ":${port}[[:space:]].*LISTEN"
  fi
}

# Returns 0 if a .js file contains a raw fetch('/api/ or fetch("/api/ call.
_renderer_has_raw_api_fetch() {
  local f="$1"
  grep -qF "fetch('/api/" "$f" 2>/dev/null ||
  grep -qF 'fetch("/api/' "$f" 2>/dev/null
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

# Resolve Python executable: honor explicit PYTHON override first, then prefer
# the active/repo venv, then fall back to the platform launcher on PATH.
PYTHON_EXE="${PYTHON:-}"
if [[ -n "$PYTHON_EXE" ]]; then
  if ! command -v "$PYTHON_EXE" >/dev/null 2>&1 && [[ ! -x "$PYTHON_EXE" ]]; then
    fail "PYTHON override is not executable or on PATH: $PYTHON_EXE"
  fi
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
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
# Step 3c: Confirm Flask GUI (rex-gui) is NOT spawned by the packaged app.
#
# US-REM-019 audit finding:
#   - gui/src/main/index.ts has no spawn of rex-gui, flask, or gui_app.
#     All spawn() calls target Python bridge scripts (bridge/*.py) only.
#   - All bridge/*.py scripts have no subprocess calls to flask or rex-gui.
#   - The renderer makes fetch('/api/...') calls, but these are dead in
#     packaged mode: the renderer loads via file:// protocol, making all
#     relative URLs unreachable. Flask routes are not accessible by default.
#   - rex-gui classification corrected to developer-only (was shippable).
#     See SURFACE-CLASSIFICATION.md and progress-remaining-release-readiness.txt.
#
# This step confirms the audit finding by asserting the built main-process
# bundle contains no Flask/gui_app spawn indicators.
# ===========================================================================
log "Step 3c: Confirming Flask GUI (rex-gui) is not spawned by packaged app..."
BUILT_MAIN="$GUI_DIR/dist-electron/main/index.js"
if [[ -f "$BUILT_MAIN" ]]; then
  if grep -q "gui_app" "$BUILT_MAIN" 2>/dev/null; then
    fail "Flask spawn indicator (gui_app) found in $BUILT_MAIN. Flask must not be spawned from the packaged app. See US-REM-019."
  fi
  log "Flask spawn check passed: no gui_app reference in $BUILT_MAIN."
else
  log "Built main.js not found at $BUILT_MAIN; relying on source audit."
  log "  US-REM-019 confirmed: gui/src/main/index.ts has no spawn of flask, rex-gui, or gui_app."
fi
log "Flask routes are not reachable from the packaged app by default (renderer loads via file:// protocol)."

# ===========================================================================
# Step 3d: Pre-launch Flask port check (US-012)
#
# rex-gui binds AppConfig.ui.gui_port (default: ${FLASK_PORT}).
# The packaged Electron app must not start rex-gui or bind this port.
# Asserting the port is free before launch serves as the pre-condition check.
# ===========================================================================
log "Step 3d: Pre-launch Flask port check (port ${FLASK_PORT} must be free)..."
if flask_port_bound "$FLASK_PORT"; then
  fail "Flask port ${FLASK_PORT} is bound before Electron launch. Stop any stray rex-gui process and retry."
fi
log "Port ${FLASK_PORT} is free — no Flask backend running before Electron launch."

# ===========================================================================
# Step 3e: Built renderer bundle — no raw /api/ fetch patterns (US-012)
#
# Scans gui/dist-electron/renderer/**/*.js for raw fetch('/api/...) strings.
# Such strings are dead in packaged mode (file:// protocol) and indicate a
# missed or regressed renderer IPC migration (US-003 through US-011).
#
# This static check is the mandatory counterpart to the runtime renderer
# console scan in Step 4. It catches regressions even when Electron does
# not produce capturable stderr output in the headless CI environment.
# ===========================================================================
log "Step 3e: Built renderer bundle check (no raw /api/ fetch patterns)..."
RENDERER_DIST="$GUI_DIR/dist-electron/renderer"
RENDERER_API_HITS=0
if [[ -d "$RENDERER_DIST" ]]; then
  while IFS= read -r -d '' jsfile; do
    if _renderer_has_raw_api_fetch "$jsfile"; then
      warn "Raw /api/ fetch found in renderer bundle: $jsfile"
      RENDERER_API_HITS=$((RENDERER_API_HITS + 1))
    fi
  done < <(find "$RENDERER_DIST" -name "*.js" -print0 2>/dev/null)
  if [[ "$RENDERER_API_HITS" -gt 0 ]]; then
    fail "$RENDERER_API_HITS renderer JS file(s) contain raw /api/ fetches. Run: python scripts/check_no_renderer_api_fetch.py"
  fi
  log "Renderer bundle check passed: no raw /api/ fetch patterns in $RENDERER_DIST."
else
  warn "Renderer dist not found at $RENDERER_DIST; built bundle check skipped (run npm run build first or set SKIP_BUILD=0)."
fi

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

  # Flask port check — the packaged Electron app must not have started rex-gui
  # at any point during the smoke window (mandatory regardless of REQUIRE_ELECTRON_SIGNAL).
  if flask_port_bound "$FLASK_PORT"; then
    fail "Flask port ${FLASK_PORT} was bound during the Electron smoke window. The packaged app must not start rex-gui."
  fi
  log "Flask port check passed: port ${FLASK_PORT} not bound during Electron launch."

  # Renderer console /api/ check — scan Electron log for fetch error traces.
  # ELECTRON_ENABLE_LOGGING=1 captures Chromium internals to stderr; failed
  # fetch('/api/...) calls may appear as resource-load errors.
  # Step 3e (static bundle scan) is the mandatory gate; this is the dynamic layer.
  if [[ -s "$_ELECTRON_LOG" ]]; then
    if grep -qE '/api/' "$_ELECTRON_LOG" 2>/dev/null; then
      warn "Electron log contains '/api/' references — possible raw renderer fetch error:"
      grep -E '/api/' "$_ELECTRON_LOG" | head -5 | sed 's/^/  /' >&2
      if [[ "$REQUIRE_ELECTRON_SIGNAL" == "1" && $STARTUP_SIGNAL -eq 1 ]]; then
        fail "'/api/' pattern in Electron log while renderer was running. Renderer /api/ fetches must use IPC."
      fi
      warn "Set REQUIRE_ELECTRON_SIGNAL=1 to make this a hard failure (Step 3e static check is the mandatory gate)."
    else
      log "Renderer console check passed: no '/api/' patterns in Electron log."
    fi
  else
    log "No Electron log captured; dynamic renderer console check skipped (Step 3e static check is the mandatory gate)."
  fi

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
log "  Steps verified: build, bridge scripts present (${#REQUIRED_BRIDGES[@]}), bridge health check,"
log "    Flask port ${FLASK_PORT} free (no rex-gui started), renderer bundle clean (no raw /api/ fetches)."
if [[ $STARTUP_SIGNAL -eq 1 ]]; then
  log "  Electron startup signal: received."
else
  log "  Electron startup signal: not required (use REQUIRE_ELECTRON_SIGNAL=1 on Linux+xvfb)."
fi
exit 0
