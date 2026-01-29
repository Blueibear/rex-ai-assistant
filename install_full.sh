#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REX_PYTHON="${REX_PYTHON:-python3}"
REX_SERVICE_PORT="${REX_SERVICE_PORT:-8765}"
REX_INSTALL_PREFIX="${REX_INSTALL_PREFIX:-}"
REX_DRY_RUN="${REX_DRY_RUN:-0}"
REX_SKIP_SERVICE="${REX_SKIP_SERVICE:-0}"

usage() {
  cat <<'USAGE'
Usage: install_full.sh [options]

Options:
  --prefix PATH        Install prefix for pip (maps to REX_INSTALL_PREFIX)
  --python PATH        Python interpreter to use
  --port PORT          Service port (default: 8765)
  --dry-run            Print commands without executing
  --skip-service       Skip systemd service setup
  --help               Show this help
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --prefix)
      REX_INSTALL_PREFIX="$2"
      shift 2
      ;;
    --python)
      REX_PYTHON="$2"
      shift 2
      ;;
    --port)
      REX_SERVICE_PORT="$2"
      shift 2
      ;;
    --dry-run)
      REX_DRY_RUN="1"
      shift
      ;;
    --skip-service)
      REX_SKIP_SERVICE="1"
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

install_cmd=("$REX_PYTHON" -m pip install "${ROOT_DIR}[sms,devtools]")
if [ -n "$REX_INSTALL_PREFIX" ]; then
  install_cmd=("$REX_PYTHON" -m pip install --prefix "$REX_INSTALL_PREFIX" "${ROOT_DIR}[sms,devtools]")
fi

echo "Installing Rex (full) with extras: sms, devtools"
if [ "$REX_DRY_RUN" = "1" ]; then
  echo "[DRY RUN] ${install_cmd[*]}"
else
  "${install_cmd[@]}"
fi

if [ "$REX_SKIP_SERVICE" = "1" ]; then
  echo "Skipping service setup (REX_SKIP_SERVICE=1)."
  exit 0
fi

if command -v systemctl >/dev/null 2>&1; then
  SERVICE_CONTENT="[Unit]
Description=Rex AI Assistant
After=network.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${REX_PYTHON} -m rex.app --port ${REX_SERVICE_PORT}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"

  SERVICE_PATH="/etc/systemd/system/rex.service"
  if [ "$REX_DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would write systemd service to ${SERVICE_PATH}"
  else
    if [ "$(id -u)" -eq 0 ]; then
      echo "$SERVICE_CONTENT" > "$SERVICE_PATH"
      systemctl daemon-reload
      systemctl enable --now rex.service
    elif command -v sudo >/dev/null 2>&1; then
      echo "$SERVICE_CONTENT" | sudo tee "$SERVICE_PATH" >/dev/null
      sudo systemctl daemon-reload
      sudo systemctl enable --now rex.service
    else
      echo "Systemd detected but sudo is unavailable. Run as root to install the service."
    fi
  fi
else
  echo "systemctl not found; run 'rex-run --port ${REX_SERVICE_PORT}' manually."
fi
