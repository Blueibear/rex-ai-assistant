#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REX_PYTHON="${REX_PYTHON:-python3}"
REX_SERVICE_PORT="${REX_SERVICE_PORT:-8765}"
REX_INSTALL_PREFIX="${REX_INSTALL_PREFIX:-}"
REX_DRY_RUN="${REX_DRY_RUN:-0}"
REX_SKIP_SERVICE="${REX_SKIP_SERVICE:-0}"
REX_SERVICES="${REX_SERVICES:-event_bus,workflow_runner,memory_store,credential_manager}"

install_cmd=("$REX_PYTHON" -m pip install "$ROOT_DIR")
if [ -n "$REX_INSTALL_PREFIX" ]; then
  install_cmd=("$REX_PYTHON" -m pip install --prefix "$REX_INSTALL_PREFIX" "$ROOT_DIR")
fi

echo "Installing Rex (lean) with minimal dependencies"
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
Description=Rex AI Assistant (Lean Node)
After=network.target

[Service]
Type=simple
WorkingDirectory=${ROOT_DIR}
ExecStart=${REX_PYTHON} -m rex.app --port ${REX_SERVICE_PORT} --services ${REX_SERVICES}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"

  SERVICE_PATH="/etc/systemd/system/rex-lean.service"
  if [ "$REX_DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would write systemd service to ${SERVICE_PATH}"
  else
    if [ "$(id -u)" -eq 0 ]; then
      echo "$SERVICE_CONTENT" > "$SERVICE_PATH"
      systemctl daemon-reload
      systemctl enable --now rex-lean.service
    elif command -v sudo >/dev/null 2>&1; then
      echo "$SERVICE_CONTENT" | sudo tee "$SERVICE_PATH" >/dev/null
      sudo systemctl daemon-reload
      sudo systemctl enable --now rex-lean.service
    else
      echo "Systemd detected but sudo is unavailable. Run as root to install the service."
    fi
  fi
else
  echo "systemctl not found; run 'rex-run --services ${REX_SERVICES} --port ${REX_SERVICE_PORT}' manually."
fi
