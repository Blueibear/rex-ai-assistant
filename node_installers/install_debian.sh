#!/usr/bin/env bash
set -euo pipefail

REX_ROOT="${REX_ROOT:-/opt/rex-node}"
REX_ENV_FILE="${REX_ENV_FILE:-$REX_ROOT/.env.node}"
REX_SERVICE_PORT="${REX_SERVICE_PORT:-8765}"
REX_SERVICES="${REX_SERVICES:-event_bus,workflow_runner,memory_store,credential_manager}"
REX_PACKAGE_SOURCE="${REX_PACKAGE_SOURCE:-rex-ai-assistant}"
REX_DRY_RUN="${REX_DRY_RUN:-0}"

if [ "$REX_DRY_RUN" = "1" ]; then
  echo "[DRY RUN] Creating $REX_ROOT and installing dependencies"
else
  sudo mkdir -p "$REX_ROOT"
  sudo chown "$(id -u)":"$(id -g)" "$REX_ROOT"
fi

if command -v apt-get >/dev/null 2>&1; then
  if [ "$REX_DRY_RUN" = "1" ]; then
    echo "[DRY RUN] sudo apt-get update"
    echo "[DRY RUN] sudo apt-get install -y python3 python3-venv python3-pip"
  else
    sudo apt-get update
    sudo apt-get install -y python3 python3-venv python3-pip
  fi
fi

cd "$REX_ROOT"

if [ "$REX_DRY_RUN" = "1" ]; then
  echo "[DRY RUN] python3 -m venv venv"
else
  python3 -m venv venv
fi

if [ "$REX_DRY_RUN" = "1" ]; then
  echo "[DRY RUN] venv/bin/pip install $REX_PACKAGE_SOURCE"
else
  venv/bin/pip install "$REX_PACKAGE_SOURCE"
fi

if [ ! -f "$REX_ENV_FILE" ]; then
  if [ "$REX_DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Copying .env.node template to $REX_ENV_FILE"
  else
    cp "$(dirname "${BASH_SOURCE[0]}")/.env.node" "$REX_ENV_FILE"
  fi
fi

if command -v systemctl >/dev/null 2>&1; then
  SERVICE_CONTENT="[Unit]
Description=Rex Lean Node
After=network.target

[Service]
Type=simple
WorkingDirectory=${REX_ROOT}
EnvironmentFile=${REX_ENV_FILE}
ExecStart=${REX_ROOT}/venv/bin/python -m rex.app --port ${REX_SERVICE_PORT} --services ${REX_SERVICES}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target"

  SERVICE_PATH="/etc/systemd/system/rex-node.service"
  if [ "$REX_DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would write systemd service to ${SERVICE_PATH}"
  else
    echo "$SERVICE_CONTENT" | sudo tee "$SERVICE_PATH" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable --now rex-node.service
  fi
fi

echo "Register the node with the gateway (stub):"
echo "  curl -X POST \"$REX_GATEWAY_URL/api/nodes/register\" -H \"Authorization: Bearer $REX_NODE_TOKEN\""
