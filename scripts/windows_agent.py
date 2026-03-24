"""Entry-point script for the Rex Windows Agent server.

Run this script on the Windows machine that Rex should be able to control:

    python scripts/windows_agent.py

All configuration is provided via environment variables.  See
``rex/computers/agent_server.py`` (or ``docs/computers_agent.md``) for the
full list of variables.

Minimum required configuration::

    set REX_AGENT_TOKEN=<your-secret-token>      # Windows cmd.exe
    $env:REX_AGENT_TOKEN="<your-secret-token>"   # PowerShell

Optional::

    set REX_AGENT_HOST=127.0.0.1       # bind address (default: 127.0.0.1)
    set REX_AGENT_PORT=7777            # listen port (default: 7777)
    set REX_AGENT_ALLOWLIST=whoami,dir,ipconfig,systeminfo
    set REX_AGENT_RATE_LIMIT=60        # max requests per minute per IP
    set REX_AGENT_TIMEOUT=30           # subprocess timeout in seconds
    set REX_AGENT_MAX_OUTPUT=65536     # max output bytes returned per run
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path when invoked directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rex.computers.agent_server import main  # noqa: E402

if __name__ == "__main__":
    main()
