"""Windows computer control client package.

This package provides the client-side foundation for remote Windows computer
control via a lightweight agent API (Cycle 5.1).

The Windows agent server is **not** included here (that is Cycle 5.3).  This
package only contains:

- Pydantic v2 config models for the ``computers[]`` config section
- An HTTP client that speaks the agent API contract
- A high-level service wrapper used by the CLI

Agent API contract (client expectations)
-----------------------------------------
- ``GET /health``  -> ``{"status": "ok"}``
- ``GET /status``  -> ``{"hostname": ..., "os": ..., "user": ..., "time": ...}``
- ``POST /run``    -> body ``{"command": "...", "args": [...], "cwd": "..."}``
                   response ``{"exit_code": 0, "stdout": "...", "stderr": "..."}``

Auth header: ``X-Auth-Token: <token>``
"""

from __future__ import annotations

from rex.computers.config import (
    ComputerAllowlists,
    ComputerConfig,
    ComputersConfig,
    load_computers_config,
)
from rex.computers.service import ComputerService, get_computer_service

__all__ = [
    "ComputerAllowlists",
    "ComputerConfig",
    "ComputersConfig",
    "ComputerService",
    "get_computer_service",
    "load_computers_config",
]
