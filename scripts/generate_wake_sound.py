"""Generate the wake acknowledgment sound."""

# ruff: noqa: E402
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env

_load_env()

from rex.wake_acknowledgment import ensure_wake_acknowledgment_sound

if __name__ == "__main__":
    path = ensure_wake_acknowledgment_sound()
    print(f"Generated wake acknowledgment sound at: {path}")
