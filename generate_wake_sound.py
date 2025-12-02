"""Generate the wake acknowledgment sound."""

# Load .env before accessing any environment variables
from utils.env_loader import load as _load_env
_load_env()

from wake_acknowledgment import ensure_wake_acknowledgment_sound

if __name__ == "__main__":
    path = ensure_wake_acknowledgment_sound()
    print(f"Generated wake acknowledgment sound at: {path}")
