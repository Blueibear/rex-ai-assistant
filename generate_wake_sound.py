"""Generate the wake acknowledgment sound."""

from utils.env_loader import load as _load_env
from wake_acknowledgment import ensure_wake_acknowledgment_sound

# Load .env before accessing any environment variables
_load_env()

if __name__ == "__main__":
    path = ensure_wake_acknowledgment_sound()
    print(f"Generated wake acknowledgment sound at: {path}")
