"""Generate the wake acknowledgment sound."""
from wake_acknowledgment import ensure_wake_acknowledgment_sound

if __name__ == "__main__":
    path = ensure_wake_acknowledgment_sound()
    print(f"Generated wake acknowledgment sound at: {path}")
