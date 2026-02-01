"""Generate the wake acknowledgment sound."""

from __future__ import annotations


def main() -> int:
    from utils.env_loader import load as _load_env

    _load_env()

    from wake_acknowledgment import ensure_wake_acknowledgment_sound

    path = ensure_wake_acknowledgment_sound()
    print(f"Generated wake acknowledgment sound at: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
