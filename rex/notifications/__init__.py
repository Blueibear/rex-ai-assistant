"""Rex notifications package."""

from rex.notifications.desktop import notify
from rex.notifications.push import send_push

__all__ = ["notify", "send_push"]
