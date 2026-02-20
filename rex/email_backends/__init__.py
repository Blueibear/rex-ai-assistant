# Rex email backends package.

from rex.email_backends.account_config import EmailConfig, load_email_config
from rex.email_backends.account_router import resolve_backend
from rex.email_backends.base import EmailBackend, EmailEnvelope, SendResult
from rex.email_backends.stub import StubEmailBackend

__all__ = [
    "EmailBackend",
    "EmailConfig",
    "EmailEnvelope",
    "SendResult",
    "StubEmailBackend",
    "load_email_config",
    "resolve_backend",
]
