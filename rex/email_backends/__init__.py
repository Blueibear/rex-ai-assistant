# Rex email backends package.

from rex.email_backends.account_config import EmailConfig, load_email_config
from rex.email_backends.account_router import resolve_backend
from rex.email_backends.base import EmailBackend, EmailEnvelope, SendResult
from rex.email_backends.inbox_stub import EmailInboxStub
from rex.email_backends.stub import StubEmailBackend
from rex.email_backends.triage import (
    TRIAGE_CATEGORIES,
    categorize,
    filter_by_category,
    triage_emails,
)
from rex.email_backends.triage_rules import (
    TriageRule,
    TriageRulesEngine,
    load_rules_from_file,
)

__all__ = [
    "EmailBackend",
    "EmailConfig",
    "EmailEnvelope",
    "EmailInboxStub",
    "SendResult",
    "StubEmailBackend",
    "TRIAGE_CATEGORIES",
    "TriageRule",
    "TriageRulesEngine",
    "categorize",
    "filter_by_category",
    "load_email_config",
    "load_rules_from_file",
    "resolve_backend",
    "triage_emails",
]
