"""Tests for US-079: Email triage categorization.

Acceptance criteria verified:
- [x] triage assigns one of four categories: urgent, action_required, fyi, newsletter
- [x] categorization logic uses sender address, subject keywords, and body patterns
- [x] triage results are queryable ("show urgent emails" returns only urgent-tagged items)
- [x] test using mock inbox data confirms at least one email correctly categorized
      into each category
- [x] Typecheck passes (enforced by mypy in CI)
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from rex.email_backends.base import EmailEnvelope
from rex.email_backends.inbox_stub import EmailInboxStub
from rex.email_backends.triage import (
    TRIAGE_CATEGORIES,
    categorize,
    filter_by_category,
    triage_emails,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 11, 9, 0, 0, tzinfo=timezone.utc)


def _env(
    *,
    sender: str = "sender@example.com",
    subject: str = "Hello",
    snippet: str = "",
) -> EmailEnvelope:
    return EmailEnvelope(
        message_id="test-id",
        from_addr=sender,
        subject=subject,
        snippet=snippet,
        received_at=_NOW,
        to_addrs=["me@example.com"],
        labels=[],
    )


# ---------------------------------------------------------------------------
# AC1 — triage assigns one of four valid categories
# ---------------------------------------------------------------------------


class TestValidCategories:
    def test_category_constants_cover_all_four(self) -> None:
        assert set(TRIAGE_CATEGORIES) == {
            "urgent",
            "action_required",
            "newsletter",
            "fyi",
        }

    def test_categorize_always_returns_valid_category(self) -> None:
        envelopes = [
            _env(subject="Normal message"),
            _env(subject="URGENT: Server down"),
            _env(subject="Action required: sign the form"),
            _env(sender="newsletter@example.com", subject="Weekly digest"),
        ]
        for env in envelopes:
            result = categorize(env)
            assert result in TRIAGE_CATEGORIES, (
                f"categorize() returned unexpected value {result!r}"
            )


# ---------------------------------------------------------------------------
# AC2 — categorization uses sender address, subject keywords, body patterns
# ---------------------------------------------------------------------------


class TestCategorizationLogic:
    # -- urgent via subject keyword --

    def test_urgent_subject_keyword_urgent(self) -> None:
        assert categorize(_env(subject="URGENT: Please act now")) == "urgent"

    def test_urgent_subject_keyword_critical(self) -> None:
        assert categorize(_env(subject="CRITICAL: Database failure")) == "urgent"

    def test_urgent_subject_keyword_outage(self) -> None:
        assert categorize(_env(subject="Outage detected on prod")) == "urgent"

    def test_urgent_subject_keyword_incident(self) -> None:
        assert categorize(_env(subject="Incident: payment service down")) == "urgent"

    # -- urgent via sender address --

    def test_urgent_sender_alerts(self) -> None:
        assert categorize(_env(sender="alerts@monitor.example.com")) == "urgent"

    def test_urgent_sender_monitoring(self) -> None:
        assert categorize(_env(sender="monitoring@infra.example.com")) == "urgent"

    def test_urgent_sender_pagerduty(self) -> None:
        assert categorize(_env(sender="notify@pagerduty.com")) == "urgent"

    # -- urgent via body / snippet --

    def test_urgent_body_join_incident_bridge(self) -> None:
        env = _env(snippet="Please join the incident bridge immediately.")
        assert categorize(env) == "urgent"

    def test_urgent_body_immediate_action_required(self) -> None:
        env = _env(snippet="Immediate action required to restore service.")
        assert categorize(env) == "urgent"

    # -- action_required via subject --

    def test_action_required_subject_action_required(self) -> None:
        env = _env(subject="Action required: complete your review")
        assert categorize(env) == "action_required"

    def test_action_required_subject_please_complete(self) -> None:
        env = _env(subject="Please complete your onboarding form")
        assert categorize(env) == "action_required"

    def test_action_required_subject_invoice(self) -> None:
        env = _env(subject="Invoice #123 due in 7 days")
        assert categorize(env) == "action_required"

    def test_action_required_subject_due_by(self) -> None:
        env = _env(subject="Form due by Friday")
        assert categorize(env) == "action_required"

    def test_action_required_subject_annual_review(self) -> None:
        env = _env(subject="Annual review form — please fill in")
        assert categorize(env) == "action_required"

    def test_action_required_subject_payment(self) -> None:
        env = _env(subject="Payment overdue — update billing details")
        assert categorize(env) == "action_required"

    # -- newsletter via sender --

    def test_newsletter_sender_newsletter_prefix(self) -> None:
        env = _env(sender="newsletter@techdigest.example.com")
        assert categorize(env) == "newsletter"

    def test_newsletter_sender_digest_prefix(self) -> None:
        env = _env(sender="digest@company.com")
        assert categorize(env) == "newsletter"

    def test_newsletter_sender_marketing(self) -> None:
        env = _env(sender="marketing@brand.com")
        assert categorize(env) == "newsletter"

    # -- newsletter via subject --

    def test_newsletter_subject_this_week_in(self) -> None:
        env = _env(subject="This week in tech — March edition")
        assert categorize(env) == "newsletter"

    def test_newsletter_subject_weekly_digest(self) -> None:
        env = _env(subject="Weekly digest: top stories")
        assert categorize(env) == "newsletter"

    def test_newsletter_subject_monthly_edition(self) -> None:
        env = _env(subject="Monthly edition: product updates")
        assert categorize(env) == "newsletter"

    def test_newsletter_subject_unsubscribe(self) -> None:
        env = _env(subject="Unsubscribe from our mailing list")
        assert categorize(env) == "newsletter"

    # -- fyi catch-all --

    def test_fyi_plain_informational(self) -> None:
        env = _env(
            sender="noreply@github.com",
            subject="Your pull request was merged",
        )
        assert categorize(env) == "fyi"

    def test_fyi_calendar_reminder(self) -> None:
        env = _env(
            sender="calendar-noreply@example.com",
            subject="Reminder: team stand-up tomorrow at 09:00",
        )
        assert categorize(env) == "fyi"


# ---------------------------------------------------------------------------
# AC3 — triage results are queryable
# ---------------------------------------------------------------------------


class TestFilterByCategory:
    def _sample_pool(self) -> list[EmailEnvelope]:
        return [
            _env(subject="URGENT: Server is down"),
            _env(subject="Action required: sign the NDA"),
            _env(
                sender="newsletter@example.com",
                subject="This week in AI — March 2026 edition",
            ),
            _env(sender="noreply@github.com", subject="Your PR was merged"),
            _env(subject="CRITICAL: Disk at 98%"),
            _env(subject="Invoice #999 due in 3 days"),
        ]

    def test_filter_urgent_returns_only_urgent(self) -> None:
        pool = self._sample_pool()
        results = filter_by_category(pool, "urgent")
        assert len(results) >= 1
        for env in results:
            assert categorize(env) == "urgent"

    def test_filter_action_required_returns_only_action_required(self) -> None:
        pool = self._sample_pool()
        results = filter_by_category(pool, "action_required")
        assert len(results) >= 1
        for env in results:
            assert categorize(env) == "action_required"

    def test_filter_newsletter_returns_only_newsletter(self) -> None:
        pool = self._sample_pool()
        results = filter_by_category(pool, "newsletter")
        assert len(results) >= 1
        for env in results:
            assert categorize(env) == "newsletter"

    def test_filter_fyi_returns_only_fyi(self) -> None:
        pool = self._sample_pool()
        results = filter_by_category(pool, "fyi")
        assert len(results) >= 1
        for env in results:
            assert categorize(env) == "fyi"

    def test_filter_unknown_category_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown triage category"):
            filter_by_category([], "spam")

    def test_filter_does_not_return_wrong_categories(self) -> None:
        pool = self._sample_pool()
        urgent = filter_by_category(pool, "urgent")
        non_urgent = [e for e in pool if categorize(e) != "urgent"]
        assert len(urgent) + len(non_urgent) == len(pool)


class TestTriageEmails:
    def test_returns_all_four_keys(self) -> None:
        result = triage_emails([])
        assert set(result.keys()) == {"urgent", "action_required", "newsletter", "fyi"}

    def test_all_emails_placed_in_exactly_one_bucket(self) -> None:
        pool = [
            _env(subject="URGENT: fire"),
            _env(subject="Action required: sign now"),
            _env(sender="newsletter@example.com", subject="Monthly edition"),
            _env(sender="noreply@github.com", subject="PR merged"),
        ]
        result = triage_emails(pool)
        total = sum(len(v) for v in result.values())
        assert total == len(pool)

    def test_empty_list_returns_empty_buckets(self) -> None:
        result = triage_emails([])
        assert all(len(v) == 0 for v in result.values())


# ---------------------------------------------------------------------------
# AC4 — mock inbox data has at least one email in each category
# ---------------------------------------------------------------------------


class TestMockInboxCoverage:
    """Confirm the built-in mock data yields at least one email per category."""

    def setup_method(self) -> None:
        stub = EmailInboxStub()
        self.all_emails = stub.all_emails
        self.grouped = triage_emails(self.all_emails)

    def test_mock_inbox_has_urgent_emails(self) -> None:
        assert len(self.grouped["urgent"]) >= 1, (
            "Mock inbox must contain at least one urgent email"
        )

    def test_mock_inbox_has_action_required_emails(self) -> None:
        assert len(self.grouped["action_required"]) >= 1, (
            "Mock inbox must contain at least one action_required email"
        )

    def test_mock_inbox_has_newsletter_emails(self) -> None:
        assert len(self.grouped["newsletter"]) >= 1, (
            "Mock inbox must contain at least one newsletter email"
        )

    def test_mock_inbox_has_fyi_emails(self) -> None:
        assert len(self.grouped["fyi"]) >= 1, (
            "Mock inbox must contain at least one fyi email"
        )

    def test_mock_urgent_emails_are_correct(self) -> None:
        """Both 'mock-urgent-*' emails categorize as urgent."""
        urgent_ids = {e.message_id for e in self.grouped["urgent"]}
        assert "mock-urgent-001" in urgent_ids
        assert "mock-urgent-002" in urgent_ids

    def test_mock_action_required_emails_are_correct(self) -> None:
        """Both 'mock-action-*' emails categorize as action_required."""
        action_ids = {e.message_id for e in self.grouped["action_required"]}
        assert "mock-action-001" in action_ids
        assert "mock-action-002" in action_ids

    def test_mock_newsletter_email_is_correct(self) -> None:
        """mock-fyi-001 (newsletter sender + edition subject) → newsletter."""
        newsletter_ids = {e.message_id for e in self.grouped["newsletter"]}
        assert "mock-fyi-001" in newsletter_ids

    def test_mock_fyi_emails_are_correct(self) -> None:
        """mock-fyi-002 and mock-fyi-003 categorize as fyi."""
        fyi_ids = {e.message_id for e in self.grouped["fyi"]}
        assert "mock-fyi-002" in fyi_ids
        assert "mock-fyi-003" in fyi_ids

    def test_filter_by_category_urgent_on_mock_data(self) -> None:
        """filter_by_category('urgent') returns only urgent emails from mock inbox."""
        results = filter_by_category(self.all_emails, "urgent")
        assert len(results) >= 1
        for env in results:
            assert categorize(env) == "urgent"

    def test_filter_by_category_newsletter_on_mock_data(self) -> None:
        """filter_by_category('newsletter') returns only newsletter emails."""
        results = filter_by_category(self.all_emails, "newsletter")
        assert len(results) >= 1
        for env in results:
            assert categorize(env) == "newsletter"
