"""
Email service module for Rex AI Assistant.

Provides email triage functionality with classification and summarization.
Currently implements stub/mock functionality for testing; real IMAP/SMTP
integration can be added later.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from rex.credentials import get_credential_manager

logger = logging.getLogger(__name__)


class EmailSummary(BaseModel):
    """Summary of an email message."""

    id: str = Field(..., description="Unique email identifier")
    from_addr: str = Field(..., description="Sender email address")
    subject: str = Field(..., description="Email subject")
    snippet: str = Field(..., description="Brief preview of email body")
    received_at: datetime = Field(..., description="When the email was received")
    labels: list[str] = Field(default_factory=list, description="Email labels/tags")
    importance_score: float = Field(default=0.5, description="Importance score (0.0-1.0)")
    category: Optional[str] = Field(default=None, description="Email category (important, promo, social, etc.)")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EmailService:
    """
    Email service for reading and categorizing emails.

    Currently uses stub implementation with mock data.
    Real IMAP/SMTP integration will be added in future iterations.
    """

    def __init__(self, mock_data_file: Optional[Path] = None):
        """
        Initialize the email service.

        Args:
            mock_data_file: Path to mock email data file
        """
        self.mock_data_file = mock_data_file or Path("data/mock_emails.json")
        self.connected = False
        self.credential_manager = get_credential_manager()
        self._mock_emails: list[EmailSummary] = []

    def connect(self) -> bool:
        """
        Connect to email service.

        For stub implementation, this loads mock data and validates credentials exist.

        Returns:
            True if connection successful
        """
        try:
            # Check if email credentials exist
            email_creds = self.credential_manager.get_credential("email")
            if not email_creds:
                logger.warning("No email credentials configured")
                # Continue anyway for testing purposes

            # Load mock data
            self._load_mock_data()

            self.connected = True
            logger.info("Email service connected (stub mode)")
            return True

        except Exception as e:
            logger.error(f"Failed to connect email service: {e}", exc_info=True)
            return False

    def _load_mock_data(self) -> None:
        """Load mock email data from file."""
        if not self.mock_data_file.exists():
            logger.warning(f"No mock email data at {self.mock_data_file}")
            self._mock_emails = []
            return

        try:
            with open(self.mock_data_file, 'r') as f:
                data = json.load(f)

            self._mock_emails = []
            for email_data in data:
                # Parse datetime from ISO format
                if 'received_at' in email_data and isinstance(email_data['received_at'], str):
                    email_data['received_at'] = datetime.fromisoformat(email_data['received_at'])

                email = EmailSummary(**email_data)
                self._mock_emails.append(email)

            logger.info(f"Loaded {len(self._mock_emails)} mock emails")

        except Exception as e:
            logger.error(f"Failed to load mock email data: {e}", exc_info=True)
            self._mock_emails = []

    def fetch_unread(self, limit: int = 10) -> list[EmailSummary]:
        """
        Fetch unread email summaries.

        Args:
            limit: Maximum number of emails to return

        Returns:
            List of EmailSummary objects
        """
        if not self.connected:
            logger.warning("Email service not connected")
            return []

        # Return mock emails (stub implementation)
        unread = [email for email in self._mock_emails if 'unread' in email.labels]
        result = unread[:limit]

        logger.info(f"Fetched {len(result)} unread emails")
        return result

    def mark_as_read(self, email_id: str) -> bool:
        """
        Mark an email as read.

        Args:
            email_id: Email identifier

        Returns:
            True if successful
        """
        if not self.connected:
            logger.warning("Email service not connected")
            return False

        # Find and update mock email
        for email in self._mock_emails:
            if email.id == email_id:
                if 'unread' in email.labels:
                    email.labels.remove('unread')
                logger.info(f"Marked email {email_id} as read")
                return True

        logger.warning(f"Email not found: {email_id}")
        return False

    def categorize(self, email: EmailSummary) -> str:
        """
        Categorize an email based on heuristics.

        Args:
            email: Email to categorize

        Returns:
            Category string (important, promo, social, newsletter, etc.)
        """
        subject_lower = email.subject.lower()
        from_lower = email.from_addr.lower()
        snippet_lower = email.snippet.lower()

        # Promo indicators
        promo_keywords = ['sale', 'discount', 'offer', 'deal', 'promotion', 'coupon', 'free shipping']
        if any(kw in subject_lower or kw in snippet_lower for kw in promo_keywords):
            return 'promo'

        # Social indicators
        social_keywords = ['liked your', 'commented on', 'mentioned you', 'friend request', 'connection']
        social_domains = ['facebook.com', 'twitter.com', 'linkedin.com', 'instagram.com']
        if any(kw in subject_lower or kw in snippet_lower for kw in social_keywords):
            return 'social'
        if any(domain in from_lower for domain in social_domains):
            return 'social'

        # Newsletter indicators
        if 'unsubscribe' in snippet_lower or 'newsletter' in subject_lower:
            return 'newsletter'

        # Important indicators
        important_keywords = ['urgent', 'important', 'asap', 'action required', 'deadline']
        if any(kw in subject_lower for kw in important_keywords):
            return 'important'

        # Check importance score
        if email.importance_score >= 0.8:
            return 'important'

        # Default
        return 'general'

    def summarize(self, email_id: str) -> str:
        """
        Get a summary of an email.

        Args:
            email_id: Email identifier

        Returns:
            Summary text (currently returns the snippet)
        """
        if not self.connected:
            return "Email service not connected"

        # Find email
        for email in self._mock_emails:
            if email.id == email_id:
                return f"From: {email.from_addr}\nSubject: {email.subject}\n\n{email.snippet}"

        return f"Email not found: {email_id}"

    def get_all_emails(self) -> list[EmailSummary]:
        """
        Get all emails (for testing purposes).

        Returns:
            List of all EmailSummary objects
        """
        return self._mock_emails.copy()


# Global email service instance
_email_service: Optional[EmailService] = None


def get_email_service() -> EmailService:
    """Get the global email service instance."""
    global _email_service
    if _email_service is None:
        _email_service = EmailService()
    return _email_service


def set_email_service(service: EmailService) -> None:
    """Set the global email service instance (for testing)."""
    global _email_service
    _email_service = service
