"""
Calendar service module for Rex AI Assistant.

Provides calendar integration with read/write capabilities.
Currently implements stub/mock functionality for testing; real calendar API
integration (Google Calendar, Outlook, etc.) can be added later.
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from rex.credentials import get_credential_manager

logger = logging.getLogger(__name__)


class CalendarEvent(BaseModel):
    """A calendar event."""

    id: str = Field(..., description="Unique event identifier")
    title: str = Field(..., description="Event title/summary")
    start_time: datetime = Field(..., description="Event start time")
    end_time: datetime = Field(..., description="Event end time")
    attendees: list[str] = Field(default_factory=list, description="List of attendee email addresses")
    location: Optional[str] = Field(default=None, description="Event location")
    description: Optional[str] = Field(default=None, description="Event description")
    all_day: bool = Field(default=False, description="Whether this is an all-day event")

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    def overlaps_with(self, other: "CalendarEvent") -> bool:
        """
        Check if this event overlaps with another event.

        Args:
            other: Another calendar event

        Returns:
            True if events overlap
        """
        # Events overlap if one starts before the other ends
        return (self.start_time < other.end_time and
                self.end_time > other.start_time)


class CalendarService:
    """
    Calendar service for reading and writing calendar events.

    Currently uses stub implementation with mock data.
    Real calendar API integration will be added in future iterations.
    """

    def __init__(self, mock_data_file: Optional[Path] = None):
        """
        Initialize the calendar service.

        Args:
            mock_data_file: Path to mock calendar data file
        """
        self.mock_data_file = mock_data_file or Path("data/mock_calendar.json")
        self.connected = False
        self.credential_manager = get_credential_manager()
        self._mock_events: list[CalendarEvent] = []

    def connect(self) -> bool:
        """
        Connect to calendar service.

        For stub implementation, this loads mock data and validates credentials exist.

        Returns:
            True if connection successful
        """
        try:
            # Check if calendar credentials exist
            calendar_creds = self.credential_manager.get_credential("calendar")
            if not calendar_creds:
                logger.warning("No calendar credentials configured")
                # Continue anyway for testing purposes

            # Load mock data
            self._load_mock_data()

            self.connected = True
            logger.info("Calendar service connected (stub mode)")
            return True

        except Exception as e:
            logger.error(f"Failed to connect calendar service: {e}", exc_info=True)
            return False

    def _load_mock_data(self) -> None:
        """Load mock calendar data from file."""
        if not self.mock_data_file.exists():
            logger.warning(f"No mock calendar data at {self.mock_data_file}")
            self._mock_events = []
            return

        try:
            with open(self.mock_data_file, 'r') as f:
                data = json.load(f)

            self._mock_events = []
            for event_data in data:
                # Parse datetime from ISO format
                if 'start_time' in event_data and isinstance(event_data['start_time'], str):
                    event_data['start_time'] = datetime.fromisoformat(event_data['start_time'])
                if 'end_time' in event_data and isinstance(event_data['end_time'], str):
                    event_data['end_time'] = datetime.fromisoformat(event_data['end_time'])

                event = CalendarEvent(**event_data)
                self._mock_events.append(event)

            logger.info(f"Loaded {len(self._mock_events)} mock calendar events")

        except Exception as e:
            logger.error(f"Failed to load mock calendar data: {e}", exc_info=True)
            self._mock_events = []

    def _save_mock_data(self) -> None:
        """Save mock calendar data to file."""
        try:
            # Ensure directory exists
            self.mock_data_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert events to dict format
            events_data = []
            for event in self._mock_events:
                event_dict = event.model_dump()
                # Convert datetime to ISO format
                if isinstance(event_dict.get('start_time'), datetime):
                    event_dict['start_time'] = event_dict['start_time'].isoformat()
                if isinstance(event_dict.get('end_time'), datetime):
                    event_dict['end_time'] = event_dict['end_time'].isoformat()
                events_data.append(event_dict)

            with open(self.mock_data_file, 'w') as f:
                json.dump(events_data, f, indent=2)

            logger.debug(f"Saved {len(self._mock_events)} events to {self.mock_data_file}")

        except Exception as e:
            logger.error(f"Failed to save mock calendar data: {e}", exc_info=True)

    def get_events(self, start: datetime, end: datetime) -> list[CalendarEvent]:
        """
        Get calendar events in a time range.

        Args:
            start: Start of time range
            end: End of time range

        Returns:
            List of CalendarEvent objects in the time range
        """
        if not self.connected:
            logger.warning("Calendar service not connected")
            return []

        # Filter events that overlap with the time range
        result = [
            event for event in self._mock_events
            if event.start_time < end and event.end_time > start
        ]

        # Sort by start time
        result.sort(key=lambda e: e.start_time)

        logger.info(f"Found {len(result)} events between {start} and {end}")
        return result

    def create_event(self, event: CalendarEvent) -> CalendarEvent:
        """
        Create a new calendar event.

        Args:
            event: Event to create (id will be generated if not provided)

        Returns:
            Created event with assigned ID
        """
        if not self.connected:
            raise RuntimeError("Calendar service not connected")

        # Generate ID if not provided
        if not event.id:
            event.id = str(uuid.uuid4())

        # Add to mock events
        self._mock_events.append(event)

        # Save to file
        self._save_mock_data()

        logger.info(f"Created calendar event: {event.id} ({event.title})")
        return event

    def update_event(self, event_id: str, updates: dict) -> Optional[CalendarEvent]:
        """
        Update an existing calendar event.

        Args:
            event_id: Event identifier
            updates: Dictionary of fields to update

        Returns:
            Updated event if found, None otherwise
        """
        if not self.connected:
            logger.warning("Calendar service not connected")
            return None

        # Find event
        for event in self._mock_events:
            if event.id == event_id:
                # Apply updates
                for key, value in updates.items():
                    if hasattr(event, key):
                        setattr(event, key, value)

                # Save to file
                self._save_mock_data()

                logger.info(f"Updated calendar event: {event_id}")
                return event

        logger.warning(f"Event not found: {event_id}")
        return None

    def delete_event(self, event_id: str) -> bool:
        """
        Delete a calendar event.

        Args:
            event_id: Event identifier

        Returns:
            True if event was deleted, False if not found
        """
        if not self.connected:
            logger.warning("Calendar service not connected")
            return False

        # Find and remove event
        for i, event in enumerate(self._mock_events):
            if event.id == event_id:
                self._mock_events.pop(i)
                self._save_mock_data()
                logger.info(f"Deleted calendar event: {event_id}")
                return True

        logger.warning(f"Event not found: {event_id}")
        return False

    def find_conflicts(self, events: Optional[list[CalendarEvent]] = None) -> list[tuple[CalendarEvent, CalendarEvent]]:
        """
        Find overlapping events.

        Args:
            events: Events to check for conflicts. If None, checks all events.

        Returns:
            List of tuples containing conflicting event pairs
        """
        if events is None:
            events = self._mock_events

        conflicts = []

        # Compare each pair of events
        for i, event1 in enumerate(events):
            for event2 in events[i + 1:]:
                if event1.overlaps_with(event2):
                    conflicts.append((event1, event2))

        logger.info(f"Found {len(conflicts)} conflicts")
        return conflicts

    def get_upcoming_events(self, days: int = 7) -> list[CalendarEvent]:
        """
        Get upcoming events in the next N days.

        Args:
            days: Number of days to look ahead

        Returns:
            List of upcoming CalendarEvent objects
        """
        now = datetime.now()
        end = now + timedelta(days=days)
        return self.get_events(now, end)

    def get_all_events(self) -> list[CalendarEvent]:
        """
        Get all events (for testing purposes).

        Returns:
            List of all CalendarEvent objects
        """
        return self._mock_events.copy()


# Global calendar service instance
_calendar_service: Optional[CalendarService] = None


def get_calendar_service() -> CalendarService:
    """Get the global calendar service instance."""
    global _calendar_service
    if _calendar_service is None:
        _calendar_service = CalendarService()
    return _calendar_service


def set_calendar_service(service: CalendarService) -> None:
    """Set the global calendar service instance (for testing)."""
    global _calendar_service
    _calendar_service = service
