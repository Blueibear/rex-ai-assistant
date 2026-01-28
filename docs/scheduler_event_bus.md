# Scheduler, Event Bus, Email, and Calendar Integrations

This document describes the Phase 4/5 scheduling and integration layer in Rex.

## Scheduler

The scheduler persists job definitions in `data/scheduler/jobs.json` and supports:

- Recurring execution with fixed intervals (in seconds).
- Manual execution via CLI (`rex scheduler run <job_id>`).
- Handlers that trigger service functions (e.g., email triage).

Default jobs are registered at startup:

- **Email triage** (every 15 minutes)
- **Calendar sync** (every 30 minutes)

## Event Bus

The event bus provides a lightweight publish/subscribe mechanism for internal
signals. Services publish events, and listeners can subscribe to react. The
event bus is process-local and does not persist data.

Event types used by the integrations:

- `email.unread`
- `email.triaged`
- `calendar.upcoming`
- `calendar.created`

## Email Service (Mock)

`EmailService` is read-only and uses mock messages by default. It:

- Fetches unread messages.
- Categorizes and summarizes them.
- Publishes `email.unread` and `email.triaged` events.

The CLI (`rex email unread`) prints summaries without exposing full message
bodies.

## Calendar Service (Mock)

`CalendarService` is backed by mock data and supports:

- Listing upcoming events.
- Creating events (with conflict detection support).
- Publishing `calendar.upcoming` and `calendar.created` events.

The CLI (`rex calendar upcoming`) lists upcoming events without including
private descriptions.
