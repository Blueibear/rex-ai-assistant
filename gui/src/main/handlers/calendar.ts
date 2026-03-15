import { ipcMain } from 'electron'
import type { CalendarEvent, CalendarEventInput } from '../../types/ipc'

// Stub events returned when no real calendar credentials are configured.
function makeStubEvents(): CalendarEvent[] {
  const now = new Date()
  const today = now.toISOString().slice(0, 10)
  const inTwoDays = new Date(now)
  inTwoDays.setDate(inTwoDays.getDate() + 2)
  const twoDaysStr = inTwoDays.toISOString().slice(0, 10)

  return [
    {
      id: 'stub-ev1',
      title: 'Team standup',
      start: `${today}T09:00:00.000Z`,
      end: `${today}T09:30:00.000Z`,
      color: '#3B82F6',
      location: 'Zoom',
      description: 'Daily sync with the team.',
      attendees: ['alice@example.com', 'bob@example.com'],
      source: 'synced'
    },
    {
      id: 'stub-ev2',
      title: 'Lunch with Alex',
      start: `${today}T12:00:00.000Z`,
      end: `${today}T13:00:00.000Z`,
      color: '#22C55E',
      location: 'The Depot Café',
      source: 'rex'
    },
    {
      id: 'stub-ev3',
      title: 'Project review',
      start: `${twoDaysStr}T14:00:00.000Z`,
      end: `${twoDaysStr}T15:30:00.000Z`,
      color: '#A855F7',
      description: 'Quarterly review of the Rex AI roadmap.',
      attendees: ['james@example.com', 'sarah@example.com'],
      source: 'rex'
    }
  ]
}

// In-memory store for stub calendar events (resets on restart).
let calendarStore: CalendarEvent[] = makeStubEvents()

function getCalendarEvents(_start: string, _end: string): CalendarEvent[] {
  // Stub: returns all events regardless of range.
  // With real credentials, filter events between start and end via CalendarService.
  return calendarStore
}

function createCalendarEvent(input: CalendarEventInput): CalendarEvent {
  const event: CalendarEvent = {
    id: `cal-${Date.now()}`,
    title: input.title,
    start: input.start,
    end: input.end,
    color: input.color ?? '#3B82F6',
    location: input.location,
    description: input.description,
    source: 'rex'
  }
  calendarStore = [...calendarStore, event]
  return event
}

function updateCalendarEvent(updated: CalendarEvent): CalendarEvent {
  calendarStore = calendarStore.map((ev) => (ev.id === updated.id ? updated : ev))
  return updated
}

function deleteCalendarEvent(id: string): void {
  calendarStore = calendarStore.filter((ev) => ev.id !== id)
}

export function registerCalendarHandlers(): void {
  ipcMain.handle(
    'rex:getCalendarEvents',
    (_event, start: string, end: string): CalendarEvent[] => {
      return getCalendarEvents(start, end)
    }
  )

  ipcMain.handle(
    'rex:createCalendarEvent',
    (_event, input: CalendarEventInput): CalendarEvent => {
      return createCalendarEvent(input)
    }
  )

  ipcMain.handle(
    'rex:updateCalendarEvent',
    (_event, updated: CalendarEvent): CalendarEvent => {
      return updateCalendarEvent(updated)
    }
  )

  ipcMain.handle('rex:deleteCalendarEvent', (_event, id: string): void => {
    deleteCalendarEvent(id)
  })
}
