import { ipcMain } from 'electron'
import type { CalendarEvent, CalendarEventInput, FindMeetingSlotsParams, TimeSlot } from '../../types/ipc'

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

  ipcMain.handle(
    'rex:findMeetingSlots',
    (_event, params: FindMeetingSlotsParams): TimeSlot[] => {
      // Stub: returns 3 future slots spaced 2 hours apart starting from earliest.
      // With real credentials this would call SchedulingEngine.find_slots() via Python.
      const base = new Date(params.earliest)
      base.setMinutes(0, 0, 0)
      base.setHours(base.getHours() + 1)

      const slots: TimeSlot[] = []
      for (let i = 0; i < 3; i++) {
        const start = new Date(base.getTime() + i * 2 * 3600 * 1000)
        const end = new Date(start.getTime() + params.durationMinutes * 60 * 1000)
        slots.push({ start: start.toISOString(), end: end.toISOString(), confidence: 0.9 - i * 0.1 })
      }
      return slots
    }
  )
}
