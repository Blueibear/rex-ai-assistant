import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import { resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import type { CalendarEvent, CalendarEventInput, FindMeetingSlotsParams, TimeSlot } from '../../types/ipc'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

// In-memory store for locally-created events (created via this GUI session).
let localEvents: CalendarEvent[] = []

function callCalendarBridge(session: ElectronSessionIdentity, command: string, extra: object = {}): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const scriptPath = resolveBridgePath('rex_calendar_bridge.py')
    const py = spawn(resolvePythonCommand(), [scriptPath], {
      stdio: ['pipe', 'pipe', 'pipe']
    })

    let stdout = ''
    let stderr = ''

    py.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString()
    })
    py.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString()
    })

    py.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Calendar bridge exited ${code}: ${stderr.slice(0, 300)}`))
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()))
      } catch {
        reject(new Error(`Failed to parse calendar bridge response: ${stdout.slice(0, 200)}`))
      }
    })

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn calendar bridge: ${err.message}`))
    })

    py.stdin.write(JSON.stringify(privateSessionPayload(session, { command, ...extra })))
    py.stdin.end()
  })
}

async function getCalendarEvents(session: ElectronSessionIdentity, start: string, end: string): Promise<CalendarEvent[]> {
  const result = (await callCalendarBridge(session, 'list', {
    start,
    end
  })) as {
    ok: boolean
    events?: CalendarEvent[]
    error?: string
  }
  if (result.ok && Array.isArray(result.events)) {
    // Merge backend events with locally-created events for the session.
    return [...result.events, ...localEvents]
  }
  throw new Error(result.error ?? 'Calendar bridge did not return events.')
}

async function createCalendarEvent(session: ElectronSessionIdentity, input: CalendarEventInput): Promise<CalendarEvent> {
  const result = (await callCalendarBridge(session, 'create', {
    event: input
  })) as {
    ok: boolean
    event?: CalendarEvent
    error?: string
  }
  if (!result.ok || !result.event) {
    throw new Error(result.error ?? 'Calendar bridge did not create the event.')
  }
  const event: CalendarEvent = {
    ...result.event,
    color: input.color ?? result.event.color ?? '#3B82F6'
  }
  localEvents = [...localEvents, event]
  return event
}

function updateCalendarEvent(updated: CalendarEvent): CalendarEvent {
  localEvents = localEvents.map((ev) => (ev.id === updated.id ? updated : ev))
  return updated
}

function deleteCalendarEvent(id: string): void {
  localEvents = localEvents.filter((ev) => ev.id !== id)
}

export function registerCalendarHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getCalendarEvents', async (_event, start: string, end: string): Promise<CalendarEvent[]> => {
    return getCalendarEvents(session, start, end)
  })

  ipcMain.handle('rex:createCalendarEvent', async (_event, input: CalendarEventInput): Promise<CalendarEvent> => {
    return createCalendarEvent(session, input)
  })

  ipcMain.handle('rex:updateCalendarEvent', (_event, updated: CalendarEvent): CalendarEvent => {
    return updateCalendarEvent(updated)
  })

  ipcMain.handle('rex:deleteCalendarEvent', (_event, id: string): void => {
    deleteCalendarEvent(id)
  })

  ipcMain.handle('rex:findMeetingSlots', (_event, params: FindMeetingSlotsParams): TimeSlot[] => {
    // Returns 3 future slots spaced 2 hours apart starting from earliest.
    const base = new Date(params.earliest)
    base.setMinutes(0, 0, 0)
    base.setHours(base.getHours() + 1)

    const slots: TimeSlot[] = []
    for (let i = 0; i < 3; i++) {
      const start = new Date(base.getTime() + i * 2 * 3600 * 1000)
      const end = new Date(start.getTime() + params.durationMinutes * 60 * 1000)
      slots.push({
        start: start.toISOString(),
        end: end.toISOString(),
        confidence: 0.9 - i * 0.1
      })
    }
    return slots
  })
}
