import React, { useState, useEffect, useCallback } from 'react'
import { CalendarGrid } from '../components/calendar/CalendarGrid'
import type { CalendarEvent } from '../types/ipc'
import type { CalendarEventInput } from '../types/ipc'
import { WeekView } from '../components/calendar/WeekView'
import { EventDetailPanel } from '../components/calendar/EventDetailPanel'
import { Modal } from '../components/ui/Modal'
import { useToast } from '../components/ui/Toast'
import { Spinner } from '../components/ui/Spinner'

type ViewMode = 'month' | 'week'

// Helper: get Sunday of the week containing a given date
function getWeekStart(date: Date): Date {
  const d = new Date(date)
  d.setDate(d.getDate() - d.getDay())
  d.setHours(0, 0, 0, 0)
  return d
}

// Get ISO range for visible calendar period
function getRangeForPeriod(viewMode: ViewMode, year: number, month: number, weekStart: Date): { start: string; end: string } {
  if (viewMode === 'month') {
    const start = new Date(year, month, 1)
    const end = new Date(year, month + 1, 0, 23, 59, 59)
    return { start: start.toISOString(), end: end.toISOString() }
  } else {
    const end = new Date(weekStart)
    end.setDate(end.getDate() + 6)
    end.setHours(23, 59, 59)
    return { start: weekStart.toISOString(), end: end.toISOString() }
  }
}

const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
]

// Zero-pad helper for datetime-local inputs
function pad(n: number): string {
  return n.toString().padStart(2, '0')
}

function toDatetimeLocal(iso: string): string {
  const d = new Date(iso)
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
}

function fromDatetimeLocal(val: string): string {
  return new Date(val).toISOString()
}

interface EventFormState {
  title: string
  start: string // datetime-local format
  end: string
  location: string
  description: string
  color: string
}

function defaultFormState(base?: CalendarEvent): EventFormState {
  if (base) {
    return {
      title: base.title,
      start: toDatetimeLocal(base.start),
      end: toDatetimeLocal(base.end),
      location: base.location ?? '',
      description: base.description ?? '',
      color: base.color ?? '#3B82F6'
    }
  }
  const now = new Date()
  const later = new Date(now.getTime() + 60 * 60 * 1000)
  return {
    title: '',
    start: toDatetimeLocal(now.toISOString()),
    end: toDatetimeLocal(later.toISOString()),
    location: '',
    description: '',
    color: '#3B82F6'
  }
}

const COLOR_OPTIONS = [
  { label: 'Blue', value: '#3B82F6' },
  { label: 'Green', value: '#22C55E' },
  { label: 'Purple', value: '#A855F7' },
  { label: 'Red', value: '#EF4444' },
  { label: 'Amber', value: '#F59E0B' }
]

interface EventFormProps {
  form: EventFormState
  onChange: (f: EventFormState) => void
  error: string
}

function EventForm({ form, onChange, error }: EventFormProps): React.ReactElement {
  function set(key: keyof EventFormState, val: string): void {
    onChange({ ...form, [key]: val })
  }

  return (
    <div className="space-y-3">
      {error && (
        <p className="text-sm text-danger">{error}</p>
      )}
      <div>
        <label className="block text-xs font-medium text-text-secondary mb-1">Title *</label>
        <input
          type="text"
          value={form.title}
          onChange={(e) => set('title', e.target.value)}
          className="w-full px-3 py-2 rounded bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent"
          placeholder="Event title"
        />
      </div>
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">Start *</label>
          <input
            type="datetime-local"
            value={form.start}
            onChange={(e) => set('start', e.target.value)}
            className="w-full px-3 py-2 rounded bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-text-secondary mb-1">End *</label>
          <input
            type="datetime-local"
            value={form.end}
            onChange={(e) => set('end', e.target.value)}
            className="w-full px-3 py-2 rounded bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent"
          />
        </div>
      </div>
      <div>
        <label className="block text-xs font-medium text-text-secondary mb-1">Location</label>
        <input
          type="text"
          value={form.location}
          onChange={(e) => set('location', e.target.value)}
          className="w-full px-3 py-2 rounded bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent"
          placeholder="Location (optional)"
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-text-secondary mb-1">Description</label>
        <textarea
          value={form.description}
          onChange={(e) => set('description', e.target.value)}
          rows={3}
          className="w-full px-3 py-2 rounded bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent resize-none"
          placeholder="Description (optional)"
        />
      </div>
      <div>
        <label className="block text-xs font-medium text-text-secondary mb-1">Color</label>
        <div className="flex gap-2">
          {COLOR_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => set('color', opt.value)}
              className={[
                'w-7 h-7 rounded-full transition-all',
                form.color === opt.value ? 'ring-2 ring-offset-2 ring-accent ring-offset-surface' : ''
              ].join(' ')}
              style={{ backgroundColor: opt.value }}
              aria-label={opt.label}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

export function CalendarPage(): React.ReactElement {
  const today = new Date()
  const addToast = useToast()

  const [viewMode, setViewMode] = useState<ViewMode>('month')
  const [year, setYear] = useState(today.getFullYear())
  const [month, setMonth] = useState(today.getMonth())
  const [weekStart, setWeekStart] = useState<Date>(() => getWeekStart(today))

  const [events, setEvents] = useState<CalendarEvent[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | null>(null)

  // Create-event modal state
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [createForm, setCreateForm] = useState<EventFormState>(() => defaultFormState())
  const [createError, setCreateError] = useState('')
  const [creating, setCreating] = useState(false)

  // Edit-event modal state
  const [showEditModal, setShowEditModal] = useState(false)
  const [editForm, setEditForm] = useState<EventFormState>(() => defaultFormState())
  const [editError, setEditError] = useState('')
  const [editing, setEditing] = useState(false)
  const [editingEventId, setEditingEventId] = useState<string | null>(null)

  // Load events for the current visible range
  const loadEvents = useCallback(async (vm: ViewMode, y: number, m: number, ws: Date) => {
    setLoading(true)
    try {
      const { start, end } = getRangeForPeriod(vm, y, m, ws)
      const fetched = await window.rex.getCalendarEvents(start, end)
      setEvents(fetched)
    } catch (err) {
      addToast('Failed to load calendar events', 'error')
    } finally {
      setLoading(false)
    }
  }, [addToast])

  // Reload when the visible period changes
  useEffect(() => {
    void loadEvents(viewMode, year, month, weekStart)
  }, [viewMode, year, month, weekStart, loadEvents])

  function prevPeriod(): void {
    if (viewMode === 'month') {
      if (month === 0) { setMonth(11); setYear((y) => y - 1) }
      else setMonth((m) => m - 1)
    } else {
      const prev = new Date(weekStart)
      prev.setDate(prev.getDate() - 7)
      setWeekStart(prev)
    }
  }

  function nextPeriod(): void {
    if (viewMode === 'month') {
      if (month === 11) { setMonth(0); setYear((y) => y + 1) }
      else setMonth((m) => m + 1)
    } else {
      const next = new Date(weekStart)
      next.setDate(next.getDate() + 7)
      setWeekStart(next)
    }
  }

  function goToToday(): void {
    setYear(today.getFullYear())
    setMonth(today.getMonth())
    setWeekStart(getWeekStart(today))
  }

  function periodLabel(): string {
    if (viewMode === 'month') return `${MONTH_NAMES[month]} ${year}`
    const weekEnd = new Date(weekStart)
    weekEnd.setDate(weekEnd.getDate() + 6)
    const startLabel = `${MONTH_NAMES[weekStart.getMonth()]} ${weekStart.getDate()}`
    const endLabel = weekEnd.getMonth() !== weekStart.getMonth()
      ? `${MONTH_NAMES[weekEnd.getMonth()]} ${weekEnd.getDate()}, ${weekEnd.getFullYear()}`
      : `${weekEnd.getDate()}, ${weekEnd.getFullYear()}`
    return `${startLabel} – ${endLabel}`
  }

  // ---- Create event ----
  function openCreateModal(): void {
    setCreateForm(defaultFormState())
    setCreateError('')
    setShowCreateModal(true)
  }

  async function handleCreate(): Promise<void> {
    if (!createForm.title.trim()) { setCreateError('Title is required.'); return }
    if (!createForm.start || !createForm.end) { setCreateError('Start and end are required.'); return }
    if (new Date(createForm.start) >= new Date(createForm.end)) {
      setCreateError('End must be after start.')
      return
    }
    setCreating(true)
    setCreateError('')
    try {
      const input: CalendarEventInput = {
        title: createForm.title.trim(),
        start: fromDatetimeLocal(createForm.start),
        end: fromDatetimeLocal(createForm.end),
        color: createForm.color,
        location: createForm.location.trim() || undefined,
        description: createForm.description.trim() || undefined
      }
      const created = await window.rex.createCalendarEvent(input)
      setEvents((prev) => [...prev, created])
      setShowCreateModal(false)
      addToast('Event created', 'success')
    } catch {
      setCreateError('Failed to create event.')
    } finally {
      setCreating(false)
    }
  }

  // ---- Edit event ----
  function openEditModal(event: CalendarEvent): void {
    setEditingEventId(event.id)
    setEditForm(defaultFormState(event))
    setEditError('')
    setShowEditModal(true)
    setSelectedEvent(null)
  }

  async function handleEdit(): Promise<void> {
    if (!editForm.title.trim()) { setEditError('Title is required.'); return }
    if (!editForm.start || !editForm.end) { setEditError('Start and end are required.'); return }
    if (new Date(editForm.start) >= new Date(editForm.end)) {
      setEditError('End must be after start.')
      return
    }
    if (!editingEventId) return
    setEditing(true)
    setEditError('')
    try {
      const existing = events.find((e) => e.id === editingEventId)
      const updated: CalendarEvent = {
        ...existing,
        id: editingEventId,
        title: editForm.title.trim(),
        start: fromDatetimeLocal(editForm.start),
        end: fromDatetimeLocal(editForm.end),
        color: editForm.color,
        location: editForm.location.trim() || undefined,
        description: editForm.description.trim() || undefined
      }
      const result = await window.rex.updateCalendarEvent(updated)
      setEvents((prev) => prev.map((e) => (e.id === result.id ? result : e)))
      setShowEditModal(false)
      setEditingEventId(null)
      addToast('Event updated', 'success')
    } catch {
      setEditError('Failed to update event.')
    } finally {
      setEditing(false)
    }
  }

  // ---- Delete event ----
  async function handleDelete(event: CalendarEvent): Promise<void> {
    try {
      await window.rex.deleteCalendarEvent(event.id)
      setEvents((prev) => prev.filter((e) => e.id !== event.id))
      setSelectedEvent(null)
      addToast('Event deleted', 'success')
    } catch {
      addToast('Failed to delete event', 'error')
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-3 border-b border-border shrink-0">
        <button
          onClick={goToToday}
          className="px-3 py-1.5 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors"
        >
          Today
        </button>
        <button
          onClick={prevPeriod}
          className="p-1.5 rounded text-text-secondary hover:text-text-primary hover:bg-surface-raised transition-colors"
          aria-label="Previous"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path
              fillRule="evenodd"
              d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </button>
        <button
          onClick={nextPeriod}
          className="p-1.5 rounded text-text-secondary hover:text-text-primary hover:bg-surface-raised transition-colors"
          aria-label="Next"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path
              fillRule="evenodd"
              d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
              clipRule="evenodd"
            />
          </svg>
        </button>
        <h2 className="flex-1 text-base font-semibold text-text-primary">
          {periodLabel()}
          {loading && <Spinner size="sm" className="inline-block ml-2 align-middle" />}
        </h2>

        {/* New Event button */}
        <button
          onClick={openCreateModal}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path
              fillRule="evenodd"
              d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z"
              clipRule="evenodd"
            />
          </svg>
          New Event
        </button>

        {/* View toggle segmented control */}
        <div className="flex rounded overflow-hidden border border-border">
          {(['month', 'week'] as ViewMode[]).map((mode) => (
            <button
              key={mode}
              onClick={() => setViewMode(mode)}
              className={[
                'px-3 py-1.5 text-sm font-medium capitalize transition-colors',
                viewMode === mode
                  ? 'bg-accent text-white'
                  : 'bg-surface text-text-secondary hover:text-text-primary'
              ].join(' ')}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {/* Calendar content */}
      <div className="flex-1 overflow-hidden">
        {viewMode === 'month' ? (
          <CalendarGrid
            year={year}
            month={month}
            events={events}
            onEventClick={setSelectedEvent}
          />
        ) : (
          <WeekView
            weekStart={weekStart}
            events={events}
            onEventClick={setSelectedEvent}
          />
        )}
      </div>

      {/* Event detail slide-in panel */}
      <EventDetailPanel
        event={selectedEvent}
        onClose={() => setSelectedEvent(null)}
        onEdit={openEditModal}
        onDelete={(ev) => void handleDelete(ev)}
      />

      {/* Create Event Modal */}
      {showCreateModal && <Modal
        onClose={() => setShowCreateModal(false)}
        title="New Event"
        footer={
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => setShowCreateModal(false)}
              className="px-4 py-2 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors"
              disabled={creating}
            >
              Cancel
            </button>
            <button
              onClick={() => void handleCreate()}
              disabled={creating}
              className="px-4 py-2 rounded text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors disabled:opacity-60"
            >
              {creating ? 'Creating…' : 'Create Event'}
            </button>
          </div>
        }
      >
        <EventForm form={createForm} onChange={setCreateForm} error={createError} />
      </Modal>}

      {/* Edit Event Modal */}
      {showEditModal && <Modal
        onClose={() => { setShowEditModal(false); setEditingEventId(null) }}
        title="Edit Event"
        footer={
          <div className="flex gap-2 justify-end">
            <button
              onClick={() => { setShowEditModal(false); setEditingEventId(null) }}
              className="px-4 py-2 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors"
              disabled={editing}
            >
              Cancel
            </button>
            <button
              onClick={() => void handleEdit()}
              disabled={editing}
              className="px-4 py-2 rounded text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors disabled:opacity-60"
            >
              {editing ? 'Saving…' : 'Save Changes'}
            </button>
          </div>
        }
      >
        <EventForm form={editForm} onChange={setEditForm} error={editError} />
      </Modal>}
    </div>
  )
}
