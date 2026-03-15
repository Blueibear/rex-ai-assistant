import React, { useState } from 'react'
import { CalendarGrid, CalendarEvent } from '../components/calendar/CalendarGrid'
import { WeekView } from '../components/calendar/WeekView'

type ViewMode = 'month' | 'week'

// Helper: get Sunday of the week containing a given date
function getWeekStart(date: Date): Date {
  const d = new Date(date)
  d.setDate(d.getDate() - d.getDay())
  d.setHours(0, 0, 0, 0)
  return d
}

const MONTH_NAMES = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
]

// Sample stub events for display
const STUB_EVENTS: CalendarEvent[] = [
  {
    id: 'ev1',
    title: 'Team standup',
    start: (() => {
      const d = new Date(); d.setHours(9, 0, 0, 0); return d.toISOString()
    })(),
    end: (() => {
      const d = new Date(); d.setHours(9, 30, 0, 0); return d.toISOString()
    })(),
    color: '#3B82F6'
  },
  {
    id: 'ev2',
    title: 'Lunch with Alex',
    start: (() => {
      const d = new Date(); d.setHours(12, 0, 0, 0); return d.toISOString()
    })(),
    end: (() => {
      const d = new Date(); d.setHours(13, 0, 0, 0); return d.toISOString()
    })(),
    color: '#22C55E'
  },
  {
    id: 'ev3',
    title: 'Project review',
    start: (() => {
      const d = new Date(); d.setDate(d.getDate() + 2); d.setHours(14, 0, 0, 0); return d.toISOString()
    })(),
    end: (() => {
      const d = new Date(); d.setDate(d.getDate() + 2); d.setHours(15, 30, 0, 0); return d.toISOString()
    })(),
    color: '#A855F7'
  }
]

export function CalendarPage(): React.ReactElement {
  const today = new Date()
  const [viewMode, setViewMode] = useState<ViewMode>('month')
  const [year, setYear] = useState(today.getFullYear())
  const [month, setMonth] = useState(today.getMonth())
  const [weekStart, setWeekStart] = useState<Date>(() => getWeekStart(today))
  const [selectedEvent, setSelectedEvent] = useState<CalendarEvent | null>(null)

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
        <h2 className="flex-1 text-base font-semibold text-text-primary">{periodLabel()}</h2>

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
            events={STUB_EVENTS}
            onEventClick={setSelectedEvent}
          />
        ) : (
          <WeekView
            weekStart={weekStart}
            events={STUB_EVENTS}
            onEventClick={setSelectedEvent}
          />
        )}
      </div>

      {/* Simple event detail popover (temporary, replaced in US-197) */}
      {selectedEvent && (
        <div className="fixed inset-0 z-50 flex items-end justify-center pb-8 pointer-events-none">
          <div
            className="bg-surface border border-border rounded-xl shadow-2xl p-4 w-80 pointer-events-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-2">
              <div
                className="w-3 h-3 rounded-full mt-1 shrink-0"
                style={{ backgroundColor: selectedEvent.color ?? '#3B82F6' }}
              />
              <div className="flex-1">
                <p className="font-semibold text-text-primary">{selectedEvent.title}</p>
                <p className="text-xs text-text-secondary mt-0.5">
                  {new Date(selectedEvent.start).toLocaleString(undefined, {
                    weekday: 'short', month: 'short', day: 'numeric',
                    hour: 'numeric', minute: '2-digit'
                  })}
                  {' – '}
                  {new Date(selectedEvent.end).toLocaleTimeString(undefined, {
                    hour: 'numeric', minute: '2-digit'
                  })}
                </p>
              </div>
              <button
                onClick={() => setSelectedEvent(null)}
                className="text-text-secondary hover:text-text-primary"
                aria-label="Close"
              >
                <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
