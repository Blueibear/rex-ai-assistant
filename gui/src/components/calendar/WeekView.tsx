import React, { useEffect, useRef } from 'react'
import type { CalendarEvent } from './CalendarGrid'

interface WeekViewProps {
  weekStart: Date // Sunday of the displayed week
  events: CalendarEvent[]
  onEventClick?: (event: CalendarEvent) => void
}

const HOUR_START = 6
const HOUR_END = 22 // 10pm exclusive end (renders 6am – 10pm)
const HOUR_COUNT = HOUR_END - HOUR_START
const SLOT_HEIGHT = 48 // px per hour

const DAY_NAMES_SHORT = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function isSameDay(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  )
}

function formatHour(h: number): string {
  if (h === 0 || h === 24) return '12 AM'
  if (h === 12) return '12 PM'
  return h < 12 ? `${h} AM` : `${h - 12} PM`
}

export const WeekView: React.FC<WeekViewProps> = ({ weekStart, events, onEventClick }) => {
  const today = new Date()
  const scrollRef = useRef<HTMLDivElement>(null)
  const nowLineRef = useRef<HTMLDivElement>(null)

  const days: Date[] = []
  for (let i = 0; i < 7; i++) {
    const d = new Date(weekStart)
    d.setDate(d.getDate() + i)
    days.push(d)
  }

  // Scroll to current time on mount
  useEffect(() => {
    if (scrollRef.current) {
      const now = new Date()
      const offsetHours = now.getHours() + now.getMinutes() / 60 - HOUR_START
      const scrollTop = Math.max(0, offsetHours * SLOT_HEIGHT - 80)
      scrollRef.current.scrollTop = scrollTop
    }
  }, [])

  const nowHour = new Date().getHours() + new Date().getMinutes() / 60
  const nowVisible = nowHour >= HOUR_START && nowHour < HOUR_END
  const nowTop = (nowHour - HOUR_START) * SLOT_HEIGHT

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Day header row */}
      <div className="flex border-b border-border shrink-0">
        {/* Gutter */}
        <div className="w-14 shrink-0" />
        {days.map((day, i) => {
          const isToday = isSameDay(day, today)
          return (
            <div key={i} className="flex-1 text-center py-2">
              <div className="text-xs text-text-secondary uppercase tracking-wider">
                {DAY_NAMES_SHORT[day.getDay()]}
              </div>
              <div
                className={[
                  'mx-auto mt-0.5 w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium',
                  isToday ? 'bg-accent text-white' : 'text-text-primary'
                ].join(' ')}
              >
                {day.getDate()}
              </div>
            </div>
          )
        })}
      </div>

      {/* Scrollable grid */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        <div
          className="relative flex"
          style={{ height: HOUR_COUNT * SLOT_HEIGHT }}
        >
          {/* Time gutter */}
          <div className="w-14 shrink-0 relative">
            {Array.from({ length: HOUR_COUNT }, (_, i) => i + HOUR_START).map((h) => (
              <div
                key={h}
                className="absolute right-2 text-xs text-text-secondary"
                style={{ top: (h - HOUR_START) * SLOT_HEIGHT - 8 }}
              >
                {formatHour(h)}
              </div>
            ))}
          </div>

          {/* Day columns */}
          <div className="flex flex-1 relative">
            {/* Horizontal hour lines */}
            {Array.from({ length: HOUR_COUNT }, (_, i) => (
              <div
                key={i}
                className="absolute left-0 right-0 border-t border-border"
                style={{ top: i * SLOT_HEIGHT }}
              />
            ))}

            {/* Current time line */}
            {nowVisible && isSameDay(today, today) && (
              <div
                ref={nowLineRef}
                className="absolute left-0 right-0 z-10 flex items-center pointer-events-none"
                style={{ top: nowTop }}
              >
                <div className="w-2 h-2 rounded-full bg-accent -ml-1 shrink-0" />
                <div className="flex-1 border-t-2 border-accent" />
              </div>
            )}

            {/* Events per day column */}
            {days.map((day, colIdx) => {
              const dayEvents = events.filter((e) => isSameDay(new Date(e.start), day))
              return (
                <div key={colIdx} className="flex-1 relative border-l border-border">
                  {dayEvents.map((event) => {
                    const start = new Date(event.start)
                    const end = new Date(event.end)
                    const startH = start.getHours() + start.getMinutes() / 60
                    const endH = end.getHours() + end.getMinutes() / 60
                    const clampedStart = Math.max(startH, HOUR_START)
                    const clampedEnd = Math.min(endH, HOUR_END)
                    if (clampedEnd <= clampedStart) return null
                    const top = (clampedStart - HOUR_START) * SLOT_HEIGHT
                    const height = Math.max((clampedEnd - clampedStart) * SLOT_HEIGHT, 20)
                    return (
                      <button
                        key={event.id}
                        onClick={() => onEventClick?.(event)}
                        className="absolute left-0.5 right-0.5 rounded px-1 py-0.5 text-xs text-left overflow-hidden"
                        style={{
                          top,
                          height,
                          backgroundColor: event.color ? event.color + '33' : 'rgba(59,130,246,0.2)',
                          color: event.color ?? '#3B82F6',
                          borderLeft: `3px solid ${event.color ?? '#3B82F6'}`
                        }}
                      >
                        <div className="font-medium truncate">{event.title}</div>
                      </button>
                    )
                  })}
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

export default WeekView
