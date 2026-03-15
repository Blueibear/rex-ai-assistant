import React from 'react'

export interface CalendarEvent {
  id: string
  title: string
  start: string // ISO date string
  end: string
  color?: string
  location?: string
  description?: string
  attendees?: string[]
  source?: 'rex' | 'synced'
}

interface CalendarGridProps {
  year: number
  month: number // 0-based
  events: CalendarEvent[]
  onEventClick?: (event: CalendarEvent) => void
}

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

function isSameDay(a: Date, b: Date): boolean {
  return (
    a.getFullYear() === b.getFullYear() &&
    a.getMonth() === b.getMonth() &&
    a.getDate() === b.getDate()
  )
}

export const CalendarGrid: React.FC<CalendarGridProps> = ({
  year,
  month,
  events,
  onEventClick
}) => {
  const today = new Date()
  const firstDay = new Date(year, month, 1)
  const lastDay = new Date(year, month + 1, 0)
  const startOffset = firstDay.getDay() // 0 = Sunday

  // Build grid cells: leading empty + days + trailing empty
  const cells: (number | null)[] = []
  for (let i = 0; i < startOffset; i++) cells.push(null)
  for (let d = 1; d <= lastDay.getDate(); d++) cells.push(d)
  // Pad to complete last row
  while (cells.length % 7 !== 0) cells.push(null)

  return (
    <div className="flex flex-col h-full">
      {/* Day headers */}
      <div className="grid grid-cols-7 border-b border-border">
        {DAY_NAMES.map((day) => (
          <div
            key={day}
            className="py-2 text-center text-xs font-medium text-text-secondary uppercase tracking-wider"
          >
            {day}
          </div>
        ))}
      </div>

      {/* Grid rows */}
      <div className="flex-1 grid grid-cols-7 grid-rows-[repeat(auto-fill,minmax(80px,1fr))]">
        {cells.map((day, idx) => {
          if (day === null) {
            return (
              <div key={`empty-${idx}`} className="border-r border-b border-border bg-bg/30" />
            )
          }
          const cellDate = new Date(year, month, day)
          const isToday = isSameDay(cellDate, today)
          const dayEvents = events.filter((e) => isSameDay(new Date(e.start), cellDate))

          return (
            <div
              key={day}
              className="border-r border-b border-border p-1 min-h-[80px] flex flex-col gap-0.5"
            >
              <span
                className={[
                  'inline-flex items-center justify-center w-6 h-6 rounded-full text-sm font-medium self-start mb-0.5',
                  isToday
                    ? 'bg-accent text-white'
                    : 'text-text-primary'
                ].join(' ')}
              >
                {day}
              </span>
              {dayEvents.slice(0, 3).map((event) => (
                <button
                  key={event.id}
                  onClick={() => onEventClick?.(event)}
                  className={[
                    'w-full text-left text-xs px-1 py-0.5 rounded truncate',
                    event.color
                      ? ''
                      : 'bg-accent/20 text-accent hover:bg-accent/30'
                  ].join(' ')}
                  style={event.color ? { backgroundColor: event.color + '33', color: event.color } : undefined}
                >
                  {event.title}
                </button>
              ))}
              {dayEvents.length > 3 && (
                <span className="text-xs text-text-secondary pl-1">
                  +{dayEvents.length - 3} more
                </span>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default CalendarGrid
