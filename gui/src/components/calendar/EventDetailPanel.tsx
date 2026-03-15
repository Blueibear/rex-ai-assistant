import React, { useEffect, useRef } from 'react'
import { CalendarEvent } from './CalendarGrid'

interface EventDetailPanelProps {
  event: CalendarEvent | null
  onClose: () => void
  onEdit?: (event: CalendarEvent) => void
  onDelete?: (event: CalendarEvent) => void
}

function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  })
}

function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit'
  })
}

function durationLabel(start: string, end: string): string {
  const ms = new Date(end).getTime() - new Date(start).getTime()
  const totalMin = Math.round(ms / 60000)
  if (totalMin < 60) return `${totalMin} min`
  const h = Math.floor(totalMin / 60)
  const m = totalMin % 60
  return m === 0 ? `${h} hr` : `${h} hr ${m} min`
}

export const EventDetailPanel: React.FC<EventDetailPanelProps> = ({
  event,
  onClose,
  onEdit,
  onDelete
}) => {
  const panelRef = useRef<HTMLDivElement>(null)
  const isOpen = event !== null

  // Close on Escape
  useEffect(() => {
    if (!isOpen) return
    function handleKey(e: KeyboardEvent): void {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [isOpen, onClose])

  // Close on backdrop click
  function handleBackdropClick(e: React.MouseEvent<HTMLDivElement>): void {
    if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
      onClose()
    }
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className={[
          'fixed inset-0 z-40 transition-opacity duration-200',
          isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        ].join(' ')}
        onClick={handleBackdropClick}
        aria-hidden="true"
      />

      {/* Slide-in panel */}
      <div
        ref={panelRef}
        className={[
          'fixed top-0 right-0 bottom-0 z-50 w-80 bg-surface border-l border-border shadow-2xl',
          'flex flex-col transition-transform duration-250 ease-in-out',
          isOpen ? 'translate-x-0' : 'translate-x-full'
        ].join(' ')}
        role="dialog"
        aria-modal="true"
        aria-label="Event details"
      >
        {event && (
          <>
            {/* Panel header */}
            <div className="flex items-start gap-3 px-4 pt-5 pb-4 border-b border-border shrink-0">
              <div
                className="w-3 h-3 rounded-full mt-1 shrink-0"
                style={{ backgroundColor: event.color ?? '#3B82F6' }}
              />
              <div className="flex-1 min-w-0">
                <h2 className="font-semibold text-text-primary text-base leading-snug">
                  {event.title}
                </h2>
                {event.source && (
                  <span className={[
                    'inline-block mt-1 text-xs px-1.5 py-0.5 rounded font-medium',
                    event.source === 'rex'
                      ? 'bg-accent/20 text-accent'
                      : 'bg-surface-raised text-text-secondary'
                  ].join(' ')}>
                    {event.source === 'rex' ? 'Rex-created' : 'Synced'}
                  </span>
                )}
              </div>
              <button
                onClick={onClose}
                className="text-text-secondary hover:text-text-primary transition-colors p-1 rounded shrink-0"
                aria-label="Close panel"
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

            {/* Panel body */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
              {/* Date / Time */}
              <div className="flex gap-3">
                <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-text-secondary shrink-0 mt-0.5">
                  <path
                    fillRule="evenodd"
                    d="M6 2a1 1 0 00-1 1v1H4a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2h-1V3a1 1 0 10-2 0v1H7V3a1 1 0 00-1-1zm0 5a1 1 0 000 2h8a1 1 0 100-2H6z"
                    clipRule="evenodd"
                  />
                </svg>
                <div>
                  <p className="text-sm text-text-primary">{formatDateTime(event.start)}</p>
                  <p className="text-xs text-text-secondary mt-0.5">
                    until {formatTime(event.end)} · {durationLabel(event.start, event.end)}
                  </p>
                </div>
              </div>

              {/* Location */}
              {event.location && (
                <div className="flex gap-3">
                  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-text-secondary shrink-0 mt-0.5">
                    <path
                      fillRule="evenodd"
                      d="M5.05 4.05a7 7 0 119.9 9.9L10 18.9l-4.95-4.95a7 7 0 010-9.9zM10 11a2 2 0 100-4 2 2 0 000 4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <p className="text-sm text-text-primary">{event.location}</p>
                </div>
              )}

              {/* Description */}
              {event.description && (
                <div className="flex gap-3">
                  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-text-secondary shrink-0 mt-0.5">
                    <path
                      fillRule="evenodd"
                      d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <p className="text-sm text-text-primary leading-relaxed">{event.description}</p>
                </div>
              )}

              {/* Attendees */}
              {event.attendees && event.attendees.length > 0 && (
                <div className="flex gap-3">
                  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4 text-text-secondary shrink-0 mt-0.5">
                    <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
                  </svg>
                  <div>
                    <p className="text-xs font-medium text-text-secondary uppercase tracking-wider mb-1">
                      Attendees
                    </p>
                    <ul className="space-y-1">
                      {event.attendees.map((a) => (
                        <li key={a} className="text-sm text-text-primary">{a}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </div>

            {/* Panel footer — action buttons */}
            <div className="shrink-0 px-4 py-4 border-t border-border flex gap-2">
              <button
                onClick={() => onEdit?.(event)}
                className="flex-1 px-3 py-2 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors"
              >
                Edit
              </button>
              <button
                onClick={() => onDelete?.(event)}
                className="flex-1 px-3 py-2 rounded text-sm font-medium bg-danger/20 text-danger hover:bg-danger/30 transition-colors"
              >
                Delete
              </button>
            </div>
          </>
        )}
      </div>
    </>
  )
}

export default EventDetailPanel
