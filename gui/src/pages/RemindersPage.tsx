import React, { useEffect, useState, useCallback } from 'react'
import type { Reminder } from '../types/ipc'
import { Spinner } from '../components/ui/Spinner'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'

type BadgeVariant = 'default' | 'accent' | 'success' | 'warning' | 'danger'

function priorityVariant(priority: Reminder['priority']): BadgeVariant {
  if (priority === 'high') return 'danger'
  if (priority === 'medium') return 'warning'
  return 'default'
}

function formatDueTime(iso: string): string {
  const d = new Date(iso)
  const now = new Date()
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  const dueStart = new Date(d.getFullYear(), d.getMonth(), d.getDate())
  const diffDays = Math.round((dueStart.getTime() - todayStart.getTime()) / (24 * 60 * 60 * 1000))
  const timeStr = d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  if (diffDays < 0) {
    const days = Math.abs(diffDays)
    return days === 1 ? `Yesterday at ${timeStr}` : `${days} days ago at ${timeStr}`
  }
  if (diffDays === 0) return `Today at ${timeStr}`
  if (diffDays === 1) return `Tomorrow at ${timeStr}`
  return `${d.toLocaleDateString([], { month: 'short', day: 'numeric' })} at ${timeStr}`
}

function isOverdue(reminder: Reminder): boolean {
  return new Date(reminder.dueAt) < new Date()
}

function isToday(reminder: Reminder): boolean {
  const d = new Date(reminder.dueAt)
  const now = new Date()
  return (
    d.getFullYear() === now.getFullYear() &&
    d.getMonth() === now.getMonth() &&
    d.getDate() === now.getDate() &&
    !isOverdue(reminder)
  )
}

interface ReminderCardProps {
  reminder: Reminder
  onComplete: (id: string) => void
  showDangerBorder: boolean
}

function ReminderCard({ reminder, onComplete, showDangerBorder }: ReminderCardProps): React.ReactElement {
  const [checked, setChecked] = useState(false)

  function handleCheck(): void {
    setChecked(true)
    setTimeout(() => onComplete(reminder.id), 300)
  }

  return (
    <div
      className={[
        'bg-surface-raised rounded-lg px-4 py-3 flex items-start gap-3 transition-opacity',
        showDangerBorder ? 'border-l-4 border-l-danger' : '',
        checked ? 'opacity-40' : ''
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {/* Checkbox */}
      <button
        type="button"
        onClick={handleCheck}
        aria-label="Mark as done"
        disabled={checked}
        className={[
          'mt-0.5 w-5 h-5 rounded-full border-2 flex-shrink-0 flex items-center justify-center transition-colors',
          checked
            ? 'border-success bg-success'
            : 'border-border hover:border-accent'
        ].join(' ')}
      >
        {checked && (
          <svg viewBox="0 0 12 12" fill="none" className="w-3 h-3">
            <path d="M2 6l3 3 5-5" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </button>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className={['text-sm font-medium', checked ? 'line-through text-text-secondary' : 'text-text-primary'].join(' ')}>
            {reminder.title}
          </span>
          <Badge variant={priorityVariant(reminder.priority)}>
            {reminder.priority}
          </Badge>
        </div>
        {reminder.notes && (
          <p className="text-xs text-text-secondary mt-0.5 truncate">{reminder.notes}</p>
        )}
        <p className="text-xs text-text-secondary mt-1">{formatDueTime(reminder.dueAt)}</p>
      </div>
    </div>
  )
}

interface SectionProps {
  title: string
  reminders: Reminder[]
  onComplete: (id: string) => void
  overdueBorders?: boolean
}

function Section({ title, reminders, onComplete, overdueBorders = false }: SectionProps): React.ReactElement {
  return (
    <div className="mb-8">
      <h2 className="text-xs font-semibold uppercase tracking-widest text-text-secondary mb-3">{title}</h2>
      {reminders.length === 0 ? (
        <div className="bg-surface-raised rounded-lg px-4 py-3 text-sm text-text-secondary">
          No reminders in this group
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          {reminders.map((r) => (
            <ReminderCard
              key={r.id}
              reminder={r}
              onComplete={onComplete}
              showDangerBorder={overdueBorders}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export function RemindersPage(): React.ReactElement {
  const [reminders, setReminders] = useState<Reminder[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchReminders = useCallback(async () => {
    try {
      const data = await window.rex.getReminders()
      setReminders(data)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load reminders')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchReminders()
  }, [fetchReminders])

  async function handleComplete(id: string): Promise<void> {
    try {
      await window.rex.completeReminder(id)
      setReminders((prev) => prev.filter((r) => r.id !== id))
    } catch {
      // silently ignore — card already shows checked state
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner size="lg" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center h-full">
        <EmptyState
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z" />
            </svg>
          }
          heading="Could not load reminders"
          subtext={error}
        />
      </div>
    )
  }

  const overdue = reminders.filter((r) => isOverdue(r))
  const today = reminders.filter((r) => isToday(r))
  const upcoming = reminders.filter((r) => {
    const d = new Date(r.dueAt)
    const now = new Date()
    const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate())
    const tomorrowStart = new Date(todayStart.getTime() + 24 * 60 * 60 * 1000)
    return d >= tomorrowStart
  })

  const allEmpty = overdue.length === 0 && today.length === 0 && upcoming.length === 0

  if (allEmpty) {
    return (
      <div className="flex items-center justify-center h-full">
        <EmptyState
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-10 h-10">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V4a2 2 0 10-4 0v1.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
          }
          heading="No reminders"
          subtext="You're all caught up. Rex will surface reminders here as they arrive."
        />
      </div>
    )
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-6">
      <Section
        title="Overdue"
        reminders={overdue}
        onComplete={handleComplete}
        overdueBorders
      />
      <Section
        title="Today"
        reminders={today}
        onComplete={handleComplete}
      />
      <Section
        title="Upcoming"
        reminders={upcoming}
        onComplete={handleComplete}
      />
    </div>
  )
}
