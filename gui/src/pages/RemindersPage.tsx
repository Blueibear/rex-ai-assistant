import React, { useEffect, useState, useCallback } from 'react'
import type { Reminder, ReminderInput } from '../types/ipc'
import { PageLoadingFallback } from '../components/ui/PageLoadingFallback'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { Modal } from '../components/ui/Modal'
import { useToast } from '../components/ui/Toast'

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

function pad(n: number): string {
  return String(n).padStart(2, '0')
}

function toDueDateParts(iso: string): { dueDate: string; dueTime: string } {
  const d = new Date(iso)
  return {
    dueDate: `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`,
    dueTime: `${pad(d.getHours())}:${pad(d.getMinutes())}`
  }
}

function fromDueDateParts(dueDate: string, dueTime: string): string {
  return new Date(`${dueDate}T${dueTime || '00:00'}`).toISOString()
}

interface FormState {
  title: string
  notes: string
  dueDate: string
  dueTime: string
  priority: 'low' | 'medium' | 'high'
  repeat: 'none' | 'daily' | 'weekly' | 'custom'
}

function emptyForm(): FormState {
  const now = new Date()
  return {
    title: '',
    notes: '',
    dueDate: `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`,
    dueTime: `${pad(now.getHours() + 1)}:00`,
    priority: 'medium',
    repeat: 'none'
  }
}

function reminderToForm(r: Reminder): FormState {
  const { dueDate, dueTime } = toDueDateParts(r.dueAt)
  return {
    title: r.title,
    notes: r.notes ?? '',
    dueDate,
    dueTime,
    priority: r.priority,
    repeat: r.repeat ?? 'none'
  }
}

interface ReminderCardProps {
  reminder: Reminder
  onComplete: (id: string) => void
  onEdit: (reminder: Reminder) => void
  showDangerBorder: boolean
}

function ReminderCard({ reminder, onComplete, onEdit, showDangerBorder }: ReminderCardProps): React.ReactElement {
  const [checked, setChecked] = useState(false)

  function handleCheck(e: React.MouseEvent): void {
    e.stopPropagation()
    setChecked(true)
    setTimeout(() => onComplete(reminder.id), 300)
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={() => onEdit(reminder)}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onEdit(reminder) }}
      className={[
        'bg-surface-raised rounded-lg px-4 py-3 flex items-start gap-3 transition-opacity cursor-pointer hover:opacity-80',
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
  onEdit: (reminder: Reminder) => void
  overdueBorders?: boolean
}

function Section({ title, reminders, onComplete, onEdit, overdueBorders = false }: SectionProps): React.ReactElement {
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
              onEdit={onEdit}
              showDangerBorder={overdueBorders}
            />
          ))}
        </div>
      )}
    </div>
  )
}

interface ReminderModalProps {
  editingReminder: Reminder | null
  onClose: () => void
  onSaved: (reminder: Reminder) => void
  onDeleted: (id: string) => void
  onError: (msg: string) => void
}

function ReminderModal({ editingReminder, onClose, onSaved, onDeleted, onError }: ReminderModalProps): React.ReactElement {
  const [form, setForm] = useState<FormState>(
    editingReminder ? reminderToForm(editingReminder) : emptyForm()
  )
  const [errors, setErrors] = useState<{ title?: string; dueDate?: string }>({})
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)

  function setField<K extends keyof FormState>(key: K, value: FormState[K]): void {
    setForm((prev) => ({ ...prev, [key]: value }))
    if (key === 'title' || key === 'dueDate') {
      setErrors((prev) => ({ ...prev, [key]: undefined }))
    }
  }

  async function handleSubmit(e: React.FormEvent): Promise<void> {
    e.preventDefault()
    const newErrors: { title?: string; dueDate?: string } = {}
    if (!form.title.trim()) newErrors.title = 'Title is required'
    if (!form.dueDate) newErrors.dueDate = 'Due date is required'
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors)
      return
    }
    setSaving(true)
    try {
      const input: ReminderInput = {
        id: editingReminder?.id,
        title: form.title.trim(),
        notes: form.notes.trim() || undefined,
        dueAt: fromDueDateParts(form.dueDate, form.dueTime),
        priority: form.priority,
        repeat: form.repeat
      }
      const saved = await window.rex.saveReminder(input)
      onSaved(saved)
    } catch (err) {
      onError(err instanceof Error ? err.message : 'Failed to save reminder')
    } finally {
      setSaving(false)
    }
  }

  async function handleDelete(): Promise<void> {
    if (!editingReminder) return
    setDeleting(true)
    try {
      await window.rex.deleteReminder(editingReminder.id)
      onDeleted(editingReminder.id)
    } catch (err) {
      onError(err instanceof Error ? err.message : 'Failed to delete reminder')
    } finally {
      setDeleting(false)
    }
  }

  const labelCls = 'block text-xs font-medium text-text-secondary mb-1'
  const inputCls =
    'w-full bg-surface border border-border rounded-md px-3 py-2 text-sm text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-accent'

  return (
    <Modal
      title={editingReminder ? 'Edit Reminder' : 'New Reminder'}
      onClose={onClose}
      footer={
        <div className="flex items-center justify-between w-full">
          <div>
            {editingReminder && (
              <button
                type="button"
                onClick={() => void handleDelete()}
                disabled={deleting}
                className="px-4 py-2 text-sm rounded-md text-danger hover:opacity-80 disabled:opacity-50 transition-opacity"
              >
                {deleting ? 'Deleting…' : 'Delete'}
              </button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm rounded-md text-text-secondary hover:text-text-primary transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              form="reminder-form"
              disabled={saving}
              className="px-4 py-2 text-sm rounded-md bg-accent text-white hover:opacity-90 disabled:opacity-50 transition-opacity"
            >
              {saving ? 'Saving…' : editingReminder ? 'Save Changes' : 'Create Reminder'}
            </button>
          </div>
        </div>
      }
    >
      <form id="reminder-form" onSubmit={(e) => void handleSubmit(e)} className="flex flex-col gap-4">
        {/* Title */}
        <div>
          <label htmlFor="rem-title" className={labelCls}>Title *</label>
          <input
            id="rem-title"
            type="text"
            value={form.title}
            onChange={(e) => setField('title', e.target.value)}
            placeholder="Reminder title"
            className={inputCls}
          />
          {errors.title && (
            <p className="mt-1 text-xs text-danger">{errors.title}</p>
          )}
        </div>

        {/* Notes */}
        <div>
          <label htmlFor="rem-notes" className={labelCls}>Notes</label>
          <textarea
            id="rem-notes"
            value={form.notes}
            onChange={(e) => setField('notes', e.target.value)}
            placeholder="Optional notes"
            rows={3}
            className={inputCls + ' resize-none'}
          />
        </div>

        {/* Due date + time */}
        <div className="flex gap-3">
          <div className="flex-1">
            <label htmlFor="rem-date" className={labelCls}>Due Date *</label>
            <input
              id="rem-date"
              type="date"
              value={form.dueDate}
              onChange={(e) => setField('dueDate', e.target.value)}
              className={inputCls}
            />
            {errors.dueDate && (
              <p className="mt-1 text-xs text-danger">{errors.dueDate}</p>
            )}
          </div>
          <div className="flex-1">
            <label htmlFor="rem-time" className={labelCls}>Due Time</label>
            <input
              id="rem-time"
              type="time"
              value={form.dueTime}
              onChange={(e) => setField('dueTime', e.target.value)}
              className={inputCls}
            />
          </div>
        </div>

        {/* Priority */}
        <div>
          <label htmlFor="rem-priority" className={labelCls}>Priority</label>
          <select
            id="rem-priority"
            value={form.priority}
            onChange={(e) => setField('priority', e.target.value as FormState['priority'])}
            className={inputCls}
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>

        {/* Repeat */}
        <div>
          <label htmlFor="rem-repeat" className={labelCls}>Repeat</label>
          <select
            id="rem-repeat"
            value={form.repeat}
            onChange={(e) => setField('repeat', e.target.value as FormState['repeat'])}
            className={inputCls}
          >
            <option value="none">None</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="custom">Custom</option>
          </select>
        </div>
      </form>
    </Modal>
  )
}

export function RemindersPage(): React.ReactElement {
  const [reminders, setReminders] = useState<Reminder[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [showModal, setShowModal] = useState(false)
  const [editingReminder, setEditingReminder] = useState<Reminder | null>(null)
  const addToast = useToast()

  const fetchReminders = useCallback(async () => {
    try {
      const data = await window.rex.getReminders()
      setReminders(data)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load reminders')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void fetchReminders()
  }, [fetchReminders])

  // Auto-refresh every 60 seconds to catch changes made by Rex autonomously
  useEffect(() => {
    const interval = setInterval(() => {
      void fetchReminders()
    }, 60_000)
    return () => clearInterval(interval)
  }, [fetchReminders])

  async function handleComplete(id: string): Promise<void> {
    try {
      await window.rex.completeReminder(id)
      void fetchReminders()
    } catch (e) {
      addToast(e instanceof Error ? e.message : 'Failed to complete reminder', 'error')
    }
  }

  function handleEdit(reminder: Reminder): void {
    setEditingReminder(reminder)
    setShowModal(true)
  }

  function handleNewReminder(): void {
    setEditingReminder(null)
    setShowModal(true)
  }

  function handleModalClose(): void {
    setShowModal(false)
    setEditingReminder(null)
  }

  function handleSaved(_saved: Reminder): void {
    handleModalClose()
    void fetchReminders()
  }

  function handleDeleted(_id: string): void {
    handleModalClose()
    void fetchReminders()
  }

  function handleModalError(msg: string): void {
    addToast(msg, 'error')
  }

  if (loading) {
    return <PageLoadingFallback lines={6} />
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

  return (
    <div className="max-w-2xl mx-auto px-6 py-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-lg font-semibold text-text-primary">Reminders</h1>
        <button
          type="button"
          onClick={handleNewReminder}
          className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded-md bg-accent text-white hover:opacity-90 transition-opacity"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4" aria-hidden="true">
            <path d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" />
          </svg>
          New Reminder
        </button>
      </div>

      {allEmpty ? (
        <div className="flex items-center justify-center" style={{ minHeight: '60vh' }}>
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
      ) : (
        <>
          <Section
            title="Overdue"
            reminders={overdue}
            onComplete={handleComplete}
            onEdit={handleEdit}
            overdueBorders
          />
          <Section
            title="Today"
            reminders={today}
            onComplete={handleComplete}
            onEdit={handleEdit}
          />
          <Section
            title="Upcoming"
            reminders={upcoming}
            onComplete={handleComplete}
            onEdit={handleEdit}
          />
        </>
      )}

      {showModal && (
        <ReminderModal
          editingReminder={editingReminder}
          onClose={handleModalClose}
          onSaved={handleSaved}
          onDeleted={handleDeleted}
          onError={handleModalError}
        />
      )}
    </div>
  )
}
