import React, { useEffect, useState } from 'react'
import type { Task, TaskInput } from '../types/ipc'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { EmptyState } from '../components/ui/EmptyState'
import { Input } from '../components/ui/Input'
import { Modal } from '../components/ui/Modal'
import { Spinner } from '../components/ui/Spinner'
import { Textarea } from '../components/ui/Textarea'

// ---------------------------------------------------------------------------
// Schedule helpers
// ---------------------------------------------------------------------------

type ScheduleType = 'every-hour' | 'every-day' | 'every-week' | 'custom-cron'

function buildScheduleString(
  type: ScheduleType,
  time: string,
  day: string,
  cron: string
): string {
  switch (type) {
    case 'every-hour':
      return 'Every hour'
    case 'every-day':
      return `Every day at ${time}`
    case 'every-week':
      return `Every ${day} at ${time}`
    case 'custom-cron':
      return cron.trim() || 'Custom cron'
  }
}

function parseScheduleType(schedule: string): ScheduleType {
  if (schedule === 'Every hour') return 'every-hour'
  if (schedule.startsWith('Every day at')) return 'every-day'
  if (/^Every \w+ at/.test(schedule)) return 'every-week'
  return 'custom-cron'
}

function parseScheduleTime(schedule: string): string {
  const m = schedule.match(/at (.+)$/)
  return m ? m[1] : '08:00'
}

function parseScheduleDay(schedule: string): string {
  // "Every monday at 08:00" → "monday"
  const m = schedule.match(/^Every (\w+) at/)
  const days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
  if (m && days.includes(m[1].toLowerCase())) return m[1].toLowerCase()
  return 'monday'
}

// ---------------------------------------------------------------------------
// Icons
// ---------------------------------------------------------------------------

function CalendarIcon(): React.ReactElement {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="w-8 h-8"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.5}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M8 7V3m8 4V3M3 11h18M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
      />
    </svg>
  )
}

function PlusIcon(): React.ReactElement {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      className="w-4 h-4"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Status badge helper
// ---------------------------------------------------------------------------

function statusVariant(status: Task['status']): 'success' | 'warning' | 'danger' {
  if (status === 'active') return 'success'
  if (status === 'paused') return 'warning'
  return 'danger'
}

// ---------------------------------------------------------------------------
// Form state type
// ---------------------------------------------------------------------------

interface FormState {
  name: string
  prompt: string
  scheduleType: ScheduleType
  scheduleTime: string
  scheduleDay: string
  scheduleCron: string
  active: boolean
}

interface FormErrors {
  name?: string
  schedule?: string
}

function emptyForm(): FormState {
  return {
    name: '',
    prompt: '',
    scheduleType: 'every-hour',
    scheduleTime: '08:00',
    scheduleDay: 'monday',
    scheduleCron: '',
    active: true
  }
}

function taskToForm(task: Task): FormState {
  const scheduleType = parseScheduleType(task.schedule)
  return {
    name: task.name,
    prompt: task.prompt,
    scheduleType,
    scheduleTime: parseScheduleTime(task.schedule),
    scheduleDay: parseScheduleDay(task.schedule),
    scheduleCron: scheduleType === 'custom-cron' ? task.schedule : '',
    active: task.status === 'active'
  }
}

// ---------------------------------------------------------------------------
// Task form (inside modal)
// ---------------------------------------------------------------------------

interface TaskFormProps {
  form: FormState
  errors: FormErrors
  saving: boolean
  onChange: (patch: Partial<FormState>) => void
  onSubmit: () => void
  onCancel: () => void
}

const DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

function TaskForm({ form, errors, saving, onChange, onSubmit, onCancel }: TaskFormProps): React.ReactElement {
  const selectClass =
    'bg-surface-raised border border-border rounded-md px-3 py-2 text-sm text-text-primary ' +
    'transition-colors duration-150 outline-none focus:ring-2 focus:ring-accent focus:border-accent w-full'

  return (
    <div className="space-y-4">
      <Input
        label="Name"
        value={form.name}
        onChange={(e) => onChange({ name: e.target.value })}
        placeholder="e.g. Morning briefing"
        error={errors.name}
      />

      <Textarea
        label="Prompt / Command"
        value={form.prompt}
        onChange={(e) => onChange({ prompt: e.target.value })}
        placeholder="What should Rex do when this task runs?"
      />

      {/* Schedule type select */}
      <div className="flex flex-col gap-1">
        <label className="text-sm font-medium text-text-secondary">Schedule</label>
        <select
          className={selectClass}
          value={form.scheduleType}
          onChange={(e) => onChange({ scheduleType: e.target.value as ScheduleType })}
        >
          <option value="every-hour">Every hour</option>
          <option value="every-day">Every day at…</option>
          <option value="every-week">Every week on…</option>
          <option value="custom-cron">Custom cron expression</option>
        </select>
        {errors.schedule && <span className="text-xs text-danger">{errors.schedule}</span>}
      </div>

      {/* Time picker for every-day and every-week */}
      {(form.scheduleType === 'every-day' || form.scheduleType === 'every-week') && (
        <Input
          label="Time"
          type="time"
          value={form.scheduleTime}
          onChange={(e) => onChange({ scheduleTime: e.target.value })}
        />
      )}

      {/* Day select for every-week */}
      {form.scheduleType === 'every-week' && (
        <div className="flex flex-col gap-1">
          <label className="text-sm font-medium text-text-secondary">Day of week</label>
          <select
            className={selectClass}
            value={form.scheduleDay}
            onChange={(e) => onChange({ scheduleDay: e.target.value })}
          >
            {DAYS.map((d) => (
              <option key={d} value={d}>
                {d.charAt(0).toUpperCase() + d.slice(1)}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Custom cron input */}
      {form.scheduleType === 'custom-cron' && (
        <Input
          label="Cron expression"
          value={form.scheduleCron}
          onChange={(e) => onChange({ scheduleCron: e.target.value })}
          placeholder="e.g. 0 9 * * 1"
          helperText="Standard 5-field cron syntax"
          error={errors.schedule}
        />
      )}

      {/* Active toggle */}
      <div className="flex items-center gap-3">
        <button
          type="button"
          role="switch"
          aria-checked={form.active}
          onClick={() => onChange({ active: !form.active })}
          className={[
            'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent',
            'transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg',
            form.active ? 'bg-accent' : 'bg-surface-raised'
          ].join(' ')}
        >
          <span
            className={[
              'pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow transform transition-transform duration-200',
              form.active ? 'translate-x-5' : 'translate-x-0'
            ].join(' ')}
          />
        </button>
        <span className="text-sm text-text-secondary">Active</span>
      </div>

      {/* Footer buttons (rendered here so they live inside the modal content area) */}
      <div className="flex justify-end gap-2 pt-2">
        <Button variant="secondary" size="sm" onClick={onCancel} disabled={saving}>
          Cancel
        </Button>
        <Button variant="primary" size="sm" onClick={onSubmit} loading={saving}>
          Save task
        </Button>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function TasksPage(): React.ReactElement {
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(true)
  const [modalOpen, setModalOpen] = useState(false)
  const [editTask, setEditTask] = useState<Task | null>(null)
  const [form, setForm] = useState<FormState>(emptyForm())
  const [errors, setErrors] = useState<FormErrors>({})
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    window.rex
      .getTasks()
      .then((result) => setTasks(result))
      .catch(() => setTasks([]))
      .finally(() => setLoading(false))
  }, [])

  function openNew(): void {
    setEditTask(null)
    setForm(emptyForm())
    setErrors({})
    setModalOpen(true)
  }

  function openEdit(task: Task): void {
    setEditTask(task)
    setForm(taskToForm(task))
    setErrors({})
    setModalOpen(true)
  }

  function closeModal(): void {
    if (saving) return
    setModalOpen(false)
  }

  function patchForm(patch: Partial<FormState>): void {
    setForm((prev) => ({ ...prev, ...patch }))
  }

  function validate(): boolean {
    const errs: FormErrors = {}
    if (!form.name.trim()) errs.name = 'Name is required'
    if (form.scheduleType === 'custom-cron' && !form.scheduleCron.trim()) {
      errs.schedule = 'Cron expression is required'
    }
    setErrors(errs)
    return Object.keys(errs).length === 0
  }

  function handleSubmit(): void {
    if (!validate()) return

    const schedule = buildScheduleString(
      form.scheduleType,
      form.scheduleTime,
      form.scheduleDay,
      form.scheduleCron
    )

    const input: TaskInput = {
      id: editTask?.id,
      name: form.name.trim(),
      prompt: form.prompt.trim(),
      schedule,
      active: form.active
    }

    setSaving(true)
    window.rex
      .saveTask(input)
      .then((saved) => {
        setTasks((prev) => {
          if (editTask) {
            return prev.map((t) => (t.id === saved.id ? saved : t))
          }
          return [...prev, saved]
        })
        setModalOpen(false)
      })
      .catch(() => {
        setErrors({ name: 'Failed to save task. Please try again.' })
      })
      .finally(() => setSaving(false))
  }

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner size="md" />
      </div>
    )
  }

  return (
    <>
      <div className="p-6 max-w-2xl mx-auto">
        {/* Header row */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-text-primary text-lg font-semibold">Scheduled Tasks</h2>
          <Button variant="primary" size="sm" onClick={openNew}>
            <span className="mr-1.5">
              <PlusIcon />
            </span>
            New Task
          </Button>
        </div>

        {/* Empty state */}
        {tasks.length === 0 && (
          <div className="flex items-center justify-center mt-16">
            <EmptyState
              icon={<CalendarIcon />}
              heading="No scheduled tasks"
              subtext="Create a task to automate Rex actions on a schedule."
              action={{ label: 'New Task', onClick: openNew }}
            />
          </div>
        )}

        {/* Task list */}
        <div className="space-y-3">
          {tasks.map((task) => (
            <button
              key={task.id}
              onClick={() => openEdit(task)}
              className="w-full text-left bg-surface-raised border border-border rounded-lg px-4 py-4 flex items-start justify-between gap-4 hover:border-accent/50 transition-colors"
            >
              <div className="min-w-0 flex-1">
                <p className="text-text-primary text-sm font-medium truncate">{task.name}</p>
                <p className="text-text-secondary text-xs mt-0.5">{task.schedule}</p>
                <p className="text-text-secondary text-xs mt-0.5">
                  <span>Next run:</span> {task.nextRun}
                </p>
              </div>
              <Badge variant={statusVariant(task.status)} className="shrink-0 mt-0.5 capitalize">
                {task.status}
              </Badge>
            </button>
          ))}
        </div>
      </div>

      {/* Create / Edit modal */}
      {modalOpen && (
        <Modal
          title={editTask ? 'Edit Task' : 'New Task'}
          onClose={closeModal}
        >
          <TaskForm
            form={form}
            errors={errors}
            saving={saving}
            onChange={patchForm}
            onSubmit={handleSubmit}
            onCancel={closeModal}
          />
        </Modal>
      )}
    </>
  )
}
