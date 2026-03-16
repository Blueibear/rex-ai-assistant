import React, { useCallback, useEffect, useState } from 'react'
import type { Task, TaskInput, TaskRun } from '../types/ipc'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { EmptyState } from '../components/ui/EmptyState'
import { Input } from '../components/ui/Input'
import { Modal } from '../components/ui/Modal'
import { Spinner } from '../components/ui/Spinner'
import { PageLoadingFallback } from '../components/ui/PageLoadingFallback'
import { Textarea } from '../components/ui/Textarea'
import { useToast } from '../components/ui/Toast'

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
  const m = schedule.match(/^Every (\w+) at/)
  const days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
  if (m && days.includes(m[1].toLowerCase())) return m[1].toLowerCase()
  return 'monday'
}

// ---------------------------------------------------------------------------
// Time helper
// ---------------------------------------------------------------------------

function relativeTime(date: Date): string {
  const diff = Math.floor((Date.now() - date.getTime()) / 1000)
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`
  return `${Math.floor(diff / 86400)}d ago`
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

function TrashIcon(): React.ReactElement {
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
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
      />
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
  deleting: boolean
  isEdit: boolean
  onChange: (patch: Partial<FormState>) => void
  onSubmit: () => void
  onCancel: () => void
  onDelete: () => void
}

const DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']

function TaskForm({
  form,
  errors,
  saving,
  deleting,
  isEdit,
  onChange,
  onSubmit,
  onCancel,
  onDelete
}: TaskFormProps): React.ReactElement {
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
        {errors.schedule && (
          <span className="text-xs text-danger">{errors.schedule}</span>
        )}
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

      {/* Footer buttons */}
      <div className="flex items-center justify-between pt-2">
        {isEdit ? (
          <Button
            variant="danger"
            size="sm"
            onClick={onDelete}
            loading={deleting}
            disabled={saving}
          >
            <span className="mr-1.5">
              <TrashIcon />
            </span>
            Delete
          </Button>
        ) : (
          <span />
        )}
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={onCancel} disabled={saving || deleting}>
            Cancel
          </Button>
          <Button variant="primary" size="sm" onClick={onSubmit} loading={saving} disabled={deleting}>
            Save task
          </Button>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inline enable/disable toggle for task cards
// ---------------------------------------------------------------------------

interface EnableToggleProps {
  enabled: boolean
  busy: boolean
  onToggle: (e: React.MouseEvent) => void
}

function EnableToggle({ enabled, busy, onToggle }: EnableToggleProps): React.ReactElement {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      onClick={onToggle}
      disabled={busy}
      title={enabled ? 'Disable task' : 'Enable task'}
      className={[
        'relative inline-flex h-5 w-9 shrink-0 cursor-pointer rounded-full border-2 border-transparent',
        'transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-1 focus:ring-offset-bg',
        busy ? 'opacity-50 cursor-not-allowed' : '',
        enabled ? 'bg-accent' : 'bg-surface-raised'
      ].join(' ')}
    >
      <span
        className={[
          'pointer-events-none inline-block h-4 w-4 rounded-full bg-white shadow transform transition-transform duration-200',
          enabled ? 'translate-x-4' : 'translate-x-0'
        ].join(' ')}
      />
    </button>
  )
}

// ---------------------------------------------------------------------------
// Log panel for task run history
// ---------------------------------------------------------------------------

interface LogPanelProps {
  history: TaskRun[] | 'loading'
}

function LogPanel({ history }: LogPanelProps): React.ReactElement {
  if (history === 'loading') {
    return (
      <div className="flex items-center gap-2 px-4 py-3 bg-surface border border-t-0 border-border rounded-b-lg">
        <Spinner size="sm" />
        <span className="text-xs text-text-secondary">Loading history…</span>
      </div>
    )
  }

  if (history.length === 0) {
    return (
      <div className="px-4 py-3 bg-surface border border-t-0 border-border rounded-b-lg">
        <p className="text-xs text-text-secondary">No run history available.</p>
      </div>
    )
  }

  return (
    <div className="bg-surface border border-t-0 border-border rounded-b-lg px-4 py-3 space-y-3">
      {history.map((run) => (
        <div key={run.id}>
          <div className="flex items-center gap-2 mb-1">
            <Badge variant={run.result === 'success' ? 'success' : 'danger'}>
              {run.result}
            </Badge>
            <span className="text-xs text-text-secondary">{relativeTime(new Date(run.timestamp))}</span>
          </div>
          <pre className="text-xs text-text-secondary bg-surface-raised rounded px-3 py-2 overflow-x-auto max-h-40 overflow-y-auto font-mono whitespace-pre-wrap">
            {run.output.slice(-20).join('\n')}
          </pre>
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export function TasksPage(): React.ReactElement {
  const addToast = useToast()
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(true)
  const [modalOpen, setModalOpen] = useState(false)
  const [editTask, setEditTask] = useState<Task | null>(null)
  const [form, setForm] = useState<FormState>(emptyForm())
  const [errors, setErrors] = useState<FormErrors>({})
  const [saving, setSaving] = useState(false)
  const [deleting, setDeleting] = useState(false)
  // Map of taskId -> true when setTaskEnabled is in-flight
  const [togglingIds, setTogglingIds] = useState<Set<string>>(new Set())
  // Task run history expansion state
  const [expandedTaskId, setExpandedTaskId] = useState<string | null>(null)
  const [historyMap, setHistoryMap] = useState<Record<string, TaskRun[] | 'loading'>>({})

  const fetchTasks = useCallback((): void => {
    window.rex
      .getTasks()
      .then((result) => setTasks(result))
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to load tasks'
        addToast(msg, 'error')
        setTasks([])
      })
      .finally(() => setLoading(false))
  }, [addToast])

  useEffect(() => {
    fetchTasks()
  }, [fetchTasks])

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
    if (saving || deleting) return
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
      .then(() => {
        setModalOpen(false)
        fetchTasks()
        addToast(editTask ? 'Task updated' : 'Task created', 'success')
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to save task'
        addToast(msg, 'error')
        setErrors({ name: msg })
      })
      .finally(() => setSaving(false))
  }

  function handleDelete(): void {
    if (!editTask) return

    setDeleting(true)
    window.rex
      .deleteTask(editTask.id)
      .then(() => {
        setModalOpen(false)
        fetchTasks()
        addToast('Task deleted', 'success')
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to delete task'
        addToast(msg, 'error')
      })
      .finally(() => setDeleting(false))
  }

  function handleToggleEnabled(task: Task, e: React.MouseEvent): void {
    e.stopPropagation()
    const newEnabled = task.status !== 'active'
    setTogglingIds((prev) => new Set(prev).add(task.id))
    window.rex
      .setTaskEnabled(task.id, newEnabled)
      .then((updated) => {
        setTasks((prev) => prev.map((t) => (t.id === updated.id ? updated : t)))
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to update task'
        addToast(msg, 'error')
      })
      .finally(() => {
        setTogglingIds((prev) => {
          const next = new Set(prev)
          next.delete(task.id)
          return next
        })
      })
  }

  function handleLastRunExpand(task: Task, e: React.MouseEvent): void {
    e.stopPropagation()

    if (expandedTaskId === task.id) {
      setExpandedTaskId(null)
      return
    }

    setExpandedTaskId(task.id)

    // Don't re-fetch if already loaded
    if (historyMap[task.id] !== undefined) return

    setHistoryMap((prev) => ({ ...prev, [task.id]: 'loading' }))
    window.rex
      .getTaskHistory(task.id)
      .then((runs) => {
        setHistoryMap((prev) => ({ ...prev, [task.id]: runs }))
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : 'Failed to load run history'
        addToast(msg, 'error')
        setHistoryMap((prev) => {
          const next = { ...prev }
          delete next[task.id]
          return next
        })
      })
  }

  // Loading state
  if (loading) {
    return <PageLoadingFallback lines={6} />
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
          {tasks.map((task) => {
            const isExpanded = expandedTaskId === task.id
            const history = historyMap[task.id]
            const isFailed = task.status === 'error'

            return (
              <div key={task.id}>
                {/* Card — using div+role to allow nested interactive elements */}
                <div
                  role="button"
                  tabIndex={0}
                  onClick={() => openEdit(task)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') openEdit(task)
                  }}
                  className={[
                    'w-full text-left bg-surface-raised border border-border px-4 py-4',
                    'flex items-start justify-between gap-4 hover:border-accent/50 transition-colors cursor-pointer',
                    isExpanded ? 'rounded-t-lg' : 'rounded-lg',
                    isFailed ? 'border-l-4 border-l-danger' : ''
                  ].join(' ')}
                >
                  <div className="min-w-0 flex-1">
                    <p className="text-text-primary text-sm font-medium truncate">{task.name}</p>
                    <p className="text-text-secondary text-xs mt-0.5">{task.schedule}</p>
                    <p className="text-text-secondary text-xs mt-0.5">
                      <span>Next run:</span> {task.nextRun}
                    </p>

                    {/* Last run row */}
                    <button
                      type="button"
                      onClick={(e) => handleLastRunExpand(task, e)}
                      className="mt-1.5 flex items-center gap-1.5 text-xs text-text-secondary hover:text-text-primary transition-colors"
                    >
                      <span>Last run:</span>
                      <span>
                        {task.lastRun
                          ? relativeTime(new Date(task.lastRun.timestamp))
                          : 'Never'}
                      </span>
                      {task.lastRun ? (
                        <Badge variant={task.lastRun.result === 'success' ? 'success' : 'danger'}>
                          {task.lastRun.result}
                        </Badge>
                      ) : (
                        <Badge variant="default">never</Badge>
                      )}
                      <span className="ml-0.5 text-[10px]">{isExpanded ? '▲' : '▼'}</span>
                    </button>
                  </div>

                  <div className="flex items-center gap-3 shrink-0 mt-0.5">
                    <Badge variant={statusVariant(task.status)} className="capitalize">
                      {task.status}
                    </Badge>
                    <EnableToggle
                      enabled={task.status === 'active'}
                      busy={togglingIds.has(task.id)}
                      onToggle={(e) => handleToggleEnabled(task, e)}
                    />
                  </div>
                </div>

                {/* Expanded log panel */}
                {isExpanded && history !== undefined && <LogPanel history={history} />}
              </div>
            )
          })}
        </div>
      </div>

      {/* Create / Edit modal */}
      {modalOpen && (
        <Modal title={editTask ? 'Edit Task' : 'New Task'} onClose={closeModal}>
          <TaskForm
            form={form}
            errors={errors}
            saving={saving}
            deleting={deleting}
            isEdit={editTask !== null}
            onChange={patchForm}
            onSubmit={handleSubmit}
            onCancel={closeModal}
            onDelete={handleDelete}
          />
        </Modal>
      )}
    </>
  )
}
