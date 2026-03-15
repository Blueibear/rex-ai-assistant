import React, { useEffect, useState } from 'react'
import type { Task } from '../types/ipc'
import { Badge } from '../components/ui/Badge'
import { EmptyState } from '../components/ui/EmptyState'
import { Spinner } from '../components/ui/Spinner'

function statusVariant(status: Task['status']): 'success' | 'warning' | 'danger' {
  if (status === 'active') return 'success'
  if (status === 'paused') return 'warning'
  return 'danger'
}

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

export function TasksPage(): React.ReactElement {
  const [tasks, setTasks] = useState<Task[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    window.rex
      .getTasks()
      .then((result) => setTasks(result))
      .catch(() => setTasks([]))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner size="md" />
      </div>
    )
  }

  if (tasks.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <EmptyState
          icon={<CalendarIcon />}
          heading="No scheduled tasks"
          subtext="Create a task to automate Rex actions on a schedule."
        />
      </div>
    )
  }

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-3">
      <h2 className="text-text-primary text-lg font-semibold mb-4">Scheduled Tasks</h2>
      {tasks.map((task) => (
        <div
          key={task.id}
          className="bg-surface-raised border border-border rounded-lg px-4 py-4 flex items-start justify-between gap-4"
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
        </div>
      ))}
    </div>
  )
}
