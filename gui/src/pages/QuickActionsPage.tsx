import React, { useEffect, useState } from 'react'
import type { QuickAction } from '../types/ipc'

export function QuickActionsPage(): React.ReactElement {
  const [actions, setActions] = useState<QuickAction[]>([])
  const [loading, setLoading] = useState(true)
  const [newLabel, setNewLabel] = useState('')
  const [newCommand, setNewCommand] = useState('')
  const [addError, setAddError] = useState('')
  const [runningId, setRunningId] = useState<string | null>(null)
  const [runResult, setRunResult] = useState<Record<string, string>>({})

  const loadActions = (): void => {
    window.rex.listQuickActions()
      .then((d) => setActions(d.quick_actions ?? []))
      .catch(() => {/* ignore */})
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadActions()
  }, [])

  const handleAdd = async (): Promise<void> => {
    setAddError('')
    if (!newLabel.trim() || !newCommand.trim()) {
      setAddError('Both label and command are required.')
      return
    }
    const result = await window.rex.createQuickAction(newLabel.trim(), newCommand.trim())
    if (result.ok && result.action) {
      setActions((prev) => [...prev, result.action!])
      setNewLabel('')
      setNewCommand('')
    } else {
      setAddError(result.error ?? 'Failed to add action.')
    }
  }

  const handleDelete = async (id: string): Promise<void> => {
    await window.rex.deleteQuickAction(id)
    setActions((prev) => prev.filter((a) => a.id !== id))
  }

  const handleRun = async (action: QuickAction): Promise<void> => {
    setRunningId(action.id)
    try {
      const result = await window.rex.runQuickAction(action.id)
      setRunResult((prev) => ({ ...prev, [action.id]: result.detail ?? '…' }))
    } catch {
      setRunResult((prev) => ({ ...prev, [action.id]: 'Failed to run action.' }))
    } finally {
      setRunningId(null)
    }
  }

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-8">
      {/* Actions list */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Quick Actions</h2>
        <p className="text-text-secondary text-sm mb-4">
          One-click buttons for common commands. Each action sends a text command to Rex.
        </p>

        {loading && <p className="text-text-muted text-sm">Loading…</p>}

        {!loading && actions.length === 0 && (
          <p className="text-text-muted text-sm">No quick actions yet. Add one below.</p>
        )}

        {!loading && actions.length > 0 && (
          <div className="space-y-2">
            {actions.map((action) => (
              <div
                key={action.id}
                className="flex items-center gap-3 px-4 py-3 bg-surface border border-border rounded-xl"
              >
                <button
                  type="button"
                  onClick={() => void handleRun(action)}
                  disabled={runningId === action.id}
                  className="flex-1 text-left text-sm font-medium text-text-primary hover:text-accent transition-colors disabled:opacity-50"
                >
                  {action.label}
                </button>
                {runResult[action.id] && (
                  <span className="text-xs text-text-muted truncate max-w-[200px]">
                    {runResult[action.id]}
                  </span>
                )}
                <button
                  type="button"
                  onClick={() => void handleDelete(action.id)}
                  className="flex-shrink-0 text-text-muted hover:text-red-400 transition-colors text-xs"
                  aria-label={`Remove ${action.label}`}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Add new action */}
      <section>
        <h3 className="text-sm font-semibold text-text-primary mb-3">Add Quick Action</h3>
        <div className="space-y-3">
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">
              Button label
            </label>
            <input
              type="text"
              value={newLabel}
              onChange={(e) => setNewLabel(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
              placeholder="e.g. Lights off"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">
              Command (sent to Rex)
            </label>
            <input
              type="text"
              value={newCommand}
              onChange={(e) => setNewCommand(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-border bg-surface text-text-primary text-sm focus:outline-none focus:ring-2 focus:ring-accent/50"
              placeholder="e.g. Turn off all the lights"
            />
          </div>
          {addError && <p className="text-red-400 text-sm">{addError}</p>}
          <button
            type="button"
            onClick={() => void handleAdd()}
            className="px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors"
          >
            Add Action
          </button>
        </div>
      </section>
    </div>
  )
}
