import React, { useEffect, useState } from 'react'

interface HistoryEntry {
  id: number
  timestamp: string
  command: string
  result: string
  success: boolean
}

function formatTimestamp(iso: string): string {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export function CommandHistoryPage(): React.ReactElement {
  const [entries, setEntries] = useState<HistoryEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    const token = localStorage.getItem('rex_token') ?? ''
    fetch('/api/history?limit=50', {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then((d) => {
        setEntries((d as { history: HistoryEntry[] }).history ?? [])
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false))
  }, [])

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-6">
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-2">Command History</h2>
        <p className="text-text-secondary text-sm mb-4">
          Recent commands sent to Rex, showing the last 50 entries.
        </p>

        {loading && <p className="text-text-muted text-sm">Loading…</p>}
        {error && <p className="text-red-400 text-sm">{error}</p>}

        {!loading && !error && entries.length === 0 && (
          <p className="text-text-muted text-sm">No command history yet.</p>
        )}

        {!loading && !error && entries.length > 0 && (
          <div className="divide-y divide-border border border-border rounded-xl overflow-hidden">
            {entries.map((entry) => (
              <div key={entry.id} className="px-4 py-3 bg-surface">
                <div className="flex items-start justify-between gap-4 mb-1">
                  <p className="text-text-primary text-sm font-medium truncate">{entry.command}</p>
                  <span
                    className={`flex-shrink-0 text-xs font-medium px-2 py-0.5 rounded-full ${
                      entry.success
                        ? 'bg-green-500/15 text-green-400'
                        : 'bg-red-500/15 text-red-400'
                    }`}
                  >
                    {entry.success ? 'Success' : 'Failed'}
                  </span>
                </div>
                {entry.result && (
                  <p className="text-text-secondary text-xs truncate">{entry.result}</p>
                )}
                <p className="text-text-muted text-xs mt-1">{formatTimestamp(entry.timestamp)}</p>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
