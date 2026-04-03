import React, { useState, useEffect, useRef, useCallback } from 'react'
import type { LogEntry } from '../types/ipc'

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: 'text-text-secondary',
  INFO: 'text-blue-400',
  WARNING: 'text-yellow-400',
  WARN: 'text-yellow-400',
  ERROR: 'text-red-400',
  CRITICAL: 'text-red-600'
}

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] as const
type LogLevel = (typeof LOG_LEVELS)[number]

const LEVEL_ORDER: Record<string, number> = {
  DEBUG: 0,
  INFO: 1,
  WARNING: 2,
  WARN: 2,
  ERROR: 3,
  CRITICAL: 4
}

function levelColor(level: string): string {
  return LEVEL_COLORS[level.toUpperCase()] ?? 'text-text-primary'
}

function matchesLevel(entry: LogEntry, minLevel: LogLevel): boolean {
  return (LEVEL_ORDER[entry.level?.toUpperCase()] ?? 1) >= (LEVEL_ORDER[minLevel] ?? 0)
}

export function LogsPage(): React.ReactElement {
  const [entries, setEntries] = useState<LogEntry[]>([])
  const [search, setSearch] = useState('')
  const [minLevel, setMinLevel] = useState<LogLevel>('DEBUG')
  const [paused, setPaused] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)
  const pausedRef = useRef(paused)
  pausedRef.current = paused

  const appendEntry = useCallback((entry: LogEntry) => {
    if (!pausedRef.current) {
      setEntries((prev) => {
        const next = [...prev, entry]
        // Keep at most 2000 entries to avoid memory growth.
        return next.length > 2000 ? next.slice(-2000) : next
      })
    }
  }, [])

  // Load existing entries on mount.
  useEffect(() => {
    setLoading(true)
    window.rex
      .getLogs(200)
      .then((res) => {
        if (res.ok) {
          setEntries(res.entries)
        } else {
          setError(res.error ?? 'Failed to load logs')
        }
      })
      .catch((err: unknown) => setError(String(err)))
      .finally(() => setLoading(false))

    // Subscribe to live log entries.
    window.rex.onLogEntry(appendEntry)

    // Start tailing.
    void window.rex.startLogTail()

    return () => {
      void window.rex.stopLogTail()
    }
  }, [appendEntry])

  // Auto-scroll to bottom when not paused.
  useEffect(() => {
    if (!paused) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [entries, paused])

  const filtered = entries.filter((e) => {
    if (!matchesLevel(e, minLevel)) return false
    if (search) {
      const q = search.toLowerCase()
      return (
        e.message?.toLowerCase().includes(q) ||
        e.logger?.toLowerCase().includes(q) ||
        e.level?.toLowerCase().includes(q)
      )
    }
    return true
  })

  function handleDownload(): void {
    window.rex
      .downloadLogs()
      .then((res) => {
        if (res.ok && res.content) {
          const blob = new Blob([res.content], { type: 'text/plain' })
          const url = URL.createObjectURL(blob)
          const a = document.createElement('a')
          a.href = url
          a.download = res.filename ?? 'rex.log'
          a.click()
          URL.revokeObjectURL(url)
        }
      })
      .catch(() => {/* silently fail */})
  }

  return (
    <div className="flex flex-col h-full p-4 gap-3">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Level filter */}
        <select
          value={minLevel}
          onChange={(e) => setMinLevel(e.target.value as LogLevel)}
          className="bg-surface-raised border border-border rounded-lg px-2 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        >
          {LOG_LEVELS.map((l) => (
            <option key={l} value={l}>{l}</option>
          ))}
        </select>

        {/* Search */}
        <input
          type="text"
          placeholder="Search logs…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 min-w-[160px] bg-surface-raised border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary placeholder-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
        />

        {/* Pause / Resume */}
        <button
          onClick={() => setPaused((v) => !v)}
          className="flex items-center gap-1.5 bg-surface-raised hover:bg-surface border border-border text-text-primary text-sm font-medium px-3 py-1.5 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent"
        >
          {paused ? (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              Resume
            </>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" />
              </svg>
              Pause
            </>
          )}
        </button>

        {/* Download */}
        <button
          onClick={handleDownload}
          className="flex items-center gap-1.5 bg-surface-raised hover:bg-surface border border-border text-text-primary text-sm font-medium px-3 py-1.5 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent"
          title="Download log file"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Download
        </button>
      </div>

      {paused && (
        <div className="text-xs text-yellow-400 bg-yellow-400/10 border border-yellow-400/20 rounded-lg px-3 py-1.5">
          Stream paused — new entries are not shown. Click Resume to continue.
        </div>
      )}

      {/* Log entries */}
      <div className="flex-1 overflow-y-auto bg-surface-raised border border-border rounded-lg font-mono text-xs">
        {loading && (
          <div className="flex items-center justify-center h-24 text-text-secondary">Loading logs…</div>
        )}
        {!loading && error && (
          <div className="p-4 text-red-400">{error}</div>
        )}
        {!loading && !error && filtered.length === 0 && (
          <div className="flex items-center justify-center h-24 text-text-secondary">
            No log entries{search || minLevel !== 'DEBUG' ? ' matching filters' : ' yet'}.
          </div>
        )}
        {filtered.map((entry, i) => (
          <div
            key={i}
            className="flex gap-2 px-3 py-0.5 hover:bg-surface border-b border-border/30 last:border-0"
          >
            <span className="text-text-secondary shrink-0 w-[160px] truncate">{entry.timestamp}</span>
            <span className={`${levelColor(entry.level)} shrink-0 w-[60px]`}>{entry.level}</span>
            <span className="text-text-secondary shrink-0 w-[120px] truncate" title={entry.logger}>{entry.logger}</span>
            <span className="text-text-primary break-all">{entry.message}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="text-xs text-text-secondary">
        Showing {filtered.length} of {entries.length} entries
      </div>
    </div>
  )
}
