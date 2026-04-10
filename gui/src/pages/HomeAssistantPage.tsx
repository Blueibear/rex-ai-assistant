import React, { useCallback, useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { Spinner } from '../components/ui/Spinner'

interface HaState {
  entity_id: string
  state: string
  friendly_name: string
  last_updated: string
}

interface HaStatesResponse {
  ok: boolean
  states?: HaState[]
  not_configured?: boolean
  error?: string
}

function formatLastUpdated(iso: string): string {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function stateBadgeClass(state: string): string {
  const s = state.toLowerCase()
  if (s === 'on' || s === 'open' || s === 'unlocked' || s === 'home') {
    return 'bg-green-500/15 text-green-400'
  }
  if (s === 'off' || s === 'closed' || s === 'locked' || s === 'away') {
    return 'bg-surface-raised text-text-muted'
  }
  if (s === 'unavailable' || s === 'unknown') {
    return 'bg-red-500/10 text-red-400'
  }
  return 'bg-blue-500/10 text-blue-400'
}

export function HomeAssistantPage(): React.ReactElement {
  const [states, setStates] = useState<HaState[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [notConfigured, setNotConfigured] = useState(false)
  const [filter, setFilter] = useState('')
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null)

  const fetchStates = useCallback((): void => {
    setLoading(true)
    setError(null)
    const token = localStorage.getItem('rex_token') ?? ''
    fetch('/api/ha/states', {
      headers: token ? { Authorization: `Bearer ${token}` } : {}
    })
      .then((r) => r.json())
      .then((data: HaStatesResponse) => {
        if (data.not_configured) {
          setNotConfigured(true)
          setStates([])
        } else if (!data.ok) {
          setError(data.error ?? 'Failed to load device states')
          setStates([])
        } else {
          setNotConfigured(false)
          setStates(data.states ?? [])
          setRefreshedAt(new Date())
        }
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Network error')
      })
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    fetchStates()
  }, [fetchStates])

  const filtered = filter.trim()
    ? states.filter(
        (s) =>
          s.entity_id.toLowerCase().includes(filter.toLowerCase()) ||
          s.friendly_name.toLowerCase().includes(filter.toLowerCase()) ||
          s.state.toLowerCase().includes(filter.toLowerCase())
      )
    : states

  if (loading && states.length === 0) {
    return (
      <div className="flex items-center justify-center h-full min-h-64">
        <Spinner size="lg" />
      </div>
    )
  }

  if (notConfigured) {
    return (
      <div className="p-6 max-w-xl mx-auto mt-12 text-center">
        <div className="w-14 h-14 mx-auto mb-4 rounded-full bg-surface-raised flex items-center justify-center">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-text-secondary">
            <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            <polyline points="9 22 9 12 15 12 15 22" />
          </svg>
        </div>
        <h2 className="text-text-primary font-semibold text-lg mb-2">Home Assistant not configured</h2>
        <p className="text-text-secondary text-sm mb-5">
          Set your Home Assistant URL and access token to view and control devices.
        </p>
        <NavLink
          to="/settings/home-assistant"
          className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90 transition-colors"
        >
          Configure Home Assistant →
        </NavLink>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-6 max-w-xl mx-auto mt-12 text-center">
        <p className="text-danger text-sm mb-4">{error}</p>
        <button
          type="button"
          onClick={fetchStates}
          className="px-4 py-2 rounded-lg bg-surface-raised text-text-primary text-sm hover:bg-border transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  return (
    <div className="p-6 max-w-4xl">
      {/* Header row */}
      <div className="flex items-center gap-3 mb-5 flex-wrap">
        <div className="flex-1 min-w-0">
          <input
            type="search"
            placeholder="Filter by name or state…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="w-full max-w-sm bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
          />
        </div>
        <div className="flex items-center gap-3 flex-shrink-0">
          {refreshedAt && (
            <span className="text-xs text-text-muted hidden sm:inline">
              Updated {refreshedAt.toLocaleTimeString()}
            </span>
          )}
          <button
            type="button"
            onClick={fetchStates}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-surface-raised border border-border text-sm text-text-primary hover:bg-border transition-colors disabled:opacity-50"
          >
            {loading ? (
              <Spinner size="sm" />
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="23 4 23 10 17 10" />
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
              </svg>
            )}
            Refresh
          </button>
        </div>
      </div>

      {/* Count */}
      <p className="text-xs text-text-muted mb-4">
        {filtered.length === states.length
          ? `${states.length} entities`
          : `${filtered.length} of ${states.length} entities`}
      </p>

      {filtered.length === 0 ? (
        <p className="text-text-secondary text-sm">No entities match your filter.</p>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border bg-surface">
                <th className="text-left px-4 py-3 text-text-secondary font-medium">Entity</th>
                <th className="text-left px-4 py-3 text-text-secondary font-medium">State</th>
                <th className="text-left px-4 py-3 text-text-secondary font-medium hidden md:table-cell">Last Updated</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((s) => (
                <tr key={s.entity_id} className="border-b border-border last:border-0 hover:bg-surface-raised/50 transition-colors">
                  <td className="px-4 py-3">
                    <div className="font-medium text-text-primary">{s.friendly_name}</div>
                    <div className="text-xs text-text-muted font-mono mt-0.5">{s.entity_id}</div>
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${stateBadgeClass(s.state)}`}>
                      {s.state}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-text-secondary text-xs hidden md:table-cell">
                    {formatLastUpdated(s.last_updated)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
