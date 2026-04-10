import React, { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'

interface Integration {
  name: string
  key: string
  configured: boolean
  configure_url?: string
}

interface Capability {
  name: string
  description: string
  category: string
  enabled: boolean
}

function StatusBadge({ configured }: { configured: boolean }): React.ReactElement {
  return configured ? (
    <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-green-500/15 text-green-400">
      Configured
    </span>
  ) : (
    <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-surface-raised text-text-muted">
      Not configured
    </span>
  )
}

export function IntegrationsPage(): React.ReactElement {
  const [integrations, setIntegrations] = useState<Integration[]>([])
  const [capabilities, setCapabilities] = useState<Capability[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      fetch('/api/integrations').then((r) => r.json()),
      fetch('/api/capabilities').then((r) => r.json()),
    ])
      .then(([intData, capData]) => {
        setIntegrations((intData as { integrations: Integration[] }).integrations ?? [])
        setCapabilities((capData as { capabilities: Capability[] }).capabilities ?? [])
      })
      .catch(() => {
        /* ignore */
      })
      .finally(() => setLoading(false))
  }, [])

  const grouped = capabilities.reduce<Record<string, Capability[]>>((acc, cap) => {
    const cat = cap.category ?? 'General'
    if (!acc[cat]) acc[cat] = []
    acc[cat].push(cap)
    return acc
  }, {})

  return (
    <div className="p-6 max-w-2xl mx-auto space-y-8">
      {/* Integrations */}
      <section>
        <h2 className="text-xl font-semibold text-text-primary mb-4">Integrations</h2>

        {loading ? (
          <p className="text-text-muted text-sm">Loading…</p>
        ) : integrations.length === 0 ? (
          <p className="text-text-muted text-sm">No integrations found.</p>
        ) : (
          <div className="divide-y divide-border border border-border rounded-xl overflow-hidden">
            {integrations.map((int) => (
              <div key={int.key} className="flex items-center justify-between px-4 py-3 bg-surface gap-3">
                <span className="text-text-primary text-sm font-medium flex-1 min-w-0 truncate">
                  {int.name}
                </span>
                <div className="flex items-center gap-3 flex-shrink-0">
                  <StatusBadge configured={int.configured} />
                  {int.configure_url && (
                    <NavLink
                      to={int.configure_url}
                      className="text-xs text-accent hover:underline"
                    >
                      Configure →
                    </NavLink>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Capabilities — only rendered when the registry returns entries */}
      {!loading && Object.keys(grouped).length > 0 && (
        <section>
          <h2 className="text-xl font-semibold text-text-primary mb-2">Capabilities</h2>
          <p className="text-text-secondary text-sm mb-4">
            All tools Rex can use. Only enabled capabilities are available during interactions.
          </p>

          <div className="space-y-4">
            {Object.entries(grouped).map(([category, caps]) => (
              <div key={category}>
                <h3 className="text-xs font-semibold uppercase tracking-wider text-text-muted mb-2">
                  {category}
                </h3>
                <div className="divide-y divide-border border border-border rounded-xl overflow-hidden">
                  {caps.map((cap) => (
                    <div key={cap.name} className="flex items-start justify-between px-4 py-3 bg-surface gap-4">
                      <div className="min-w-0">
                        <p className="text-text-primary text-sm font-medium">{cap.name}</p>
                        {cap.description && (
                          <p className="text-text-muted text-xs mt-0.5 truncate">{cap.description}</p>
                        )}
                      </div>
                      <span
                        className={`flex-shrink-0 text-xs font-medium px-2 py-0.5 rounded-full ${
                          cap.enabled
                            ? 'bg-green-500/15 text-green-400'
                            : 'bg-surface-raised text-text-muted'
                        }`}
                      >
                        {cap.enabled ? 'Enabled' : 'Disabled'}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}
