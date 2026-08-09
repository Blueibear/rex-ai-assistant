import React from 'react'
import type { TestStatus } from './useIntegrationsSettingsController'

export function ConnectionBadge({
  status,
  hasCredentials
}: {
  status: TestStatus
  hasCredentials: boolean
}): React.ReactElement {
  if (status === 'ok') {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-success/15 px-2 py-0.5 text-xs font-medium text-success">
        <span className="h-1.5 w-1.5 rounded-full bg-success" />
        Live test passed
      </span>
    )
  }
  if (status === 'error') {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-danger/15 px-2 py-0.5 text-xs font-medium text-danger">
        <span className="h-1.5 w-1.5 rounded-full bg-danger" />
        Error
      </span>
    )
  }
  if (!hasCredentials) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-border px-2 py-0.5 text-xs font-medium text-text-secondary">
        <span className="h-1.5 w-1.5 rounded-full bg-text-secondary" />
        Not Configured
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-warning/15 px-2 py-0.5 text-xs font-medium text-warning">
      <span className="h-1.5 w-1.5 rounded-full bg-warning" />
      Configured only
    </span>
  )
}


export function TestConnectionButton({
  status,
  onTest,
  error
}: {
  status: TestStatus
  onTest: () => void
  error?: string
}): React.ReactElement {
  return (
    <div className="mt-3">
      <div className="flex items-center gap-3">
        <button
          onClick={onTest}
          disabled={status === 'testing'}
          className="flex items-center gap-2 bg-surface-raised hover:bg-border disabled:opacity-50 text-text-primary text-xs font-medium px-3 py-1.5 rounded-lg border border-border transition-colors focus:outline-none focus:ring-2 focus:ring-accent"
        >
          {status === 'testing' ? (
            <>
              <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
              Testing…
            </>
          ) : 'Check Status'}
        </button>
        {status === 'ok' && (
          <span className="flex items-center gap-1 text-xs text-success">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Authenticated
          </span>
        )}
        {status === 'error' && (
          <span className="flex items-center gap-1 text-xs text-danger">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
            Check failed
          </span>
        )}
        {status === 'configured' && (
          <span className="text-xs text-warning">Configured only</span>
        )}
        {status === 'idle' && (
          <span className="text-xs text-text-secondary">Not tested</span>
        )}
      </div>
      {(status === 'error' || status === 'configured') && error && (
        <p className={`mt-1 text-xs ${status === 'error' ? 'text-danger' : 'text-warning'}`}>{error}</p>
      )}
    </div>
  )
}
