import React from 'react'
import { useNavigate } from 'react-router-dom'

interface SetupOptionalExtensionsPageProps {
  onContinue: () => void
}

export function SetupOptionalExtensionsPage({
  onContinue
}: SetupOptionalExtensionsPageProps): React.ReactElement {
  const navigate = useNavigate()

  const openExistingSurface = (path: string): void => {
    onContinue()
    navigate(path)
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-bg p-4">
      <div className="w-full max-w-lg rounded-2xl border border-border bg-surface p-8 space-y-6">
        <div className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-text-muted">Optional</p>
          <h2 className="text-xl font-semibold text-text-primary">Optional household setup</h2>
          <p className="text-sm text-text-secondary">
            Basic Rex conversation is already configured. These household additions can be done
            now or later and are not required to open the dashboard.
          </p>
        </div>

        <div className="space-y-3">
          <button
            type="button"
            onClick={() => openExistingSurface('/settings/home-assistant')}
            className="w-full rounded-xl border border-border bg-surface-raised p-4 text-left hover:border-accent/50"
          >
            <span className="block text-sm font-medium text-text-primary">Home Assistant</span>
            <span className="mt-1 block text-xs text-text-secondary">
              Connect smart-home control using the existing Home Assistant settings.
            </span>
          </button>

          <button
            type="button"
            onClick={() => openExistingSurface('/settings?section=users')}
            className="w-full rounded-xl border border-border bg-surface-raised p-4 text-left hover:border-accent/50"
          >
            <span className="block text-sm font-medium text-text-primary">
              Additional household voice
            </span>
            <span className="mt-1 block text-xs text-text-secondary">
              Add another household member through the existing user and voice-enrollment tools.
            </span>
          </button>

          <button
            type="button"
            onClick={() => openExistingSurface('/pairing')}
            className="w-full rounded-xl border border-border bg-surface-raised p-4 text-left hover:border-accent/50"
          >
            <span className="block text-sm font-medium text-text-primary">
              Additional room endpoint
            </span>
            <span className="mt-1 block text-xs text-text-secondary">
              Pair a trusted Rex Room endpoint using the existing device-pairing flow.
            </span>
          </button>
        </div>

        <button
          type="button"
          onClick={onContinue}
          className="w-full px-6 py-2 rounded-lg bg-accent text-white text-sm font-medium hover:bg-accent/90"
        >
          Not now, open dashboard
        </button>
      </div>
    </div>
  )
}
