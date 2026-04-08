import React from 'react'
import { NavLink } from 'react-router-dom'

export function HomePage(): React.ReactElement {
  return (
    <div className="p-6 max-w-2xl mx-auto">
      <h2 className="text-xl font-semibold text-text-primary mb-2">Home</h2>
      <p className="text-text-secondary mb-6">
        Control your smart home devices and automations.
      </p>

      <div className="grid gap-4">
        <div className="bg-surface border border-border rounded-xl p-5">
          <h3 className="text-text-primary font-medium mb-1">Device Control</h3>
          <p className="text-text-secondary text-sm mb-3">
            Toggle lights, switches, and media players connected via Home Assistant.
          </p>
          <NavLink
            to="/settings"
            className="inline-flex items-center gap-1.5 text-sm text-accent hover:underline"
          >
            Configure Home Assistant in Settings
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="M2 6h8M6 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </NavLink>
        </div>

        <div className="bg-surface border border-border rounded-xl p-5">
          <h3 className="text-text-primary font-medium mb-1">Suggestions</h3>
          <p className="text-text-secondary text-sm">
            Rex will suggest automations based on your usage patterns. Suggestions appear
            after 3 or more repeated commands at the same time of day.
          </p>
        </div>

        <div className="bg-surface border border-border rounded-xl p-5">
          <h3 className="text-text-primary font-medium mb-1">Music</h3>
          <p className="text-text-secondary text-sm mb-3">
            Control Music Assistant playback by voice or from the Chat page.
          </p>
          <NavLink
            to="/chat"
            className="inline-flex items-center gap-1.5 text-sm text-accent hover:underline"
          >
            Go to Chat
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="M2 6h8M6 2l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </NavLink>
        </div>
      </div>
    </div>
  )
}
