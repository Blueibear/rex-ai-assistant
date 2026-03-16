import React, { useState, useEffect } from 'react'
import type { VersionInfo } from '../types/ipc'

type CategoryId = 'general' | 'voice' | 'ai' | 'integrations' | 'notifications' | 'about'

interface Category {
  id: CategoryId
  label: string
  icon: React.ReactElement
}

const categories: Category[] = [
  {
    id: 'general',
    label: 'General',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.07 4.93l-1.41 1.41M4.93 19.07l1.41-1.41M4.93 4.93l1.41 1.41M19.07 19.07l-1.41-1.41M12 2v2M12 20v2M2 12h2M20 12h2" />
      </svg>
    )
  },
  {
    id: 'voice',
    label: 'Voice',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="23" />
        <line x1="8" y1="23" x2="16" y2="23" />
      </svg>
    )
  },
  {
    id: 'ai',
    label: 'AI',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M9 9h6M9 12h6M9 15h4" />
      </svg>
    )
  },
  {
    id: 'integrations',
    label: 'Integrations',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
        <polyline points="15 3 21 3 21 9" />
        <line x1="10" y1="14" x2="21" y2="3" />
      </svg>
    )
  },
  {
    id: 'notifications',
    label: 'Notifications',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
        <path d="M13.73 21a2 2 0 0 1-3.46 0" />
      </svg>
    )
  },
  {
    id: 'about',
    label: 'About',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
    )
  }
]

function PlaceholderPanel({ title }: { title: string }): React.ReactElement {
  return (
    <div className="p-6">
      <h2 className="text-lg font-semibold text-text-primary mb-2">{title}</h2>
      <p className="text-sm text-text-secondary">
        {title} settings will be configured in a future story.
      </p>
    </div>
  )
}

function AboutPanel(): React.ReactElement {
  const [info, setInfo] = useState<VersionInfo | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    window.rex
      .getVersionInfo()
      .then((v) => {
        setInfo(v)
      })
      .catch(() => {
        setInfo({ rex: 'unknown', electron: 'unknown', node: 'unknown' })
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  return (
    <div className="p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-accent/20 flex items-center justify-center">
          <span className="text-accent font-bold text-lg">R</span>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-text-primary">Rex AI Assistant</h2>
          <p className="text-sm text-text-secondary">Local-first AI companion</p>
        </div>
      </div>

      {loading ? (
        <div className="space-y-3">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-5 rounded bg-surface-raised animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="space-y-3">
          <VersionRow label="Rex version" value={info?.rex ?? 'unknown'} />
          <VersionRow label="Electron version" value={info?.electron ?? 'unknown'} />
          <VersionRow label="Node version" value={info?.node ?? 'unknown'} />
        </div>
      )}

      <div className="mt-8 pt-6 border-t border-border">
        <p className="text-xs text-text-secondary">
          Rex AI Assistant is a local-first, voice-activated AI companion. All data is stored on
          your device.
        </p>
      </div>
    </div>
  )
}

function VersionRow({ label, value }: { label: string; value: string }): React.ReactElement {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50">
      <span className="text-sm text-text-secondary">{label}</span>
      <span className="text-sm font-mono text-text-primary">{value}</span>
    </div>
  )
}

function renderPanel(categoryId: CategoryId): React.ReactElement {
  switch (categoryId) {
    case 'general':
      return <PlaceholderPanel title="General" />
    case 'voice':
      return <PlaceholderPanel title="Voice" />
    case 'ai':
      return <PlaceholderPanel title="AI" />
    case 'integrations':
      return <PlaceholderPanel title="Integrations" />
    case 'notifications':
      return <PlaceholderPanel title="Notifications" />
    case 'about':
      return <AboutPanel />
  }
}

export function SettingsPage(): React.ReactElement {
  const [activeCategory, setActiveCategory] = useState<CategoryId>('general')

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left: category list */}
      <nav className="w-48 shrink-0 border-r border-border overflow-y-auto py-2">
        {categories.map((cat) => {
          const isActive = cat.id === activeCategory
          return (
            <button
              key={cat.id}
              onClick={() => setActiveCategory(cat.id)}
              className={[
                'w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-left transition-colors',
                isActive
                  ? 'bg-accent/10 text-accent font-medium'
                  : 'text-text-secondary hover:bg-surface-raised hover:text-text-primary'
              ].join(' ')}
            >
              <span className={isActive ? 'text-accent' : 'text-text-secondary'}>{cat.icon}</span>
              {cat.label}
            </button>
          )
        })}
      </nav>

      {/* Right: content area */}
      <main className="flex-1 overflow-y-auto">{renderPanel(activeCategory)}</main>
    </div>
  )
}
