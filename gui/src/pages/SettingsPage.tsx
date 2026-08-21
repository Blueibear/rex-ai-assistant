import React, { useState } from 'react'
import { AboutSettingsSection } from './settings/AboutSettingsSection'
import { AiSettingsSection } from './settings/AiSettingsSection'
import { AudioOutputSettingsSection } from './settings/AudioOutputSettingsSection'
import { GeneralSettingsSection } from './settings/GeneralSettingsSection'
import { IntegrationsSettingsSection } from './settings/integrations/IntegrationsSettingsSection'
import { NotificationsSettingsSection } from './settings/NotificationsSettingsSection'
import { OutputRoutingSettingsSection } from './settings/OutputRoutingSettingsSection'
import { SystemSettingsSection } from './settings/SystemSettingsSection'
import { UsersSettingsSection } from './settings/UsersSettingsSection'
import { VoiceSettingsSection } from './settings/voice/VoiceSettingsSection'
import { settingsCategoryFromHash, type SettingsCategoryId } from '../types/settingsRouting'

type CategoryId = SettingsCategoryId

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
    id: 'users',
    label: 'Users',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    )
  },
  {
    id: 'audio',
    label: 'Audio Output',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
        <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      </svg>
    )
  },
  {
    id: 'system',
    label: 'System',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <line x1="8" y1="21" x2="16" y2="21" />
        <line x1="12" y1="17" x2="12" y2="21" />
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

function renderPanel(categoryId: CategoryId): React.ReactElement {
  switch (categoryId) {
    case 'general':
      return <GeneralSettingsSection />
    case 'voice':
      return <VoiceSettingsSection />
    case 'ai':
      return <AiSettingsSection />
    case 'integrations':
      return <IntegrationsSettingsSection />
    case 'notifications':
      return <NotificationsSettingsSection />
    case 'users':
      return <UsersSettingsSection />
    case 'audio':
      return (
        <>
          <AudioOutputSettingsSection />
          <OutputRoutingSettingsSection />
        </>
      )
    case 'system':
      return <SystemSettingsSection />
    case 'about':
      return <AboutSettingsSection />
  }
}

export function SettingsPage(): React.ReactElement {
  const [activeCategory, setActiveCategory] = useState<CategoryId>(() =>
    settingsCategoryFromHash(window.location.hash)
  )

  return (
    <div className="flex h-full overflow-hidden">
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
      <main className="flex-1 overflow-y-auto">{renderPanel(activeCategory)}</main>
    </div>
  )
}
