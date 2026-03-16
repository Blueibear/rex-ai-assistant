import React, { useEffect, useState } from 'react'
import { NavLink, useNavigate } from 'react-router-dom'
import { useNotificationsStore } from '../store/notificationsStore'

interface AppLayoutProps {
  children: React.ReactNode
}

interface NavItem {
  path: string
  label: string
  icon: React.ReactElement
  beta?: boolean
  showUnread?: boolean
}

function useIsNarrow(): boolean {
  const [narrow, setNarrow] = useState(() => window.innerWidth < 900)
  useEffect(() => {
    const handler = (): void => setNarrow(window.innerWidth < 900)
    window.addEventListener('resize', handler)
    return () => window.removeEventListener('resize', handler)
  }, [])
  return narrow
}

const navItems: NavItem[] = [
  {
    path: '/chat',
    label: 'Chat',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <path
          d="M3 3h12a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H6l-3 3V4a1 1 0 0 1 1-1z"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
      </svg>
    )
  },
  {
    path: '/voice',
    label: 'Voice',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <rect x="7" y="2" width="4" height="8" rx="2" stroke="currentColor" strokeWidth="1.4" />
        <path
          d="M4 9a5 5 0 0 0 10 0"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinecap="round"
        />
        <line x1="9" y1="14" x2="9" y2="16" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      </svg>
    )
  },
  {
    path: '/tasks',
    label: 'Tasks',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <rect x="2" y="2" width="14" height="14" rx="2" stroke="currentColor" strokeWidth="1.4" />
        <path d="M5 9l3 3 5-5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    )
  },
  {
    path: '/calendar',
    label: 'Calendar',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <rect x="2" y="4" width="14" height="12" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
        <path d="M6 2v4M12 2v4M2 8h14" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      </svg>
    )
  },
  {
    path: '/reminders',
    label: 'Reminders',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <path
          d="M9 2a6 6 0 0 1 6 6v3l1 2H2l1-2V8a6 6 0 0 1 6-6z"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
        <path d="M7 15a2 2 0 0 0 4 0" stroke="currentColor" strokeWidth="1.4" />
      </svg>
    )
  },
  {
    path: '/memories',
    label: 'Memories',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <circle cx="9" cy="9" r="7" stroke="currentColor" strokeWidth="1.4" />
        <circle cx="9" cy="9" r="3" stroke="currentColor" strokeWidth="1.4" />
      </svg>
    )
  },
  {
    path: '/email',
    label: 'Email',
    beta: true,
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <rect x="2" y="4" width="14" height="10" rx="1.5" stroke="currentColor" strokeWidth="1.4" />
        <path d="M2 6l7 5 7-5" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      </svg>
    )
  },
  {
    path: '/sms',
    label: 'SMS',
    beta: true,
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <path
          d="M3 3h12a1 1 0 0 1 1 1v7a1 1 0 0 1-1 1H9l-4 3v-3H3a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"
          stroke="currentColor"
          strokeWidth="1.4"
          strokeLinejoin="round"
        />
      </svg>
    )
  },
  {
    path: '/notifications',
    label: 'Notifications',
    showUnread: true,
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <circle cx="9" cy="3" r="1.5" fill="currentColor" />
        <path d="M5 7h8M4 11h10M6 15h6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
      </svg>
    )
  },
  {
    path: '/settings',
    label: 'Settings',
    icon: (
      <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
        <path
          d="M7.5 2h3l.5 1.5a5 5 0 0 1 1.2.7l1.5-.5 1.5 2.6-1.2 1a5 5 0 0 1 0 1.4l1.2 1-1.5 2.6-1.5-.5A5 5 0 0 1 11 12.3l-.5 1.5h-3L7 12.3a5 5 0 0 1-1.2-.7L4.3 12 2.8 9.4l1.2-1a5 5 0 0 1 0-1.4l-1.2-1L4.3 3.4l1.5.5A5 5 0 0 1 7 3.2L7.5 2z"
          stroke="currentColor"
          strokeWidth="1.2"
          strokeLinejoin="round"
        />
        <circle cx="9" cy="9" r="2" stroke="currentColor" strokeWidth="1.2" />
      </svg>
    )
  }
]

/** Map each route path to the human-readable section name shown in the topbar */
const sectionNames: Record<string, string> = {
  '/chat': 'Chat',
  '/voice': 'Voice',
  '/tasks': 'Tasks',
  '/calendar': 'Calendar',
  '/reminders': 'Reminders',
  '/memories': 'Memories',
  '/email': 'Email',
  '/sms': 'SMS',
  '/notifications': 'Notifications',
  '/settings': 'Settings'
}

export function AppLayout({ children }: AppLayoutProps): React.ReactElement {
  const narrow = useIsNarrow()
  const navigate = useNavigate()
  const unreadCount = useNotificationsStore((state) => state.unreadCount)

  const [sectionName, setSectionName] = useState('Chat')

  useEffect(() => {
    const path = window.location.hash.replace('#', '') || '/chat'
    setSectionName(sectionNames[path] ?? 'Rex')
  }, [])

  const handleNavClick = (label: string): void => {
    setSectionName(label)
  }

  return (
    <div className="flex h-screen bg-bg text-text-primary overflow-hidden">
      {/* Sidebar */}
      <aside
        className="flex flex-col flex-shrink-0 bg-surface border-r border-border transition-all duration-200"
        style={{ width: narrow ? 64 : 240 }}
      >
        {/* Logo / wordmark */}
        <div className="flex items-center gap-3 px-4 py-5 border-b border-border">
          <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-accent flex items-center justify-center">
            <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
              <circle cx="9" cy="9" r="7" stroke="white" strokeWidth="2" />
              <path d="M6 9h6M9 6v6" stroke="white" strokeWidth="2" strokeLinecap="round" />
            </svg>
          </div>
          {!narrow && (
            <span className="text-text-primary font-semibold text-lg tracking-tight">Rex</span>
          )}
        </div>

        {/* Nav items */}
        <nav className="flex-1 overflow-y-auto py-2" aria-label="Main navigation">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={() => handleNavClick(item.label)}
              className={({ isActive }) =>
                [
                  'flex items-center gap-3 px-4 py-2.5 mx-2 rounded-lg text-sm transition-colors',
                  'motion-safe:transition-transform motion-safe:duration-100 motion-safe:hover:scale-[1.02]',
                  isActive
                    ? 'bg-accent/15 text-accent font-medium'
                    : 'text-text-secondary hover:bg-surface-raised hover:text-text-primary'
                ].join(' ')
              }
              title={narrow ? item.label : undefined}
            >
              {/* Icon with optional unread dot when collapsed */}
              <span className="relative flex-shrink-0">
                {item.icon}
                {item.showUnread && narrow && unreadCount > 0 && (
                  <span className="absolute -top-1 -right-1 w-2 h-2 rounded-full bg-accent" aria-hidden="true" />
                )}
              </span>

              {!narrow && (
                <>
                  <span className="flex-1">{item.label}</span>

                  {/* BETA badge */}
                  {item.beta && (
                    <span className="text-[10px] font-semibold uppercase tracking-wide px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400 leading-none">
                      BETA
                    </span>
                  )}

                  {/* Unread count badge for Notifications */}
                  {item.showUnread && unreadCount > 0 && (
                    <span className="min-w-[18px] h-[18px] flex items-center justify-center rounded-full bg-accent text-white text-[10px] font-bold px-1 leading-none">
                      {unreadCount > 99 ? '99+' : unreadCount}
                    </span>
                  )}
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* Bottom: user avatar + settings shortcut */}
        <div className="flex items-center gap-3 px-4 py-4 border-t border-border">
          <div
            className="flex-shrink-0 w-8 h-8 rounded-full bg-surface-raised flex items-center justify-center text-text-secondary text-sm font-medium cursor-pointer"
            aria-label="User avatar"
          >
            U
          </div>
          {!narrow && (
            <button
              type="button"
              onClick={() => {
                setSectionName('Settings')
                navigate('/settings')
              }}
              className="flex items-center gap-2 text-text-secondary hover:text-text-primary text-sm transition-colors"
              aria-label="Settings"
            >
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path
                  d="M6.5 1.5h3l.5 1.5a5 5 0 0 1 1.2.7l1.5-.5 1.5 2.6-1.2 1a5 5 0 0 1 0 1.4l1.2 1-1.5 2.6-1.5-.5A5 5 0 0 1 10 12.3l-.5 1.5h-3L6 12.3a5 5 0 0 1-1.2-.7L3.3 12 1.8 9.4l1.2-1a5 5 0 0 1 0-1.4l-1.2-1L3.3 3.4l1.5.5A5 5 0 0 1 6 3.2L6.5 1.5z"
                  stroke="currentColor"
                  strokeWidth="1.2"
                  strokeLinejoin="round"
                />
                <circle cx="8" cy="8" r="2" stroke="currentColor" strokeWidth="1.2" />
              </svg>
              Settings
            </button>
          )}
        </div>
      </aside>

      {/* Main content */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Topbar */}
        <header className="flex items-center h-14 px-6 border-b border-border bg-surface flex-shrink-0">
          <h1 className="text-text-primary font-semibold text-base">{sectionName}</h1>
        </header>

        {/* Scrollable content area */}
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  )
}
