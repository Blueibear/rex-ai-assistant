import React, { useEffect, useState } from 'react'

interface AppLayoutProps {
  sectionName: string
  children: React.ReactNode
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

export function AppLayout({ sectionName, children }: AppLayoutProps): React.ReactElement {
  const narrow = useIsNarrow()

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
            <svg
              width="18"
              height="18"
              viewBox="0 0 18 18"
              fill="none"
              aria-hidden="true"
            >
              <circle cx="9" cy="9" r="7" stroke="white" strokeWidth="2" />
              <path
                d="M6 9h6M9 6v6"
                stroke="white"
                strokeWidth="2"
                strokeLinecap="round"
              />
            </svg>
          </div>
          {!narrow && (
            <span className="text-text-primary font-semibold text-lg tracking-tight">
              Rex
            </span>
          )}
        </div>

        {/* Nav slot — children will be placed here by a nav component */}
        <nav className="flex-1 overflow-y-auto py-2" aria-label="Main navigation">
          {/* Navigation items rendered by parent or nav component */}
        </nav>

        {/* Bottom: user avatar + settings shortcut */}
        <div className="flex items-center gap-3 px-4 py-4 border-t border-border">
          {/* Avatar placeholder */}
          <div
            className="flex-shrink-0 w-8 h-8 rounded-full bg-surface-raised flex items-center justify-center text-text-secondary text-sm font-medium"
            aria-label="User avatar"
          >
            U
          </div>
          {!narrow && (
            <button
              type="button"
              className="flex items-center gap-2 text-text-secondary hover:text-text-primary text-sm transition-colors"
              aria-label="Settings"
            >
              {/* Gear icon */}
              <svg
                width="16"
                height="16"
                viewBox="0 0 16 16"
                fill="none"
                aria-hidden="true"
              >
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
