import React, { useState, useEffect, useCallback } from 'react'
import type { EmailMessage, EmailPriority } from '../types/ipc'
import { PageLoadingFallback } from '../components/ui/PageLoadingFallback'

// ---------------------------------------------------------------------------
// Priority helpers
// ---------------------------------------------------------------------------

const PRIORITY_ORDER: Record<EmailPriority, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3
}

const PRIORITY_BADGE: Record<EmailPriority, { label: string; className: string }> = {
  critical: { label: 'Critical', className: 'bg-red-500 text-white' },
  high: { label: 'High', className: 'bg-orange-400 text-white' },
  medium: { label: 'Medium', className: 'bg-blue-500 text-white' },
  low: { label: 'Low', className: 'bg-gray-400 text-white' }
}

function sortMessages(messages: EmailMessage[]): EmailMessage[] {
  const unread = messages.filter((m) => !m.is_read)
  const read = messages.filter((m) => m.is_read)
  const byPriority = (a: EmailMessage, b: EmailMessage): number =>
    PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority]
  return [...unread.sort(byPriority), ...read.sort(byPriority)]
}

function formatRelativeTime(isoString: string): string {
  const diff = Date.now() - new Date(isoString).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  return `${days}d ago`
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface PriorityBadgeProps {
  priority: EmailPriority
}

function PriorityBadge({ priority }: PriorityBadgeProps): React.ReactElement {
  const { label, className } = PRIORITY_BADGE[priority]
  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${className}`}>
      {label}
    </span>
  )
}

interface EmailRowProps {
  message: EmailMessage
}

function EmailRow({ message }: EmailRowProps): React.ReactElement {
  const snippet =
    message.body_text.length > 100
      ? message.body_text.slice(0, 100) + '…'
      : message.body_text

  return (
    <div
      className={[
        'flex items-start gap-3 px-4 py-3 border-b border-border hover:bg-surface-raised transition-colors cursor-default',
        message.is_read ? 'opacity-70' : ''
      ].join(' ')}
    >
      {/* Unread dot */}
      <div className="mt-1.5 shrink-0 w-2 h-2">
        {!message.is_read && (
          <span className="block w-2 h-2 rounded-full bg-accent" />
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className={`text-sm truncate ${message.is_read ? 'text-text-secondary font-normal' : 'text-text-primary font-semibold'}`}
          >
            {message.sender}
          </span>
          <span className="text-xs text-text-secondary shrink-0">
            {formatRelativeTime(message.received_at)}
          </span>
        </div>
        <p
          className={`text-sm truncate mb-0.5 ${message.is_read ? 'text-text-secondary' : 'text-text-primary'}`}
        >
          {message.subject}
        </p>
        <p className="text-xs text-text-secondary truncate">{snippet}</p>
      </div>

      {/* Priority badge */}
      <div className="shrink-0 mt-1">
        <PriorityBadge priority={message.priority} />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// EmailPage
// ---------------------------------------------------------------------------

export function EmailPage(): React.ReactElement {
  const [messages, setMessages] = useState<EmailMessage[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  const loadInbox = useCallback(async (isRefresh = false): Promise<void> => {
    if (isRefresh) setRefreshing(true)
    else setLoading(true)
    try {
      const inbox = await window.rex.getEmailInbox()
      setMessages(sortMessages(inbox))
    } catch {
      // Silently keep previous state on error
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void loadInbox(false)
  }, [loadInbox])

  if (loading) {
    return <PageLoadingFallback lines={6} />
  }

  return (
    <div className="flex flex-col h-full">
      {/* BETA banner */}
      <div className="px-4 py-2 bg-surface-raised border-b border-border text-xs text-text-secondary">
        Email integration — enter credentials in Settings &gt; Integrations for live data.
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border shrink-0">
        <h2 className="flex-1 text-base font-semibold text-text-primary">Inbox</h2>
        <span className="text-xs text-text-secondary">{messages.length} messages</span>
        <button
          onClick={() => void loadInbox(true)}
          disabled={refreshing}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors disabled:opacity-60"
        >
          {/* Refresh icon */}
          <svg
            viewBox="0 0 20 20"
            fill="currentColor"
            className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`}
          >
            <path
              fillRule="evenodd"
              d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 110 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z"
              clipRule="evenodd"
            />
          </svg>
          Refresh
        </button>
      </div>

      {/* Message list */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="flex items-center justify-center h-full text-text-secondary text-sm">
            No messages
          </div>
        ) : (
          messages.map((msg) => <EmailRow key={msg.id} message={msg} />)
        )}
      </div>
    </div>
  )
}
