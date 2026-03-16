import React, { useState, useEffect, useCallback, useRef } from 'react'
import type { EmailMessage, EmailPriority } from '../types/ipc'
import { PageLoadingFallback } from '../components/ui/PageLoadingFallback'
import { Modal } from '../components/ui/Modal'

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

function formatFullDateTime(isoString: string): string {
  return new Date(isoString).toLocaleString(undefined, {
    weekday: 'long',
    month: 'long',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit'
  })
}

// ---------------------------------------------------------------------------
// PriorityBadge
// ---------------------------------------------------------------------------

function PriorityBadge({ priority }: { priority: EmailPriority }): React.ReactElement {
  const { label, className } = PRIORITY_BADGE[priority]
  return (
    <span
      className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${className}`}
    >
      {label}
    </span>
  )
}

// ---------------------------------------------------------------------------
// EmailRow
// ---------------------------------------------------------------------------

interface EmailRowProps {
  message: EmailMessage
  isSelected: boolean
  onClick: (msg: EmailMessage) => void
}

function EmailRow({ message, isSelected, onClick }: EmailRowProps): React.ReactElement {
  const snippet =
    message.body_text.length > 100
      ? message.body_text.slice(0, 100) + '…'
      : message.body_text

  return (
    <button
      type="button"
      onClick={() => onClick(message)}
      className={[
        'w-full text-left flex items-start gap-3 px-4 py-3 border-b border-border hover:bg-surface-raised transition-colors',
        message.is_read ? 'opacity-70' : '',
        isSelected ? 'bg-surface-raised' : ''
      ].join(' ')}
    >
      {/* Unread dot */}
      <div className="mt-1.5 shrink-0 w-2 h-2">
        {!message.is_read && <span className="block w-2 h-2 rounded-full bg-accent" />}
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className={`text-sm truncate ${
              message.is_read ? 'text-text-secondary font-normal' : 'text-text-primary font-semibold'
            }`}
          >
            {message.sender}
          </span>
          <span className="text-xs text-text-secondary shrink-0">
            {formatRelativeTime(message.received_at)}
          </span>
        </div>
        <p
          className={`text-sm truncate mb-0.5 ${
            message.is_read ? 'text-text-secondary' : 'text-text-primary'
          }`}
        >
          {message.subject}
        </p>
        <p className="text-xs text-text-secondary truncate">{snippet}</p>
      </div>

      {/* Priority badge */}
      <div className="shrink-0 mt-1">
        <PriorityBadge priority={message.priority} />
      </div>
    </button>
  )
}

// ---------------------------------------------------------------------------
// EmailDetailPanel
// ---------------------------------------------------------------------------

interface EmailDetailPanelProps {
  message: EmailMessage | null
  onClose: () => void
  onArchive: (id: string) => void
  onMarkRead: (id: string) => void
  onReplyDraft: (id: string) => void
  generatingReply: boolean
}

function EmailDetailPanel({
  message,
  onClose,
  onArchive,
  onMarkRead,
  onReplyDraft,
  generatingReply
}: EmailDetailPanelProps): React.ReactElement {
  const panelRef = useRef<HTMLDivElement>(null)
  const isOpen = message !== null

  useEffect(() => {
    if (!isOpen) return
    function handleKey(e: KeyboardEvent): void {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [isOpen, onClose])

  function handleBackdropClick(e: React.MouseEvent<HTMLDivElement>): void {
    if (panelRef.current && !panelRef.current.contains(e.target as Node)) onClose()
  }

  return (
    <>
      {/* Backdrop */}
      <div
        className={[
          'fixed inset-0 z-40 transition-opacity duration-200',
          isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        ].join(' ')}
        onClick={handleBackdropClick}
        aria-hidden="true"
      />

      {/* Slide-in panel */}
      <div
        ref={panelRef}
        className={[
          'fixed top-0 right-0 bottom-0 z-50 w-96 bg-surface border-l border-border shadow-2xl',
          'flex flex-col transition-transform duration-250 ease-in-out'
        ].join(' ')}
        style={{ transform: isOpen ? 'translateX(0)' : 'translateX(100%)' }}
        role="dialog"
        aria-modal="true"
        aria-label="Email details"
      >
        {message && (
          <>
            {/* Panel header */}
            <div className="flex items-start gap-3 px-4 pt-5 pb-4 border-b border-border shrink-0">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <PriorityBadge priority={message.priority} />
                  {!message.is_read && (
                    <span className="text-xs text-accent font-medium">Unread</span>
                  )}
                </div>
                <h2 className="font-semibold text-text-primary text-base leading-snug">
                  {message.subject}
                </h2>
              </div>
              <button
                onClick={onClose}
                className="text-text-secondary hover:text-text-primary transition-colors p-1 rounded shrink-0"
                aria-label="Close panel"
              >
                <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>

            {/* Panel body */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
              {/* Meta */}
              <div className="space-y-1">
                <p className="text-xs text-text-secondary">
                  <span className="font-medium text-text-primary">From:</span>{' '}
                  {message.sender}
                </p>
                <p className="text-xs text-text-secondary">
                  <span className="font-medium text-text-primary">To:</span>{' '}
                  {message.recipients.join(', ')}
                </p>
                <p className="text-xs text-text-secondary">
                  {formatFullDateTime(message.received_at)}
                </p>
              </div>

              {/* Body */}
              <div className="border-t border-border pt-4">
                <p className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
                  {message.body_text}
                </p>
              </div>
            </div>

            {/* Panel footer — action buttons */}
            <div className="shrink-0 px-4 py-4 border-t border-border space-y-2">
              <div className="flex gap-2">
                <button
                  onClick={() => onArchive(message.id)}
                  className="flex-1 px-3 py-2 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors"
                >
                  Archive
                </button>
                <button
                  onClick={() => onMarkRead(message.id)}
                  disabled={message.is_read}
                  className="flex-1 px-3 py-2 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors disabled:opacity-40"
                >
                  Mark as Read
                </button>
              </div>
              <button
                onClick={() => onReplyDraft(message.id)}
                disabled={generatingReply}
                className="w-full px-3 py-2 rounded text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors disabled:opacity-60"
              >
                {generatingReply ? 'Generating…' : 'Generate Reply Draft'}
              </button>
            </div>
          </>
        )}
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// EmailPage
// ---------------------------------------------------------------------------

export function EmailPage(): React.ReactElement {
  const [messages, setMessages] = useState<EmailMessage[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [selected, setSelected] = useState<EmailMessage | null>(null)

  // Compose modal state
  const [showCompose, setShowCompose] = useState(false)
  const [draftBody, setDraftBody] = useState('')
  const [generatingReply, setGeneratingReply] = useState(false)

  const loadInbox = useCallback(async (isRefresh = false): Promise<void> => {
    if (isRefresh) setRefreshing(true)
    else setLoading(true)
    try {
      const inbox = await window.rex.getEmailInbox()
      setMessages(sortMessages(inbox))
    } catch {
      // Keep previous state on error
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void loadInbox(false)
  }, [loadInbox])

  function handleArchive(id: string): void {
    setMessages((prev) => prev.filter((m) => m.id !== id))
    if (selected?.id === id) setSelected(null)
  }

  function handleMarkRead(id: string): void {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, is_read: true } : m))
    )
    setSelected((prev) => (prev?.id === id ? { ...prev, is_read: true } : prev))
  }

  async function handleReplyDraft(id: string): Promise<void> {
    setGeneratingReply(true)
    try {
      const draft = await window.rex.generateEmailReply(id)
      setDraftBody(draft)
      setShowCompose(true)
    } finally {
      setGeneratingReply(false)
    }
  }

  function handleSendDraft(): void {
    console.log('[Email stub] Would send draft:', draftBody)
    setShowCompose(false)
    setDraftBody('')
  }

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
          messages.map((msg) => (
            <EmailRow
              key={msg.id}
              message={msg}
              isSelected={selected?.id === msg.id}
              onClick={setSelected}
            />
          ))
        )}
      </div>

      {/* Email detail slide-in panel */}
      <EmailDetailPanel
        message={selected}
        onClose={() => setSelected(null)}
        onArchive={handleArchive}
        onMarkRead={handleMarkRead}
        onReplyDraft={(id) => void handleReplyDraft(id)}
        generatingReply={generatingReply}
      />

      {/* Compose modal */}
      {showCompose && (
        <Modal
          onClose={() => setShowCompose(false)}
          title="Reply Draft"
          footer={
            <div className="flex gap-2 justify-end">
              <button
                onClick={() => setShowCompose(false)}
                className="px-4 py-2 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors"
              >
                Discard
              </button>
              <button
                onClick={handleSendDraft}
                className="px-4 py-2 rounded text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors"
              >
                Send
              </button>
            </div>
          }
        >
          <textarea
            value={draftBody}
            onChange={(e) => setDraftBody(e.target.value)}
            rows={10}
            className="w-full px-3 py-2 rounded bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent resize-none"
          />
        </Modal>
      )}
    </div>
  )
}
