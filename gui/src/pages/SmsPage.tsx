import React, { useState, useEffect, useCallback, useRef } from 'react'
import type { SMSThread, SMSMessage } from '../types/ipc'
import { PageLoadingFallback } from '../components/ui/PageLoadingFallback'
import { Modal } from '../components/ui/Modal'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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

function formatMessageTime(isoString: string): string {
  return new Date(isoString).toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit'
  })
}

function getSnippet(thread: SMSThread): string {
  if (thread.messages.length === 0) return ''
  const last = thread.messages[thread.messages.length - 1]
  const text = last.body.length > 60 ? last.body.slice(0, 60) + '…' : last.body
  return last.direction === 'outbound' ? `You: ${text}` : text
}

// ---------------------------------------------------------------------------
// ThreadRow
// ---------------------------------------------------------------------------

interface ThreadRowProps {
  thread: SMSThread
  isSelected: boolean
  onClick: (thread: SMSThread) => void
}

function ThreadRow({ thread, isSelected, onClick }: ThreadRowProps): React.ReactElement {
  return (
    <button
      type="button"
      onClick={() => onClick(thread)}
      className={[
        'w-full text-left flex items-start gap-3 px-4 py-3 border-b border-border hover:bg-surface-raised transition-colors',
        isSelected ? 'bg-surface-raised' : ''
      ].join(' ')}
    >
      {/* Avatar circle */}
      <div className="shrink-0 w-9 h-9 rounded-full bg-accent/20 flex items-center justify-center text-sm font-semibold text-accent mt-0.5">
        {thread.contact_name.slice(0, 1).toUpperCase()}
      </div>

      {/* Main content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className={`text-sm truncate ${
              thread.unread_count > 0
                ? 'text-text-primary font-semibold'
                : 'text-text-primary font-normal'
            }`}
          >
            {thread.contact_name}
          </span>
          <span className="text-xs text-text-secondary shrink-0">
            {formatRelativeTime(thread.last_message_at)}
          </span>
        </div>
        <p className="text-xs text-text-secondary truncate">{getSnippet(thread)}</p>
      </div>

      {/* Unread badge */}
      {thread.unread_count > 0 && (
        <div className="shrink-0 mt-1">
          <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-accent text-white text-[10px] font-semibold">
            {thread.unread_count}
          </span>
        </div>
      )}
    </button>
  )
}

// ---------------------------------------------------------------------------
// MessageBubble
// ---------------------------------------------------------------------------

interface MessageBubbleProps {
  message: SMSMessage
}

function MessageBubble({ message }: MessageBubbleProps): React.ReactElement {
  const isOutbound = message.direction === 'outbound'
  return (
    <div className={`flex ${isOutbound ? 'justify-end' : 'justify-start'}`}>
      <div
        className={[
          'max-w-[75%] px-3 py-2 rounded-2xl text-sm leading-relaxed',
          isOutbound
            ? 'bg-accent text-white rounded-br-sm'
            : 'bg-surface-raised text-text-primary rounded-bl-sm'
        ].join(' ')}
      >
        <p className="whitespace-pre-wrap break-words">{message.body}</p>
        <div
          className={`flex items-center gap-1 mt-1 text-[10px] ${
            isOutbound ? 'text-white/70 justify-end' : 'text-text-secondary justify-start'
          }`}
        >
          <span>{formatMessageTime(message.sent_at)}</span>
          {message.status === 'stub' && <span>(stub)</span>}
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// ThreadDetailPanel
// ---------------------------------------------------------------------------

interface ThreadDetailPanelProps {
  thread: SMSThread | null
  sendError: string | null
  sending: boolean
  onClose: () => void
  onSend: (to: string, body: string) => Promise<void>
}

function ThreadDetailPanel({
  thread,
  sendError,
  sending,
  onClose,
  onSend
}: ThreadDetailPanelProps): React.ReactElement {
  const panelRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [composeBody, setComposeBody] = useState('')
  const isOpen = thread !== null

  useEffect(() => {
    if (!isOpen) return
    function handleKey(e: KeyboardEvent): void {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [isOpen, onClose])

  // Scroll to bottom when messages change
  useEffect(() => {
    if (thread && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [thread?.messages.length, thread])

  function handleBackdropClick(e: React.MouseEvent<HTMLDivElement>): void {
    if (panelRef.current && !panelRef.current.contains(e.target as Node)) onClose()
  }

  async function handleSend(): Promise<void> {
    if (!thread || !composeBody.trim()) return
    await onSend(thread.contact_number, composeBody.trim())
    setComposeBody('')
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>): void {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSend()
    }
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
        aria-label="SMS thread"
      >
        {thread && (
          <>
            {/* Panel header */}
            <div className="flex items-center gap-3 px-4 pt-4 pb-3 border-b border-border shrink-0">
              <div className="w-8 h-8 rounded-full bg-accent/20 flex items-center justify-center text-sm font-semibold text-accent shrink-0">
                {thread.contact_name.slice(0, 1).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <h2 className="font-semibold text-text-primary text-sm truncate">
                  {thread.contact_name}
                </h2>
                <p className="text-xs text-text-secondary truncate">{thread.contact_number}</p>
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

            {/* Messages */}
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
              {thread.messages.length === 0 ? (
                <p className="text-xs text-text-secondary text-center py-8">
                  No messages yet. Say hello!
                </p>
              ) : (
                thread.messages.map((msg) => <MessageBubble key={msg.id} message={msg} />)
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Send error */}
            {sendError && (
              <div className="shrink-0 px-4 py-1.5 bg-red-500/10 border-t border-red-500/20">
                <p className="text-xs text-red-400">{sendError}</p>
              </div>
            )}

            {/* Compose bar */}
            <div className="shrink-0 px-3 py-3 border-t border-border flex items-end gap-2">
              <textarea
                value={composeBody}
                onChange={(e) => setComposeBody(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Message…"
                rows={1}
                className="flex-1 px-3 py-2 rounded-xl bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent resize-none leading-snug max-h-28 overflow-y-auto"
                style={{ height: 'auto' }}
              />
              <button
                onClick={() => void handleSend()}
                disabled={sending || !composeBody.trim()}
                className="shrink-0 w-9 h-9 rounded-full bg-accent text-white flex items-center justify-center hover:bg-accent/90 transition-colors disabled:opacity-50"
                aria-label="Send"
              >
                {sending ? (
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8v8H4z"
                    />
                  </svg>
                ) : (
                  <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
                    <path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z" />
                  </svg>
                )}
              </button>
            </div>
          </>
        )}
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// SmsPage
// ---------------------------------------------------------------------------

export function SmsPage(): React.ReactElement {
  const [threads, setThreads] = useState<SMSThread[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [selected, setSelected] = useState<SMSThread | null>(null)
  const [sending, setSending] = useState(false)
  const [sendError, setSendError] = useState<string | null>(null)

  // New Message modal state
  const [showNewMessage, setShowNewMessage] = useState(false)
  const [newTo, setNewTo] = useState('')
  const [newBody, setNewBody] = useState('')
  const [newSending, setNewSending] = useState(false)
  const [newSendError, setNewSendError] = useState<string | null>(null)

  const loadThreads = useCallback(async (isRefresh = false): Promise<void> => {
    if (isRefresh) setRefreshing(true)
    else setLoading(true)
    try {
      const data = await window.rex.getSMSThreads()
      // Sort by last_message_at descending
      const sorted = [...data].sort(
        (a, b) => new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime()
      )
      setThreads(sorted)
    } catch {
      // Keep previous state on error
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void loadThreads(false)
  }, [loadThreads])

  async function handleSend(to: string, body: string): Promise<void> {
    if (!selected) return
    setSending(true)
    setSendError(null)
    try {
      const msg = await window.rex.sendSMS(to, body)
      // Update thread in state with new message
      setThreads((prev) =>
        prev.map((t) => {
          if (t.id !== selected.id) return t
          return {
            ...t,
            messages: [...t.messages, msg],
            last_message_at: msg.sent_at
          }
        })
      )
      setSelected((prev) => {
        if (!prev) return prev
        return {
          ...prev,
          messages: [...prev.messages, msg],
          last_message_at: msg.sent_at
        }
      })
    } catch (err) {
      setSendError(err instanceof Error ? err.message : 'Failed to send message')
    } finally {
      setSending(false)
    }
  }

  function handleSelectThread(thread: SMSThread): void {
    setSendError(null)
    setSelected(thread)
    // Mark thread as read locally
    setThreads((prev) =>
      prev.map((t) => (t.id === thread.id ? { ...t, unread_count: 0 } : t))
    )
  }

  async function handleNewMessageSend(): Promise<void> {
    if (!newTo.trim() || !newBody.trim()) return
    setNewSending(true)
    setNewSendError(null)
    try {
      const msg = await window.rex.sendSMS(newTo.trim(), newBody.trim())
      // Re-fetch thread list so new thread appears at top
      const updated = await window.rex.getSMSThreads()
      const sorted = [...updated].sort(
        (a, b) => new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime()
      )
      setThreads(sorted)
      // Select the new/updated thread
      const targetThread = sorted.find((t) => t.id === msg.thread_id) ?? null
      if (targetThread) {
        setSelected(targetThread)
        setSendError(null)
      }
      setShowNewMessage(false)
      setNewTo('')
      setNewBody('')
    } catch (err) {
      setNewSendError(err instanceof Error ? err.message : 'Failed to send message')
    } finally {
      setNewSending(false)
    }
  }

  if (loading) {
    return <PageLoadingFallback lines={5} />
  }

  return (
    <div className="flex flex-col h-full">
      {/* BETA banner */}
      <div className="px-4 py-2 bg-surface-raised border-b border-border text-xs text-text-secondary">
        SMS integration — enter Twilio credentials in Settings &gt; Integrations to send real messages.
      </div>

      {/* Toolbar */}
      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border shrink-0">
        <h2 className="flex-1 text-base font-semibold text-text-primary">Messages</h2>
        <span className="text-xs text-text-secondary">{threads.length} threads</span>
        <button
          onClick={() => {
            setNewTo('')
            setNewBody('')
            setNewSendError(null)
            setShowNewMessage(true)
          }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4">
            <path
              fillRule="evenodd"
              d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z"
              clipRule="evenodd"
            />
          </svg>
          New Message
        </button>
        <button
          onClick={() => void loadThreads(true)}
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

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto">
        {threads.length === 0 ? (
          <div className="flex items-center justify-center h-full text-text-secondary text-sm">
            No conversations
          </div>
        ) : (
          threads.map((thread) => (
            <ThreadRow
              key={thread.id}
              thread={thread}
              isSelected={selected?.id === thread.id}
              onClick={handleSelectThread}
            />
          ))
        )}
      </div>

      {/* Thread detail slide-in panel */}
      <ThreadDetailPanel
        thread={selected}
        sendError={sendError}
        sending={sending}
        onClose={() => setSelected(null)}
        onSend={handleSend}
      />

      {/* New Message modal */}
      {showNewMessage && (
        <Modal
          title="New Message"
          onClose={() => setShowNewMessage(false)}
          footer={
            <div className="flex gap-2">
              <button
                onClick={() => setShowNewMessage(false)}
                className="px-4 py-2 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => void handleNewMessageSend()}
                disabled={newSending || !newTo.trim() || !newBody.trim()}
                className="px-4 py-2 rounded text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors disabled:opacity-50"
              >
                {newSending ? 'Sending…' : 'Send'}
              </button>
            </div>
          }
        >
          <div className="space-y-3">
            <div>
              <label className="block text-xs font-medium text-text-primary mb-1" htmlFor="sms-to">
                To (phone number or contact name)
              </label>
              <input
                id="sms-to"
                type="text"
                value={newTo}
                onChange={(e) => setNewTo(e.target.value)}
                placeholder="+14155550100 or Alice"
                className="w-full px-3 py-2 rounded bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent"
              />
            </div>
            <div>
              <label
                className="block text-xs font-medium text-text-primary mb-1"
                htmlFor="sms-body"
              >
                Message
              </label>
              <textarea
                id="sms-body"
                value={newBody}
                onChange={(e) => setNewBody(e.target.value)}
                rows={4}
                className="w-full px-3 py-2 rounded bg-surface-raised border border-border text-text-primary text-sm focus:outline-none focus:border-accent resize-none"
              />
            </div>
            {newSendError && (
              <p className="text-xs text-red-400">{newSendError}</p>
            )}
          </div>
        </Modal>
      )}
    </div>
  )
}
