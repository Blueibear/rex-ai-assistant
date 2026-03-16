import React, { useState, useEffect, useCallback, useRef } from 'react'
import type { GuiNotification, NotificationPriority } from '../types/ipc'
import { PageLoadingFallback } from '../components/ui/PageLoadingFallback'
import { useNotificationsStore } from '../store/notificationsStore'

const PRIORITY_ORDER: Record<NotificationPriority, number> = {
  critical: 0,
  high: 1,
  medium: 2,
  low: 3
}

const PRIORITY_BADGE: Record<NotificationPriority, { label: string; className: string }> = {
  critical: { label: 'Critical', className: 'bg-red-500 text-white' },
  high: { label: 'High', className: 'bg-orange-400 text-white' },
  medium: { label: 'Medium', className: 'bg-blue-500 text-white' },
  low: { label: 'Low', className: 'bg-gray-400 text-white' }
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

function formatEscalation(isoString: string | undefined): string {
  if (!isoString) return 'No escalation'
  const diff = new Date(isoString).getTime() - Date.now()
  if (diff <= 0) return 'Escalating now'
  const mins = Math.ceil(diff / 60000)
  if (mins < 60) return `Escalates in ${mins}m`
  const hours = Math.ceil(mins / 60)
  return `Escalates in ${hours}h`
}

function groupByPriority(
  notifications: GuiNotification[]
): Array<{ label: string; items: GuiNotification[] }> {
  const groups: Record<NotificationPriority, GuiNotification[]> = {
    critical: [],
    high: [],
    medium: [],
    low: []
  }
  for (const n of notifications) {
    groups[n.priority].push(n)
  }
  return [
    { label: 'Critical', items: groups.critical },
    { label: 'High', items: groups.high },
    { label: 'Medium', items: groups.medium },
    { label: 'Low / Digest', items: groups.low }
  ].filter((g) => g.items.length > 0)
}

function PriorityBadge({ priority }: { priority: NotificationPriority }): React.ReactElement {
  const { label, className } = PRIORITY_BADGE[priority]
  return (
    <span
      className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wide ${className}`}
    >
      {label}
    </span>
  )
}

interface NotificationCardProps {
  notification: GuiNotification
  isSelected: boolean
  onClick: (n: GuiNotification) => void
}

function NotificationCard({
  notification,
  isSelected,
  onClick
}: NotificationCardProps): React.ReactElement {
  const isUnread = !notification.read_at
  const snippet =
    notification.body.length > 120 ? notification.body.slice(0, 120) + '...' : notification.body

  return (
    <button
      type="button"
      onClick={() => onClick(notification)}
      className={[
        'w-full text-left flex items-start gap-3 px-4 py-3 border-b border-border hover:bg-surface-raised transition-colors',
        isUnread ? '' : 'opacity-60',
        isSelected ? 'bg-surface-raised' : ''
      ].join(' ')}
    >
      <div className="mt-1.5 shrink-0 w-2 h-2">
        {isUnread && <span className="block w-2 h-2 rounded-full bg-accent" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span
            className={`text-sm truncate ${
              isUnread ? 'text-text-primary font-semibold' : 'text-text-primary font-normal'
            }`}
          >
            {notification.title}
          </span>
          <span className="text-xs text-text-secondary shrink-0">
            {formatRelativeTime(notification.created_at)}
          </span>
        </div>
        <p className="text-xs text-text-secondary line-clamp-2">{snippet}</p>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-[10px] text-text-secondary bg-surface-raised px-1.5 py-0.5 rounded">
            {notification.source}
          </span>
        </div>
      </div>
      <div className="shrink-0 mt-1">
        <PriorityBadge priority={notification.priority} />
      </div>
    </button>
  )
}

interface NotificationDetailPanelProps {
  notification: GuiNotification | null
  onClose: () => void
  onMarkRead: (id: string) => void
  onDismiss: (id: string) => void
}

function NotificationDetailPanel({
  notification,
  onClose,
  onMarkRead,
  onDismiss
}: NotificationDetailPanelProps): React.ReactElement {
  const panelRef = useRef<HTMLDivElement>(null)
  const isOpen = notification !== null

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
      <div
        className={[
          'fixed inset-0 z-40 transition-opacity duration-200',
          isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        ].join(' ')}
        onClick={handleBackdropClick}
        aria-hidden="true"
      />
      <div
        ref={panelRef}
        className={[
          'fixed top-0 right-0 bottom-0 z-50 w-96 bg-surface border-l border-border shadow-2xl',
          'flex flex-col transition-transform duration-250 ease-in-out'
        ].join(' ')}
        style={{ transform: isOpen ? 'translateX(0)' : 'translateX(100%)' }}
        role="dialog"
        aria-modal="true"
        aria-label="Notification details"
      >
        {notification && (
          <>
            <div className="flex items-start gap-3 px-4 pt-5 pb-4 border-b border-border shrink-0">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <PriorityBadge priority={notification.priority} />
                  {!notification.read_at && (
                    <span className="text-xs text-accent font-medium">Unread</span>
                  )}
                </div>
                <h2 className="font-semibold text-text-primary text-base leading-snug">
                  {notification.title}
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
            <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
              <div className="space-y-1.5 text-xs text-text-secondary">
                <p>
                  <span className="font-medium text-text-primary">Source:</span>{' '}
                  {notification.source}
                </p>
                <p>
                  <span className="font-medium text-text-primary">Channel:</span>{' '}
                  {notification.channel}
                </p>
                <p>
                  <span className="font-medium text-text-primary">Received:</span>{' '}
                  {new Date(notification.created_at).toLocaleString()}
                </p>
                <p>
                  <span className="font-medium text-text-primary">Escalation:</span>{' '}
                  {formatEscalation(notification.escalation_due_at)}
                </p>
              </div>
              <div className="border-t border-border pt-4">
                <p className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
                  {notification.body}
                </p>
              </div>
            </div>
            <div className="shrink-0 px-4 py-4 border-t border-border space-y-2">
              <div className="flex gap-2">
                <button
                  onClick={() => onMarkRead(notification.id)}
                  disabled={!!notification.read_at}
                  className="flex-1 px-3 py-2 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors disabled:opacity-40"
                >
                  Mark as Read
                </button>
                <button
                  onClick={() => onDismiss(notification.id)}
                  className="flex-1 px-3 py-2 rounded text-sm font-medium bg-accent text-white hover:bg-accent/90 transition-colors"
                >
                  Dismiss
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    </>
  )
}

export function NotificationsPage(): React.ReactElement {
  const [notifications, setNotifications] = useState<GuiNotification[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [selected, setSelected] = useState<GuiNotification | null>(null)
  const fetchUnreadCount = useNotificationsStore((state) => state.fetchUnreadCount)

  const loadNotifications = useCallback(async (isRefresh = false): Promise<void> => {
    if (isRefresh) setRefreshing(true)
    else setLoading(true)
    try {
      const data = await window.rex.getNotifications()
      setNotifications(data)
    } catch {
      // Keep previous state on error
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  useEffect(() => {
    void loadNotifications(false)
  }, [loadNotifications])

  async function handleMarkRead(id: string): Promise<void> {
    await window.rex.markNotificationRead(id)
    const now = new Date().toISOString()
    setNotifications((prev) => prev.map((n) => (n.id === id ? { ...n, read_at: now } : n)))
    setSelected((prev) => (prev?.id === id ? { ...prev, read_at: now } : prev))
    void fetchUnreadCount()
  }

  async function handleDismiss(id: string): Promise<void> {
    await window.rex.dismissNotification(id)
    const now = new Date().toISOString()
    setNotifications((prev) =>
      prev.map((n) =>
        n.id === id ? { ...n, read_at: n.read_at ?? now, escalation_due_at: undefined } : n
      )
    )
    setSelected(null)
    void fetchUnreadCount()
  }

  async function handleMarkAllRead(): Promise<void> {
    const unread = notifications.filter((n) => !n.read_at)
    await Promise.all(unread.map((n) => window.rex.markNotificationRead(n.id)))
    const now = new Date().toISOString()
    setNotifications((prev) => prev.map((n) => (n.read_at ? n : { ...n, read_at: now })))
    setSelected((prev) => (prev && !prev.read_at ? { ...prev, read_at: now } : prev))
    void fetchUnreadCount()
  }

  const unreadCount = notifications.filter((n) => !n.read_at).length

  if (loading) {
    return <PageLoadingFallback lines={6} />
  }

  const groups = groupByPriority(
    [...notifications].sort((a, b) => PRIORITY_ORDER[a.priority] - PRIORITY_ORDER[b.priority])
  )

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center gap-3 px-4 py-2.5 border-b border-border shrink-0">
        <h2 className="flex-1 text-base font-semibold text-text-primary">
          Notifications
          {unreadCount > 0 && (
            <span className="ml-2 inline-flex items-center justify-center w-5 h-5 rounded-full bg-accent text-white text-[10px] font-semibold">
              {unreadCount > 99 ? '99+' : unreadCount}
            </span>
          )}
        </h2>
        <button
          onClick={() => void handleMarkAllRead()}
          disabled={unreadCount === 0}
          className="px-3 py-1.5 rounded text-sm font-medium bg-surface-raised text-text-primary hover:bg-border transition-colors disabled:opacity-40"
        >
          Mark all read
        </button>
        <button
          onClick={() => void loadNotifications(true)}
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

      <div className="flex-1 overflow-y-auto">
        {notifications.length === 0 ? (
          <div className="flex items-center justify-center h-full text-text-secondary text-sm">
            No notifications
          </div>
        ) : (
          groups.map((group) => (
            <div key={group.label}>
              <div className="px-4 py-1.5 bg-surface-raised border-b border-border">
                <span className="text-xs font-semibold text-text-secondary uppercase tracking-wide">
                  {group.label}
                </span>
              </div>
              {group.items.map((n) => (
                <NotificationCard
                  key={n.id}
                  notification={n}
                  isSelected={selected?.id === n.id}
                  onClick={(notif) => {
                    setSelected(notif)
                    if (!notif.read_at) void handleMarkRead(notif.id)
                  }}
                />
              ))}
            </div>
          ))
        )}
      </div>

      <NotificationDetailPanel
        notification={selected}
        onClose={() => setSelected(null)}
        onMarkRead={(id) => void handleMarkRead(id)}
        onDismiss={(id) => void handleDismiss(id)}
      />
    </div>
  )
}
