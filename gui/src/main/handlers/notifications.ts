import { ipcMain } from 'electron'
import type { GuiNotification } from '../../types/ipc'

function makeStubNotifications(): GuiNotification[] {
  const now = new Date()
  function minsAgo(m: number): string {
    return new Date(now.getTime() - m * 60 * 1000).toISOString()
  }
  function minsFromNow(m: number): string {
    return new Date(now.getTime() + m * 60 * 1000).toISOString()
  }

  return [
    {
      id: 'notif-001',
      title: 'Urgent: Server disk usage at 95%',
      body: 'The production server disk is at 95% capacity. Immediate action required to prevent service disruption.',
      source: 'system',
      priority: 'critical',
      channel: 'desktop',
      digest_eligible: false,
      quiet_hours_exempt: true,
      created_at: minsAgo(5),
      escalation_due_at: minsFromNow(25)
    },
    {
      id: 'notif-002',
      title: 'Meeting starting in 15 minutes',
      body: 'Your weekly standup with the engineering team starts at 10:00 AM.',
      source: 'calendar',
      priority: 'high',
      channel: 'desktop',
      digest_eligible: false,
      quiet_hours_exempt: false,
      created_at: minsAgo(3)
    },
    {
      id: 'notif-003',
      title: 'Email from Sarah Johnson',
      body: 'Sarah replied to your proposal. She has a few questions about the timeline and budget estimates.',
      source: 'email',
      priority: 'medium',
      channel: 'desktop',
      digest_eligible: false,
      quiet_hours_exempt: false,
      created_at: minsAgo(45)
    },
    {
      id: 'notif-004',
      title: 'Task reminder: Quarterly report',
      body: 'Your quarterly report task is due in 2 days. You have 3 sections left to complete.',
      source: 'task',
      priority: 'medium',
      channel: 'desktop',
      digest_eligible: false,
      quiet_hours_exempt: false,
      created_at: minsAgo(120),
      read_at: minsAgo(60)
    },
    {
      id: 'notif-005',
      title: 'GitHub PR review requested',
      body: 'Alex Chen requested your review on PR #247: "Add dark mode support to dashboard".',
      source: 'github',
      priority: 'low',
      channel: 'digest',
      digest_eligible: true,
      quiet_hours_exempt: false,
      created_at: minsAgo(180)
    },
    {
      id: 'notif-006',
      title: 'Weekly digest: 4 low-priority updates',
      body: 'Newsletter subscription renewed. Package delivered. Blog post published. Slack workspace storage at 60%.',
      source: 'digest',
      priority: 'low',
      channel: 'digest',
      digest_eligible: true,
      quiet_hours_exempt: false,
      created_at: minsAgo(360),
      read_at: minsAgo(300)
    }
  ]
}

// In-memory store (resets on restart).
let notifications: GuiNotification[] = makeStubNotifications()

export function registerNotificationHandlers(): void {
  ipcMain.handle('rex:getNotifications', (): GuiNotification[] => {
    return [...notifications].sort(
      (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    )
  })

  ipcMain.handle('rex:markNotificationRead', (_event, id: string): void => {
    const now = new Date().toISOString()
    notifications = notifications.map((n) =>
      n.id === id ? { ...n, read_at: now } : n
    )
  })

  ipcMain.handle('rex:dismissNotification', (_event, id: string): void => {
    const now = new Date().toISOString()
    notifications = notifications.map((n) =>
      n.id === id
        ? { ...n, read_at: n.read_at ?? now, escalation_due_at: undefined }
        : n
    )
  })

  ipcMain.handle('rex:getUnreadNotificationCount', (): number => {
    return notifications.filter((n) => !n.read_at).length
  })
}
