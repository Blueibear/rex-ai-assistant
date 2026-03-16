import { ipcMain } from 'electron'
import type { EmailMessage } from '../../types/ipc'

function makeStubInbox(): EmailMessage[] {
  const now = new Date()
  function hoursAgo(h: number): string {
    return new Date(now.getTime() - h * 3600 * 1000).toISOString()
  }

  return [
    {
      id: 'stub-email-001',
      thread_id: 'thread-001',
      subject: 'Production alert: API latency spike',
      sender: 'alerts@monitoring.example.com',
      recipients: ['user@example.com'],
      body_text:
        'ALERT: p99 latency has exceeded 2000ms on the /api/v2/infer endpoint for the past 5 minutes. Please investigate immediately.',
      received_at: hoursAgo(0.1),
      labels: ['INBOX'],
      is_read: false,
      priority: 'critical'
    },
    {
      id: 'stub-email-002',
      thread_id: 'thread-002',
      subject: 'Contract renewal — action needed by Friday',
      sender: 'legal@partner.example.com',
      recipients: ['user@example.com'],
      body_text:
        'Hi, the vendor contract for the infrastructure stack is up for renewal. Please review the attached terms and reply with your decision by end of Friday.',
      received_at: hoursAgo(2),
      labels: ['INBOX'],
      is_read: false,
      priority: 'high'
    },
    {
      id: 'stub-email-003',
      thread_id: 'thread-003',
      subject: 'Q3 roadmap doc ready for review',
      sender: 'pm@company.example.com',
      recipients: ['user@example.com'],
      body_text:
        'Hey team, I\'ve published the Q3 roadmap in Notion. Please take a look when you get a chance and leave comments by next Wednesday.',
      received_at: hoursAgo(5),
      labels: ['INBOX'],
      is_read: false,
      priority: 'medium'
    },
    {
      id: 'stub-email-004',
      thread_id: 'thread-004',
      subject: 'Your Amazon receipt for order #112-4567890',
      sender: 'orders@amazon.example.com',
      recipients: ['user@example.com'],
      body_text: 'Your order has been dispatched. Estimated delivery: 2 business days.',
      received_at: hoursAgo(8),
      labels: ['INBOX'],
      is_read: true,
      priority: 'low'
    },
    {
      id: 'stub-email-005',
      thread_id: 'thread-005',
      subject: 'This week in AI — newsletter',
      sender: 'newsletter@ai-weekly.example.com',
      recipients: ['user@example.com'],
      body_text:
        'Top stories this week: advances in reasoning models, new open-source releases, and a recap of the latest benchmarks.',
      received_at: hoursAgo(20),
      labels: ['INBOX'],
      is_read: true,
      priority: 'low'
    }
  ]
}

export function registerEmailHandlers(): void {
  ipcMain.handle('rex:getEmailInbox', (): EmailMessage[] => {
    return makeStubInbox()
  })

  ipcMain.handle('rex:generateEmailReply', (_event, id: string): string => {
    // Stub: returns a template reply draft. A real implementation would call
    // the LLM via Python with the original message as context.
    return (
      `Hi,\n\nThank you for your email (ref: ${id}).\n\n` +
      `I wanted to follow up on the points you raised. ` +
      `Could we schedule a quick call this week to discuss further?\n\n` +
      `Best regards`
    )
  })
}
