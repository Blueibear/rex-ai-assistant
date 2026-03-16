import { ipcMain } from 'electron'
import type { SMSMessage, SMSThread } from '../../types/ipc'

function makeStubThreads(): SMSThread[] {
  const now = new Date()
  function minsAgo(m: number): string {
    return new Date(now.getTime() - m * 60 * 1000).toISOString()
  }

  const aliceMessages: SMSMessage[] = [
    {
      id: 'sms-alice-001',
      thread_id: 'thread-alice',
      direction: 'inbound',
      body: 'Hey, are you free for a call tomorrow?',
      from_number: '+14155550101',
      to_number: '+15559876543',
      sent_at: minsAgo(30),
      status: 'delivered'
    },
    {
      id: 'sms-alice-002',
      thread_id: 'thread-alice',
      direction: 'outbound',
      body: 'Sure, how about 3pm?',
      from_number: '+15559876543',
      to_number: '+14155550101',
      sent_at: minsAgo(25),
      status: 'delivered'
    },
    {
      id: 'sms-alice-003',
      thread_id: 'thread-alice',
      direction: 'inbound',
      body: 'Perfect, talk then!',
      from_number: '+14155550101',
      to_number: '+15559876543',
      sent_at: minsAgo(20),
      status: 'delivered'
    }
  ]

  const bobMessages: SMSMessage[] = [
    {
      id: 'sms-bob-001',
      thread_id: 'thread-bob',
      direction: 'inbound',
      body: "Don't forget the team lunch at noon.",
      from_number: '+14155550202',
      to_number: '+15559876543',
      sent_at: minsAgo(120),
      status: 'delivered'
    },
    {
      id: 'sms-bob-002',
      thread_id: 'thread-bob',
      direction: 'outbound',
      body: 'Thanks for the reminder, see you there!',
      from_number: '+15559876543',
      to_number: '+14155550202',
      sent_at: minsAgo(115),
      status: 'delivered'
    }
  ]

  return [
    {
      id: 'thread-alice',
      contact_name: 'Alice',
      contact_number: '+14155550101',
      messages: aliceMessages,
      last_message_at: aliceMessages[aliceMessages.length - 1].sent_at,
      unread_count: 1
    },
    {
      id: 'thread-bob',
      contact_name: 'Bob',
      contact_number: '+14155550202',
      messages: bobMessages,
      last_message_at: bobMessages[bobMessages.length - 1].sent_at,
      unread_count: 0
    }
  ]
}

// In-memory thread store (resets on restart).
let smsThreads: SMSThread[] = makeStubThreads()

export function registerSMSHandlers(): void {
  ipcMain.handle('rex:getSMSThreads', (): SMSThread[] => {
    return [...smsThreads].sort(
      (a, b) => new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime()
    )
  })

  ipcMain.handle('rex:sendSMS', (_event, to: string, body: string): SMSMessage => {
    const now = new Date().toISOString()
    const threadId = `thread-${to.replace(/\D/g, '')}`
    const newMsg: SMSMessage = {
      id: `stub-${Date.now()}`,
      thread_id: threadId,
      direction: 'outbound',
      body,
      from_number: '+15559876543',
      to_number: to,
      sent_at: now,
      status: 'stub'
    }

    const existing = smsThreads.find((t) => t.id === threadId)
    if (existing) {
      smsThreads = smsThreads.map((t) =>
        t.id === threadId
          ? { ...t, messages: [...t.messages, newMsg], last_message_at: now }
          : t
      )
    } else {
      const newThread: SMSThread = {
        id: threadId,
        contact_name: to,
        contact_number: to,
        messages: [newMsg],
        last_message_at: now,
        unread_count: 0
      }
      smsThreads = [...smsThreads, newThread]
    }

    return newMsg
  })
}
