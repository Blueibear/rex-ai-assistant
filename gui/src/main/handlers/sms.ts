import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import type { SMSMessage, SMSThread } from '../../types/ipc'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

// In-memory store for messages sent during this session.
let sessionThreads: SMSThread[] = []

function callSmsBridge(session: ElectronSessionIdentity, command: string, extra: object = {}): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const scriptPath = resolveBridgePath('rex_sms_bridge.py')
    const py = spawn(resolvePythonCommand(), [scriptPath], {
      ...bridgeSpawnOptions(),
      stdio: ['pipe', 'pipe', 'pipe']
    })

    let stdout = ''
    let stderr = ''

    py.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString()
    })
    py.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString()
    })

    py.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`SMS bridge exited ${code}: ${stderr.slice(0, 300)}`))
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()))
      } catch {
        reject(new Error(`Failed to parse SMS bridge response: ${stdout.slice(0, 200)}`))
      }
    })

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn SMS bridge: ${err.message}`))
    })

    py.stdin.write(JSON.stringify(privateSessionPayload(session, { command, ...extra })))
    py.stdin.end()
  })
}

async function getSMSThreads(session: ElectronSessionIdentity): Promise<SMSThread[]> {
  try {
    const result = (await callSmsBridge(session, 'list_threads')) as {
      ok: boolean
      threads?: SMSThread[]
      error?: string
    }
    const backendThreads = result.ok && Array.isArray(result.threads) ? result.threads : []
    // Merge backend threads with any locally-sent messages this session.
    const merged = [...backendThreads]
    for (const local of sessionThreads) {
      if (!merged.find((t) => t.id === local.id)) {
        merged.push(local)
      }
    }
    return merged.sort((a, b) => new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime())
  } catch {
    return [...sessionThreads].sort((a, b) => new Date(b.last_message_at).getTime() - new Date(a.last_message_at).getTime())
  }
}

export function registerSMSHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getSMSThreads', async (): Promise<SMSThread[]> => {
    return getSMSThreads(session)
  })

  ipcMain.handle('rex:getSMSThread', async (_event, threadId: string): Promise<SMSThread | undefined> => {
    const threads = await getSMSThreads(session)
    return threads.find((t) => t.id === threadId)
  })

  ipcMain.handle('rex:sendSMS', (_event, to: string, body: string): SMSMessage => {
    const now = new Date().toISOString()
    const threadId = `thread-${to.replace(/\D/g, '')}`
    const newMsg: SMSMessage = {
      id: `outbound-${Date.now()}`,
      thread_id: threadId,
      direction: 'outbound',
      body,
      from_number: '',
      to_number: to,
      sent_at: now,
      status: 'sent'
    }

    const existing = sessionThreads.find((t) => t.id === threadId)
    if (existing) {
      sessionThreads = sessionThreads.map((t) => (t.id === threadId ? { ...t, messages: [...t.messages, newMsg], last_message_at: now } : t))
    } else {
      const newThread: SMSThread = {
        id: threadId,
        contact_name: to,
        contact_number: to,
        messages: [newMsg],
        last_message_at: now,
        unread_count: 0
      }
      sessionThreads = [...sessionThreads, newThread]
    }

    return newMsg
  })
}
