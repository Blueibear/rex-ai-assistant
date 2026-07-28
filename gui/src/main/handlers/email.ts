import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import { resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import type { EmailMessage } from '../../types/ipc'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

function callEmailBridge(session: ElectronSessionIdentity, command: string, extra: object = {}): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const scriptPath = resolveBridgePath('rex_email_bridge.py')
    const py = spawn(resolvePythonCommand(), [scriptPath], {
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
        reject(new Error(`Email bridge exited ${code}: ${stderr.slice(0, 300)}`))
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()))
      } catch {
        reject(new Error(`Failed to parse email bridge response: ${stdout.slice(0, 200)}`))
      }
    })

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn email bridge: ${err.message}`))
    })

    py.stdin.write(JSON.stringify(privateSessionPayload(session, { command, ...extra })))
    py.stdin.end()
  })
}

async function getEmailInbox(session: ElectronSessionIdentity): Promise<EmailMessage[]> {
  const result = (await callEmailBridge(session, 'list', { limit: 50 })) as {
    ok: boolean
    messages?: EmailMessage[]
    error?: string
  }
  if (result.ok && Array.isArray(result.messages)) {
    return result.messages
  }
  throw new Error(result.error ?? 'Email bridge did not return inbox messages.')
}

export function registerEmailHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getEmailInbox', async (): Promise<EmailMessage[]> => {
    return getEmailInbox(session)
  })

  ipcMain.handle('rex:generateEmailReply', (_event, id: string): string => {
    // Stub: returns a template reply draft. A real implementation would call
    // the LLM via Python with the original message as context.
    return (
      `Hi,\n\nThank you for your email (ref: ${id}).\n\n` + `I wanted to follow up on the points you raised. ` + `Could we schedule a quick call this week to discuss further?\n\n` + `Best regards`
    )
  })
}
