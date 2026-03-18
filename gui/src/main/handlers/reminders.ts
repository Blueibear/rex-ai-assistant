import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'
import type { Reminder, ReminderInput } from '../../types/ipc'

/**
 * Call rex_reminders_bridge.py with a JSON payload via stdin and resolve the
 * parsed JSON response from stdout.
 */
function callRemindersBridge(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const scriptPath = join(__dirname, '../../../../rex_reminders_bridge.py')

    const py = spawn('python', [scriptPath], {
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
        reject(new Error(`Reminders bridge exited with code ${code}: ${stderr.slice(0, 300)}`))
        return
      }
      try {
        const result = JSON.parse(stdout.trim()) as Record<string, unknown>
        resolve(result)
      } catch {
        reject(new Error(`Failed to parse reminders bridge response: ${stdout.slice(0, 200)}`))
      }
    })

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn Python reminders bridge: ${err.message}`))
    })

    py.stdin.write(JSON.stringify(payload))
    py.stdin.end()
  })
}

async function getReminders(): Promise<Reminder[]> {
  const result = await callRemindersBridge({ command: 'list' })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to list reminders')
  }
  return (result.reminders as Reminder[]) ?? []
}

async function saveReminder(reminder: ReminderInput): Promise<Reminder> {
  const result = await callRemindersBridge({ command: 'save', reminder })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to save reminder')
  }
  return result.reminder as Reminder
}

async function deleteReminder(id: string): Promise<void> {
  const result = await callRemindersBridge({ command: 'delete', id })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to delete reminder')
  }
}

async function completeReminder(id: string): Promise<void> {
  const result = await callRemindersBridge({ command: 'complete', id })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to complete reminder')
  }
}

export function registerRemindersHandlers(): void {
  ipcMain.handle('rex:getReminders', async (): Promise<Reminder[]> => {
    return getReminders()
  })

  ipcMain.handle('rex:saveReminder', async (_event, input: ReminderInput): Promise<Reminder> => {
    return saveReminder(input)
  })

  ipcMain.handle('rex:deleteReminder', async (_event, id: string): Promise<void> => {
    return deleteReminder(id)
  })

  ipcMain.handle('rex:completeReminder', async (_event, id: string): Promise<void> => {
    return completeReminder(id)
  })
}
