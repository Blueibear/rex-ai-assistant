import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import type { QuickAction } from '../../types/ipc'
import { resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'

function callQuickActionsBridge(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    const scriptPath = resolveBridgePath('rex_quick_actions_bridge.py')

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
        resolve({ ok: false, error: `bridge exited ${code}: ${stderr.slice(0, 200)}` })
        return
      }
      try {
        resolve(JSON.parse(stdout.trim()) as Record<string, unknown>)
      } catch {
        resolve({ ok: false, error: `parse error: ${stdout.slice(0, 100)}` })
      }
    })

    py.on('error', (err) => {
      resolve({ ok: false, error: `spawn error: ${err.message}` })
    })

    py.stdin.write(JSON.stringify(payload))
    py.stdin.end()
  })
}

export function registerQuickActionsHandlers(): void {
  ipcMain.handle(
    'rex:listQuickActions',
    (): Promise<{ ok: boolean; quick_actions: QuickAction[]; error?: string }> =>
      callQuickActionsBridge({ command: 'list' }) as Promise<{
        ok: boolean
        quick_actions: QuickAction[]
        error?: string
      }>
  )

  ipcMain.handle(
    'rex:createQuickAction',
    (
      _event,
      label: string,
      commandText: string
    ): Promise<{ ok: boolean; action?: QuickAction; error?: string }> =>
      callQuickActionsBridge({ command: 'add', label, command_text: commandText }) as Promise<{
        ok: boolean
        action?: QuickAction
        error?: string
      }>
  )
}
