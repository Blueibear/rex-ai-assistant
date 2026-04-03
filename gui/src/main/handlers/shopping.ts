import { ipcMain, app } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'
import { existsSync } from 'fs'
import type { ShoppingItem } from '../../types/ipc'

function resolvePythonCommand(): string {
  const bundledVenvPython = join(app.getAppPath(), '..', '.venv', 'Scripts', 'python.exe')
  return existsSync(bundledVenvPython) ? bundledVenvPython : 'python'
}

function callShoppingBridge(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return new Promise((resolve) => {
    const scriptPath = join(__dirname, '../../../../rex_shopping_list_bridge.py')

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

export function registerShoppingHandlers(): void {
  ipcMain.handle(
    'rex:getShoppingItems',
    (): Promise<{ ok: boolean; items: ShoppingItem[]; error?: string }> =>
      callShoppingBridge({ command: 'list' }) as Promise<{
        ok: boolean
        items: ShoppingItem[]
        error?: string
      }>
  )

  ipcMain.handle(
    'rex:addShoppingItem',
    (_event, name: string, quantity: number, unit: string): Promise<{
      ok: boolean
      item?: ShoppingItem
      error?: string
    }> =>
      callShoppingBridge({ command: 'add', name, quantity, unit }) as Promise<{
        ok: boolean
        item?: ShoppingItem
        error?: string
      }>
  )

  ipcMain.handle(
    'rex:checkShoppingItem',
    (_event, id: string): Promise<{ ok: boolean; error?: string }> =>
      callShoppingBridge({ command: 'check', id }) as Promise<{ ok: boolean; error?: string }>
  )

  ipcMain.handle(
    'rex:uncheckShoppingItem',
    (_event, id: string): Promise<{ ok: boolean; error?: string }> =>
      callShoppingBridge({ command: 'uncheck', id }) as Promise<{ ok: boolean; error?: string }>
  )

  ipcMain.handle(
    'rex:clearCheckedShoppingItems',
    (): Promise<{ ok: boolean; count?: number; error?: string }> =>
      callShoppingBridge({ command: 'clear_checked' }) as Promise<{
        ok: boolean
        count?: number
        error?: string
      }>
  )
}
