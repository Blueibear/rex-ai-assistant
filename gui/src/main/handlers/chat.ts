import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'

/**
 * Spawn the rex_chat_bridge.py script, pass the message via stdin,
 * and resolve with the reply string from stdout.
 */
function callRexBackend(message: string): Promise<string> {
  return new Promise((resolve, reject) => {
    // Bridge script lives at repo root (two levels up from gui/src/main/)
    const scriptPath = join(__dirname, '../../../../rex_chat_bridge.py')

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
      if (code !== 0 && stdout.trim() === '') {
        reject(new Error(`Rex exited with code ${code}: ${stderr.slice(0, 300)}`))
        return
      }
      try {
        const result = JSON.parse(stdout.trim()) as { ok: boolean; reply?: string; error?: string }
        if (result.ok) {
          resolve(result.reply ?? '')
        } else {
          reject(new Error(result.error ?? 'Unknown error from Rex backend'))
        }
      } catch {
        reject(new Error(`Failed to parse Rex response: ${stdout.slice(0, 200)}`))
      }
    })

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn Python: ${err.message}`))
    })

    py.stdin.write(JSON.stringify({ message }))
    py.stdin.end()
  })
}

export function registerChatHandlers(): void {
  ipcMain.handle('rex:sendChat', async (_event, message: string): Promise<string> => {
    return callRexBackend(message)
  })
}
