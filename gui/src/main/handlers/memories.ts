import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'
import type { Memory, MemoryUpdateInput } from '../../types/ipc'

/**
 * Call rex_memories_bridge.py with a JSON payload via stdin and resolve the
 * parsed JSON response from stdout.
 */
function callMemoriesBridge(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const scriptPath = join(__dirname, '../../../../rex_memories_bridge.py')

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
        reject(new Error(`Memories bridge exited with code ${code}: ${stderr.slice(0, 300)}`))
        return
      }
      try {
        const result = JSON.parse(stdout.trim()) as Record<string, unknown>
        resolve(result)
      } catch {
        reject(new Error(`Failed to parse memories bridge response: ${stdout.slice(0, 200)}`))
      }
    })

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn Python memories bridge: ${err.message}`))
    })

    py.stdin.write(JSON.stringify(payload))
    py.stdin.end()
  })
}

async function getMemories(): Promise<Memory[]> {
  const result = await callMemoriesBridge({ command: 'list' })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to list memories')
  }
  return (result.memories as Memory[]) ?? []
}

async function addMemory(data: MemoryUpdateInput): Promise<Memory> {
  const result = await callMemoriesBridge({ command: 'add', data })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to add memory')
  }
  return result.memory as Memory
}

async function updateMemory(id: string, data: MemoryUpdateInput): Promise<Memory> {
  const result = await callMemoriesBridge({ command: 'update', id, data })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to update memory')
  }
  return result.memory as Memory
}

async function deleteMemory(id: string): Promise<void> {
  const result = await callMemoriesBridge({ command: 'delete', id })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to delete memory')
  }
}

export function registerMemoriesHandlers(): void {
  ipcMain.handle('rex:getMemories', async (): Promise<Memory[]> => {
    return getMemories()
  })

  ipcMain.handle('rex:addMemory', async (_event, data: MemoryUpdateInput): Promise<Memory> => {
    return addMemory(data)
  })

  ipcMain.handle('rex:updateMemory', async (_event, id: string, data: MemoryUpdateInput): Promise<Memory> => {
    return updateMemory(id, data)
  })

  ipcMain.handle('rex:deleteMemory', async (_event, id: string): Promise<void> => {
    return deleteMemory(id)
  })
}
