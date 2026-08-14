import { dialog, ipcMain } from 'electron'
import { spawn } from 'child_process'
import type { Memory, MemoryUpdateInput, Procedure } from '../../types/ipc'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

/**
 * Call rex_memories_bridge.py with a JSON payload via stdin and resolve the
 * parsed JSON response from stdout.
 */
function callMemoriesBridge(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const scriptPath = resolveBridgePath('rex_memories_bridge.py')

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

async function getMemories(session: ElectronSessionIdentity): Promise<Memory[]> {
  const result = await callMemoriesBridge(privateSessionPayload(session, { command: 'list' }))
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to list memories')
  }
  return (result.memories as Memory[]) ?? []
}

async function addMemory(session: ElectronSessionIdentity, data: MemoryUpdateInput): Promise<Memory> {
  const result = await callMemoriesBridge(privateSessionPayload(session, { command: 'add', data }))
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to add memory')
  }
  return result.memory as Memory
}

async function updateMemory(session: ElectronSessionIdentity, id: string, data: MemoryUpdateInput): Promise<Memory> {
  const result = await callMemoriesBridge(privateSessionPayload(session, { command: 'update', id, data }))
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to update memory')
  }
  return result.memory as Memory
}

async function deleteMemory(session: ElectronSessionIdentity, id: string): Promise<void> {
  const result = await callMemoriesBridge(privateSessionPayload(session, { command: 'delete', id }))
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to delete memory')
  }
}

async function getProcedures(session: ElectronSessionIdentity): Promise<Procedure[]> {
  const result = await callMemoriesBridge(
    privateSessionPayload(session, { command: 'procedures-list' })
  )
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to list learned procedures')
  }
  return (result.procedures as Procedure[]) ?? []
}

async function mutateProcedure(
  session: ElectronSessionIdentity,
  id: string,
  command: 'procedures-approve' | 'procedures-disable' | 'procedures-revoke',
  confirmed = false
): Promise<Procedure> {
  const result = await callMemoriesBridge(
    privateSessionPayload(session, { command, id, ...(confirmed ? { confirmed: true } : {}) })
  )
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to update learned procedure')
  }
  return result.procedure as Procedure
}

async function approveProcedure(
  session: ElectronSessionIdentity,
  id: string
): Promise<Procedure> {
  const response = await dialog.showMessageBox({
    type: 'warning',
    title: 'Approve learned procedure?',
    message: 'Approve this learned procedure for future execution?',
    detail:
      'This procedure was learned from a verified outcome. If it can mutate state or carries elevated risk, approval allows it to become active, but current permissions and safety checks still apply on every execution.',
    buttons: ['Cancel', 'Approve'],
    defaultId: 0,
    cancelId: 0,
    noLink: true
  })
  if (response.response !== 1) {
    throw new Error('Procedure approval cancelled')
  }
  return mutateProcedure(session, id, 'procedures-approve', true)
}

async function deleteProcedure(session: ElectronSessionIdentity, id: string): Promise<void> {
  const result = await callMemoriesBridge(
    privateSessionPayload(session, { command: 'procedures-delete', id })
  )
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to delete learned procedure')
  }
}

export function registerMemoriesHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getMemories', async (): Promise<Memory[]> => {
    return getMemories(session)
  })

  ipcMain.handle('rex:addMemory', async (_event, data: MemoryUpdateInput): Promise<Memory> => {
    return addMemory(session, data)
  })

  ipcMain.handle('rex:updateMemory', async (_event, id: string, data: MemoryUpdateInput): Promise<Memory> => {
    return updateMemory(session, id, data)
  })

  ipcMain.handle('rex:deleteMemory', async (_event, id: string): Promise<void> => {
    return deleteMemory(session, id)
  })

  ipcMain.handle('rex:getProcedures', async (): Promise<Procedure[]> => {
    return getProcedures(session)
  })

  ipcMain.handle('rex:approveProcedure', async (_event, id: string): Promise<Procedure> => {
    return approveProcedure(session, id)
  })

  ipcMain.handle('rex:disableProcedure', async (_event, id: string): Promise<Procedure> => {
    return mutateProcedure(session, id, 'procedures-disable')
  })

  ipcMain.handle('rex:revokeProcedure', async (_event, id: string): Promise<Procedure> => {
    return mutateProcedure(session, id, 'procedures-revoke')
  })

  ipcMain.handle('rex:deleteProcedure', async (_event, id: string): Promise<void> => {
    return deleteProcedure(session, id)
  })
}
