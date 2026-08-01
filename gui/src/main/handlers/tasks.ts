import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import type { Task, TaskInput, TaskRun } from '../../types/ipc'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from '../bridgeResolver'
import { privateSessionPayload, type ElectronSessionIdentity } from '../sessionIdentity'

/**
 * Call rex_tasks_bridge.py with a JSON payload via stdin and resolve the
 * parsed JSON response from stdout.
 */
function callTasksBridge(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const scriptPath = resolveBridgePath('rex_tasks_bridge.py')

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
        reject(new Error(`Tasks bridge exited with code ${code}: ${stderr.slice(0, 300)}`))
        return
      }
      try {
        const result = JSON.parse(stdout.trim()) as Record<string, unknown>
        resolve(result)
      } catch {
        reject(new Error(`Failed to parse tasks bridge response: ${stdout.slice(0, 200)}`))
      }
    })

    py.on('error', (err) => {
      reject(new Error(`Failed to spawn Python tasks bridge: ${err.message}`))
    })

    py.stdin.write(JSON.stringify(payload))
    py.stdin.end()
  })
}

async function getTasks(session: ElectronSessionIdentity): Promise<Task[]> {
  const result = await callTasksBridge(privateSessionPayload(session, { command: 'list' }))
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to list tasks')
  }
  return (result.tasks as Task[]) ?? []
}

async function saveTask(session: ElectronSessionIdentity, task: TaskInput): Promise<Task> {
  const result = await callTasksBridge(privateSessionPayload(session, { command: 'save', task }))
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to save task')
  }
  return result.task as Task
}

async function deleteTask(session: ElectronSessionIdentity, taskId: string): Promise<void> {
  const result = await callTasksBridge(privateSessionPayload(session, { command: 'delete', id: taskId }))
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to delete task')
  }
}

async function setTaskEnabled(session: ElectronSessionIdentity, taskId: string, enabled: boolean): Promise<Task> {
  const result = await callTasksBridge(
    privateSessionPayload(session, {
      command: 'set_enabled',
      id: taskId,
      enabled
    })
  )
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to update task')
  }
  return result.task as Task
}

async function getTaskHistory(session: ElectronSessionIdentity, taskId: string): Promise<TaskRun[]> {
  const result = await callTasksBridge(privateSessionPayload(session, { command: 'history', id: taskId }))
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to load task history')
  }
  return (result.runs as TaskRun[]) ?? []
}

export function registerTaskHandlers(session: ElectronSessionIdentity): void {
  ipcMain.handle('rex:getTasks', async (): Promise<Task[]> => {
    return getTasks(session)
  })

  ipcMain.handle('rex:saveTask', async (_event, task: TaskInput): Promise<Task> => {
    return saveTask(session, task)
  })

  ipcMain.handle('rex:deleteTask', async (_event, taskId: string): Promise<void> => {
    return deleteTask(session, taskId)
  })

  ipcMain.handle('rex:setTaskEnabled', async (_event, taskId: string, enabled: boolean): Promise<Task> => {
    return setTaskEnabled(session, taskId, enabled)
  })

  ipcMain.handle('rex:getTaskHistory', async (_event, taskId: string): Promise<TaskRun[]> => {
    return getTaskHistory(session, taskId)
  })
}
