import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'
import type { Task, TaskInput, TaskRun } from '../../types/ipc'

/**
 * Call rex_tasks_bridge.py with a JSON payload via stdin and resolve the
 * parsed JSON response from stdout.
 */
function callTasksBridge(payload: Record<string, unknown>): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const scriptPath = join(__dirname, '../../../../rex_tasks_bridge.py')

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

async function getTasks(): Promise<Task[]> {
  const result = await callTasksBridge({ command: 'list' })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to list tasks')
  }
  return (result.tasks as Task[]) ?? []
}

async function saveTask(task: TaskInput): Promise<Task> {
  const result = await callTasksBridge({ command: 'save', task })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to save task')
  }
  return result.task as Task
}

async function deleteTask(taskId: string): Promise<void> {
  const result = await callTasksBridge({ command: 'delete', id: taskId })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to delete task')
  }
}

async function setTaskEnabled(taskId: string, enabled: boolean): Promise<Task> {
  const result = await callTasksBridge({ command: 'set_enabled', id: taskId, enabled })
  if (!result.ok) {
    throw new Error((result.error as string | undefined) ?? 'Failed to update task')
  }
  return result.task as Task
}

function getTaskHistory(taskId: string): Promise<TaskRun[]> {
  // Stub: returns 1-2 sample runs so the UI can be exercised without a real backend
  const now = Date.now()
  const runs: TaskRun[] = [
    {
      id: `run-${taskId}-1`,
      taskId,
      timestamp: new Date(now - 3600_000).toISOString(),
      result: 'success',
      output: [
        '[INFO] Task started',
        '[INFO] Running prompt...',
        '[INFO] LLM responded in 1.2s',
        '[INFO] Task completed successfully'
      ]
    },
    {
      id: `run-${taskId}-2`,
      taskId,
      timestamp: new Date(now - 7200_000).toISOString(),
      result: 'failed',
      output: [
        '[INFO] Task started',
        '[ERROR] Connection refused: rex backend not running',
        '[ERROR] Task failed after 3 retries'
      ]
    }
  ]
  return Promise.resolve(runs)
}

export function registerTaskHandlers(): void {
  ipcMain.handle('rex:getTasks', async (): Promise<Task[]> => {
    return getTasks()
  })

  ipcMain.handle('rex:saveTask', async (_event, task: TaskInput): Promise<Task> => {
    return saveTask(task)
  })

  ipcMain.handle('rex:deleteTask', async (_event, taskId: string): Promise<void> => {
    return deleteTask(taskId)
  })

  ipcMain.handle(
    'rex:setTaskEnabled',
    async (_event, taskId: string, enabled: boolean): Promise<Task> => {
      return setTaskEnabled(taskId, enabled)
    }
  )

  ipcMain.handle(
    'rex:getTaskHistory',
    async (_event, taskId: string): Promise<TaskRun[]> => {
      return getTaskHistory(taskId)
    }
  )
}
