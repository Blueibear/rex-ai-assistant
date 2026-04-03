import { app, ipcMain } from 'electron'
import { existsSync, readFileSync, statSync, watch } from 'fs'
import { join } from 'path'
import type { FSWatcher } from 'fs'

export interface LogEntry {
  timestamp: string
  level: string
  logger: string
  message: string
  extra: Record<string, unknown>
  raw?: string
}

const LOG_TAIL_LINES = 200

function resolveLogFile(): string {
  // Prefer a log file relative to the app's working directory.
  return join(app.getAppPath(), '..', 'logs', 'rex.log')
}

function parseLogLine(line: string): LogEntry | null {
  const trimmed = line.trim()
  if (!trimmed) return null
  try {
    const obj = JSON.parse(trimmed) as Partial<LogEntry>
    return {
      timestamp: String(obj.timestamp ?? ''),
      level: String(obj.level ?? 'INFO'),
      logger: String(obj.logger ?? ''),
      message: String(obj.message ?? trimmed),
      extra: (obj.extra as Record<string, unknown>) ?? {}
    }
  } catch {
    // Non-JSON line — wrap as plain message.
    return { timestamp: '', level: 'INFO', logger: '', message: trimmed, extra: {} }
  }
}

function readLastLines(filePath: string, n: number): LogEntry[] {
  if (!existsSync(filePath)) return []
  try {
    const content = readFileSync(filePath, 'utf-8')
    const lines = content.split('\n')
    const tail = lines.slice(-n)
    return tail.map(parseLogLine).filter((e): e is LogEntry => e !== null)
  } catch {
    return []
  }
}

// Active watcher state — only one active at a time.
let activeWatcher: FSWatcher | null = null
let lastSize = 0

export function registerLogsHandlers(): void {
  ipcMain.handle(
    'rex:getLogs',
    async (_event, limit: number = LOG_TAIL_LINES): Promise<{ ok: boolean; entries: LogEntry[]; log_path?: string; error?: string }> => {
      const logPath = resolveLogFile()
      const entries = readLastLines(logPath, limit)
      return { ok: true, entries, log_path: logPath }
    }
  )

  ipcMain.handle(
    'rex:startLogTail',
    async (event): Promise<{ ok: boolean; error?: string }> => {
      // Stop any existing watcher.
      if (activeWatcher) {
        try { activeWatcher.close() } catch { /* ignore */ }
        activeWatcher = null
      }

      const logPath = resolveLogFile()
      if (!existsSync(logPath)) {
        // File doesn't exist yet — watch parent dir for creation.
        return { ok: true }
      }

      try {
        lastSize = statSync(logPath).size

        activeWatcher = watch(logPath, () => {
          if (event.sender.isDestroyed()) {
            activeWatcher?.close()
            activeWatcher = null
            return
          }
          try {
            const stat = statSync(logPath)
            if (stat.size <= lastSize) return
            const content = readFileSync(logPath, 'utf-8')
            const newContent = content.slice(lastSize)
            lastSize = stat.size
            const lines = newContent.split('\n')
            for (const line of lines) {
              const entry = parseLogLine(line)
              if (entry) {
                event.sender.send('rex:logEntry', entry)
              }
            }
          } catch { /* file rotated or unreadable */ }
        })

        return { ok: true }
      } catch (err) {
        return { ok: false, error: String(err) }
      }
    }
  )

  ipcMain.handle(
    'rex:stopLogTail',
    async (): Promise<{ ok: boolean }> => {
      if (activeWatcher) {
        try { activeWatcher.close() } catch { /* ignore */ }
        activeWatcher = null
      }
      return { ok: true }
    }
  )

  ipcMain.handle(
    'rex:downloadLogs',
    async (): Promise<{ ok: boolean; content?: string; filename?: string; error?: string }> => {
      const logPath = resolveLogFile()
      if (!existsSync(logPath)) {
        return { ok: false, error: 'Log file not found.' }
      }
      try {
        const content = readFileSync(logPath, 'utf-8')
        return { ok: true, content, filename: 'rex.log' }
      } catch (err) {
        return { ok: false, error: String(err) }
      }
    }
  )
}
