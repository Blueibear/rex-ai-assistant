import { app, ipcMain } from 'electron'
import { appendFileSync, existsSync, mkdirSync, readFileSync, statSync, watch } from 'fs'
import { basename, dirname, join, resolve } from 'path'
import type { FSWatcher } from 'fs'

export interface LogEntry {
  timestamp: string
  level: string
  logger: string
  message: string
  extra: Record<string, unknown>
  raw?: string
}

interface LogReadResult {
  ok: boolean
  entries: LogEntry[]
  log_path?: string
  legacy_log_path?: string
  log_source?: string
  timestamp_basis?: string
  error?: string
}

const LOG_TAIL_LINES = 200
const SESSION_ID = `${new Date().toISOString().replace(/[:.]/g, '-')}-${process.pid}`

function resolveRepoRoot(): string {
  const candidates = [
    join(app.getAppPath(), '..'),
    app.getAppPath(),
    process.cwd(),
    join(__dirname, '..', '..', '..')
  ]

  for (const candidate of candidates) {
    const root = resolve(candidate)
    if (existsSync(join(root, 'config', 'rex_config.json')) || existsSync(join(root, 'bridge', 'rex_chat_stream_bridge.py'))) {
      return root
    }
  }

  return resolve(join(app.getAppPath(), '..'))
}

export function resolveActiveLogFile(): string {
  return join(resolveRepoRoot(), 'data', 'logs', 'rex.log')
}

export function resolveLegacyLogFile(): string {
  return join(resolveRepoRoot(), 'logs', 'rex.log')
}

function ensureLogFile(logPath: string): void {
  mkdirSync(dirname(logPath), { recursive: true })
  if (!existsSync(logPath)) {
    appendFileSync(logPath, '', 'utf-8')
  }
}

export function appendElectronLog(level: string, message: string, extra: Record<string, unknown> = {}): void {
  const logPath = resolveActiveLogFile()
  const legacyPath = resolveLegacyLogFile()
  ensureLogFile(logPath)
  const entry = {
    timestamp: new Date().toISOString(),
    level,
    logger: 'electron.main',
    message,
    extra: {
      ...extra,
      component: 'electron-gui',
      session_id: SESSION_ID,
      timestamp_basis: 'UTC',
      active_log_path: logPath,
      legacy_log_path: legacyPath
    }
  }
  appendFileSync(logPath, `${JSON.stringify(entry)}\n`, 'utf-8')
}

export function writeElectronSessionStart(): void {
  appendElectronLog(
    'INFO',
    '=== AskRex Electron GUI session started ===',
    { event: 'session_start' }
  )
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
      extra: (obj.extra as Record<string, unknown>) ?? {},
      raw: trimmed
    }
  } catch {
    return {
      timestamp: '',
      level: 'INFO',
      logger: 'legacy',
      message: trimmed,
      extra: { format: 'legacy_or_plain_text' },
      raw: trimmed
    }
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

// Active watcher state. Only one active watcher is needed for the GUI page.
let activeWatcher: FSWatcher | null = null
let lastSize = 0

export function registerLogsHandlers(): void {
  ipcMain.handle(
    'rex:getLogs',
    async (_event, limit: number = LOG_TAIL_LINES): Promise<LogReadResult> => {
      const logPath = resolveActiveLogFile()
      const legacyPath = resolveLegacyLogFile()
      ensureLogFile(logPath)
      const entries = readLastLines(logPath, limit)
      return {
        ok: true,
        entries,
        log_path: logPath,
        legacy_log_path: legacyPath,
        log_source: 'active_current_session',
        timestamp_basis: 'UTC'
      }
    }
  )

  ipcMain.handle(
    'rex:startLogTail',
    async (event): Promise<{ ok: boolean; log_path?: string; error?: string }> => {
      if (activeWatcher) {
        try { activeWatcher.close() } catch { /* ignore */ }
        activeWatcher = null
      }

      const logPath = resolveActiveLogFile()
      ensureLogFile(logPath)

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
          } catch {
            // File may be rotated or temporarily unreadable.
          }
        })

        return { ok: true, log_path: logPath }
      } catch (err) {
        return { ok: false, log_path: logPath, error: String(err) }
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
    async (): Promise<{ ok: boolean; content?: string; filename?: string; log_path?: string; error?: string }> => {
      const logPath = resolveActiveLogFile()
      ensureLogFile(logPath)
      try {
        const content = readFileSync(logPath, 'utf-8')
        return { ok: true, content, filename: basename(logPath), log_path: logPath }
      } catch (err) {
        return { ok: false, log_path: logPath, error: String(err) }
      }
    }
  )
}
