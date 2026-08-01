import { join } from 'path'
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs'
import type { Settings } from '../types/ipc'
import { redactSecretSettings } from './settingsRedaction'
import { resolveRuntimeRoot } from './bridgeResolver'

// ---------------------------------------------------------------------------
// Config file helpers
// ---------------------------------------------------------------------------

export function getConfigDir(): string {
  return join(resolveRuntimeRoot(), 'config')
}

function getGuiSettingsPath(): string {
  return join(getConfigDir(), 'gui_settings.json')
}

export function getRexConfigPath(): string {
  return join(getConfigDir(), 'rex_config.json')
}

export function readGuiSettings(): Record<string, Settings> {
  try {
    const p = getGuiSettingsPath()
    if (!existsSync(p)) return {}
    return JSON.parse(readFileSync(p, 'utf8')) as Record<string, Settings>
  } catch {
    return {}
  }
}

export function writeGuiSettings(settings: Record<string, Settings>): void {
  const dir = getConfigDir()
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  // Secrets belong in .env only (US-027) — strip any secret-pattern key
  // before persisting, regardless of what a caller passes in.
  const redacted = redactSecretSettings(settings)
  writeFileSync(getGuiSettingsPath(), JSON.stringify(redacted, null, 2), 'utf8')
}

export function readRexConfig(): Record<string, unknown> {
  try {
    const p = getRexConfigPath()
    if (!existsSync(p)) return {}
    return JSON.parse(readFileSync(p, 'utf8')) as Record<string, unknown>
  } catch {
    return {}
  }
}

export function writeRexConfig(config: Record<string, unknown>): void {
  const dir = getConfigDir()
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  writeFileSync(getRexConfigPath(), JSON.stringify(config, null, 2), 'utf8')
}

// ---------------------------------------------------------------------------
// .env file helpers (API keys)
// ---------------------------------------------------------------------------

function getEnvFilePath(): string {
  return join(resolveRuntimeRoot(), '.env')
}

export function readEnvFile(): Record<string, string> {
  try {
    const p = getEnvFilePath()
    if (!existsSync(p)) return {}
    const lines = readFileSync(p, 'utf8').split('\n')
    const result: Record<string, string> = {}
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('#')) continue
      const eq = trimmed.indexOf('=')
      if (eq === -1) continue
      const key = trimmed.slice(0, eq).trim()
      const val = trimmed.slice(eq + 1).trim()
      result[key] = val
    }
    return result
  } catch {
    return {}
  }
}

export function writeEnvKey(name: string, value: string): void {
  const p = getEnvFilePath()
  let lines: string[] = []
  try {
    if (existsSync(p)) {
      lines = readFileSync(p, 'utf8').split('\n')
    }
  } catch {
    lines = []
  }
  const keyPrefix = `${name}=`
  const newLine = `${name}=${value}`
  let found = false
  lines = lines.map((line) => {
    if (line.startsWith(keyPrefix) || line.trim().startsWith(keyPrefix)) {
      found = true
      return newLine
    }
    return line
  })
  if (!found) {
    lines.push(newLine)
  }
  // Trim trailing empty lines then add single newline at end
  while (lines.length > 0 && lines[lines.length - 1].trim() === '') lines.pop()
  writeFileSync(p, lines.join('\n') + '\n', 'utf8')
}
