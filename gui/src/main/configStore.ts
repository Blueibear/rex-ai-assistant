import { join } from 'path'
import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, unlinkSync, writeFileSync } from 'fs'
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
  // Secrets belong in the OS vault — strip any secret-pattern key
  // before persisting, regardless of what a caller passes in.
  const redacted = redactSecretSettings(settings)
  atomicWriteJson(getGuiSettingsPath(), redacted)
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
export function readRexConfigStrict(): Record<string, unknown> {
  const p = getRexConfigPath()
  if (!existsSync(p)) return {}
  const parsed = JSON.parse(readFileSync(p, 'utf8')) as unknown
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error('rex_config.json must contain a JSON object')
  }
  return parsed as Record<string, unknown>
}

export function writeRexConfig(config: Record<string, unknown>): void {
  const dir = getConfigDir()
  if (!existsSync(dir)) mkdirSync(dir, { recursive: true })
  atomicWriteJson(getRexConfigPath(), config)
}
function atomicWriteJson(path: string, value: unknown): void {
  const temp = `${path}.tmp-${process.pid}-${Date.now()}`
  let fd: number | undefined
  try {
    fd = openSync(temp, 'w')
    writeFileSync(fd, `${JSON.stringify(value, null, 2)}\n`, 'utf8')
    fsyncSync(fd)
    closeSync(fd)
    fd = undefined
    renameSync(temp, path)
  } finally {
    if (fd !== undefined) closeSync(fd)
    try { unlinkSync(temp) } catch { /* already renamed or never created */ }
  }
}
