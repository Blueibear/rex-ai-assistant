import type { AiSettings, Settings } from '../types/ipc'
import { readGuiSettings, readRexConfigStrict, writeGuiSettings, writeRexConfig } from './configStore'

export type AutonomyMode = AiSettings['autonomyMode']

const AUTONOMY_RANK: Record<AutonomyMode, number> = {
  manual: 0,
  supervised: 1,
  'full-auto': 2
}

export function normalizeAutonomyMode(value: unknown): AutonomyMode | null {
  if (value === 'manual' || value === 'supervised' || value === 'full-auto') return value
  return null
}

/**
 * Resolve the single runtime autonomy mode.
 *
 * A valid canonical runtime value always wins. During migration from duplicate
 * GUI values, conflicting legacy values resolve to the more restrictive mode so
 * migration can never increase autonomy unexpectedly.
 */
export function resolveAutonomyMode(
  runtimeValue: unknown,
  legacyAiValue?: unknown,
  legacySystemValue?: unknown
): AutonomyMode {
  const runtime = normalizeAutonomyMode(runtimeValue)
  if (runtime) return runtime

  const legacy = [legacyAiValue, legacySystemValue]
    .map(normalizeAutonomyMode)
    .filter((value): value is AutonomyMode => value !== null)
  if (legacy.length === 0) return 'manual'
  return legacy.reduce((safer, candidate) =>
    AUTONOMY_RANK[candidate] < AUTONOMY_RANK[safer] ? candidate : safer
  )
}

function withoutAutonomyMode(settings: Settings | undefined): Settings | undefined {
  if (!settings || typeof settings !== 'object') return settings
  const next = { ...settings }
  delete next.autonomyMode
  return next
}

export function stripLegacyAutonomyMode(values: Settings): Settings {
  return withoutAutonomyMode(values) ?? {}
}

/**
 * Move legacy GUI autonomy values into models.autonomy_mode and remove both
 * duplicate GUI copies. The runtime config is the sole persisted authority.
 */
export function migrateLegacyAutonomySettings(): Record<string, Settings> {
  const originalStored = readGuiSettings()
  let originalConfig: Record<string, unknown>
  try {
    originalConfig = readRexConfigStrict()
  } catch {
    // A malformed runtime config must remain untouched and repairable. Defer
    // on-disk migration, but surface a conservative legacy value in memory so
    // AI Settings can still load instead of becoming another recovery blocker.
    const fallback = JSON.parse(JSON.stringify(originalStored)) as Record<string, Settings>
    const legacyAi = (originalStored.ai ?? {}) as Settings
    const legacySystem = (originalStored.system ?? {}) as Settings
    const hasLegacyValue = normalizeAutonomyMode(legacyAi.autonomyMode) !== null
      || normalizeAutonomyMode(legacySystem.autonomyMode) !== null
    if (hasLegacyValue) {
      fallback.ai = {
        ...(fallback.ai ?? {}),
        autonomyMode: resolveAutonomyMode(undefined, legacyAi.autonomyMode, legacySystem.autonomyMode)
      }
    }
    return fallback
  }
  const nextStored = JSON.parse(JSON.stringify(originalStored)) as Record<string, Settings>
  const nextConfig = JSON.parse(JSON.stringify(originalConfig)) as Record<string, unknown>
  const models = nextConfig.models && typeof nextConfig.models === 'object'
    ? { ...(nextConfig.models as Record<string, unknown>) }
    : {}

  const legacyAi = (originalStored.ai ?? {}) as Settings
  const legacySystem = (originalStored.system ?? {}) as Settings
  const hasRuntimeValue = Object.prototype.hasOwnProperty.call(models, 'autonomy_mode')
  const hasLegacyValue = normalizeAutonomyMode(legacyAi.autonomyMode) !== null
    || normalizeAutonomyMode(legacySystem.autonomyMode) !== null
  const mode = resolveAutonomyMode(models.autonomy_mode, legacyAi.autonomyMode, legacySystem.autonomyMode)
  const configChanged = (hasRuntimeValue || hasLegacyValue) && models.autonomy_mode !== mode
  models.autonomy_mode = mode
  nextConfig.models = models

  const nextAi = withoutAutonomyMode(nextStored.ai)
  const nextSystem = withoutAutonomyMode(nextStored.system)
  let storedChanged = false
  if (nextStored.ai && Object.prototype.hasOwnProperty.call(nextStored.ai, 'autonomyMode')) {
    storedChanged = true
    nextStored.ai = nextAi ?? {}
  }
  if (nextStored.system && Object.prototype.hasOwnProperty.call(nextStored.system, 'autonomyMode')) {
    storedChanged = true
    nextStored.system = nextSystem ?? {}
  }

  let configWritten = false
  let storedWritten = false
  try {
    if (configChanged) {
      writeRexConfig(nextConfig)
      configWritten = true
    }
    if (storedChanged) {
      writeGuiSettings(nextStored)
      storedWritten = true
    }
    return nextStored
  } catch (error) {
    if (storedWritten) {
      try { writeGuiSettings(originalStored) } catch { /* preserve migration error */ }
    }
    if (configWritten) {
      try { writeRexConfig(originalConfig) } catch { /* preserve migration error */ }
    }
    throw error
  }
}
