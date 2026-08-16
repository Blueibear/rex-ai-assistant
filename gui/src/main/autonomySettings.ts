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

interface AutonomyMigrationPlan {
  nextStored: Record<string, Settings>
  nextConfig: Record<string, unknown>
  configChanged: boolean
  storedChanged: boolean
}

function cloneRecord<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T
}

function legacySettings(stored: Record<string, Settings>): [Settings, Settings] {
  return [(stored.ai ?? {}) as Settings, (stored.system ?? {}) as Settings]
}

function malformedConfigFallback(
  originalStored: Record<string, Settings>
): Record<string, Settings> {
  const fallback = cloneRecord(originalStored)
  const [legacyAi, legacySystem] = legacySettings(originalStored)
  const hasLegacyValue = normalizeAutonomyMode(legacyAi.autonomyMode) !== null
    || normalizeAutonomyMode(legacySystem.autonomyMode) !== null
  if (!hasLegacyValue) return fallback

  fallback.ai = {
    ...(fallback.ai ?? {}),
    autonomyMode: resolveAutonomyMode(undefined, legacyAi.autonomyMode, legacySystem.autonomyMode)
  }
  return fallback
}

function buildMigrationPlan(
  originalStored: Record<string, Settings>,
  originalConfig: Record<string, unknown>
): AutonomyMigrationPlan {
  const nextStored = cloneRecord(originalStored)
  const nextConfig = cloneRecord(originalConfig)
  const models = nextConfig.models && typeof nextConfig.models === 'object'
    ? { ...(nextConfig.models as Record<string, unknown>) }
    : {}
  const [legacyAi, legacySystem] = legacySettings(originalStored)
  const hasRuntimeValue = Object.prototype.hasOwnProperty.call(models, 'autonomy_mode')
  const hasLegacyValue = normalizeAutonomyMode(legacyAi.autonomyMode) !== null
    || normalizeAutonomyMode(legacySystem.autonomyMode) !== null
  const mode = resolveAutonomyMode(models.autonomy_mode, legacyAi.autonomyMode, legacySystem.autonomyMode)
  const configChanged = (hasRuntimeValue || hasLegacyValue) && models.autonomy_mode !== mode
  models.autonomy_mode = mode
  nextConfig.models = models

  const aiHadLegacy = Boolean(
    nextStored.ai && Object.prototype.hasOwnProperty.call(nextStored.ai, 'autonomyMode')
  )
  const systemHadLegacy = Boolean(
    nextStored.system && Object.prototype.hasOwnProperty.call(nextStored.system, 'autonomyMode')
  )
  if (aiHadLegacy) nextStored.ai = withoutAutonomyMode(nextStored.ai) ?? {}
  if (systemHadLegacy) nextStored.system = withoutAutonomyMode(nextStored.system) ?? {}

  return {
    nextStored,
    nextConfig,
    configChanged,
    storedChanged: aiHadLegacy || systemHadLegacy
  }
}

function rollbackMigration(
  originalStored: Record<string, Settings>,
  originalConfig: Record<string, unknown>,
  state: { storedWritten: boolean; configWritten: boolean }
): void {
  const { storedWritten, configWritten } = state
  if (storedWritten) {
    try { writeGuiSettings(originalStored) } catch { /* preserve migration error */ }
  }
  if (configWritten) {
    try { writeRexConfig(originalConfig) } catch { /* preserve migration error */ }
  }
}

function applyMigrationPlan(
  plan: AutonomyMigrationPlan,
  originalStored: Record<string, Settings>,
  originalConfig: Record<string, unknown>
): Record<string, Settings> {
  let configWritten = false
  let storedWritten = false
  try {
    if (plan.configChanged) {
      writeRexConfig(plan.nextConfig)
      configWritten = true
    }
    if (plan.storedChanged) {
      writeGuiSettings(plan.nextStored)
      storedWritten = true
    }
    return plan.nextStored
  } catch (error) {
    rollbackMigration(originalStored, originalConfig, { storedWritten, configWritten })
    throw error
  }
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
    return malformedConfigFallback(originalStored)
  }

  return applyMigrationPlan(
    buildMigrationPlan(originalStored, originalConfig),
    originalStored,
    originalConfig
  )
}
