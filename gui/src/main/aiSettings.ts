import type { AiSettings, Settings } from '../types/ipc'
import { readRexConfig } from './configStore'
import { normalizeAutonomyMode, resolveAutonomyMode } from './autonomySettings'

export const OPENROUTER_DEFAULT_BASE_URL = 'https://openrouter.ai/api/v1'
export const OPENAI_DEFAULT_MODEL = 'gpt-4o'
export const OPENROUTER_DEFAULT_MODEL = 'openai/gpt-4o'

export function normalizeAiModelRouting(raw: unknown): AiSettings['modelRouting'] {
  const source = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  return {
    default: typeof source.default === 'string' ? source.default : '',
    coding: typeof source.coding === 'string' ? source.coding : '',
    reasoning: typeof source.reasoning === 'string' ? source.reasoning : '',
    search: typeof source.search === 'string' ? source.search : '',
    vision: typeof source.vision === 'string' ? source.vision : '',
    fast: typeof source.fast === 'string' ? source.fast : ''
  }
}

export function normalizeGuiAiProvider(raw: unknown): AiSettings['provider'] {
  if (raw === 'openai' || raw === 'openrouter' || raw === 'ollama' || raw === 'local') {
    return raw
  }
  if (raw === 'transformers') {
    return 'local'
  }
  return 'openai'
}

export function toRuntimeAiProvider(provider: AiSettings['provider']): string {
  return provider === 'local' ? 'transformers' : provider
}

function objectSection(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' ? (value as Record<string, unknown>) : {}
}

function nonEmptyString(value: unknown): string {
  return typeof value === 'string' && value.trim() ? value.trim() : ''
}

export function buildAiSettings(raw: Settings = {}): AiSettings {
  const rexConfig = readRexConfig()
  const models = objectSection(rexConfig.models)
  const openai = objectSection(rexConfig.openai)
  const openrouter = objectSection(rexConfig.openrouter)
  const ollama = objectSection(rexConfig.ollama)
  const rawProvider =
    typeof models.llm_provider === 'string'
      ? models.llm_provider
      : typeof raw.provider === 'string'
        ? raw.provider
        : null
  const provider = normalizeGuiAiProvider(rawProvider)
  const runtimeModelId = nonEmptyString(models.llm_model)
  const model =
    nonEmptyString(raw.model)
    || nonEmptyString(openai.model)
    || (provider === 'openai' ? runtimeModelId : '')
    || OPENAI_DEFAULT_MODEL
  const openrouterModel =
    nonEmptyString(raw.openrouterModel)
    || nonEmptyString(openrouter.model)
    || (provider === 'openrouter' ? runtimeModelId : '')
    || OPENROUTER_DEFAULT_MODEL
  const customModelId =
    nonEmptyString(raw.customModelId)
    || (provider === 'ollama' || provider === 'local' ? runtimeModelId : '')
  const openaiBaseUrl = Object.prototype.hasOwnProperty.call(raw, 'openaiBaseUrl')
    ? typeof raw.openaiBaseUrl === 'string'
      ? raw.openaiBaseUrl.trim()
      : ''
    : nonEmptyString(openai.base_url)
  const ollamaBaseUrl =
    nonEmptyString(raw.ollamaBaseUrl)
    || nonEmptyString(ollama.base_url)
    || 'http://localhost:11434'
  const openrouterBaseUrl =
    nonEmptyString(raw.openrouterBaseUrl)
    || nonEmptyString(openrouter.base_url)
    || OPENROUTER_DEFAULT_BASE_URL
  const routingSource =
    raw.modelRouting && typeof raw.modelRouting === 'object'
      ? raw.modelRouting
      : rexConfig.model_routing

  const VALID_PERSONALITIES = ['Friendly', 'Professional', 'Minimal']
  const rawPersonality = typeof raw.personality === 'string' ? raw.personality : null
  const personality = rawPersonality && VALID_PERSONALITIES.includes(rawPersonality)
    ? rawPersonality
    : typeof rexConfig.personality === 'string' && VALID_PERSONALITIES.includes(rexConfig.personality as string)
      ? rexConfig.personality as string
      : 'Friendly'

  return {
    model,
    provider,
    customModelId,
    openaiBaseUrl,
    ollamaBaseUrl,
    openrouterModel,
    openrouterBaseUrl,
    temperature:
      typeof raw.temperature === 'number'
        ? raw.temperature
        : typeof models.llm_temperature === 'string'
          ? parseFloat(models.llm_temperature) || 0.7
          : 0.7,
    maxTokens:
      typeof raw.maxTokens === 'number'
        ? raw.maxTokens
        : typeof models.llm_max_tokens === 'number'
          ? models.llm_max_tokens
          : 2048,
    systemPrompt: typeof raw.systemPrompt === 'string' ? raw.systemPrompt : '',
    autonomyMode: resolveAutonomyMode(models.autonomy_mode, raw.autonomyMode),
    budgetPerPlan: typeof raw.budgetPerPlan === 'number' ? raw.budgetPerPlan : 0,
    budgetPerStep: typeof raw.budgetPerStep === 'number' ? raw.budgetPerStep : 0,
    modelRouting: normalizeAiModelRouting(routingSource),
    personality
  }
}


export function buildAiSettingsForSave(raw: Settings = {}): AiSettings {
  const resolved = buildAiSettings(raw)
  const submittedAutonomyMode = normalizeAutonomyMode(raw.autonomyMode)
  return submittedAutonomyMode ? { ...resolved, autonomyMode: submittedAutonomyMode } : resolved
}
