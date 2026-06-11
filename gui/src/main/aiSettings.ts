import type { AiSettings, Settings } from '../types/ipc'
import { readRexConfig } from './configStore'

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
  if (raw === 'openai' || raw === 'ollama' || raw === 'local') {
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

export function buildAiSettings(raw: Settings = {}): AiSettings {
  const rexConfig = readRexConfig()
  const models = rexConfig.models && typeof rexConfig.models === 'object'
    ? (rexConfig.models as Record<string, unknown>)
    : {}
  const ollama = rexConfig.ollama && typeof rexConfig.ollama === 'object'
    ? (rexConfig.ollama as Record<string, unknown>)
    : {}
  const rawModel = typeof raw.model === 'string' ? raw.model : null
  const model = rawModel === 'gpt-4o' || rawModel === 'gpt-4-turbo' || rawModel === 'claude-opus-4' || rawModel === 'claude-sonnet-4' || rawModel === 'gemini-1.5-pro'
    ? rawModel
    : 'claude-sonnet-4'
  const routingSource =
    raw.modelRouting && typeof raw.modelRouting === 'object'
      ? raw.modelRouting
      : rexConfig.model_routing
  const rawProvider =
    typeof models.llm_provider === 'string'
      ? models.llm_provider
      : typeof raw.provider === 'string'
        ? raw.provider
        : null
  const provider = normalizeGuiAiProvider(rawProvider)
  const rawCustomModelId = typeof raw.customModelId === 'string' ? raw.customModelId : ''
  const runtimeModelId = typeof models.llm_model === 'string' ? models.llm_model : ''
  const customModelId =
    rawCustomModelId || (provider !== 'openai' ? runtimeModelId : '')
  const ollamaBaseUrl =
    typeof raw.ollamaBaseUrl === 'string' && raw.ollamaBaseUrl
      ? raw.ollamaBaseUrl
      : typeof ollama.base_url === 'string'
        ? ollama.base_url
        : 'http://localhost:11434'

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
    ollamaBaseUrl,
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
    autonomyMode:
      raw.autonomyMode === 'supervised' || raw.autonomyMode === 'full-auto'
        ? raw.autonomyMode
        : 'manual',
    budgetPerPlan: typeof raw.budgetPerPlan === 'number' ? raw.budgetPerPlan : 0,
    budgetPerStep: typeof raw.budgetPerStep === 'number' ? raw.budgetPerStep : 0,
    modelRouting: normalizeAiModelRouting(routingSource),
    personality
  }
}
