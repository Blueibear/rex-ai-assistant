import type { ModelDiscoveryProvider, ModelDiscoveryResponse } from '../types/ipc'

type FetchLike = typeof fetch

function providerLabel(provider: ModelDiscoveryProvider): string {
  return provider === 'ollama' ? 'Ollama' : 'LM Studio'
}

function validatedBaseUrl(
  provider: ModelDiscoveryProvider,
  endpoint: string
): { ok: true; value: URL } | { ok: false; error: string } {
  const label = providerLabel(provider)
  if (!endpoint.trim()) {
    return { ok: false, error: `${label} model discovery endpoint is not configured` }
  }
  try {
    const url = new URL(endpoint.trim())
    if (url.protocol !== 'http:' && url.protocol !== 'https:') {
      return { ok: false, error: `${label} model discovery endpoint must use HTTP or HTTPS` }
    }
    if (url.search || url.hash) {
      return { ok: false, error: `${label} model discovery endpoint must not include a query or fragment` }
    }
    return { ok: true, value: url }
  } catch {
    return { ok: false, error: `${label} model discovery endpoint must use HTTP or HTTPS` }
  }
}

function discoveryUrl(provider: ModelDiscoveryProvider, base: URL): string {
  const url = new URL(base.toString())
  const suffix = provider === 'ollama' ? '/api/tags' : '/models'
  url.pathname = `${url.pathname.replace(/\/+$/, '')}${suffix}`
  return url.toString()
}

function uniqueNonblank(values: unknown[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const value of values) {
    if (typeof value !== 'string') continue
    const trimmed = value.trim()
    if (!trimmed || seen.has(trimmed)) continue
    seen.add(trimmed)
    result.push(trimmed)
  }
  return result
}

function parseModels(
  provider: ModelDiscoveryProvider,
  payload: unknown
): string[] | null {
  if (!payload || typeof payload !== 'object') return null
  const source = payload as Record<string, unknown>
  const rows = provider === 'ollama' ? source.models : source.data
  if (!Array.isArray(rows)) return null
  const field = provider === 'ollama' ? 'name' : 'id'
  return uniqueNonblank(
    rows.map((row) =>
      row && typeof row === 'object' ? (row as Record<string, unknown>)[field] : null
    )
  )
}

export async function discoverAiModelsAtEndpoint(
  provider: ModelDiscoveryProvider,
  endpoint: string,
  fetchImpl: FetchLike = globalThis.fetch,
  timeoutMs = 5000
): Promise<ModelDiscoveryResponse> {
  const validated = validatedBaseUrl(provider, endpoint)
  if (!validated.ok) return { ok: false, models: [], error: validated.error }

  const label = providerLabel(provider)
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const response = await fetchImpl(discoveryUrl(provider, validated.value), {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal
    })
    if (!response.ok) {
      return {
        ok: false,
        models: [],
        error: `${label} model discovery failed (HTTP ${response.status})`
      }
    }

    let payload: unknown
    try {
      payload = await response.json()
    } catch {
      return { ok: false, models: [], error: `${label} returned an invalid model list` }
    }
    const models = parseModels(provider, payload)
    if (models === null) {
      return { ok: false, models: [], error: `${label} returned an invalid model list` }
    }
    return { ok: true, models }
  } catch {
    return {
      ok: false,
      models: [],
      error: `Unable to reach configured ${label} endpoint`
    }
  } finally {
    clearTimeout(timeout)
  }
}
