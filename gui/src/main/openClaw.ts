import { readRexConfigStrict } from './configStore'
import { getVaultReference } from './credentialReferences'
import { vaultGetSecret, type VaultContext } from './credentialVault'
import type { IntegrationConnectionStatus } from '../types/ipc'
import type { ElectronSessionIdentity } from './sessionIdentity'

export const OPENCLAW_VAULT_CONTEXT: VaultContext = {
  scope: 'household',
  integration: 'openclaw_gateway',
  account: null,
  slot: 'token'
}

export interface OpenClawConnectionResult {
  ok: boolean
  state: IntegrationConnectionStatus
  error?: string
}

export interface OpenClawConfiguration {
  gatewayUrl: string
  useTools: boolean
  useVoiceBackend: boolean
  tokenRef: string | null
  token: string
}

export async function readOpenClawConfiguration(
  session: ElectronSessionIdentity
): Promise<OpenClawConfiguration> {
  const config = readRexConfigStrict()
  const raw = config.openclaw && typeof config.openclaw === 'object'
    ? config.openclaw as Record<string, unknown>
    : {}
  const gatewayUrl = typeof raw.gateway_url === 'string' ? raw.gateway_url.trim() : ''
  const record = getVaultReference(
    config,
    'OPENCLAW_GATEWAY_TOKEN',
    OPENCLAW_VAULT_CONTEXT,
    session.userId
  )
  const token = record
    ? (await vaultGetSecret(session, record.ref, OPENCLAW_VAULT_CONTEXT)) ?? ''
    : ''
  return {
    gatewayUrl,
    useTools: raw.use_tools === true,
    useVoiceBackend: raw.use_voice_backend === true,
    tokenRef: record?.ref ?? null,
    token
  }
}

export async function testOpenClawConnection(
  session: ElectronSessionIdentity,
  fetchImpl: typeof fetch = fetch
): Promise<OpenClawConnectionResult> {
  const config = await readOpenClawConfiguration(session)
  if (!config.gatewayUrl || !config.token) {
    return {
      ok: false,
      state: 'unconfigured',
      error: 'OpenClaw gateway URL and token are required.'
    }
  }

  let healthUrl: string
  try {
    healthUrl = new URL('/healthz', `${config.gatewayUrl.replace(/\/+$/, '')}/`).toString()
  } catch {
    return { ok: false, state: 'degraded', error: 'OpenClaw gateway URL is invalid.' }
  }

  try {
    const response = await fetchImpl(healthUrl, {
      method: 'GET',
      headers: { Authorization: `Bearer ${config.token}` },
      signal: AbortSignal.timeout(5000)
    })
    if (!response.ok) {
      return {
        ok: false,
        state: 'degraded',
        error: `OpenClaw gateway health check returned HTTP ${response.status}.`
      }
    }
    return { ok: true, state: 'reachable' }
  } catch {
    return { ok: false, state: 'degraded', error: 'OpenClaw gateway health check failed.' }
  }
}