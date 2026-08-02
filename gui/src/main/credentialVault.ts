import { spawn } from 'child_process'
import { bridgeSpawnOptions, resolveBridgePath, resolvePythonCommand } from './bridgeResolver'
import type { ElectronSessionIdentity } from './sessionIdentity'

export interface VaultEntryMetadata {
  key: string
  integration: string
  account: string | null
  slot: string
  scope: 'household' | 'user'
  owner: string
  created_at: string
  updated_at: string
}

export interface VaultContext {
  integration: string
  account: string | null
  slot: string
  scope: 'household' | 'user'
}

function callVaultBridge(
  session: ElectronSessionIdentity,
  payload: Record<string, unknown>
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const scriptPath = resolveBridgePath('rex_credential_vault_bridge.py')
    const py = spawn(resolvePythonCommand(), [scriptPath], {
      ...bridgeSpawnOptions(),
      stdio: ['pipe', 'pipe', 'pipe']
    })
    let stdout = ''
    let settled = false

    const rejectOnce = (error: Error): void => {
      if (settled) return
      settled = true
      reject(error)
    }

    py.stdout.on('data', (chunk: Buffer) => { stdout += chunk.toString() })
    py.stderr.on('data', () => { /* never surface raw bridge stderr */ })
    py.on('close', (code) => {
      if (settled) return
      if (code !== 0 && !stdout.trim()) {
        rejectOnce(new Error(`Credential vault bridge exited with code ${code}`))
        return
      }
      try {
        const result = JSON.parse(stdout.trim()) as Record<string, unknown>
        if (result.ok !== true) {
          rejectOnce(new Error((result.error as string | undefined) ?? 'Credential vault operation failed'))
          return
        }
        settled = true
        resolve(result)
      } catch {
        rejectOnce(new Error('Failed to parse credential vault bridge response'))
      }
    })
    py.on('error', () => rejectOnce(new Error('Failed to start credential vault bridge')))
    py.stdin.write(JSON.stringify({ ...payload, request_user_id: session.userId }))
    py.stdin.end()
  })
}

function contextPayload(context: VaultContext): Record<string, unknown> {
  return {
    scope: context.scope,
    integration: context.integration,
    account: context.account,
    slot: context.slot
  }
}

export async function vaultSetSecret(
  session: ElectronSessionIdentity,
  value: string,
  context: VaultContext,
  ref?: string
): Promise<string> {
  const result = await callVaultBridge(session, {
    action: 'set',
    key: ref,
    value,
    ...contextPayload(context)
  })
  if (typeof result.ref !== 'string' || !result.ref) {
    throw new Error('Credential vault did not return an opaque reference')
  }
  return result.ref
}

export async function vaultGetSecret(
  session: ElectronSessionIdentity,
  ref: string,
  context: VaultContext
): Promise<string | null> {
  const result = await callVaultBridge(session, { action: 'get', key: ref, ...contextPayload(context) })
  return (result.value as string | null) ?? null
}

export async function vaultHasSecret(
  session: ElectronSessionIdentity,
  ref: string,
  context: VaultContext
): Promise<boolean> {
  const result = await callVaultBridge(session, { action: 'has', key: ref, ...contextPayload(context) })
  return result.has === true
}

export async function vaultDeleteSecret(
  session: ElectronSessionIdentity,
  ref: string,
  context: VaultContext
): Promise<boolean> {
  const result = await callVaultBridge(session, { action: 'delete', key: ref, ...contextPayload(context) })
  return result.deleted === true
}

export async function vaultListEntries(
  session: ElectronSessionIdentity,
  scope: 'household' | 'user'
): Promise<VaultEntryMetadata[]> {
  const result = await callVaultBridge(session, { action: 'list', scope })
  return (result.entries as VaultEntryMetadata[]) ?? []
}
