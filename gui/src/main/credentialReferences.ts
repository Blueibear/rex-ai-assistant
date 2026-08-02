import type { VaultContext } from './credentialVault'

const CREDENTIAL_REF_PATTERN = /^cred_[A-Za-z0-9_-]{32}$/

export interface VaultReferenceRecord {
  ref: string
  integration: string
  account: string | null
  slot: string
}

function sectionFor(
  config: Record<string, unknown>,
  context: VaultContext,
  userId: string,
  create: boolean
): Record<string, unknown> {
  let refs = config.credential_refs
  if (!refs) {
    if (!create) return {}
    refs = {}
    config.credential_refs = refs
  }
  if (!refs || typeof refs !== 'object' || Array.isArray(refs)) {
    throw new Error('Credential reference registry is invalid')
  }
  const root = refs as Record<string, unknown>
  if (context.scope === 'household') {
    let household = root.household
    if (!household) {
      if (!create) return {}
      household = {}
      root.household = household
    }
    if (typeof household !== 'object' || Array.isArray(household)) {
      throw new Error('Household credential reference registry is invalid')
    }
    return household as Record<string, unknown>
  }

  let users = root.users
  if (!users) {
    if (!create) return {}
    users = {}
    root.users = users
  }
  if (typeof users !== 'object' || Array.isArray(users)) {
    throw new Error('User credential reference registry is invalid')
  }
  const userMap = users as Record<string, unknown>
  let user = userMap[userId]
  if (!user) {
    if (!create) return {}
    user = {}
    userMap[userId] = user
  }
  if (typeof user !== 'object' || Array.isArray(user)) {
    throw new Error('User credential reference registry entry is invalid')
  }
  return user as Record<string, unknown>
}

export function getVaultReference(
  config: Record<string, unknown>,
  logicalName: string,
  context: VaultContext,
  userId: string
): VaultReferenceRecord | null {
  const raw = sectionFor(config, context, userId, false)[logicalName]
  if (raw === undefined) return null
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error(`Credential reference for ${logicalName} is invalid`)
  }
  const record = raw as Record<string, unknown>
  const allowedKeys = new Set(['ref', 'integration', 'account', 'slot'])
  if ('migrated_from' in record) allowedKeys.add('migrated_from')
  if (
    Object.keys(record).some((key) => !allowedKeys.has(key)) ||
    Object.keys(record).length !== allowedKeys.size ||
    typeof record.ref !== 'string' || !CREDENTIAL_REF_PATTERN.test(record.ref) ||
    record.integration !== context.integration ||
    record.account !== context.account ||
    record.slot !== context.slot ||
    ('migrated_from' in record && !['env', 'credentials.json'].includes(String(record.migrated_from)))
  ) {
    throw new Error(`Credential reference context for ${logicalName} is invalid`)
  }
  return {
    ref: record.ref,
    integration: context.integration,
    account: context.account,
    slot: context.slot
  }
}

export function putVaultReference(
  config: Record<string, unknown>,
  logicalName: string,
  ref: string,
  context: VaultContext,
  userId: string
): void {
  if (!CREDENTIAL_REF_PATTERN.test(ref)) {
    throw new Error(`Credential reference for ${logicalName} is invalid`)
  }
  sectionFor(config, context, userId, true)[logicalName] = {
    ref,
    integration: context.integration,
    account: context.account,
    slot: context.slot
  }
}

export function deleteVaultReference(
  config: Record<string, unknown>,
  logicalName: string,
  context: VaultContext,
  userId: string
): void {
  const section = sectionFor(config, context, userId, false)
  const existing = getVaultReference(config, logicalName, context, userId)
  if (existing) delete section[logicalName]
}
