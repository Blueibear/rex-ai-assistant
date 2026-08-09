import { useEffect, useRef, useState } from 'react'
import type { EmailAccount, IntegrationInventoryItem, IntegrationsSettings, Settings } from '../../../types/ipc'
import { useToast } from '../../../components/ui/Toast'

export type IntegrationSection = 'email' | 'calendar' | 'sms' | 'homeassistant' | 'phone' | 'openclaw'
export type TestStatus = 'idle' | 'testing' | 'configured' | 'ok' | 'error'

function integrationStatusToTestStatus(state: IntegrationInventoryItem['state']): TestStatus {
  if (['authenticated', 'read_only', 'write_capable', 'write_tested', 'verified'].includes(state)) return 'ok'
  if (state === 'configured' || state === 'reachable') return 'configured'
  if (state === 'degraded' || state === 'unavailable') return 'error'
  return 'idle'
}

function integrationKeyToSection(key: string): IntegrationSection | null {
  if (key === 'email' || key === 'calendar' || key === 'sms' || key === 'homeassistant' || key === 'phone' || key === 'openclaw') {
    return key
  }
  return null
}

const INTEGRATION_SECRET_FIELDS = new Set<keyof IntegrationsSettings>([
  'emailClientSecret', 'calendarClientSecret', 'smsSid', 'smsAuthToken',
  'smsFromNumber', 'haToken', 'phoneSid', 'phoneAuthToken', 'phoneNumber',
  'phoneTransferNumber', 'telegramBotToken', 'openclawToken'
])

function isIntegrationSecretField(field: keyof IntegrationsSettings): boolean {
  return INTEGRATION_SECRET_FIELDS.has(field)
}

export function hasStoredCredential(form: IntegrationsSettings, field: keyof IntegrationsSettings): boolean {
  return form.credentialStatus[field]?.hasCredential === true
}

const INTEGRATION_FIELD_SECTIONS: Partial<Record<keyof IntegrationsSettings, IntegrationSection[]>> = {
  emailProvider: ['email'],
  emailClientId: ['email'],
  emailClientSecret: ['email'],
  emailAccounts: ['email'],
  calendarProvider: ['calendar'],
  calendarClientId: ['calendar'],
  calendarClientSecret: ['calendar'],
  smsSid: ['sms'],
  smsAuthToken: ['sms'],
  smsFromNumber: ['sms'],
  haUrl: ['homeassistant'],
  haToken: ['homeassistant'],
  phoneSid: ['phone'],
  phoneAuthToken: ['phone'],
  phoneNumber: ['phone'],
  openclawGatewayUrl: ['openclaw'],
  openclawToolsEnabled: ['openclaw'],
  openclawVoiceEnabled: ['openclaw'],
  openclawToken: ['openclaw']
}

function sectionsForIntegrationField(field: keyof IntegrationsSettings): IntegrationSection[] {
  return INTEGRATION_FIELD_SECTIONS[field] ?? []
}

function stringSetting(settings: Settings, field: string): string {
  const value = settings[field]
  return typeof value === 'string' ? value : ''
}

function normalizeEmailAccount(account: EmailAccount): EmailAccount {
  return {
    id: account.id,
    backend: account.backend ?? (account as unknown as { provider?: string }).provider ?? 'gmail',
    displayName: account.displayName ?? '',
    clientId: account.clientId ?? '',
    clientSecret: account.clientSecret ?? '',
    host: account.host ?? '',
    port: typeof account.port === 'number' ? account.port : 993,
    username: account.username ?? '',
    password: account.password ?? '',
    credentialRef: account.credentialRef,
    hasCredential: account.hasCredential === true,
    lastSynced: account.lastSynced
  }
}

function normalizeEmailAccounts(raw: unknown): EmailAccount[] {
  if (!Array.isArray(raw)) return []
  return (raw as EmailAccount[])
    .filter((account) => typeof account === 'object' && account !== null && typeof account.id === 'string')
    .map(normalizeEmailAccount)
}

function integrationsFormFromSettings(settings: Settings): IntegrationsSettings {
  const credentialStatus = settings.credentialStatus && typeof settings.credentialStatus === 'object'
    ? settings.credentialStatus as IntegrationsSettings['credentialStatus']
    : {}
  return {
    emailProvider: settings.emailProvider === 'outlook' ? 'outlook' : 'gmail',
    emailClientId: stringSetting(settings, 'emailClientId'),
    emailClientSecret: stringSetting(settings, 'emailClientSecret'), // pragma: allowlist secret
    emailAccounts: normalizeEmailAccounts(settings.emailAccounts),
    calendarProvider: settings.calendarProvider === 'outlook' ? 'outlook' : 'gmail',
    calendarClientId: stringSetting(settings, 'calendarClientId'),
    calendarClientSecret: stringSetting(settings, 'calendarClientSecret'), // pragma: allowlist secret
    smsSid: stringSetting(settings, 'smsSid'),
    smsAuthToken: stringSetting(settings, 'smsAuthToken'),
    smsFromNumber: stringSetting(settings, 'smsFromNumber'),
    haUrl: stringSetting(settings, 'haUrl'),
    haToken: stringSetting(settings, 'haToken'),
    phoneSid: stringSetting(settings, 'phoneSid'),
    phoneAuthToken: stringSetting(settings, 'phoneAuthToken'),
    phoneNumber: stringSetting(settings, 'phoneNumber'),
    phoneTransferNumber: stringSetting(settings, 'phoneTransferNumber'),
    voicemailNotificationsEnabled: settings.voicemailNotificationsEnabled === true,
    contactsFilePath: stringSetting(settings, 'contactsFilePath'),
    telegramBotToken: stringSetting(settings, 'telegramBotToken'),
    telegramChatId: stringSetting(settings, 'telegramChatId'),
    openclawGatewayUrl: stringSetting(settings, 'openclawGatewayUrl'),
    openclawToolsEnabled: settings.openclawToolsEnabled === true,
    openclawVoiceEnabled: settings.openclawVoiceEnabled === true,
    openclawToken: stringSetting(settings, 'openclawToken'),
    credentialStatus
  }
}


export function useIntegrationsSettingsController() {
  const addToast = useToast()
  const [form, setForm] = useState<IntegrationsSettings>({
    emailProvider: 'gmail',
    emailClientId: '',
    emailClientSecret: '',
    emailAccounts: [],
    calendarProvider: 'gmail',
    calendarClientId: '',
    calendarClientSecret: '',
    smsSid: '',
    smsAuthToken: '',
    smsFromNumber: '',
    haUrl: '',
    haToken: '',
    phoneSid: '',
    phoneAuthToken: '',
    phoneNumber: '',
    phoneTransferNumber: '',
    voicemailNotificationsEnabled: false,
    contactsFilePath: '',
    telegramBotToken: '',
    telegramChatId: '',
    openclawGatewayUrl: '',
    openclawToolsEnabled: false,
    openclawVoiceEnabled: false,
    openclawToken: '',
    credentialStatus: {}
  })
  const [loading, setLoading] = useState(true)
  const [savedField, setSavedField] = useState<keyof IntegrationsSettings | null>(null)
  const [testStatus, setTestStatus] = useState<Record<IntegrationSection, TestStatus>>({
    email: 'idle',
    calendar: 'idle',
    sms: 'idle',
    homeassistant: 'idle',
    phone: 'idle',
    openclaw: 'idle'
  })
  const [testErrors, setTestErrors] = useState<Partial<Record<IntegrationSection, string>>>({})
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [accountTestStatus, setAccountTestStatus] = useState<Record<string, TestStatus>>({})
  const accountTestTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  function handleTestEmailAccount(id: string): void {
    setAccountTestStatus((s) => ({ ...s, [id]: 'testing' }))
    window.rex
      .testEmailAccount(id)
      .then((res) => {
        const nextStatus = res.state
          ? integrationStatusToTestStatus(res.state)
          : res.ok ? 'ok' : 'error'
        setAccountTestStatus((s) => ({ ...s, [id]: nextStatus }))
      })
      .catch(() => {
        setAccountTestStatus((s) => ({ ...s, [id]: 'error' }))
      })
      .finally(() => {
        if (accountTestTimers.current[id]) clearTimeout(accountTestTimers.current[id])
        accountTestTimers.current[id] = setTimeout(
          () => setAccountTestStatus((s) => ({ ...s, [id]: 'idle' })),
          5000
        )
      })
  }

  function applyIntegrationStatuses(inventory: IntegrationInventoryItem[]): void {
    setTestStatus((current) => {
      const next = { ...current }
      for (const item of inventory) {
        const section = integrationKeyToSection(item.key)
        if (section) next[section] = integrationStatusToTestStatus(item.state)
      }
      return next
    })
    setTestErrors((current) => {
      const next = { ...current }
      for (const item of inventory) {
        const section = integrationKeyToSection(item.key)
        if (!section) continue
        if ((item.state === 'degraded' || item.state === 'unavailable') && item.error) {
          next[section] = item.error
        } else if (item.state !== 'degraded' && item.state !== 'unavailable') {
          delete next[section]
        }
      }
      return next
    })
  }

  function resetStatusForField(field: keyof IntegrationsSettings): void {
    const sections = sectionsForIntegrationField(field)
    if (sections.length === 0) return
    setTestStatus((current) => {
      const next = { ...current }
      for (const section of sections) next[section] = 'idle'
      return next
    })
    setTestErrors((current) => {
      const next = { ...current }
      for (const section of sections) delete next[section]
      return next
    })
  }

  useEffect(() => {
    window.rex
      .getSettings('integrations')
      .then(async (settings: Settings) => {
        setForm(integrationsFormFromSettings(settings))

        const inventory = await window.rex.getIntegrations().catch(() => null)
        if (inventory?.ok) {
          applyIntegrationStatuses(inventory.integrations)
        }
      })
      .catch(() => {
        addToast('Failed to load integrations settings', 'error')
      })
      .finally(() => setLoading(false))
  }, [addToast])

  function showSaved(field: keyof IntegrationsSettings): void {
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
    setSavedField(field)
    savedTimerRef.current = setTimeout(() => setSavedField(null), 2000)
  }

  function saveField(field: keyof IntegrationsSettings, updatedForm: IntegrationsSettings): void {
    window.rex
      .setSettings('integrations', updatedForm as unknown as Settings)
      .then((result) => {
        if (result.ok) {
          showSaved(field)
          if (isIntegrationSecretField(field)) {
            setForm((current) => ({
              ...current,
              [field]: '',
              credentialStatus: {
                ...current.credentialStatus,
                [field]: { ref: current.credentialStatus[field]?.ref ?? '', hasCredential: true }
              }
            }))
          }
        } else {
          addToast(result.error ?? 'Failed to save integrations settings', 'error')
        }
      })
      .catch(() => {
        addToast('Failed to save integrations settings', 'error')
      })
  }

  function handleFieldChange<K extends keyof IntegrationsSettings>(
    field: K,
    value: IntegrationsSettings[K]
  ): void {
    const updated = { ...form, [field]: value }
    setForm(updated)
    resetStatusForField(field)
    saveField(field, updated)
  }

  function handleTest(section: IntegrationSection): void {
    setTestStatus((s) => ({ ...s, [section]: 'testing' }))
    setTestErrors((s) => {
      const next = { ...s }
      delete next[section]
      return next
    })
    const testRequest =
      section === 'homeassistant'
        ? window.rex.testHomeAssistant(form.haUrl, form.haToken)
        : window.rex.testIntegration(section)

    testRequest
      .then((res) => {
        const evidenceState = 'state' in res
          ? res.state as IntegrationInventoryItem['state'] | undefined
          : undefined
        const nextStatus = evidenceState
          ? integrationStatusToTestStatus(evidenceState)
          : res.ok ? 'ok' : 'error'
        setTestStatus((s) => ({ ...s, [section]: nextStatus }))
        setTestErrors((s) => {
          const next = { ...s }
          if ((!res.ok || evidenceState === 'configured') && res.error) {
            next[section] = res.error
          } else {
            delete next[section]
          }
          return next
        })
      })
      .catch((err) => {
        setTestStatus((s) => ({ ...s, [section]: 'error' }))
        setTestErrors((s) => ({
          ...s,
          [section]: err instanceof Error ? err.message : 'Connection test failed'
        }))
      })
  }

  function handleAddEmailAccount(): void {
    const newAccount: EmailAccount = {
      id: `${Date.now()}`,
      backend: 'gmail',
      displayName: '',
      clientId: '',
      clientSecret: '',
      host: '',
      port: 993,
      username: '',
      password: ''
    }
    const updated = { ...form, emailAccounts: [...form.emailAccounts, newAccount] }
    setForm(updated)
    resetStatusForField('emailAccounts')
    window.rex
      .setSettings('integrations', updated as unknown as Settings)
      .then((result) => {
        if (!result.ok) {
          setForm(form)
          addToast(result.error ?? 'Failed to save email account', 'error')
        }
      })
      .catch(() => {
        setForm(form)
        addToast('Failed to save email account', 'error')
      })
  }

  function updateEmailAccountDraft(id: string, patch: Partial<EmailAccount>): void {
    setForm((current) => ({
      ...current,
      emailAccounts: current.emailAccounts.map((account) =>
        account.id === id ? { ...account, ...patch } : account
      )
    }))
    resetStatusForField('emailAccounts')
  }

  function handleUpdateEmailAccount(id: string, patch: Partial<EmailAccount>): void {
    const updated = {
      ...form,
      emailAccounts: form.emailAccounts.map((a) => (a.id === id ? { ...a, ...patch } : a))
    }
    setForm(updated)
    resetStatusForField('emailAccounts')
    window.rex
      .removeEmailAccount(id, true)
      .then((result) => {
        if (!result.ok) {
          setForm(form)
          addToast(result.error ?? 'Failed to save email account', 'error')
        } else if (patch.password || patch.clientSecret) {
          setForm((current) => ({
            ...current,
            emailAccounts: current.emailAccounts.map((account) =>
              account.id === id
                ? { ...account, password: '', clientSecret: '', hasCredential: true }
                : account
            )
          }))
        }
      })
      .catch(() => {
        setForm(form)
        addToast('Failed to save email account', 'error')
      })
  }

  function handleRemoveEmailAccount(id: string): void {
    if (!window.confirm('Remove this email account and delete its stored credential?')) return
    const updated = {
      ...form,
      emailAccounts: form.emailAccounts.filter((a) => a.id !== id)
    }
    setForm(updated)
    resetStatusForField('emailAccounts')
    window.rex
      .setSettings('integrations', updated as unknown as Settings)
      .then((result) => {
        if (!result.ok) {
          setForm(form)
          addToast(result.error ?? 'Failed to remove email account', 'error')
        }
      })
      .catch(() => {
        setForm(form)
        addToast('Failed to remove email account', 'error')
      })
  }

  const inputClass =
    'w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent'

  return {
    loading,
    hasStoredCredential,
    form,
    setForm,
    savedField,
    testStatus,
    setTestStatus,
    testErrors,
    accountTestStatus,
    handleTestEmailAccount,
    handleFieldChange,
    handleTest,
    handleAddEmailAccount,
    updateEmailAccountDraft,
    handleUpdateEmailAccount,
    handleRemoveEmailAccount,
    inputClass
  }
}
