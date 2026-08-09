import React from 'react'
import type { EmailAccount } from '../../../types/ipc'
import { PasswordInput } from '../shared'
import type { TestStatus } from './useIntegrationsSettingsController'

interface EmailAccountsListProps {
  accounts: EmailAccount[]
  accountTestStatus: Record<string, TestStatus>
  inputClass: string
  onAdd: () => void
  onTest: (id: string) => void
  onUpdateDraft: (id: string, patch: Partial<EmailAccount>) => void
  onUpdate: (id: string, patch: Partial<EmailAccount>) => void
  onRemove: (id: string) => void
}

function AccountEvidence({ account, status }: { account: EmailAccount; status: TestStatus }): React.ReactElement {
  return (
    <>
      {status === 'ok' && <span className="text-xs font-medium text-success">Authenticated</span>}
      {status === 'configured' && <span className="text-xs font-medium text-warning">Configured only</span>}
      {status === 'error' && <span className="text-xs font-medium text-danger">Failed</span>}
      {account.lastSynced && (
        <span className="ml-auto text-xs text-text-secondary">
          Synced {new Date(account.lastSynced).toLocaleString()}
        </span>
      )}
    </>
  )
}

function AccountCredentialFields({
  account,
  inputClass,
  onUpdateDraft,
  onUpdate
}: {
  account: EmailAccount
  inputClass: string
  onUpdateDraft: (id: string, patch: Partial<EmailAccount>) => void
  onUpdate: (id: string, patch: Partial<EmailAccount>) => void
}): React.ReactElement {
  if (account.backend === 'imap') {
    return (
      <div className="space-y-2 mb-2">
        <input type="text" value={account.host} placeholder="IMAP host (e.g. imap.gmail.com)"
          onChange={(e) => onUpdate(account.id, { host: e.target.value })} className={inputClass} />
        <input type="number" value={account.port} placeholder="Port (993)"
          onChange={(e) => onUpdate(account.id, { port: parseInt(e.target.value, 10) || 993 })} className={inputClass} />
        <input type="text" value={account.username} placeholder="Username / email address"
          onChange={(e) => onUpdate(account.id, { username: e.target.value })} className={inputClass} />
        <PasswordInput
          id={`imap-pass-${account.id}`}
          value={account.password}
          placeholder={account.hasCredential ? 'Stored credential (enter to replace)' : 'Password or app password'}
          onChange={(value) => onUpdateDraft(account.id, { password: value })}
          onBlur={() => { if (account.password) onUpdate(account.id, { password: account.password }) }}
        />
      </div>
    )
  }
  return (
    <div className="space-y-2 mb-2">
      <input type="text" value={account.clientId} placeholder="OAuth Client ID"
        onChange={(e) => onUpdate(account.id, { clientId: e.target.value })} className={inputClass} />
      <PasswordInput
        id={`email-secret-${account.id}`}
        value={account.clientSecret}
        placeholder={account.hasCredential ? 'Stored credential (enter to replace)' : 'OAuth Client Secret'}
        onChange={(value) => onUpdateDraft(account.id, { clientSecret: value })}
        onBlur={() => { if (account.clientSecret) onUpdate(account.id, { clientSecret: account.clientSecret }) }}
      />
    </div>
  )
}

function EmailAccountCard({
  account,
  status,
  inputClass,
  onTest,
  onUpdateDraft,
  onUpdate,
  onRemove
}: {
  account: EmailAccount
  status: TestStatus
  inputClass: string
  onTest: (id: string) => void
  onUpdateDraft: (id: string, patch: Partial<EmailAccount>) => void
  onUpdate: (id: string, patch: Partial<EmailAccount>) => void
  onRemove: (id: string) => void
}): React.ReactElement {
  return (
    <div className="rounded-xl border border-border bg-surface-raised p-4">
      <div className="mb-3 flex items-center justify-between gap-2">
        <input type="text" value={account.displayName} placeholder="Account label (e.g. Work Gmail)"
          onChange={(e) => onUpdate(account.id, { displayName: e.target.value })}
          className="flex-1 bg-bg border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent" />
        <button type="button" onClick={() => onRemove(account.id)}
          className="rounded-lg border border-danger/30 px-2.5 py-1.5 text-xs font-medium text-danger transition-colors hover:bg-danger/10 focus:outline-none">
          Remove
        </button>
      </div>
      <div className="mb-2">
        <select value={account.backend}
          onChange={(e) => onUpdate(account.id, { backend: e.target.value as EmailAccount['backend'] })}
          className="w-full bg-bg border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent">
          <option value="gmail">Gmail OAuth</option>
          <option value="outlook">Outlook OAuth</option>
          <option value="imap">IMAP</option>
        </select>
      </div>
      <AccountCredentialFields account={account} inputClass={inputClass} onUpdateDraft={onUpdateDraft} onUpdate={onUpdate} />
      <div className="flex items-center gap-3">
        <button type="button" onClick={() => onTest(account.id)} disabled={status === 'testing'}
          className="flex items-center gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-border focus:outline-none focus:ring-2 focus:ring-accent disabled:opacity-50">
          {status === 'testing' ? 'Checking?' : 'Check Configuration'}
        </button>
        <AccountEvidence account={account} status={status} />
      </div>
    </div>
  )
}

export function EmailAccountsList(props: EmailAccountsListProps): React.ReactElement {
  const { accounts, accountTestStatus, inputClass, onAdd, onTest, onUpdateDraft, onUpdate, onRemove } = props
  return (
    <div className="mt-5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-text-primary">Email Accounts</span>
        <button type="button" onClick={onAdd}
          className="flex items-center gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-border focus:outline-none focus:ring-2 focus:ring-accent">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
          </svg>
          Add Account
        </button>
      </div>
      {accounts.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-raised/40 px-4 py-4 text-sm text-text-secondary">
          No additional accounts. Click "Add Account" to connect another inbox.
        </div>
      ) : (
        <div className="space-y-3">
          {accounts.map((account) => (
            <EmailAccountCard key={account.id} account={account} status={accountTestStatus[account.id] ?? 'idle'}
              inputClass={inputClass} onTest={onTest} onUpdateDraft={onUpdateDraft} onUpdate={onUpdate} onRemove={onRemove} />
          ))}
        </div>
      )}
    </div>
  )
}
