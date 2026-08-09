import React from 'react'
import { NavLink } from 'react-router-dom'
import type { IntegrationsSettings } from '../../../types/ipc'
import { PageLoadingFallback } from '../../../components/ui/PageLoadingFallback'
import { PasswordInput, SavedIndicator } from '../shared'
import { ConnectionBadge, TestConnectionButton } from './IntegrationControls'
import { EmailAccountsList } from './EmailAccountsList'
import { OpenClawIntegrationSection } from './OpenClawIntegrationSection'
import { useIntegrationsSettingsController } from './useIntegrationsSettingsController'

export function IntegrationsSettingsSection(): React.ReactElement {
  const {
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
  } = useIntegrationsSettingsController()

  if (loading) {
    return <PageLoadingFallback lines={6} />
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">Integrations</h2>

      {/* Email section */}
      <section className="mb-7">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
              <polyline points="22,6 12,13 2,6" />
            </svg>
            Email
          </h3>
          <div className="flex items-center gap-2">
            <a
              href="https://console.cloud.google.com/apis/credentials"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline"
            >
              Google Console →
            </a>
            <ConnectionBadge
              status={testStatus.email}
              hasCredentials={
                (form.emailClientId.trim() !== '' && hasStoredCredential(form, 'emailClientSecret')) ||
                form.emailAccounts.some((account) => account.hasCredential === true)
              }
            />
          </div>
        </div>

        <p className="text-xs text-text-secondary mb-4">
          Optional. Lets Rex read and send mail. Create OAuth credentials in the Google Console linked above.
        </p>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="emailProvider" className="text-sm font-medium text-text-primary">Provider</label>
            <SavedIndicator visible={savedField === 'emailProvider'} />
          </div>
          <select
            id="emailProvider"
            value={form.emailProvider}
            onChange={(e) => handleFieldChange('emailProvider', e.target.value as IntegrationsSettings['emailProvider'])}
            className={inputClass}
          >
            <option value="gmail">Gmail</option>
            <option value="outlook">Outlook</option>
          </select>
          {form.emailProvider === 'outlook' && (
            <p className="mt-2 text-xs text-text-secondary">
              Outlook mailbox sync is not live yet. These fields store app credentials only.
            </p>
          )}
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <label htmlFor="emailClientId" className="text-sm font-medium text-text-primary">OAuth Client ID</label>
              <span
                title="Create OAuth 2.0 credentials in Google Cloud Console → APIs & Services → Credentials. Enable the Gmail API first."
                className="flex-shrink-0 w-4 h-4 rounded-full bg-surface-raised text-text-muted flex items-center justify-center text-[10px] font-bold cursor-help select-none"
                aria-label="OAuth Client ID help"
              >
                ?
              </span>
            </div>
            <SavedIndicator visible={savedField === 'emailClientId'} />
          </div>
          <input
            id="emailClientId"
            type="text"
            value={form.emailClientId}
            placeholder="Enter client ID"
            onChange={(e) => setForm((f) => ({ ...f, emailClientId: e.target.value }))}
            onBlur={(e) => handleFieldChange('emailClientId', e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="mb-2">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="emailClientSecret" className="text-sm font-medium text-text-primary">OAuth Client Secret</label>
            <SavedIndicator visible={savedField === 'emailClientSecret'} />
          </div>
          <PasswordInput
            id="emailClientSecret"
            value={form.emailClientSecret}
            placeholder="Enter client secret (example)" // pragma: allowlist secret
            onChange={(v) => setForm((f) => ({ ...f, emailClientSecret: v }))}
            onBlur={() => handleFieldChange('emailClientSecret', form.emailClientSecret)}
          />
        </div>

        <TestConnectionButton status={testStatus.email} error={testErrors.email} onTest={() => handleTest('email')} />

        <EmailAccountsList
          accounts={form.emailAccounts}
          accountTestStatus={accountTestStatus}
          inputClass={inputClass}
          onAdd={handleAddEmailAccount}
          onTest={handleTestEmailAccount}
          onUpdateDraft={updateEmailAccountDraft}
          onUpdate={handleUpdateEmailAccount}
          onRemove={handleRemoveEmailAccount}
        />
      </section>

      <div className="border-t border-border mb-7" />

      {/* Calendar section */}
      <section className="mb-7">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
              <line x1="16" y1="2" x2="16" y2="6" />
              <line x1="8" y1="2" x2="8" y2="6" />
              <line x1="3" y1="10" x2="21" y2="10" />
            </svg>
            Calendar
          </h3>
          <div className="flex items-center gap-2">
            <a
              href="https://console.cloud.google.com/apis/credentials"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline"
            >
              Google Console →
            </a>
            <ConnectionBadge
              status={testStatus.calendar}
              hasCredentials={form.calendarClientId.trim() !== '' && hasStoredCredential(form, 'calendarClientSecret')}
            />
          </div>
        </div>

        <p className="text-xs text-text-secondary mb-4">
          Optional. Lets Rex read and create events on your calendar. Use the Google Console link above for OAuth credentials.
        </p>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="calendarProvider" className="text-sm font-medium text-text-primary">Provider</label>
            <SavedIndicator visible={savedField === 'calendarProvider'} />
          </div>
          <select
            id="calendarProvider"
            value={form.calendarProvider}
            onChange={(e) => handleFieldChange('calendarProvider', e.target.value as IntegrationsSettings['calendarProvider'])}
            className={inputClass}
          >
            <option value="gmail">Google Calendar</option>
            <option value="outlook">Outlook Calendar</option>
          </select>
          {form.calendarProvider === 'outlook' && (
            <p className="mt-2 text-xs text-text-secondary">
              Outlook calendar sync is not live yet. Rex cannot read or write Outlook events from these fields.
            </p>
          )}
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <label htmlFor="calendarClientId" className="text-sm font-medium text-text-primary">OAuth Client ID</label>
              <span
                title="Create OAuth 2.0 credentials in Google Cloud Console → APIs & Services → Credentials. Enable the Google Calendar API first."
                className="flex-shrink-0 w-4 h-4 rounded-full bg-surface-raised text-text-muted flex items-center justify-center text-[10px] font-bold cursor-help select-none"
                aria-label="OAuth Client ID help"
              >
                ?
              </span>
            </div>
            <SavedIndicator visible={savedField === 'calendarClientId'} />
          </div>
          <input
            id="calendarClientId"
            type="text"
            value={form.calendarClientId}
            placeholder="Enter client ID"
            onChange={(e) => setForm((f) => ({ ...f, calendarClientId: e.target.value }))}
            onBlur={(e) => handleFieldChange('calendarClientId', e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="mb-2">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="calendarClientSecret" className="text-sm font-medium text-text-primary">OAuth Client Secret</label>
            <SavedIndicator visible={savedField === 'calendarClientSecret'} />
          </div>
          <PasswordInput
            id="calendarClientSecret"
            value={form.calendarClientSecret}
            placeholder="Enter client secret (example)" // pragma: allowlist secret
            onChange={(v) => setForm((f) => ({ ...f, calendarClientSecret: v }))}
            onBlur={() => handleFieldChange('calendarClientSecret', form.calendarClientSecret)}
          />
        </div>

        <TestConnectionButton status={testStatus.calendar} error={testErrors.calendar} onTest={() => handleTest('calendar')} />
      </section>

      <div className="border-t border-border mb-7" />

      {/* SMS section */}
      <section className="mb-7">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            SMS (Twilio)
          </h3>
          <div className="flex items-center gap-2">
            <a
              href="https://console.twilio.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline"
            >
              Twilio Console →
            </a>
            <ConnectionBadge
              status={testStatus.sms}
              hasCredentials={hasStoredCredential(form, 'smsSid') && hasStoredCredential(form, 'smsAuthToken') && hasStoredCredential(form, 'smsFromNumber')}
            />
          </div>
        </div>

        <p className="text-xs text-text-secondary mb-4">
          Optional. Lets Rex send text messages through Twilio. Grab the SID, token, and a sender number from the Twilio Console link above.
        </p>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <label htmlFor="smsSid" className="text-sm font-medium text-text-primary">Account SID</label>
              <span
                title="Find your Account SID in the Twilio Console dashboard under 'Account Info'."
                className="flex-shrink-0 w-4 h-4 rounded-full bg-surface-raised text-text-muted flex items-center justify-center text-[10px] font-bold cursor-help select-none"
                aria-label="Where to find Account SID"
              >
                ?
              </span>
            </div>
            <SavedIndicator visible={savedField === 'smsSid'} />
          </div>
          <input
            id="smsSid"
            type="text"
            value={form.smsSid}
            placeholder="Example Account SID" // pragma: allowlist secret
            onChange={(e) => setForm((f) => ({ ...f, smsSid: e.target.value }))}
            onBlur={(e) => handleFieldChange('smsSid', e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="smsAuthToken" className="text-sm font-medium text-text-primary">Auth Token</label>
            <SavedIndicator visible={savedField === 'smsAuthToken'} />
          </div>
          <PasswordInput
            id="smsAuthToken"
            value={form.smsAuthToken}
            placeholder="Enter auth token (example)" // pragma: allowlist secret
            onChange={(v) => setForm((f) => ({ ...f, smsAuthToken: v }))}
            onBlur={() => handleFieldChange('smsAuthToken', form.smsAuthToken)}
          />
        </div>

        <div className="mb-2">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="smsFromNumber" className="text-sm font-medium text-text-primary">From Phone Number</label>
            <SavedIndicator visible={savedField === 'smsFromNumber'} />
          </div>
          <input
            id="smsFromNumber"
            type="text"
            value={form.smsFromNumber}
            placeholder="+15551234567"
            onChange={(e) => setForm((f) => ({ ...f, smsFromNumber: e.target.value }))}
            onBlur={(e) => handleFieldChange('smsFromNumber', e.target.value)}
            className={inputClass}
          />
        </div>

        <TestConnectionButton status={testStatus.sms} error={testErrors.sms} onTest={() => handleTest('sms')} />
      </section>

      <div className="border-t border-border mb-7" />

      {/* Home Assistant section */}
      <section className="mb-2">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
              <polyline points="9 22 9 12 15 12 15 22" />
            </svg>
            Home Assistant
          </h3>
          <div className="flex items-center gap-2">
            <NavLink
              to="/settings/home-assistant"
              className="text-xs text-accent hover:underline"
            >
              Full setup →
            </NavLink>
            <ConnectionBadge
              status={testStatus.homeassistant}
              hasCredentials={form.haUrl.trim() !== '' && hasStoredCredential(form, 'haToken')}
            />
          </div>
        </div>

        <p className="text-xs text-text-secondary mb-4">
          Optional. Lets Rex control your smart home. Create a long-lived access token from your Home Assistant user profile page.
        </p>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="haUrl" className="text-sm font-medium text-text-primary">Base URL</label>
            <SavedIndicator visible={savedField === 'haUrl'} />
          </div>
          <input
            id="haUrl"
            type="text"
            value={form.haUrl}
            placeholder="http://homeassistant.local:8123"
            onChange={(e) => {
              setForm((f) => ({ ...f, haUrl: e.target.value }))
              setTestStatus((s) => ({ ...s, homeassistant: 'idle' }))
            }}
            onBlur={(e) => handleFieldChange('haUrl', e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="mb-2">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="haToken" className="text-sm font-medium text-text-primary">Long-Lived Access Token</label>
            <SavedIndicator visible={savedField === 'haToken'} />
          </div>
          <PasswordInput
            id="haToken"
            value={form.haToken}
            placeholder="Enter access token" // pragma: allowlist secret
            onChange={(v) => {
              setForm((f) => ({ ...f, haToken: v }))
              setTestStatus((s) => ({ ...s, homeassistant: 'idle' }))
            }}
            onBlur={() => handleFieldChange('haToken', form.haToken)}
          />
        </div>

        <TestConnectionButton status={testStatus.homeassistant} error={testErrors.homeassistant} onTest={() => handleTest('homeassistant')} />
      </section>

      <div className="border-t border-border mb-7" />

      <OpenClawIntegrationSection
        form={form}
        savedField={savedField}
        status={testStatus.openclaw}
        error={testErrors.openclaw}
        inputClass={inputClass}
        hasStoredCredential={hasStoredCredential}
        setForm={setForm}
        onFieldChange={handleFieldChange}
        onTest={() => handleTest('openclaw')}
      />

      {/* Phone (Twilio) section */}
      <section className="mb-2">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.69 13.6 19.79 19.79 0 0 1 1.61 5.08 2 2 0 0 1 3.58 3h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L7.91 10.09a16 16 0 0 0 6 6l.91-.91a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z" />
            </svg>
            Phone (Twilio)
          </h3>
          <div className="flex items-center gap-2">
            <a
              href="https://console.twilio.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline"
            >
              Twilio Console →
            </a>
            <ConnectionBadge
              status={testStatus.phone}
              hasCredentials={hasStoredCredential(form, 'phoneSid') && hasStoredCredential(form, 'phoneAuthToken') && hasStoredCredential(form, 'phoneNumber')}
            />
          </div>
        </div>

        <p className="text-xs text-text-secondary mb-4">
          Optional. Lets Rex place voice calls through Twilio. Use the same Twilio Console link above to copy your credentials.
        </p>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <label htmlFor="phoneSid" className="text-sm font-medium text-text-primary">Account SID</label>
              <span
                title="Find your Account SID in the Twilio Console dashboard under 'Account Info'."
                className="flex-shrink-0 w-4 h-4 rounded-full bg-surface-raised text-text-muted flex items-center justify-center text-[10px] font-bold cursor-help select-none"
                aria-label="Where to find Account SID"
              >
                ?
              </span>
            </div>
            <SavedIndicator visible={savedField === 'phoneSid'} />
          </div>
          <input
            id="phoneSid"
            type="text"
            value={form.phoneSid}
            placeholder="Example Account SID" // pragma: allowlist secret
            onChange={(e) => setForm((f) => ({ ...f, phoneSid: e.target.value }))}
            onBlur={(e) => handleFieldChange('phoneSid', e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="phoneAuthToken" className="text-sm font-medium text-text-primary">Auth Token</label>
            <SavedIndicator visible={savedField === 'phoneAuthToken'} />
          </div>
          <PasswordInput
            id="phoneAuthToken"
            value={form.phoneAuthToken}
            placeholder="Enter auth token (example)" // pragma: allowlist secret
            onChange={(v) => setForm((f) => ({ ...f, phoneAuthToken: v }))}
            onBlur={() => handleFieldChange('phoneAuthToken', form.phoneAuthToken)}
          />
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="phoneNumber" className="text-sm font-medium text-text-primary">Twilio Phone Number</label>
            <SavedIndicator visible={savedField === 'phoneNumber'} />
          </div>
          <input
            id="phoneNumber"
            type="text"
            value={form.phoneNumber}
            placeholder="+15551234567"
            onChange={(e) => setForm((f) => ({ ...f, phoneNumber: e.target.value }))}
            onBlur={(e) => handleFieldChange('phoneNumber', e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="phoneTransferNumber" className="text-sm font-medium text-text-primary">Transfer-to Number (optional)</label>
            <SavedIndicator visible={savedField === 'phoneTransferNumber'} />
          </div>
          <input
            id="phoneTransferNumber"
            type="text"
            value={form.phoneTransferNumber}
            placeholder="+15559876543"
            onChange={(e) => setForm((f) => ({ ...f, phoneTransferNumber: e.target.value }))}
            onBlur={(e) => handleFieldChange('phoneTransferNumber', e.target.value)}
            className={inputClass}
          />
        </div>

        <div className="mb-4 flex items-center justify-between">
          <label htmlFor="voicemailNotificationsEnabled" className="text-sm font-medium text-text-primary">Voicemail notifications</label>
          <div className="flex items-center gap-2">
            <SavedIndicator visible={savedField === 'voicemailNotificationsEnabled'} />
            <input
              id="voicemailNotificationsEnabled"
              type="checkbox"
              checked={form.voicemailNotificationsEnabled}
              onChange={(e) => handleFieldChange('voicemailNotificationsEnabled', e.target.checked)}
              className="h-4 w-4 rounded border-border text-accent accent-accent cursor-pointer"
            />
          </div>
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label className="text-sm font-medium text-text-primary">Contacts File</label>
            <SavedIndicator visible={savedField === 'contactsFilePath'} />
          </div>
          <div className="flex items-center gap-2">
            <input
              type="text"
              value={form.contactsFilePath}
              readOnly
              placeholder="No file selected"
              className={`${inputClass} flex-1 cursor-default`}
            />
            <button
              type="button"
              onClick={() => {
                window.rex.uploadContactsFile().then((res) => {
                  if (res.ok && res.path) {
                    handleFieldChange('contactsFilePath', res.path)
                  }
                })
              }}
              className="shrink-0 rounded-md border border-border bg-surface px-3 py-1.5 text-xs font-medium text-text-primary hover:bg-hover transition-colors"
            >
              Browse…
            </button>
          </div>
          <p className="mt-1 text-xs text-text-secondary">Accepts .json or .vcf (vCard) contact files for outbound calling.</p>
        </div>

        <TestConnectionButton status={testStatus.phone} error={testErrors.phone} onTest={() => handleTest('phone')} />
      </section>

      <div className="border-t border-border mb-7" />

      {/* Telegram section */}
      <section className="mb-2">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            Telegram
          </h3>
          <ConnectionBadge
            status="idle"
            hasCredentials={hasStoredCredential(form, 'telegramBotToken') && form.telegramChatId.trim() !== ''}
          />
        </div>

        <p className="text-xs text-text-secondary mb-4">
          Optional. Lets Rex send messages to a Telegram chat. Create a bot through @BotFather to get a token, then message it to find your chat ID.
        </p>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <label htmlFor="telegramBotToken" className="text-sm font-medium text-text-primary">Bot Token</label>
              <span
              title="Create a bot via @BotFather on Telegram to get a token. Rex stores it in the Windows credential vault."
                className="flex-shrink-0 w-4 h-4 rounded-full bg-surface-raised text-text-muted flex items-center justify-center text-[10px] font-bold cursor-help select-none"
                aria-label="Where to find Bot Token"
              >
                ?
              </span>
            </div>
            <SavedIndicator visible={savedField === 'telegramBotToken'} />
          </div>
          <PasswordInput
            id="telegramBotToken"
            value={form.telegramBotToken}
            placeholder="Enter access token (example)" // pragma: allowlist secret
            onChange={(v) => setForm((f) => ({ ...f, telegramBotToken: v }))}
            onBlur={() => handleFieldChange('telegramBotToken', form.telegramBotToken)}
          />
        </div>

        <div className="mb-2">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-1.5">
              <label htmlFor="telegramChatId" className="text-sm font-medium text-text-primary">Chat ID</label>
              <span
                title="Your personal chat ID or group chat ID. Send a message to your bot and check the Telegram Bot API getUpdates response."
                className="flex-shrink-0 w-4 h-4 rounded-full bg-surface-raised text-text-muted flex items-center justify-center text-[10px] font-bold cursor-help select-none"
                aria-label="Where to find Chat ID"
              >
                ?
              </span>
            </div>
            <SavedIndicator visible={savedField === 'telegramChatId'} />
          </div>
          <input
            id="telegramChatId"
            type="text"
            value={form.telegramChatId}
            placeholder="-1001234567890"
            onChange={(e) => setForm((f) => ({ ...f, telegramChatId: e.target.value }))}
            onBlur={(e) => handleFieldChange('telegramChatId', e.target.value)}
            className={inputClass}
          />
          <p className="mt-1 text-xs text-text-secondary">
            The token is stored in the Windows credential vault and is never loaded back into this field.
          </p>
        </div>
      </section>
    </div>
  )
}
