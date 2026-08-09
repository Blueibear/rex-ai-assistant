import React, { useEffect, useRef, useState } from 'react'
import type { Settings, NotificationsSettings } from '../../types/ipc'
import { PageLoadingFallback } from '../../components/ui/PageLoadingFallback'
import { useToast } from '../../components/ui/Toast'
import { Toggle, SavedIndicator } from './shared'

export function NotificationsSettingsSection(): React.ReactElement {
  const addToast = useToast()
  const [form, setForm] = useState<NotificationsSettings>({
    quietHoursEnabled: false,
    quietHoursStart: '22:00',
    quietHoursEnd: '08:00',
    digestModeEnabled: false,
    digestDeliveryTime: '08:00',
    highPriorityThreshold: 'high_and_critical',
    autoEscalationDelay: 30,
    desktopNotificationsEnabled: true,
    soundAlertsEnabled: true
  })
  const [loading, setLoading] = useState(true)
  const [savedField, setSavedField] = useState<keyof NotificationsSettings | null>(null)
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    window.rex
      .getSettings('notifications')
      .then((settings: Settings) => {
        setForm({
          quietHoursEnabled:
            typeof settings.quietHoursEnabled === 'boolean' ? settings.quietHoursEnabled : false,
          quietHoursStart:
            typeof settings.quietHoursStart === 'string' ? settings.quietHoursStart : '22:00',
          quietHoursEnd:
            typeof settings.quietHoursEnd === 'string' ? settings.quietHoursEnd : '08:00',
          digestModeEnabled:
            typeof settings.digestModeEnabled === 'boolean' ? settings.digestModeEnabled : false,
          digestDeliveryTime:
            typeof settings.digestDeliveryTime === 'string' ? settings.digestDeliveryTime : '08:00',
          highPriorityThreshold:
            settings.highPriorityThreshold === 'critical_only'
              ? 'critical_only'
              : 'high_and_critical',
          autoEscalationDelay:
            typeof settings.autoEscalationDelay === 'number' ? settings.autoEscalationDelay : 30,
          desktopNotificationsEnabled:
            typeof settings.desktopNotificationsEnabled === 'boolean'
              ? settings.desktopNotificationsEnabled
              : true,
          soundAlertsEnabled:
            typeof settings.soundAlertsEnabled === 'boolean' ? settings.soundAlertsEnabled : true
        })
      })
      .catch(() => {
        addToast('Failed to load notifications settings', 'error')
      })
      .finally(() => setLoading(false))
  }, [addToast])

  function showSaved(field: keyof NotificationsSettings): void {
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
    setSavedField(field)
    savedTimerRef.current = setTimeout(() => setSavedField(null), 2000)
  }

  function saveField(
    field: keyof NotificationsSettings,
    value: NotificationsSettings[keyof NotificationsSettings],
    updatedForm?: NotificationsSettings
  ): void {
    const updated: NotificationsSettings = { ...(updatedForm ?? form), [field]: value }
    window.rex
      .setSettings('notifications', updated as unknown as Settings)
      .then((result) => {
        if (result.ok) {
          showSaved(field)
        } else {
          addToast(result.error ?? 'Failed to save notifications settings', 'error')
        }
      })
      .catch(() => {
        addToast('Failed to save notifications settings', 'error')
      })
  }

  function handleChange<K extends keyof NotificationsSettings>(
    field: K,
    value: NotificationsSettings[K]
  ): void {
    const updated = { ...form, [field]: value }
    setForm(updated)
    saveField(field, value, updated)
  }

  if (loading) {
    return <PageLoadingFallback lines={9} />
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">Notifications</h2>

      {/* Quiet Hours section */}
      <section className="mb-6">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-4">
          Quiet Hours
        </h3>

        {/* Quiet hours enabled toggle */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <label htmlFor="quietHoursEnabled" className="text-sm font-medium text-text-primary">
              Enable Quiet Hours
            </label>
            <p className="text-xs text-text-secondary mt-0.5">
              Suppress non-critical notifications during quiet hours
            </p>
          </div>
          <div className="flex items-center gap-3">
            <SavedIndicator visible={savedField === 'quietHoursEnabled'} />
            <Toggle
              id="quietHoursEnabled"
              checked={form.quietHoursEnabled}
              onChange={(v) => handleChange('quietHoursEnabled', v)}
            />
          </div>
        </div>

        {/* Quiet hours start */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label
              htmlFor="quietHoursStart"
              className={[
                'text-sm font-medium',
                form.quietHoursEnabled ? 'text-text-primary' : 'text-text-secondary'
              ].join(' ')}
            >
              Quiet Hours Start
            </label>
            <SavedIndicator visible={savedField === 'quietHoursStart'} />
          </div>
          <input
            id="quietHoursStart"
            type="time"
            value={form.quietHoursStart}
            disabled={!form.quietHoursEnabled}
            onChange={(e) => handleChange('quietHoursStart', e.target.value)}
            className={[
              'w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent',
              form.quietHoursEnabled
                ? 'text-text-primary'
                : 'text-text-secondary opacity-50 cursor-not-allowed'
            ].join(' ')}
          />
        </div>

        {/* Quiet hours end */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label
              htmlFor="quietHoursEnd"
              className={[
                'text-sm font-medium',
                form.quietHoursEnabled ? 'text-text-primary' : 'text-text-secondary'
              ].join(' ')}
            >
              Quiet Hours End
            </label>
            <SavedIndicator visible={savedField === 'quietHoursEnd'} />
          </div>
          <input
            id="quietHoursEnd"
            type="time"
            value={form.quietHoursEnd}
            disabled={!form.quietHoursEnabled}
            onChange={(e) => handleChange('quietHoursEnd', e.target.value)}
            className={[
              'w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent',
              form.quietHoursEnabled
                ? 'text-text-primary'
                : 'text-text-secondary opacity-50 cursor-not-allowed'
            ].join(' ')}
          />
        </div>
      </section>

      <div className="border-t border-border mb-6" />

      {/* Digest section */}
      <section className="mb-6">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-4">
          Digest Mode
        </h3>

        {/* Digest mode enabled toggle */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <label htmlFor="digestModeEnabled" className="text-sm font-medium text-text-primary">
              Enable Digest Mode
            </label>
            <p className="text-xs text-text-secondary mt-0.5">
              Batch low-priority notifications into a daily digest
            </p>
          </div>
          <div className="flex items-center gap-3">
            <SavedIndicator visible={savedField === 'digestModeEnabled'} />
            <Toggle
              id="digestModeEnabled"
              checked={form.digestModeEnabled}
              onChange={(v) => handleChange('digestModeEnabled', v)}
            />
          </div>
        </div>

        {/* Digest delivery time */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label
              htmlFor="digestDeliveryTime"
              className={[
                'text-sm font-medium',
                form.digestModeEnabled ? 'text-text-primary' : 'text-text-secondary'
              ].join(' ')}
            >
              Digest Delivery Time
            </label>
            <SavedIndicator visible={savedField === 'digestDeliveryTime'} />
          </div>
          <input
            id="digestDeliveryTime"
            type="time"
            value={form.digestDeliveryTime}
            disabled={!form.digestModeEnabled}
            onChange={(e) => handleChange('digestDeliveryTime', e.target.value)}
            className={[
              'w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-accent',
              form.digestModeEnabled
                ? 'text-text-primary'
                : 'text-text-secondary opacity-50 cursor-not-allowed'
            ].join(' ')}
          />
        </div>
      </section>

      <div className="border-t border-border mb-6" />

      {/* Priority & escalation section */}
      <section className="mb-6">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-4">
          Priority &amp; Escalation
        </h3>

        {/* High-priority threshold */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="highPriorityThreshold" className="text-sm font-medium text-text-primary">
              High-Priority Threshold
            </label>
            <SavedIndicator visible={savedField === 'highPriorityThreshold'} />
          </div>
          <select
            id="highPriorityThreshold"
            value={form.highPriorityThreshold}
            onChange={(e) =>
              handleChange(
                'highPriorityThreshold',
                e.target.value as NotificationsSettings['highPriorityThreshold']
              )
            }
            className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="critical_only">Critical only</option>
            <option value="high_and_critical">High and critical</option>
          </select>
        </div>

        {/* Auto-escalation delay */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label
              htmlFor="autoEscalationDelay"
              className="text-sm font-medium text-text-primary"
            >
              Auto-Escalation Delay (minutes)
            </label>
            <SavedIndicator visible={savedField === 'autoEscalationDelay'} />
          </div>
          <input
            id="autoEscalationDelay"
            type="number"
            min={1}
            max={1440}
            value={form.autoEscalationDelay}
            onChange={(e) => {
              const val = parseInt(e.target.value, 10)
              if (!isNaN(val) && val >= 1) handleChange('autoEscalationDelay', val)
            }}
            className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
          />
        </div>
      </section>

      <div className="border-t border-border mb-6" />

      {/* Delivery section */}
      <section>
        <h3 className="text-xs font-semibold uppercase tracking-wider text-text-secondary mb-4">
          Delivery
        </h3>

        {/* Desktop notifications */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <label
              htmlFor="desktopNotificationsEnabled"
              className="text-sm font-medium text-text-primary"
            >
              Desktop Notifications
            </label>
            <p className="text-xs text-text-secondary mt-0.5">Show system notifications</p>
          </div>
          <div className="flex items-center gap-3">
            <SavedIndicator visible={savedField === 'desktopNotificationsEnabled'} />
            <Toggle
              id="desktopNotificationsEnabled"
              checked={form.desktopNotificationsEnabled}
              onChange={(v) => handleChange('desktopNotificationsEnabled', v)}
            />
          </div>
        </div>

        {/* Sound alerts */}
        <div className="flex items-center justify-between">
          <div>
            <label htmlFor="soundAlertsEnabled" className="text-sm font-medium text-text-primary">
              Sound Alerts
            </label>
            <p className="text-xs text-text-secondary mt-0.5">Play audio cues with notifications</p>
          </div>
          <div className="flex items-center gap-3">
            <SavedIndicator visible={savedField === 'soundAlertsEnabled'} />
            <Toggle
              id="soundAlertsEnabled"
              checked={form.soundAlertsEnabled}
              onChange={(v) => handleChange('soundAlertsEnabled', v)}
            />
          </div>
        </div>
      </section>
    </div>
  )
}
