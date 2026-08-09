import React, { useEffect, useRef, useState } from 'react'
import type { GeneralSettings, Settings } from '../../types/ipc'
import { PageLoadingFallback } from '../../components/ui/PageLoadingFallback'
import { useToast } from '../../components/ui/Toast'
import { SavedIndicator, Toggle } from './shared'

const TIMEZONE_LIST: string[] = (() => {
  try {
    return (
      Intl as unknown as { supportedValuesOf: (key: string) => string[] }
    ).supportedValuesOf('timeZone')
  } catch {
    return [
      'UTC',
      'America/New_York',
      'America/Chicago',
      'America/Denver',
      'America/Los_Angeles',
      'America/Anchorage',
      'America/Honolulu',
      'America/Toronto',
      'America/Vancouver',
      'America/Mexico_City',
      'America/Sao_Paulo',
      'America/Buenos_Aires',
      'Europe/London',
      'Europe/Paris',
      'Europe/Berlin',
      'Europe/Madrid',
      'Europe/Rome',
      'Europe/Amsterdam',
      'Europe/Stockholm',
      'Europe/Moscow',
      'Africa/Cairo',
      'Africa/Johannesburg',
      'Asia/Dubai',
      'Asia/Kolkata',
      'Asia/Dhaka',
      'Asia/Bangkok',
      'Asia/Singapore',
      'Asia/Shanghai',
      'Asia/Tokyo',
      'Asia/Seoul',
      'Australia/Sydney',
      'Australia/Melbourne',
      'Pacific/Auckland',
      'Pacific/Honolulu'
    ]
  }
})()

const LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Japanese']

export function GeneralSettingsSection(): React.ReactElement {
  const addToast = useToast()
  const [form, setForm] = useState<GeneralSettings>({
    displayName: '',
    timezone: 'UTC',
    language: 'English',
    launchAtLogin: false,
    startMinimized: false
  })
  const [loading, setLoading] = useState(true)
  const [savedField, setSavedField] = useState<keyof GeneralSettings | null>(null)
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    window.rex
      .getSettings('general')
      .then((settings: Settings) => {
        setForm({
          displayName: typeof settings.displayName === 'string' ? settings.displayName : '',
          timezone:
            typeof settings.timezone === 'string' ? settings.timezone : 'UTC',
          language: typeof settings.language === 'string' ? settings.language : 'English',
          launchAtLogin:
            typeof settings.launchAtLogin === 'boolean' ? settings.launchAtLogin : false,
          startMinimized:
            typeof settings.startMinimized === 'boolean' ? settings.startMinimized : false
        })
      })
      .catch(() => {
        addToast('Failed to load general settings', 'error')
      })
      .finally(() => setLoading(false))
  }, [addToast])

  function showSaved(field: keyof GeneralSettings): void {
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
    setSavedField(field)
    savedTimerRef.current = setTimeout(() => setSavedField(null), 2000)
  }

  function saveField(field: keyof GeneralSettings, value: GeneralSettings[keyof GeneralSettings]): void {
    const updated: GeneralSettings = { ...form, [field]: value }
    window.rex
      .setSettings('general', updated as unknown as Settings)
      .then((result) => {
        if (result.ok) {
          showSaved(field)
        } else {
          addToast(result.error ?? 'Failed to save general settings', 'error')
        }
      })
      .catch(() => {
        addToast('Failed to save general settings', 'error')
      })
  }

  if (loading) {
    return <PageLoadingFallback lines={5} />
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">General</h2>

      {/* Display Name */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="displayName" className="text-sm font-medium text-text-primary">
            Display Name
          </label>
          <SavedIndicator visible={savedField === 'displayName'} />
        </div>
        <input
          id="displayName"
          type="text"
          value={form.displayName}
          placeholder="Your name"
          onChange={(e) => setForm((f) => ({ ...f, displayName: e.target.value }))}
          onBlur={(e) => saveField('displayName', e.target.value)}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
        />
      </div>

      {/* Timezone */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="timezone" className="text-sm font-medium text-text-primary">
            Timezone
          </label>
          <SavedIndicator visible={savedField === 'timezone'} />
        </div>
        <input
          id="timezone"
          list="timezone-list"
          type="text"
          value={form.timezone}
          onChange={(e) => setForm((f) => ({ ...f, timezone: e.target.value }))}
          onBlur={(e) => {
            const val = e.target.value.trim()
            if (TIMEZONE_LIST.includes(val)) {
              saveField('timezone', val)
            } else {
              // reset to last saved value on invalid entry
              window.rex
                .getSettings('general')
                .then((s: Settings) => {
                  const tz = typeof s.timezone === 'string' ? s.timezone : form.timezone
                  setForm((f) => ({ ...f, timezone: tz }))
                })
                .catch(() => {
                  setForm((f) => ({ ...f, timezone: 'UTC' }))
                })
            }
          }}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
        />
        <datalist id="timezone-list">
          {TIMEZONE_LIST.map((tz) => (
            <option key={tz} value={tz} />
          ))}
        </datalist>
      </div>

      {/* Language */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="language" className="text-sm font-medium text-text-primary">
            Language
          </label>
          <SavedIndicator visible={savedField === 'language'} />
        </div>
        <select
          id="language"
          value={form.language}
          onChange={(e) => {
            setForm((f) => ({ ...f, language: e.target.value }))
            saveField('language', e.target.value)
          }}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        >
          {LANGUAGES.map((lang) => (
            <option key={lang} value={lang}>
              {lang}
            </option>
          ))}
        </select>
      </div>

      <div className="border-t border-border pt-5 space-y-4">
        {/* Launch at login */}
        <div className="flex items-center justify-between">
          <div>
            <label htmlFor="launchAtLogin" className="text-sm font-medium text-text-primary">
              Launch at login
            </label>
            <p className="text-xs text-text-secondary mt-0.5">Start Rex automatically when you log in</p>
          </div>
          <div className="flex items-center gap-3">
            <SavedIndicator visible={savedField === 'launchAtLogin'} />
            <Toggle
              id="launchAtLogin"
              checked={form.launchAtLogin}
              onChange={(v) => {
                setForm((f) => ({ ...f, launchAtLogin: v }))
                saveField('launchAtLogin', v)
              }}
            />
          </div>
        </div>

        {/* Start minimized */}
        <div className="flex items-center justify-between">
          <div>
            <label htmlFor="startMinimized" className="text-sm font-medium text-text-primary">
              Start minimized to tray
            </label>
            <p className="text-xs text-text-secondary mt-0.5">
              Open in system tray on startup
            </p>
          </div>
          <div className="flex items-center gap-3">
            <SavedIndicator visible={savedField === 'startMinimized'} />
            <Toggle
              id="startMinimized"
              checked={form.startMinimized}
              onChange={(v) => {
                setForm((f) => ({ ...f, startMinimized: v }))
                saveField('startMinimized', v)
              }}
            />
          </div>
        </div>
      </div>
    </div>
  )
}
