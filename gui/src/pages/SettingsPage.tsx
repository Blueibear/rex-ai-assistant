import React, { useState, useEffect, useRef } from 'react'
import type { GeneralSettings, VoiceSettings, AiSettings, IntegrationsSettings, NotificationsSettings, Settings, VersionInfo, PreferenceSuggestion, VoiceInfo } from '../types/ipc'
import { useToast } from '../components/ui/Toast'
import { PageLoadingFallback } from '../components/ui/PageLoadingFallback'
import { SkeletonLine } from '../components/ui/SkeletonLine'

type CategoryId = 'general' | 'voice' | 'ai' | 'integrations' | 'notifications' | 'about'

interface Category {
  id: CategoryId
  label: string
  icon: React.ReactElement
}

const categories: Category[] = [
  {
    id: 'general',
    label: 'General',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.07 4.93l-1.41 1.41M4.93 19.07l1.41-1.41M4.93 4.93l1.41 1.41M19.07 19.07l-1.41-1.41M12 2v2M12 20v2M2 12h2M20 12h2" />
      </svg>
    )
  },
  {
    id: 'voice',
    label: 'Voice',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
        <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        <line x1="12" y1="19" x2="12" y2="23" />
        <line x1="8" y1="23" x2="16" y2="23" />
      </svg>
    )
  },
  {
    id: 'ai',
    label: 'AI',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="3" y="3" width="18" height="18" rx="2" />
        <path d="M9 9h6M9 12h6M9 15h4" />
      </svg>
    )
  },
  {
    id: 'integrations',
    label: 'Integrations',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6" />
        <polyline points="15 3 21 3 21 9" />
        <line x1="10" y1="14" x2="21" y2="3" />
      </svg>
    )
  },
  {
    id: 'notifications',
    label: 'Notifications',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
        <path d="M13.73 21a2 2 0 0 1-3.46 0" />
      </svg>
    )
  },
  {
    id: 'about',
    label: 'About',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="8" x2="12" y2="12" />
        <line x1="12" y1="16" x2="12.01" y2="16" />
      </svg>
    )
  }
]

// Common IANA timezone list (falls back to a curated list if Intl.supportedValuesOf unavailable)
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
      'Pacific/Auckland'
    ]
  }
})()

const LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Japanese']

function Toggle({
  checked,
  onChange,
  id
}: {
  checked: boolean
  onChange: (v: boolean) => void
  id: string
}): React.ReactElement {
  return (
    <button
      id={id}
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={[
        'relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg',
        checked ? 'bg-accent' : 'bg-surface-raised border border-border'
      ].join(' ')}
    >
      <span
        className={[
          'inline-block h-4 w-4 rounded-full bg-white shadow transition-transform',
          checked ? 'translate-x-6' : 'translate-x-1'
        ].join(' ')}
      />
    </button>
  )
}

function SavedIndicator({ visible }: { visible: boolean }): React.ReactElement {
  return (
    <span
      className={[
        'flex items-center gap-1 text-xs text-success transition-opacity duration-300',
        visible ? 'opacity-100' : 'opacity-0'
      ].join(' ')}
    >
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
        <polyline points="20 6 9 17 4 12" />
      </svg>
      Saved
    </span>
  )
}

function GeneralPanel(): React.ReactElement {
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
      .then(() => showSaved(field))
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

interface MediaDeviceOption {
  deviceId: string
  label: string
}

function VoicePanel(): React.ReactElement {
  const addToast = useToast()
  const [form, setForm] = useState<VoiceSettings>({
    microphoneDeviceId: '',
    speakerDeviceId: '',
    ttsEngine: 'system',
    ttsVoice: '',
    speechRate: 1.0,
    volume: 1.0
  })
  const [loading, setLoading] = useState(true)
  const [mics, setMics] = useState<MediaDeviceOption[]>([])
  const [speakers, setSpeakers] = useState<MediaDeviceOption[]>([])
  const [savedField, setSavedField] = useState<keyof VoiceSettings | null>(null)
  const [testing, setTesting] = useState(false)
  const [testResult, setTestResult] = useState<'ok' | 'error' | null>(null)
  const [voices, setVoices] = useState<VoiceInfo[]>([])
  const [voicesLoading, setVoicesLoading] = useState(false)
  const [previewing, setPreviewing] = useState(false)
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const testResultTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  function engineToProvider(engine: VoiceSettings['ttsEngine']): string {
    if (engine === 'openai') return 'edge-tts'
    if (engine === 'elevenlabs') return 'xtts'
    return 'pyttsx3'
  }

  function loadVoices(engine: VoiceSettings['ttsEngine']): void {
    setVoicesLoading(true)
    setVoices([])
    window.rex
      .listVoices(engineToProvider(engine))
      .then((res) => {
        setVoices(res.voices ?? [])
      })
      .catch(() => {
        setVoices([])
      })
      .finally(() => setVoicesLoading(false))
  }

  useEffect(() => {
    // Load devices
    if (navigator.mediaDevices?.enumerateDevices) {
      navigator.mediaDevices
        .enumerateDevices()
        .then((devices) => {
          const micList = devices
            .filter((d) => d.kind === 'audioinput')
            .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Microphone ${i + 1}` }))
          const speakerList = devices
            .filter((d) => d.kind === 'audiooutput')
            .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Speaker ${i + 1}` }))
          setMics(micList)
          setSpeakers(speakerList)
        })
        .catch(() => {
          /* no devices available */
        })
    }

    // Load settings
    window.rex
      .getSettings('voice')
      .then((settings: Settings) => {
        setForm({
          microphoneDeviceId:
            typeof settings.microphoneDeviceId === 'string' ? settings.microphoneDeviceId : '',
          speakerDeviceId:
            typeof settings.speakerDeviceId === 'string' ? settings.speakerDeviceId : '',
          ttsEngine:
            settings.ttsEngine === 'openai' || settings.ttsEngine === 'elevenlabs'
              ? settings.ttsEngine
              : 'system',
          ttsVoice: typeof settings.ttsVoice === 'string' ? settings.ttsVoice : '',
          speechRate: typeof settings.speechRate === 'number' ? settings.speechRate : 1.0,
          volume: typeof settings.volume === 'number' ? settings.volume : 1.0
        })
      })
      .catch(() => {
        addToast('Failed to load voice settings', 'error')
      })
      .finally(() => setLoading(false))
  }, [addToast])

  useEffect(() => {
    if (!loading) {
      loadVoices(form.ttsEngine)
    }
  }, [form.ttsEngine, loading]) // eslint-disable-line react-hooks/exhaustive-deps

  function showSaved(field: keyof VoiceSettings): void {
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
    setSavedField(field)
    savedTimerRef.current = setTimeout(() => setSavedField(null), 2000)
  }

  function saveField(
    field: keyof VoiceSettings,
    value: VoiceSettings[keyof VoiceSettings],
    updatedForm?: VoiceSettings
  ): void {
    const updated: VoiceSettings = { ...(updatedForm ?? form), [field]: value }
    window.rex
      .setSettings('voice', updated as unknown as Settings)
      .then(() => showSaved(field))
      .catch(() => {
        addToast('Failed to save voice settings', 'error')
      })
  }

  function handleFieldChange<K extends keyof VoiceSettings>(
    field: K,
    value: VoiceSettings[K]
  ): void {
    const updated = { ...form, [field]: value }
    setForm(updated)
    saveField(field, value, updated)
  }

  function handleTestVoice(): void {
    setTesting(true)
    setTestResult(null)
    window.rex
      .testVoice(form)
      .then((res) => {
        setTestResult(res.ok ? 'ok' : 'error')
      })
      .catch(() => {
        setTestResult('error')
      })
      .finally(() => {
        setTesting(false)
        if (testResultTimerRef.current) clearTimeout(testResultTimerRef.current)
        testResultTimerRef.current = setTimeout(() => setTestResult(null), 3000)
      })
  }

  function handlePreviewVoice(): void {
    if (!form.ttsVoice) return
    setPreviewing(true)
    window.rex
      .previewVoice(engineToProvider(form.ttsEngine), form.ttsVoice)
      .then((res) => {
        if (res.ok && res.audio_base64) {
          const binary = atob(res.audio_base64)
          const bytes = new Uint8Array(binary.length)
          for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i)
          }
          const ctx = new AudioContext()
          ctx.decodeAudioData(bytes.buffer).then((buf) => {
            const src = ctx.createBufferSource()
            src.buffer = buf
            src.connect(ctx.destination)
            src.start()
          }).catch(() => {
            addToast('Could not decode audio preview', 'error')
          })
        } else {
          addToast(res.error ?? 'Preview failed', 'error')
        }
      })
      .catch(() => {
        addToast('Preview failed', 'error')
      })
      .finally(() => setPreviewing(false))
  }

  if (loading) {
    return <PageLoadingFallback lines={6} />
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">Voice</h2>

      {/* Microphone device */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="microphoneDeviceId" className="text-sm font-medium text-text-primary">
            Microphone
          </label>
          <SavedIndicator visible={savedField === 'microphoneDeviceId'} />
        </div>
        <select
          id="microphoneDeviceId"
          value={form.microphoneDeviceId}
          onChange={(e) => handleFieldChange('microphoneDeviceId', e.target.value)}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        >
          <option value="">System default</option>
          {mics.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label}
            </option>
          ))}
        </select>
      </div>

      {/* Speaker device */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="speakerDeviceId" className="text-sm font-medium text-text-primary">
            Speaker
          </label>
          <SavedIndicator visible={savedField === 'speakerDeviceId'} />
        </div>
        <select
          id="speakerDeviceId"
          value={form.speakerDeviceId}
          onChange={(e) => handleFieldChange('speakerDeviceId', e.target.value)}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        >
          <option value="">System default</option>
          {speakers.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label}
            </option>
          ))}
        </select>
      </div>

      {/* TTS engine */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="ttsEngine" className="text-sm font-medium text-text-primary">
            TTS Engine
          </label>
          <SavedIndicator visible={savedField === 'ttsEngine'} />
        </div>
        <select
          id="ttsEngine"
          value={form.ttsEngine}
          onChange={(e) => {
            const v = e.target.value as VoiceSettings['ttsEngine']
            handleFieldChange('ttsEngine', v)
          }}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        >
          <option value="system">System</option>
          <option value="openai">OpenAI</option>
          <option value="elevenlabs">ElevenLabs</option>
        </select>
      </div>

      {/* TTS voice selector */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="ttsVoice" className="text-sm font-medium text-text-primary">
            Voice
          </label>
          <SavedIndicator visible={savedField === 'ttsVoice'} />
        </div>
        <div className="flex items-center gap-2">
          {voicesLoading ? (
            <div className="flex-1 flex items-center gap-2 bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-secondary">
              <svg className="animate-spin h-4 w-4 shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
              Loading voices…
            </div>
          ) : voices.length > 0 ? (
            <select
              id="ttsVoice"
              value={form.ttsVoice}
              onChange={(e) => handleFieldChange('ttsVoice', e.target.value)}
              className="flex-1 bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
            >
              <option value="">Select a voice…</option>
              {voices.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name}{v.language ? ` (${v.language})` : ''}
                </option>
              ))}
            </select>
          ) : (
            <input
              id="ttsVoice"
              type="text"
              value={form.ttsVoice}
              placeholder="Enter voice ID or name"
              onChange={(e) => setForm((f) => ({ ...f, ttsVoice: e.target.value }))}
              onBlur={(e) => saveField('ttsVoice', e.target.value)}
              className="flex-1 bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
            />
          )}
          <button
            onClick={handlePreviewVoice}
            disabled={previewing || !form.ttsVoice}
            title="Preview voice"
            className="flex items-center gap-1.5 bg-surface-raised hover:bg-surface border border-border disabled:opacity-40 text-text-primary text-sm font-medium px-3 py-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg shrink-0"
          >
            {previewing ? (
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
            )}
            Preview
          </button>
        </div>
      </div>

      {/* Speech rate */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="speechRate" className="text-sm font-medium text-text-primary">
            Speech Rate
            <span className="ml-2 text-xs text-text-secondary font-normal">
              {form.speechRate.toFixed(1)}×
            </span>
          </label>
          <SavedIndicator visible={savedField === 'speechRate'} />
        </div>
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>Slow</span>
          <input
            id="speechRate"
            type="range"
            min={0.5}
            max={2.0}
            step={0.1}
            value={form.speechRate}
            onChange={(e) => handleFieldChange('speechRate', parseFloat(e.target.value))}
            className="flex-1 accent-accent"
          />
          <span>Fast</span>
        </div>
      </div>

      {/* Volume */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="volume" className="text-sm font-medium text-text-primary">
            Volume
            <span className="ml-2 text-xs text-text-secondary font-normal">
              {Math.round(form.volume * 100)}%
            </span>
          </label>
          <SavedIndicator visible={savedField === 'volume'} />
        </div>
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>0%</span>
          <input
            id="volume"
            type="range"
            min={0}
            max={1.0}
            step={0.05}
            value={form.volume}
            onChange={(e) => handleFieldChange('volume', parseFloat(e.target.value))}
            className="flex-1 accent-accent"
          />
          <span>100%</span>
        </div>
      </div>

      {/* Test Voice button */}
      <div className="border-t border-border pt-5 flex items-center gap-3">
        <button
          onClick={handleTestVoice}
          disabled={testing}
          className="flex items-center gap-2 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg"
        >
          {testing ? (
            <>
              <svg
                className="animate-spin h-4 w-4"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
              Testing…
            </>
          ) : (
            <>
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
              >
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
              Test Voice
            </>
          )}
        </button>
        {testResult === 'ok' && (
          <span className="flex items-center gap-1 text-xs text-success">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Playing sample
          </span>
        )}
        {testResult === 'error' && (
          <span className="text-xs text-danger">Failed to play sample</span>
        )}
      </div>
    </div>
  )
}

const AI_MODELS: Array<{ value: AiSettings['model']; label: string }> = [
  { value: 'gpt-4o', label: 'GPT-4o' },
  { value: 'gpt-4-turbo', label: 'GPT-4 Turbo' },
  { value: 'claude-opus-4', label: 'Claude Opus 4' },
  { value: 'claude-sonnet-4', label: 'Claude Sonnet 4' },
  { value: 'gemini-1.5-pro', label: 'Gemini 1.5 Pro' }
]

function AiPanel(): React.ReactElement {
  const addToast = useToast()
  const [form, setForm] = useState<AiSettings>({
    model: 'claude-sonnet-4',
    temperature: 0.7,
    maxTokens: 2048,
    systemPrompt: '',
    autonomyMode: 'manual',
    budgetPerPlan: 0,
    budgetPerStep: 0
  })
  const [loading, setLoading] = useState(true)
  const [savedField, setSavedField] = useState<keyof AiSettings | null>(null)
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [suggestions, setSuggestions] = useState<PreferenceSuggestion[]>([])
  const [dismissedFields, setDismissedFields] = useState<Set<string>>(new Set())

  function loadSuggestions(): void {
    window.rex
      .getPreferenceSuggestions()
      .then((s) => setSuggestions(s))
      .catch(() => {
        // Non-fatal — suggestions are best-effort
      })
  }

  useEffect(() => {
    window.rex
      .getSettings('ai')
      .then((settings: Settings) => {
        setForm({
          model: (AI_MODELS.some((m) => m.value === settings.model)
            ? settings.model
            : 'claude-sonnet-4') as AiSettings['model'],
          temperature: typeof settings.temperature === 'number' ? settings.temperature : 0.7,
          maxTokens: typeof settings.maxTokens === 'number' ? settings.maxTokens : 2048,
          systemPrompt: typeof settings.systemPrompt === 'string' ? settings.systemPrompt : '',
          autonomyMode:
            settings.autonomyMode === 'supervised' || settings.autonomyMode === 'full-auto'
              ? settings.autonomyMode
              : 'manual',
          budgetPerPlan: typeof settings.budgetPerPlan === 'number' ? settings.budgetPerPlan : 0,
          budgetPerStep: typeof settings.budgetPerStep === 'number' ? settings.budgetPerStep : 0
        })
      })
      .catch(() => {
        addToast('Failed to load AI settings', 'error')
      })
      .finally(() => setLoading(false))

    loadSuggestions()
  }, [addToast])

  function showSaved(field: keyof AiSettings): void {
    if (savedTimerRef.current) clearTimeout(savedTimerRef.current)
    setSavedField(field)
    savedTimerRef.current = setTimeout(() => setSavedField(null), 2000)
  }

  function handleFieldChange<K extends keyof AiSettings>(field: K, value: AiSettings[K]): void {
    const updated = { ...form, [field]: value }
    setForm(updated)
    window.rex
      .setSettings('ai', updated as unknown as Settings)
      .then(() => showSaved(field))
      .catch(() => {
        addToast('Failed to save AI settings', 'error')
      })
  }

  function handleApplySuggestion(suggestion: PreferenceSuggestion): void {
    window.rex
      .applyPreferenceSuggestion(suggestion.field, suggestion.suggested_value)
      .then(() => {
        setForm((f) => ({ ...f, [suggestion.field]: suggestion.suggested_value }))
        setDismissedFields((prev) => new Set(prev).add(suggestion.field))
        loadSuggestions()
      })
      .catch(() => {
        addToast('Failed to apply suggestion', 'error')
      })
  }

  function handleDismissSuggestion(field: string): void {
    setDismissedFields((prev) => new Set(prev).add(field))
  }

  const activeSuggestion = suggestions.find((s) => !dismissedFields.has(s.field)) ?? null

  if (loading) {
    return <PageLoadingFallback lines={5} />
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">AI</h2>

      {/* AI Model */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="aiModel" className="text-sm font-medium text-text-primary">
            AI Model
          </label>
          <SavedIndicator visible={savedField === 'model'} />
        </div>
        <select
          id="aiModel"
          value={form.model}
          onChange={(e) => handleFieldChange('model', e.target.value as AiSettings['model'])}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        >
          {AI_MODELS.map((m) => (
            <option key={m.value} value={m.value}>
              {m.label}
            </option>
          ))}
        </select>
      </div>

      {/* Temperature */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="temperature" className="text-sm font-medium text-text-primary">
            Temperature
            <span className="ml-2 text-xs text-text-secondary font-normal">
              {form.temperature.toFixed(2)}
            </span>
          </label>
          <SavedIndicator visible={savedField === 'temperature'} />
        </div>
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>Precise</span>
          <input
            id="temperature"
            type="range"
            min={0}
            max={1.0}
            step={0.01}
            value={form.temperature}
            onChange={(e) => handleFieldChange('temperature', parseFloat(e.target.value))}
            className="flex-1 accent-accent"
          />
          <span>Creative</span>
        </div>
      </div>

      {/* Max tokens */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="maxTokens" className="text-sm font-medium text-text-primary">
            Max Tokens
          </label>
          <SavedIndicator visible={savedField === 'maxTokens'} />
        </div>
        <input
          id="maxTokens"
          type="number"
          min={1}
          max={128000}
          step={256}
          value={form.maxTokens}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10)
            if (!isNaN(val) && val > 0) handleFieldChange('maxTokens', val)
          }}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        />
      </div>

      {/* System prompt override */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="systemPrompt" className="text-sm font-medium text-text-primary">
            System Prompt Override
          </label>
          <SavedIndicator visible={savedField === 'systemPrompt'} />
        </div>
        <textarea
          id="systemPrompt"
          rows={4}
          value={form.systemPrompt}
          placeholder="Leave blank to use the default system prompt"
          onChange={(e) => setForm((f) => ({ ...f, systemPrompt: e.target.value }))}
          onBlur={(e) => handleFieldChange('systemPrompt', e.target.value)}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent resize-none"
        />
      </div>

      {/* Autonomy mode */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="autonomyMode" className="text-sm font-medium text-text-primary">
            Autonomy Mode
          </label>
          <SavedIndicator visible={savedField === 'autonomyMode'} />
        </div>
        <select
          id="autonomyMode"
          value={form.autonomyMode}
          onChange={(e) =>
            handleFieldChange('autonomyMode', e.target.value as AiSettings['autonomyMode'])
          }
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        >
          <option value="manual">Manual — confirm every action</option>
          <option value="supervised">Supervised — confirm risky actions</option>
          <option value="full-auto">Full Auto — act without confirmation</option>
        </select>
      </div>

      {/* Full-auto warning */}
      {form.autonomyMode === 'full-auto' && (
        <div className="flex items-start gap-2.5 rounded-lg border border-warning/40 bg-warning/10 px-4 py-3 text-sm text-warning">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="shrink-0 mt-0.5"
          >
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          Rex will act without confirmation. Review task history regularly.
        </div>
      )}

      {/* Budget per plan */}
      <div className="mb-5 mt-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="budgetPerPlan" className="text-sm font-medium text-text-primary">
            Budget per Plan (USD)
          </label>
          <SavedIndicator visible={savedField === 'budgetPerPlan'} />
        </div>
        <input
          id="budgetPerPlan"
          type="number"
          min="0"
          step="0.01"
          value={form.budgetPerPlan}
          onChange={(e) => handleFieldChange('budgetPerPlan', parseFloat(e.target.value) || 0)}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        />
        <p className="mt-1 text-xs text-text-secondary">Maximum estimated cost per plan run in USD. Set to 0 for unlimited.</p>
      </div>

      {/* Budget per step */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="budgetPerStep" className="text-sm font-medium text-text-primary">
            Budget per Step (USD)
          </label>
          <SavedIndicator visible={savedField === 'budgetPerStep'} />
        </div>
        <input
          id="budgetPerStep"
          type="number"
          min="0"
          step="0.001"
          value={form.budgetPerStep}
          onChange={(e) => handleFieldChange('budgetPerStep', parseFloat(e.target.value) || 0)}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        />
        <p className="mt-1 text-xs text-text-secondary">Maximum estimated cost per individual step in USD. Steps over this limit are skipped. Set to 0 for unlimited.</p>
      </div>

      {/* Preference suggestion banner */}
      {activeSuggestion !== null && (
        <div className="flex items-start gap-3 rounded-lg border border-accent/40 bg-accent/10 px-4 py-3 text-sm text-accent">
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            className="shrink-0 mt-0.5"
          >
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <div className="flex-1">
            <p>Based on your usage: {activeSuggestion.reason}.</p>
            <div className="flex items-center gap-2 mt-2">
              <button
                onClick={() => handleApplySuggestion(activeSuggestion)}
                className="text-xs font-medium bg-accent text-white px-3 py-1 rounded-md hover:bg-accent/90 transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg"
              >
                Apply
              </button>
              <button
                onClick={() => handleDismissSuggestion(activeSuggestion.field)}
                className="text-xs font-medium text-accent hover:text-accent/80 transition-colors focus:outline-none"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

type IntegrationSection = 'email' | 'calendar' | 'sms'
type TestStatus = 'idle' | 'testing' | 'ok' | 'error'

function EyeIcon({ open }: { open: boolean }): React.ReactElement {
  if (open) {
    return (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
        <circle cx="12" cy="12" r="3" />
      </svg>
    )
  }
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94" />
      <path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19" />
      <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
  )
}

function PasswordInput({
  id,
  value,
  placeholder,
  onChange,
  onBlur
}: {
  id: string
  value: string
  placeholder?: string
  onChange: (v: string) => void
  onBlur?: () => void
}): React.ReactElement {
  const [show, setShow] = useState(false)
  return (
    <div className="relative">
      <input
        id={id}
        type={show ? 'text' : 'password'}
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        onBlur={onBlur}
        className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 pr-10 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
      />
      <button
        type="button"
        onClick={() => setShow((s) => !s)}
        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary transition-colors"
        aria-label={show ? 'Hide' : 'Show'}
      >
        <EyeIcon open={show} />
      </button>
    </div>
  )
}

function TestConnectionButton({
  status,
  onTest
}: {
  status: TestStatus
  onTest: () => void
}): React.ReactElement {
  return (
    <div className="flex items-center gap-3 mt-3">
      <button
        onClick={onTest}
        disabled={status === 'testing'}
        className="flex items-center gap-2 bg-surface-raised hover:bg-border disabled:opacity-50 text-text-primary text-xs font-medium px-3 py-1.5 rounded-lg border border-border transition-colors focus:outline-none focus:ring-2 focus:ring-accent"
      >
        {status === 'testing' ? (
          <>
            <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
            Testing…
          </>
        ) : 'Test Connection'}
      </button>
      {status === 'ok' && (
        <span className="flex items-center gap-1 text-xs text-success">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
            <polyline points="20 6 9 17 4 12" />
          </svg>
          Connected
        </span>
      )}
      {status === 'error' && (
        <span className="flex items-center gap-1 text-xs text-danger">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
          Not connected
        </span>
      )}
      {status === 'idle' && (
        <span className="text-xs text-text-secondary">Not tested</span>
      )}
    </div>
  )
}

function IntegrationsPanel(): React.ReactElement {
  const addToast = useToast()
  const [form, setForm] = useState<IntegrationsSettings>({
    emailProvider: 'gmail',
    emailClientId: '',
    emailClientSecret: '',
    calendarProvider: 'gmail',
    calendarClientId: '',
    calendarClientSecret: '',
    smsSid: '',
    smsAuthToken: '',
    smsFromNumber: ''
  })
  const [loading, setLoading] = useState(true)
  const [savedField, setSavedField] = useState<keyof IntegrationsSettings | null>(null)
  const [testStatus, setTestStatus] = useState<Record<IntegrationSection, TestStatus>>({
    email: 'idle',
    calendar: 'idle',
    sms: 'idle'
  })
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const testTimers = useRef<Partial<Record<IntegrationSection, ReturnType<typeof setTimeout>>>>({})

  useEffect(() => {
    window.rex
      .getSettings('integrations')
      .then((settings: Settings) => {
        setForm({
          emailProvider:
            settings.emailProvider === 'outlook' ? 'outlook' : 'gmail',
          emailClientId: typeof settings.emailClientId === 'string' ? settings.emailClientId : '',
          emailClientSecret:
            typeof settings.emailClientSecret === 'string' ? settings.emailClientSecret : '',
          calendarProvider:
            settings.calendarProvider === 'outlook' ? 'outlook' : 'gmail',
          calendarClientId:
            typeof settings.calendarClientId === 'string' ? settings.calendarClientId : '',
          calendarClientSecret:
            typeof settings.calendarClientSecret === 'string' ? settings.calendarClientSecret : '',
          smsSid: typeof settings.smsSid === 'string' ? settings.smsSid : '',
          smsAuthToken: typeof settings.smsAuthToken === 'string' ? settings.smsAuthToken : '',
          smsFromNumber: typeof settings.smsFromNumber === 'string' ? settings.smsFromNumber : ''
        })
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
      .then(() => showSaved(field))
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
    saveField(field, updated)
  }

  function handleTest(section: IntegrationSection): void {
    setTestStatus((s) => ({ ...s, [section]: 'testing' }))
    window.rex
      .testIntegration(section)
      .then((res) => {
        setTestStatus((s) => ({ ...s, [section]: res.ok ? 'ok' : 'error' }))
      })
      .catch(() => {
        setTestStatus((s) => ({ ...s, [section]: 'error' }))
      })
      .finally(() => {
        if (testTimers.current[section]) clearTimeout(testTimers.current[section])
        testTimers.current[section] = setTimeout(
          () => setTestStatus((s) => ({ ...s, [section]: 'idle' })),
          5000
        )
      })
  }

  const inputClass =
    'w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent'

  if (loading) {
    return <PageLoadingFallback lines={6} />
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">Integrations</h2>

      {/* Email section */}
      <section className="mb-7">
        <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
            <polyline points="22,6 12,13 2,6" />
          </svg>
          Email
        </h3>

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
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="emailClientId" className="text-sm font-medium text-text-primary">OAuth Client ID</label>
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
            placeholder="Enter client secret"
            onChange={(v) => setForm((f) => ({ ...f, emailClientSecret: v }))}
            onBlur={() => handleFieldChange('emailClientSecret', form.emailClientSecret)}
          />
        </div>

        <TestConnectionButton status={testStatus.email} onTest={() => handleTest('email')} />
      </section>

      <div className="border-t border-border mb-7" />

      {/* Calendar section */}
      <section className="mb-7">
        <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" />
          </svg>
          Calendar
        </h3>

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
        </div>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="calendarClientId" className="text-sm font-medium text-text-primary">OAuth Client ID</label>
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
            placeholder="Enter client secret"
            onChange={(v) => setForm((f) => ({ ...f, calendarClientSecret: v }))}
            onBlur={() => handleFieldChange('calendarClientSecret', form.calendarClientSecret)}
          />
        </div>

        <TestConnectionButton status={testStatus.calendar} onTest={() => handleTest('calendar')} />
      </section>

      <div className="border-t border-border mb-7" />

      {/* SMS section */}
      <section className="mb-2">
        <h3 className="text-sm font-semibold text-text-primary mb-4 flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
          </svg>
          SMS (Twilio)
        </h3>

        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="smsSid" className="text-sm font-medium text-text-primary">Account SID</label>
            <SavedIndicator visible={savedField === 'smsSid'} />
          </div>
          <input
            id="smsSid"
            type="text"
            value={form.smsSid}
            placeholder="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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
            placeholder="Enter auth token"
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

        <TestConnectionButton status={testStatus.sms} onTest={() => handleTest('sms')} />
      </section>
    </div>
  )
}

function NotificationsPanel(): React.ReactElement {
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
      .then(() => showSaved(field))
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

function AboutPanel(): React.ReactElement {
  const [info, setInfo] = useState<VersionInfo | null>(null)
  const [loading, setLoading] = useState(true)
  const [slow, setSlow] = useState(false)

  useEffect(() => {
    window.rex
      .getVersionInfo()
      .then((v) => {
        setInfo(v)
      })
      .catch(() => {
        setInfo({ rex: 'unknown', electron: 'unknown', node: 'unknown' })
      })
      .finally(() => {
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    if (!loading) return
    const id = setTimeout(() => setSlow(true), 5000)
    return () => clearTimeout(id)
  }, [loading])

  return (
    <div className="p-6">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-12 h-12 rounded-xl bg-accent/20 flex items-center justify-center">
          <span className="text-accent font-bold text-lg">R</span>
        </div>
        <div>
          <h2 className="text-lg font-semibold text-text-primary">Rex AI Assistant</h2>
          <p className="text-sm text-text-secondary">Local-first AI companion</p>
        </div>
      </div>

      {loading ? (
        <div className="space-y-3">
          <SkeletonLine width="100%" height="1.25rem" />
          <SkeletonLine width="80%" height="1.25rem" />
          <SkeletonLine width="60%" height="1.25rem" />
          {slow && (
            <p className="text-xs text-text-secondary mt-1 animate-pulse">
              Taking longer than expected…
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          <VersionRow label="Rex version" value={info?.rex ?? 'unknown'} />
          <VersionRow label="Electron version" value={info?.electron ?? 'unknown'} />
          <VersionRow label="Node version" value={info?.node ?? 'unknown'} />
        </div>
      )}

      <div className="mt-8 pt-6 border-t border-border">
        <p className="text-xs text-text-secondary">
          Rex AI Assistant is a local-first, voice-activated AI companion. All data is stored on
          your device.
        </p>
      </div>
    </div>
  )
}

function VersionRow({ label, value }: { label: string; value: string }): React.ReactElement {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border/50">
      <span className="text-sm text-text-secondary">{label}</span>
      <span className="text-sm font-mono text-text-primary">{value}</span>
    </div>
  )
}

function renderPanel(categoryId: CategoryId): React.ReactElement {
  switch (categoryId) {
    case 'general':
      return <GeneralPanel />
    case 'voice':
      return <VoicePanel />
    case 'ai':
      return <AiPanel />
    case 'integrations':
      return <IntegrationsPanel />
    case 'notifications':
      return <NotificationsPanel />
    case 'about':
      return <AboutPanel />
  }
}

export function SettingsPage(): React.ReactElement {
  const [activeCategory, setActiveCategory] = useState<CategoryId>('general')

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left: category list */}
      <nav className="w-48 shrink-0 border-r border-border overflow-y-auto py-2">
        {categories.map((cat) => {
          const isActive = cat.id === activeCategory
          return (
            <button
              key={cat.id}
              onClick={() => setActiveCategory(cat.id)}
              className={[
                'w-full flex items-center gap-2.5 px-4 py-2.5 text-sm text-left transition-colors',
                isActive
                  ? 'bg-accent/10 text-accent font-medium'
                  : 'text-text-secondary hover:bg-surface-raised hover:text-text-primary'
              ].join(' ')}
            >
              <span className={isActive ? 'text-accent' : 'text-text-secondary'}>{cat.icon}</span>
              {cat.label}
            </button>
          )
        })}
      </nav>

      {/* Right: content area */}
      <main className="flex-1 overflow-y-auto">{renderPanel(activeCategory)}</main>
    </div>
  )
}
