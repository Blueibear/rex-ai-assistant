import React, { useState, useEffect, useRef } from 'react'
import type { GeneralSettings, VoiceSettings, AiSettings, IntegrationsSettings, EmailAccount, NotificationsSettings, Settings, VersionInfo, PreferenceSuggestion, VoiceInfo, WakeWordInfo, VoiceEnrollment, Memory, SmartSpeaker, SystemSettings } from '../types/ipc'
import { useToast } from '../components/ui/Toast'
import { PageLoadingFallback } from '../components/ui/PageLoadingFallback'
import { SkeletonLine } from '../components/ui/SkeletonLine'

type CategoryId = 'general' | 'voice' | 'ai' | 'integrations' | 'notifications' | 'users' | 'audio' | 'system' | 'about'

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
    id: 'users',
    label: 'Users',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
        <circle cx="9" cy="7" r="4" />
        <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
        <path d="M16 3.13a4 4 0 0 1 0 7.75" />
      </svg>
    )
  },
  {
    id: 'audio',
    label: 'Audio Output',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
        <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
        <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
      </svg>
    )
  },
  {
    id: 'system',
    label: 'System',
    icon: (
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <rect x="2" y="3" width="20" height="14" rx="2" />
        <line x1="8" y1="21" x2="16" y2="21" />
        <line x1="12" y1="17" x2="12" y2="21" />
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

const ENROLLMENT_SAMPLE_TARGET = 3
const ENROLLMENT_SAMPLE_DURATION_MS = 1600
const ENROLLMENT_COUNTDOWN_SECONDS = 3
const ENROLLMENT_SAMPLE_RATE = 16000

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms)
  })
}

function mergeFloat32Arrays(chunks: Float32Array[]): Float32Array {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  const merged = new Float32Array(totalLength)
  let offset = 0
  for (const chunk of chunks) {
    merged.set(chunk, offset)
    offset += chunk.length
  }
  return merged
}

function downsampleFloat32(
  samples: Float32Array,
  inputSampleRate: number,
  outputSampleRate: number
): Float32Array {
  if (inputSampleRate === outputSampleRate) {
    return samples
  }
  if (inputSampleRate < outputSampleRate) {
    throw new Error('Microphone sample rate is lower than the enrollment target rate')
  }

  const ratio = inputSampleRate / outputSampleRate
  const outputLength = Math.max(1, Math.round(samples.length / ratio))
  const output = new Float32Array(outputLength)

  for (let i = 0; i < outputLength; i += 1) {
    const start = Math.floor(i * ratio)
    const end = Math.min(samples.length, Math.floor((i + 1) * ratio))
    let sum = 0
    let count = 0
    for (let j = start; j < end; j += 1) {
      sum += samples[j]
      count += 1
    }
    output[i] = count > 0 ? sum / count : samples[Math.min(start, samples.length - 1)]
  }

  return output
}

async function captureEnrollmentSample(stream: MediaStream): Promise<number[]> {
  const audioContext = new AudioContext()
  const source = audioContext.createMediaStreamSource(stream)
  const processor = audioContext.createScriptProcessor(4096, 1, 1)
  const sink = audioContext.createGain()
  const chunks: Float32Array[] = []

  sink.gain.value = 0

  return new Promise<number[]>((resolve, reject) => {
    let settled = false

    const cleanup = (): void => {
      if (!settled) {
        settled = true
        processor.disconnect()
        sink.disconnect()
        source.disconnect()
        void audioContext.close()
      }
    }

    processor.onaudioprocess = (event) => {
      const channel = event.inputBuffer.getChannelData(0)
      chunks.push(new Float32Array(channel))
    }

    source.connect(processor)
    processor.connect(sink)
    sink.connect(audioContext.destination)

    window.setTimeout(() => {
      try {
        if (chunks.length === 0) {
          throw new Error('No microphone audio was captured')
        }
        const merged = mergeFloat32Arrays(chunks)
        const downsampled = downsampleFloat32(
          merged,
          audioContext.sampleRate,
          ENROLLMENT_SAMPLE_RATE
        )
        resolve(Array.from(downsampled))
      } catch (error) {
        reject(error)
      } finally {
        cleanup()
      }
    }, ENROLLMENT_SAMPLE_DURATION_MS)
  })
}

async function runEnrollmentCountdown(
  onTick: (remaining: number) => void
): Promise<void> {
  for (let remaining = ENROLLMENT_COUNTDOWN_SECONDS; remaining > 0; remaining -= 1) {
    onTick(remaining)
    await sleep(1000)
  }
  onTick(0)
}

function VoicePanel(): React.ReactElement {
  const addToast = useToast()
  const [form, setForm] = useState<VoiceSettings>({
    microphoneDeviceId: '',
    speakerDeviceId: '',
    ttsEngine: 'pyttsx3',
    ttsVoice: '',
    speechRate: 1.0,
    volume: 1.0,
    sttModel: 'base',
    sttLanguage: 'auto',
    sttDevice: 'auto',
    wakeWord: ''
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
  const [wakeWords, setWakeWords] = useState<WakeWordInfo[]>([])
  const [previewingWakeWord, setPreviewingWakeWord] = useState(false)
  const [activeUserId, setActiveUserId] = useState('default')
  const [enrollments, setEnrollments] = useState<VoiceEnrollment[]>([])
  const [enrollmentCountdown, setEnrollmentCountdown] = useState(0)
  const [capturedSamples, setCapturedSamples] = useState(0)
  const [enrollmentMessage, setEnrollmentMessage] = useState<string | null>(null)
  const [enrollmentError, setEnrollmentError] = useState<string | null>(null)
  const [enrolling, setEnrolling] = useState(false)
  const [deletingUserId, setDeletingUserId] = useState<string | null>(null)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadFileDuration, setUploadFileDuration] = useState<number | null>(null)
  const [uploadVoiceName, setUploadVoiceName] = useState('')
  const [uploading, setUploading] = useState(false)
  const [uploadResult, setUploadResult] = useState<{ ok: boolean; message: string } | null>(null)
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const testResultTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  function engineToProvider(engine: VoiceSettings['ttsEngine']): string {
    if (engine === 'xtts' || engine === 'elevenlabs') return 'xtts'
    if (engine === 'edge-tts' || engine === 'openai') return 'edge-tts'
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

  function loadWakeWords(): void {
    window.rex
      .listWakeWords()
      .then((res) => {
        setWakeWords(res.wake_words ?? [])
      })
      .catch(() => setWakeWords([]))
  }

  function loadEnrollmentState(): void {
    window.rex
      .getVoiceEnrollments()
      .then((result) => {
        if (!result.ok) {
          throw new Error(result.error ?? 'Failed to load enrollments')
        }
        setActiveUserId(result.active_user_id || 'default')
        setEnrollments(result.enrollments ?? [])
      })
      .catch((error: unknown) => {
        const message = error instanceof Error ? error.message : 'Failed to load voice enrollments'
        setEnrollmentError(message)
      })
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
        const rawEngine = settings.ttsEngine
        const ttsEngine: VoiceSettings['ttsEngine'] =
          rawEngine === 'xtts' || rawEngine === 'edge-tts' || rawEngine === 'pyttsx3'
            ? rawEngine
            : rawEngine === 'elevenlabs'
              ? 'xtts'
              : rawEngine === 'openai'
                ? 'edge-tts'
                : 'pyttsx3'
        const rawSttDevice = settings.sttDevice
        const sttDevice: VoiceSettings['sttDevice'] =
          rawSttDevice === 'cpu' || rawSttDevice === 'cuda' ? rawSttDevice : 'auto'
        setForm({
          microphoneDeviceId:
            typeof settings.microphoneDeviceId === 'string' ? settings.microphoneDeviceId : '',
          speakerDeviceId:
            typeof settings.speakerDeviceId === 'string' ? settings.speakerDeviceId : '',
          ttsEngine,
          ttsVoice: typeof settings.ttsVoice === 'string' ? settings.ttsVoice : '',
          speechRate: typeof settings.speechRate === 'number' ? settings.speechRate : 1.0,
          volume: typeof settings.volume === 'number' ? settings.volume : 1.0,
          sttModel: typeof settings.sttModel === 'string' ? settings.sttModel : 'base',
          sttLanguage: typeof settings.sttLanguage === 'string' ? settings.sttLanguage : 'auto',
          sttDevice,
          wakeWord: typeof settings.wakeWord === 'string' ? settings.wakeWord : ''
        })
      })
      .catch(() => {
        addToast('Failed to load voice settings', 'error')
      })
      .finally(() => setLoading(false))

    loadEnrollmentState()
    loadWakeWords()
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

  function handlePreviewWakeWord(): void {
    if (!form.wakeWord) return
    setPreviewingWakeWord(true)
    const phrase = form.wakeWord.replace(/_/g, ' ')
    window.rex
      .previewVoice('pyttsx3', phrase)
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
            addToast('Could not play wake word sample', 'error')
          })
        } else {
          addToast(res.error ?? 'Preview failed', 'error')
        }
      })
      .catch(() => addToast('Preview failed', 'error'))
      .finally(() => setPreviewingWakeWord(false))
  }

  function handleUploadFileChange(e: React.ChangeEvent<HTMLInputElement>): void {
    const file = e.target.files?.[0] ?? null
    setUploadFile(file)
    setUploadFileDuration(null)
    setUploadResult(null)
    if (file) {
      setUploadVoiceName(file.name.replace(/\.[^.]+$/, '').replace(/[_-]/g, ' '))
      const audio = new Audio(URL.createObjectURL(file))
      audio.addEventListener('loadedmetadata', () => {
        setUploadFileDuration(audio.duration)
      })
      audio.addEventListener('error', () => {
        setUploadFileDuration(0)
      })
    } else {
      setUploadVoiceName('')
    }
  }

  async function handleUploadCustomVoice(): Promise<void> {
    if (!uploadFile || !uploadVoiceName.trim()) return
    setUploading(true)
    setUploadResult(null)
    try {
      // Write file to a temp path via the file system API is unavailable in the
      // renderer; instead we use a blob URL and pass the file path from the
      // webkitRelativePath or name. For Electron, we read the file as an
      // ArrayBuffer and write to a temp path via the main process.
      // Simpler approach: use the native file path exposed by Electron.
      const nativePath: string = (uploadFile as File & { path?: string }).path ?? ''
      if (!nativePath) {
        setUploadResult({ ok: false, message: 'Cannot read file path. Try again.' })
        setUploading(false)
        return
      }
      const res = await window.rex.uploadCustomVoice(nativePath, uploadVoiceName.trim())
      if (res.ok) {
        setUploadResult({ ok: true, message: `Voice "${res.voice_name}" saved successfully.` })
        setUploadFile(null)
        setUploadVoiceName('')
        setUploadFileDuration(null)
        // Refresh voice list so the new voice appears in the dropdown.
        if (form.ttsEngine === 'xtts') {
          loadVoices('xtts')
        }
      } else {
        setUploadResult({ ok: false, message: res.error ?? 'Upload failed.' })
      }
    } catch (err) {
      setUploadResult({ ok: false, message: String(err) })
    } finally {
      setUploading(false)
    }
  }

  async function handleStartEnrollment(): Promise<void> {
    setEnrolling(true)
    setEnrollmentCountdown(0)
    setCapturedSamples(0)
    setEnrollmentMessage(null)
    setEnrollmentError(null)

    let stream: MediaStream | null = null

    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: form.microphoneDeviceId
          ? { deviceId: { exact: form.microphoneDeviceId } }
          : true
      })

      const samples: number[][] = []
      for (let index = 0; index < ENROLLMENT_SAMPLE_TARGET; index += 1) {
        await runEnrollmentCountdown(setEnrollmentCountdown)
        const sample = await captureEnrollmentSample(stream)
        samples.push(sample)
        setCapturedSamples(index + 1)
        setEnrollmentMessage(`Captured sample ${index + 1} of ${ENROLLMENT_SAMPLE_TARGET}.`)
        await sleep(250)
      }

      const result = await window.rex.enrollVoice(activeUserId, samples)
      if (!result.ok) {
        throw new Error(result.error ?? 'Voice enrollment failed')
      }

      setEnrollmentMessage(`Voice enrollment saved for ${activeUserId}.`)
      addToast('Voice enrollment saved', 'success')
      loadEnrollmentState()
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Voice enrollment failed'
      setEnrollmentError(message)
      addToast(message, 'error')
    } finally {
      setEnrollmentCountdown(0)
      setEnrolling(false)
      stream?.getTracks().forEach((track) => track.stop())
    }
  }

  function handleDeleteEnrollment(userId: string): void {
    setDeletingUserId(userId)
    setEnrollmentError(null)
    setEnrollmentMessage(null)
    window.rex
      .deleteVoiceEnrollment(userId)
      .then((result) => {
        if (!result.ok) {
          throw new Error(result.error ?? 'Failed to delete voice enrollment')
        }
        setEnrollmentMessage(`Deleted enrollment for ${userId}.`)
        addToast('Voice enrollment deleted', 'success')
        loadEnrollmentState()
      })
      .catch((error: unknown) => {
        const message =
          error instanceof Error ? error.message : 'Failed to delete voice enrollment'
        setEnrollmentError(message)
        addToast(message, 'error')
      })
      .finally(() => {
        setDeletingUserId(null)
      })
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

      {/* STT section */}
      <div className="mb-6 rounded-xl border border-border bg-surface-raised/40 p-4">
        <h3 className="mb-4 text-sm font-semibold text-text-primary">Speech-to-Text (Whisper)</h3>
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="sttModel" className="text-sm font-medium text-text-primary">
              Model Size
            </label>
            <SavedIndicator visible={savedField === 'sttModel'} />
          </div>
          <select
            id="sttModel"
            value={form.sttModel}
            onChange={(e) => handleFieldChange('sttModel', e.target.value)}
            className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="tiny">tiny — fastest, lowest accuracy</option>
            <option value="base">base — good balance</option>
            <option value="small">small</option>
            <option value="medium">medium</option>
            <option value="large">large</option>
            <option value="large-v2">large-v2</option>
            <option value="large-v3">large-v3 — best accuracy</option>
          </select>
        </div>
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="sttLanguage" className="text-sm font-medium text-text-primary">
              Language
            </label>
            <SavedIndicator visible={savedField === 'sttLanguage'} />
          </div>
          <select
            id="sttLanguage"
            value={form.sttLanguage}
            onChange={(e) => handleFieldChange('sttLanguage', e.target.value)}
            className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="auto">Auto-detect</option>
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
            <option value="it">Italian</option>
            <option value="pt">Portuguese</option>
            <option value="zh">Chinese</option>
            <option value="ja">Japanese</option>
            <option value="ko">Korean</option>
            <option value="ar">Arabic</option>
            <option value="ru">Russian</option>
            <option value="nl">Dutch</option>
            <option value="pl">Polish</option>
            <option value="tr">Turkish</option>
          </select>
        </div>
        <div>
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="sttDevice" className="text-sm font-medium text-text-primary">
              Device
            </label>
            <SavedIndicator visible={savedField === 'sttDevice'} />
          </div>
          <select
            id="sttDevice"
            value={form.sttDevice}
            onChange={(e) => handleFieldChange('sttDevice', e.target.value as VoiceSettings['sttDevice'])}
            className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="auto">Auto (prefer GPU)</option>
            <option value="cpu">CPU</option>
            <option value="cuda">GPU (CUDA)</option>
          </select>
        </div>
      </div>

      {/* Wake word */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="wakeWord" className="text-sm font-medium text-text-primary">
            Wake Word
          </label>
          <SavedIndicator visible={savedField === 'wakeWord'} />
        </div>
        <div className="flex items-center gap-2">
          <select
            id="wakeWord"
            value={form.wakeWord}
            onChange={(e) => handleFieldChange('wakeWord', e.target.value)}
            className="flex-1 bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <option value="">Disabled</option>
            {wakeWords.length > 0
              ? wakeWords.map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.name} [{w.engine}]
                  </option>
                ))
              : /* fallback hardcoded list when bridge hasn't loaded yet */
                [
                  { id: 'hey_jarvis', name: 'Hey Jarvis' },
                  { id: 'hey_mycroft', name: 'Hey Mycroft' },
                  { id: 'hey_rhasspy', name: 'Hey Rhasspy' },
                  { id: 'ok_nabu', name: 'OK Nabu' },
                  { id: 'alexa', name: 'Alexa' },
                ].map((w) => (
                  <option key={w.id} value={w.id}>
                    {w.name}
                  </option>
                ))}
          </select>
          <button
            onClick={handlePreviewWakeWord}
            disabled={previewingWakeWord || !form.wakeWord}
            title="Play a sample of this wake word"
            className="flex items-center gap-1.5 bg-surface-raised hover:bg-surface border border-border disabled:opacity-40 text-text-primary text-sm font-medium px-3 py-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg shrink-0"
          >
            {previewingWakeWord ? (
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
            ) : (
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="5 3 19 12 5 21 5 3" />
              </svg>
            )}
            Sample
          </button>
        </div>
        <p className="mt-1 text-xs text-text-secondary">
          Uses openWakeWord. Select a model or leave disabled to start Rex manually.
          Changes take effect when the voice loop restarts.
        </p>
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
          <option value="pyttsx3">pyttsx3 (offline, system voices)</option>
          <option value="edge-tts">edge-tts (Microsoft, requires internet)</option>
          <option value="xtts">XTTS (Coqui, voice cloning)</option>
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
                  {v.name}{v.language ? ` (${v.language})` : ''}{v.engine ? ` [${v.engine}]` : ''}
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

      {/* Custom Voice Upload (XTTS only) */}
      {form.ttsEngine === 'xtts' && (
        <div className="mb-5 p-4 bg-surface-raised border border-border rounded-lg">
          <p className="text-sm font-medium text-text-primary mb-3">Upload Custom Voice (XTTS)</p>
          <p className="text-xs text-text-secondary mb-3">
            Upload a WAV or MP3 recording (minimum 10 seconds) to create a custom speaker voice.
          </p>
          <div className="space-y-3">
            <input
              type="file"
              accept=".wav,.mp3"
              onChange={handleUploadFileChange}
              className="block w-full text-sm text-text-secondary file:mr-3 file:py-1.5 file:px-3 file:rounded file:border-0 file:text-xs file:font-medium file:bg-accent file:text-white hover:file:bg-accent/80 cursor-pointer"
            />
            {uploadFile && uploadFileDuration !== null && (
              <div className="text-xs">
                {uploadFileDuration >= 10 ? (
                  <span className="text-green-500">{uploadFileDuration.toFixed(1)}s — ready</span>
                ) : (
                  <span className="text-amber-500">
                    {uploadFileDuration.toFixed(1)}s — need {(10 - uploadFileDuration).toFixed(1)}s more
                  </span>
                )}
              </div>
            )}
            {uploadFile && (
              <input
                type="text"
                value={uploadVoiceName}
                placeholder="Voice name"
                onChange={(e) => setUploadVoiceName(e.target.value)}
                className="w-full bg-surface border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
              />
            )}
            {uploadFile && (
              <button
                onClick={handleUploadCustomVoice}
                disabled={uploading || !uploadVoiceName.trim() || (uploadFileDuration !== null && uploadFileDuration < 10)}
                className="flex items-center gap-2 bg-accent hover:bg-accent/80 disabled:opacity-40 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent"
              >
                {uploading ? (
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                  </svg>
                ) : null}
                {uploading ? 'Saving…' : 'Create Voice'}
              </button>
            )}
            {uploadResult && (
              <p className={`text-xs ${uploadResult.ok ? 'text-green-500' : 'text-red-400'}`}>
                {uploadResult.message}
              </p>
            )}
          </div>
        </div>
      )}

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

      <div className="mt-8 border-t border-border pt-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">Enroll Voice</h3>
            <p className="mt-1 text-sm text-text-secondary">
              Record three short samples for the active user so Rex can recognize your voice.
            </p>
            <p className="mt-1 text-xs text-text-secondary">
              Active user: <span className="font-medium text-text-primary">{activeUserId}</span>
            </p>
          </div>
          <button
            onClick={() => {
              void handleStartEnrollment()
            }}
            disabled={enrolling}
            className="flex items-center gap-2 bg-accent hover:bg-accent/90 disabled:opacity-50 text-white text-sm font-medium px-4 py-2 rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg"
          >
            {enrolling ? (
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
                Recording...
              </>
            ) : (
              'Start Enrollment'
            )}
          </button>
        </div>

        <div className="mt-4 rounded-xl border border-border bg-surface-raised p-4">
          <div className="flex items-center justify-between text-sm text-text-primary">
            <span>Samples captured</span>
            <span>
              {capturedSamples}/{ENROLLMENT_SAMPLE_TARGET}
            </span>
          </div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-surface">
            <div
              className="h-full rounded-full bg-accent transition-all duration-300"
              style={{
                width: `${(capturedSamples / ENROLLMENT_SAMPLE_TARGET) * 100}%`
              }}
            />
          </div>
          <div className="mt-4 flex items-center justify-between text-sm">
            <span className="text-text-secondary">
              {enrollmentCountdown > 0
                ? `Sample ${Math.min(capturedSamples + 1, ENROLLMENT_SAMPLE_TARGET)} starts in`
                : enrolling
                  ? 'Recording now'
                  : 'Ready to record'}
            </span>
            <span className="text-2xl font-semibold text-text-primary tabular-nums">
              {enrollmentCountdown > 0 ? enrollmentCountdown : enrolling ? 'REC' : '--'}
            </span>
          </div>
          {enrollmentMessage && (
            <p className="mt-3 text-sm text-success">{enrollmentMessage}</p>
          )}
          {enrollmentError && (
            <p className="mt-3 text-sm text-danger">{enrollmentError}</p>
          )}
        </div>

        <div className="mt-6">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-semibold text-text-primary">Enrolled Users</h4>
            <span className="text-xs text-text-secondary">{enrollments.length} total</span>
          </div>
          <div className="mt-3 space-y-3">
            {enrollments.length === 0 ? (
              <div className="rounded-xl border border-dashed border-border bg-surface-raised px-4 py-5 text-sm text-text-secondary">
                No voice enrollments yet.
              </div>
            ) : (
              enrollments.map((enrollment) => (
                <div
                  key={enrollment.user_id}
                  className="flex items-center justify-between gap-4 rounded-xl border border-border bg-surface-raised px-4 py-3"
                >
                  <div>
                    <div className="text-sm font-medium text-text-primary">
                      {enrollment.user_id}
                    </div>
                    <div className="mt-1 text-xs text-text-secondary">
                      {enrollment.sample_count} samples, {enrollment.model_id}
                      {enrollment.updated_at ? `, updated ${new Date(enrollment.updated_at).toLocaleString()}` : ''}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteEnrollment(enrollment.user_id)}
                    disabled={deletingUserId === enrollment.user_id}
                    className="rounded-lg border border-danger/30 px-3 py-2 text-sm font-medium text-danger transition-colors hover:bg-danger/10 disabled:opacity-50"
                  >
                    {deletingUserId === enrollment.user_id ? 'Deleting...' : 'Delete Enrollment'}
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
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

const MODEL_ROUTING_FIELDS: Array<{
  key: keyof AiSettings['modelRouting']
  label: string
  placeholder: string
}> = [
  { key: 'default', label: 'Default', placeholder: 'gpt-4o' },
  { key: 'coding', label: 'Coding', placeholder: 'claude-sonnet-4' },
  { key: 'reasoning', label: 'Reasoning', placeholder: 'o3-mini' },
  { key: 'search', label: 'Search', placeholder: 'gpt-4o-mini' },
  { key: 'vision', label: 'Vision', placeholder: 'gpt-4o' },
  { key: 'fast', label: 'Fast', placeholder: 'llama3.2' }
]

type SavedField = keyof AiSettings | 'modelRouting'

function AiPanel(): React.ReactElement {
  const addToast = useToast()
  const [form, setForm] = useState<AiSettings>({
    model: 'claude-sonnet-4',
    provider: 'openai',
    customModelId: '',
    ollamaBaseUrl: 'http://localhost:11434',
    temperature: 0.7,
    maxTokens: 2048,
    systemPrompt: '',
    autonomyMode: 'manual',
    budgetPerPlan: 0,
    budgetPerStep: 0,
    modelRouting: {
      default: '',
      coding: '',
      reasoning: '',
      search: '',
      vision: '',
      fast: ''
    }
  })
  const [loading, setLoading] = useState(true)
  const [savedField, setSavedField] = useState<SavedField | null>(null)
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [suggestions, setSuggestions] = useState<PreferenceSuggestion[]>([])
  const [dismissedFields, setDismissedFields] = useState<Set<string>>(new Set())
  const [routingDirty, setRoutingDirty] = useState(false)
  const [savingRouting, setSavingRouting] = useState(false)
  const [apiKeySet, setApiKeySet] = useState(false)
  const [apiKeyValue, setApiKeyValue] = useState('')
  const [apiKeySaving, setApiKeySaving] = useState(false)

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
        const modelRouting =
          settings.modelRouting && typeof settings.modelRouting === 'object'
            ? (settings.modelRouting as Record<string, unknown>)
            : {}
        const rawProvider = settings.provider
        const provider: AiSettings['provider'] =
          rawProvider === 'openai' || rawProvider === 'ollama' || rawProvider === 'local'
            ? rawProvider
            : 'openai'
        setForm({
          model: (AI_MODELS.some((m) => m.value === settings.model)
            ? settings.model
            : 'claude-sonnet-4') as AiSettings['model'],
          provider,
          customModelId: typeof settings.customModelId === 'string' ? settings.customModelId : '',
          ollamaBaseUrl: typeof settings.ollamaBaseUrl === 'string' ? settings.ollamaBaseUrl : 'http://localhost:11434',
          temperature: typeof settings.temperature === 'number' ? settings.temperature : 0.7,
          maxTokens: typeof settings.maxTokens === 'number' ? settings.maxTokens : 2048,
          systemPrompt: typeof settings.systemPrompt === 'string' ? settings.systemPrompt : '',
          autonomyMode:
            settings.autonomyMode === 'supervised' || settings.autonomyMode === 'full-auto'
              ? settings.autonomyMode
              : 'manual',
          budgetPerPlan: typeof settings.budgetPerPlan === 'number' ? settings.budgetPerPlan : 0,
          budgetPerStep: typeof settings.budgetPerStep === 'number' ? settings.budgetPerStep : 0,
          modelRouting: {
            default: typeof modelRouting.default === 'string' ? modelRouting.default : '',
            coding: typeof modelRouting.coding === 'string' ? modelRouting.coding : '',
            reasoning: typeof modelRouting.reasoning === 'string' ? modelRouting.reasoning : '',
            search: typeof modelRouting.search === 'string' ? modelRouting.search : '',
            vision: typeof modelRouting.vision === 'string' ? modelRouting.vision : '',
            fast: typeof modelRouting.fast === 'string' ? modelRouting.fast : ''
          }
        })
        setRoutingDirty(false)
      })
      .catch(() => {
        addToast('Failed to load AI settings', 'error')
      })
      .finally(() => setLoading(false))

    window.rex
      .getApiKeys()
      .then((keys) => setApiKeySet(keys.openai_key_set))
      .catch(() => {
        // Non-fatal — API key status will show as unset
      })

    loadSuggestions()
  }, [addToast])

  function showSaved(field: SavedField): void {
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

  function handleRoutingChange(
    field: keyof AiSettings['modelRouting'],
    value: string
  ): void {
    setForm((current) => ({
      ...current,
      modelRouting: {
        ...current.modelRouting,
        [field]: value
      }
    }))
    setRoutingDirty(true)
  }

  function handleSaveRouting(): void {
    const updated = {
      ...form,
      modelRouting: { ...form.modelRouting }
    }
    setSavingRouting(true)
    window.rex
      .setSettings('ai', updated as unknown as Settings)
      .then(() => {
        setForm(updated)
        setRoutingDirty(false)
        showSaved('modelRouting')
      })
      .catch(() => {
        addToast('Failed to save model routing', 'error')
      })
      .finally(() => setSavingRouting(false))
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

  function handleSaveApiKey(): void {
    if (!apiKeyValue.trim()) return
    setApiKeySaving(true)
    window.rex
      .setApiKey('OPENAI_API_KEY', apiKeyValue.trim())
      .then((result) => {
        if (result.ok) {
          setApiKeySet(true)
          setApiKeyValue('')
          addToast('API key saved', 'success')
        } else {
          addToast(result.error ?? 'Failed to save API key', 'error')
        }
      })
      .catch(() => {
        addToast('Failed to save API key', 'error')
      })
      .finally(() => setApiKeySaving(false))
  }

  const activeSuggestion = suggestions.find((s) => !dismissedFields.has(s.field)) ?? null

  if (loading) {
    return <PageLoadingFallback lines={5} />
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">AI</h2>

      {/* LLM Provider */}
      <div className="mb-5">
        <div className="flex items-center justify-between mb-1.5">
          <label htmlFor="llmProvider" className="text-sm font-medium text-text-primary">
            LLM Provider
          </label>
          <SavedIndicator visible={savedField === 'provider'} />
        </div>
        <select
          id="llmProvider"
          value={form.provider}
          onChange={(e) => handleFieldChange('provider', e.target.value as AiSettings['provider'])}
          className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
        >
          <option value="openai">OpenAI</option>
          <option value="ollama">Ollama (local)</option>
          <option value="local">Local Transformers</option>
        </select>
      </div>

      {/* OpenAI: model dropdown + API key */}
      {form.provider === 'openai' && (
        <>
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

          {/* OpenAI API Key */}
          <div className="mb-5">
            <div className="flex items-center justify-between mb-1.5">
              <label htmlFor="openaiApiKey" className="text-sm font-medium text-text-primary">
                OpenAI API Key
              </label>
              {apiKeySet && (
                <span className="text-xs text-success font-medium">Key set</span>
              )}
            </div>
            <div className="flex gap-2">
              <div className="flex-1">
                <PasswordInput
                  id="openaiApiKey"
                  value={apiKeyValue}
                  placeholder={apiKeySet ? '••••••••••••••••' : 'sk-…'}
                  onChange={setApiKeyValue}
                />
              </div>
              <button
                type="button"
                onClick={handleSaveApiKey}
                disabled={apiKeySaving || !apiKeyValue.trim()}
                className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-50 shrink-0"
              >
                {apiKeySaving ? 'Saving…' : 'Save'}
              </button>
            </div>
            <p className="mt-1 text-xs text-text-secondary">
              Saved to .env at the repo root. Never stored in gui_settings.json.
            </p>
          </div>
        </>
      )}

      {/* Ollama: base URL + model name */}
      {form.provider === 'ollama' && (
        <>
          <div className="mb-5">
            <div className="flex items-center justify-between mb-1.5">
              <label htmlFor="ollamaBaseUrl" className="text-sm font-medium text-text-primary">
                Ollama Base URL
              </label>
              <SavedIndicator visible={savedField === 'ollamaBaseUrl'} />
            </div>
            <input
              id="ollamaBaseUrl"
              type="text"
              value={form.ollamaBaseUrl}
              placeholder="http://localhost:11434"
              onChange={(e) => setForm((f) => ({ ...f, ollamaBaseUrl: e.target.value }))}
              onBlur={(e) => handleFieldChange('ollamaBaseUrl', e.target.value)}
              className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
            />
          </div>
          <div className="mb-5">
            <div className="flex items-center justify-between mb-1.5">
              <label htmlFor="customModelId" className="text-sm font-medium text-text-primary">
                Model Name / Tag
              </label>
              <SavedIndicator visible={savedField === 'customModelId'} />
            </div>
            <input
              id="customModelId"
              type="text"
              value={form.customModelId}
              placeholder="e.g. llama3:8b"
              onChange={(e) => setForm((f) => ({ ...f, customModelId: e.target.value }))}
              onBlur={(e) => handleFieldChange('customModelId', e.target.value)}
              className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
            />
          </div>
        </>
      )}

      {/* Local Transformers: model path */}
      {form.provider === 'local' && (
        <div className="mb-5">
          <div className="flex items-center justify-between mb-1.5">
            <label htmlFor="customModelId" className="text-sm font-medium text-text-primary">
              Model Name or Path
            </label>
            <SavedIndicator visible={savedField === 'customModelId'} />
          </div>
          <input
            id="customModelId"
            type="text"
            value={form.customModelId}
            placeholder="e.g. mistralai/Mistral-7B-Instruct-v0.3"
            onChange={(e) => setForm((f) => ({ ...f, customModelId: e.target.value }))}
            onBlur={(e) => handleFieldChange('customModelId', e.target.value)}
            className="w-full bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
          />
        </div>
      )}

      <div className="mb-6 rounded-xl border border-border bg-surface-raised/40 p-4">
        <div className="mb-4 flex items-start justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">Model Routing</h3>
            <p className="mt-1 text-xs text-text-secondary">
              Override the model used for each task category. Leave a field blank to fall back to Rex’s default routing.
            </p>
          </div>
          <SavedIndicator visible={savedField === 'modelRouting'} />
        </div>
        <div className="grid gap-4 md:grid-cols-2">
          {MODEL_ROUTING_FIELDS.map((field) => (
            <div key={field.key}>
              <label
                htmlFor={`model-routing-${field.key}`}
                className="mb-1.5 block text-sm font-medium text-text-primary"
              >
                {field.label}
              </label>
              <input
                id={`model-routing-${field.key}`}
                type="text"
                value={form.modelRouting[field.key]}
                placeholder={field.placeholder}
                onChange={(e) => handleRoutingChange(field.key, e.target.value)}
                className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
              />
            </div>
          ))}
        </div>
        <div className="mt-4 flex items-center justify-between gap-3">
          <p className="text-xs text-text-secondary">
            Supported values include OpenAI model IDs, Claude model names, or local model identifiers such as Ollama tags.
          </p>
          <button
            type="button"
            onClick={handleSaveRouting}
            disabled={!routingDirty || savingRouting}
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent/90 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {savingRouting ? 'Saving…' : 'Save Routing'}
          </button>
        </div>
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

type IntegrationSection = 'email' | 'calendar' | 'sms' | 'homeassistant'
type TestStatus = 'idle' | 'testing' | 'ok' | 'error'

function ConnectionBadge({
  status,
  hasCredentials
}: {
  status: TestStatus
  hasCredentials: boolean
}): React.ReactElement {
  if (status === 'ok') {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-success/15 px-2 py-0.5 text-xs font-medium text-success">
        <span className="h-1.5 w-1.5 rounded-full bg-success" />
        Connected
      </span>
    )
  }
  if (status === 'error') {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-danger/15 px-2 py-0.5 text-xs font-medium text-danger">
        <span className="h-1.5 w-1.5 rounded-full bg-danger" />
        Error
      </span>
    )
  }
  if (!hasCredentials) {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-border px-2 py-0.5 text-xs font-medium text-text-secondary">
        <span className="h-1.5 w-1.5 rounded-full bg-text-secondary" />
        Not Configured
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-warning/15 px-2 py-0.5 text-xs font-medium text-warning">
      <span className="h-1.5 w-1.5 rounded-full bg-warning" />
      Untested
    </span>
  )
}

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
    emailAccounts: [],
    calendarProvider: 'gmail',
    calendarClientId: '',
    calendarClientSecret: '',
    smsSid: '',
    smsAuthToken: '',
    smsFromNumber: '',
    haUrl: '',
    haToken: ''
  })
  const [loading, setLoading] = useState(true)
  const [savedField, setSavedField] = useState<keyof IntegrationsSettings | null>(null)
  const [testStatus, setTestStatus] = useState<Record<IntegrationSection, TestStatus>>({
    email: 'idle',
    calendar: 'idle',
    sms: 'idle',
    homeassistant: 'idle'
  })
  const savedTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const testTimers = useRef<Partial<Record<IntegrationSection, ReturnType<typeof setTimeout>>>>({})
  const [accountTestStatus, setAccountTestStatus] = useState<Record<string, TestStatus>>({})
  const accountTestTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})

  function handleTestEmailAccount(id: string): void {
    setAccountTestStatus((s) => ({ ...s, [id]: 'testing' }))
    window.rex
      .testEmailAccount(id)
      .then((res) => {
        setAccountTestStatus((s) => ({ ...s, [id]: res.ok ? 'ok' : 'error' }))
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

  useEffect(() => {
    window.rex
      .getSettings('integrations')
      .then((settings: Settings) => {
        const rawAccounts = settings.emailAccounts
        const emailAccounts: EmailAccount[] = Array.isArray(rawAccounts)
          ? (rawAccounts as EmailAccount[])
              .filter((a) => typeof a === 'object' && a !== null && typeof a.id === 'string')
              .map((a) => ({
                id: a.id,
                backend: a.backend ?? (a as unknown as { provider?: string }).provider ?? 'gmail',
                displayName: a.displayName ?? '',
                clientId: a.clientId ?? '',
                clientSecret: a.clientSecret ?? '',
                host: a.host ?? '',
                port: typeof a.port === 'number' ? a.port : 993,
                username: a.username ?? '',
                password: a.password ?? '',
                lastSynced: a.lastSynced
              } as EmailAccount))
          : []
        setForm({
          emailProvider:
            settings.emailProvider === 'outlook' ? 'outlook' : 'gmail',
          emailClientId: typeof settings.emailClientId === 'string' ? settings.emailClientId : '',
          emailClientSecret:
            typeof settings.emailClientSecret === 'string' ? settings.emailClientSecret : '',
          emailAccounts,
          calendarProvider:
            settings.calendarProvider === 'outlook' ? 'outlook' : 'gmail',
          calendarClientId:
            typeof settings.calendarClientId === 'string' ? settings.calendarClientId : '',
          calendarClientSecret:
            typeof settings.calendarClientSecret === 'string' ? settings.calendarClientSecret : '',
          smsSid: typeof settings.smsSid === 'string' ? settings.smsSid : '',
          smsAuthToken: typeof settings.smsAuthToken === 'string' ? settings.smsAuthToken : '',
          smsFromNumber: typeof settings.smsFromNumber === 'string' ? settings.smsFromNumber : '',
          haUrl: typeof settings.haUrl === 'string' ? settings.haUrl : '',
          haToken: typeof settings.haToken === 'string' ? settings.haToken : ''
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
    window.rex.setSettings('integrations', updated as unknown as Settings).catch(() => {
      addToast('Failed to save email account', 'error')
    })
  }

  function handleUpdateEmailAccount(id: string, patch: Partial<EmailAccount>): void {
    const updated = {
      ...form,
      emailAccounts: form.emailAccounts.map((a) => (a.id === id ? { ...a, ...patch } : a))
    }
    setForm(updated)
    window.rex.setSettings('integrations', updated as unknown as Settings).catch(() => {
      addToast('Failed to save email account', 'error')
    })
  }

  function handleRemoveEmailAccount(id: string): void {
    const updated = {
      ...form,
      emailAccounts: form.emailAccounts.filter((a) => a.id !== id)
    }
    setForm(updated)
    window.rex.setSettings('integrations', updated as unknown as Settings).catch(() => {
      addToast('Failed to remove email account', 'error')
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
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
              <polyline points="22,6 12,13 2,6" />
            </svg>
            Email
          </h3>
          <ConnectionBadge
            status={testStatus.email}
            hasCredentials={form.emailClientId.trim() !== '' || form.emailAccounts.length > 0}
          />
        </div>

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

        {/* Multi-account email list */}
        <div className="mt-5">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-text-primary">Email Accounts</span>
            <button
              type="button"
              onClick={handleAddEmailAccount}
              className="flex items-center gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-border focus:outline-none focus:ring-2 focus:ring-accent"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
              Add Account
            </button>
          </div>
          {form.emailAccounts.length === 0 ? (
            <div className="rounded-xl border border-dashed border-border bg-surface-raised/40 px-4 py-4 text-sm text-text-secondary">
              No additional accounts. Click "Add Account" to connect another inbox.
            </div>
          ) : (
            <div className="space-y-3">
              {form.emailAccounts.map((account) => {
                const acctTestStatus = accountTestStatus[account.id] ?? 'idle'
                return (
                  <div key={account.id} className="rounded-xl border border-border bg-surface-raised p-4">
                    {/* Header row: display name + remove */}
                    <div className="mb-3 flex items-center justify-between gap-2">
                      <input
                        type="text"
                        value={account.displayName}
                        placeholder="Account label (e.g. Work Gmail)"
                        onChange={(e) =>
                          handleUpdateEmailAccount(account.id, { displayName: e.target.value })
                        }
                        className="flex-1 bg-bg border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
                      />
                      <button
                        type="button"
                        onClick={() => handleRemoveEmailAccount(account.id)}
                        className="rounded-lg border border-danger/30 px-2.5 py-1.5 text-xs font-medium text-danger transition-colors hover:bg-danger/10 focus:outline-none"
                      >
                        Remove
                      </button>
                    </div>

                    {/* Backend selector */}
                    <div className="mb-2">
                      <select
                        value={account.backend}
                        onChange={(e) =>
                          handleUpdateEmailAccount(account.id, {
                            backend: e.target.value as EmailAccount['backend']
                          })
                        }
                        className="w-full bg-bg border border-border rounded-lg px-3 py-1.5 text-sm text-text-primary focus:outline-none focus:ring-2 focus:ring-accent"
                      >
                        <option value="gmail">Gmail OAuth</option>
                        <option value="outlook">Outlook OAuth</option>
                        <option value="imap">IMAP</option>
                      </select>
                    </div>

                    {/* Credential fields */}
                    {account.backend === 'imap' ? (
                      <div className="space-y-2 mb-2">
                        <input
                          type="text"
                          value={account.host}
                          placeholder="IMAP host (e.g. imap.gmail.com)"
                          onChange={(e) =>
                            handleUpdateEmailAccount(account.id, { host: e.target.value })
                          }
                          className={inputClass}
                        />
                        <input
                          type="number"
                          value={account.port}
                          placeholder="Port (993)"
                          onChange={(e) =>
                            handleUpdateEmailAccount(account.id, { port: parseInt(e.target.value, 10) || 993 })
                          }
                          className={inputClass}
                        />
                        <input
                          type="text"
                          value={account.username}
                          placeholder="Username / email address"
                          onChange={(e) =>
                            handleUpdateEmailAccount(account.id, { username: e.target.value })
                          }
                          className={inputClass}
                        />
                        <PasswordInput
                          id={`imap-pass-${account.id}`}
                          value={account.password}
                          placeholder="Password or app password"
                          onChange={(v) => handleUpdateEmailAccount(account.id, { password: v })}
                        />
                      </div>
                    ) : (
                      <div className="space-y-2 mb-2">
                        <input
                          type="text"
                          value={account.clientId}
                          placeholder="OAuth Client ID"
                          onChange={(e) =>
                            handleUpdateEmailAccount(account.id, { clientId: e.target.value })
                          }
                          className={inputClass}
                        />
                        <PasswordInput
                          id={`email-secret-${account.id}`}
                          value={account.clientSecret}
                          placeholder="OAuth Client Secret"
                          onChange={(v) => handleUpdateEmailAccount(account.id, { clientSecret: v })}
                        />
                      </div>
                    )}

                    {/* Test connection + last-synced */}
                    <div className="flex items-center gap-3">
                      <button
                        type="button"
                        onClick={() => handleTestEmailAccount(account.id)}
                        disabled={acctTestStatus === 'testing'}
                        className="flex items-center gap-1.5 rounded-lg border border-border bg-surface-raised px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-border focus:outline-none focus:ring-2 focus:ring-accent disabled:opacity-50"
                      >
                        {acctTestStatus === 'testing' ? 'Testing…' : 'Test Connection'}
                      </button>
                      {acctTestStatus === 'ok' && (
                        <span className="text-xs font-medium text-success">Connected</span>
                      )}
                      {acctTestStatus === 'error' && (
                        <span className="text-xs font-medium text-danger">Failed</span>
                      )}
                      {account.lastSynced && (
                        <span className="ml-auto text-xs text-text-secondary">
                          Synced {new Date(account.lastSynced).toLocaleString()}
                        </span>
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
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
          <ConnectionBadge
            status={testStatus.calendar}
            hasCredentials={form.calendarClientId.trim() !== ''}
          />
        </div>

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
      <section className="mb-7">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h3 className="text-sm font-semibold text-text-primary flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
            </svg>
            SMS (Twilio)
          </h3>
          <ConnectionBadge
            status={testStatus.sms}
            hasCredentials={form.smsSid.trim() !== ''}
          />
        </div>

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
          <ConnectionBadge
            status={testStatus.homeassistant}
            hasCredentials={form.haUrl.trim() !== ''}
          />
        </div>

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
            onChange={(e) => setForm((f) => ({ ...f, haUrl: e.target.value }))}
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
            placeholder="Enter access token"
            onChange={(v) => setForm((f) => ({ ...f, haToken: v }))}
            onBlur={() => handleFieldChange('haToken', form.haToken)}
          />
        </div>

        <TestConnectionButton status={testStatus.homeassistant} onTest={() => handleTest('homeassistant')} />
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

interface AudioDevice {
  deviceId: string
  label: string
}

function AudioOutputPanel(): React.ReactElement {
  const addToast = useToast()
  const [speakers, setSpeakers] = useState<AudioDevice[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [volume, setVolume] = useState(1.0)
  const [testing, setTesting] = useState<string | null>(null) // deviceId being tested
  const [savedVolume, setSavedVolume] = useState(false)
  const volumeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [smartSpeakers, setSmartSpeakers] = useState<SmartSpeaker[]>([])
  const [loadingSmartSpeakers, setLoadingSmartSpeakers] = useState(false)

  function loadSmartSpeakers(): void {
    setLoadingSmartSpeakers(true)
    window.rex
      .getSmartSpeakers()
      .then((res) => {
        setSmartSpeakers(res.speakers ?? [])
        if (!res.ok && res.error) addToast(`Smart speaker discovery: ${res.error}`, 'error')
      })
      .catch(() => {
        setSmartSpeakers([])
      })
      .finally(() => setLoadingSmartSpeakers(false))
  }

  useEffect(() => {
    // Load saved voice settings to get current speaker + volume
    window.rex
      .getSettings('voice')
      .then((s: Settings) => {
        if (typeof s.speakerDeviceId === 'string') setSelectedDeviceId(s.speakerDeviceId)
        if (typeof s.volume === 'number') setVolume(s.volume)
      })
      .catch(() => {
        // Non-fatal
      })

    // Enumerate output devices
    if (navigator.mediaDevices?.enumerateDevices) {
      navigator.mediaDevices
        .enumerateDevices()
        .then((devices) => {
          const list = devices
            .filter((d) => d.kind === 'audiooutput')
            .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Speaker ${i + 1}` }))
          setSpeakers(list)
        })
        .catch(() => {
          setSpeakers([])
        })
    }

    // Discover smart speakers
    loadSmartSpeakers()
  }, [])

  function handleSelectDevice(deviceId: string): void {
    setSelectedDeviceId(deviceId)
    window.rex
      .getSettings('voice')
      .then((s: Settings) => {
        return window.rex.setSettings('voice', { ...s, speakerDeviceId: deviceId } as Settings)
      })
      .catch(() => {
        addToast('Failed to save speaker selection', 'error')
      })
  }

  function handleVolumeChange(v: number): void {
    setVolume(v)
    if (volumeTimerRef.current) clearTimeout(volumeTimerRef.current)
    volumeTimerRef.current = setTimeout(() => {
      window.rex
        .getSettings('voice')
        .then((s: Settings) => window.rex.setSettings('voice', { ...s, volume: v } as Settings))
        .then(() => {
          setSavedVolume(true)
          setTimeout(() => setSavedVolume(false), 2000)
        })
        .catch(() => {
          addToast('Failed to save volume', 'error')
        })
    }, 400)
  }

  function handleTestDevice(deviceId: string): void {
    setTesting(deviceId)
    window.rex
      .testVoice({ speakerDeviceId: deviceId } as unknown as VoiceSettings)
      .then((res) => {
        if (!res.ok) addToast(res.error ?? 'Test failed', 'error')
      })
      .catch(() => {
        addToast('Test failed', 'error')
      })
      .finally(() => setTesting(null))
  }

  const displayDevices: AudioDevice[] =
    speakers.length > 0
      ? speakers
      : [{ deviceId: '', label: 'System Default' }]

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">Audio Output</h2>

      {/* System output devices */}
      <div className="mb-6">
        <h3 className="mb-4 text-sm font-semibold text-text-primary">System Speakers</h3>
        <div className="space-y-3">
          {displayDevices.map((device) => {
            const isSelected = device.deviceId === selectedDeviceId || (selectedDeviceId === '' && device.deviceId === '')
            return (
              <div
                key={device.deviceId || 'default'}
                className={[
                  'flex items-center gap-3 rounded-xl border p-4 transition-colors',
                  isSelected ? 'border-accent bg-accent/5' : 'border-border bg-surface-raised'
                ].join(' ')}
              >
                <button
                  type="button"
                  onClick={() => handleSelectDevice(device.deviceId)}
                  className={[
                    'h-4 w-4 shrink-0 rounded-full border-2 transition-colors focus:outline-none',
                    isSelected ? 'border-accent bg-accent' : 'border-border bg-bg'
                  ].join(' ')}
                  aria-label={`Select ${device.label}`}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-text-primary truncate">{device.label}</div>
                  {isSelected && (
                    <div className="text-xs text-accent mt-0.5">Default TTS output</div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => handleTestDevice(device.deviceId)}
                  disabled={testing === device.deviceId}
                  className="flex items-center gap-1.5 rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
                >
                  {testing === device.deviceId ? (
                    <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                  )}
                  Test
                </button>
              </div>
            )
          })}
        </div>
      </div>

      {/* Volume */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-text-primary">
            TTS Volume
            <span className="ml-2 text-xs text-text-secondary font-normal">
              {Math.round(volume * 100)}%
            </span>
          </label>
          {savedVolume && (
            <span className="text-xs text-success flex items-center gap-1">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              Saved
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>0%</span>
          <input
            type="range"
            min={0}
            max={1.0}
            step={0.05}
            value={volume}
            onChange={(e) => handleVolumeChange(parseFloat(e.target.value))}
            className="flex-1 accent-accent"
          />
          <span>100%</span>
        </div>
      </div>

      {/* Smart speakers */}
      <div className="border-t border-border pt-6">
        <div className="flex items-center justify-between mb-1">
          <h3 className="text-sm font-semibold text-text-primary">Smart Speakers</h3>
          <button
            type="button"
            onClick={loadSmartSpeakers}
            disabled={loadingSmartSpeakers}
            className="flex items-center gap-1 rounded-lg border border-border bg-bg px-2.5 py-1 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
          >
            {loadingSmartSpeakers ? (
              <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="23 4 23 10 17 10" />
                <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
              </svg>
            )}
            Refresh
          </button>
        </div>
        <p className="mb-4 text-xs text-text-secondary">
          Sonos and Bose devices discovered on the local network.
        </p>
        {smartSpeakers.length === 0 ? (
          <div className="rounded-xl border border-dashed border-border bg-surface-raised/40 px-4 py-6 text-center">
            <p className="text-sm text-text-secondary">No smart speakers discovered on the network.</p>
            <p className="mt-1 text-xs text-text-secondary">
              Ensure your Sonos or Bose device is on the same network and click Refresh.
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            {smartSpeakers.map((ss) => (
              <div
                key={`${ss.provider}-${ss.ip}`}
                className="flex items-center gap-3 rounded-xl border border-border bg-surface-raised p-4"
              >
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-text-primary truncate">{ss.name}</div>
                  <div className="text-xs text-text-secondary mt-0.5 capitalize">
                    {ss.provider} · {ss.model} · {ss.ip}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => handleTestDevice(`smart:${ss.provider}:${ss.ip}`)}
                  disabled={testing === `smart:${ss.provider}:${ss.ip}`}
                  className="flex items-center gap-1.5 rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
                >
                  {testing === `smart:${ss.provider}:${ss.ip}` ? (
                    <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                    </svg>
                  ) : (
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                  )}
                  Test
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function UsersPanel(): React.ReactElement {
  const addToast = useToast()
  const [enrollments, setEnrollments] = useState<VoiceEnrollment[]>([])
  const [activeUserId, setActiveUserId] = useState('default')
  const [userNames, setUserNames] = useState<Record<string, string>>({})
  const [editingName, setEditingName] = useState<string | null>(null)
  const [editNameValue, setEditNameValue] = useState('')
  const [selectedUserId, setSelectedUserId] = useState<string | null>(null)
  const [memories, setMemories] = useState<Memory[]>([])
  const [memoriesLoading, setMemoriesLoading] = useState(false)
  const [addingUser, setAddingUser] = useState(false)
  const [newUserId, setNewUserId] = useState('')
  const [enrollingUserId, setEnrollingUserId] = useState<string | null>(null)
  const [enrollmentCountdown, setEnrollmentCountdown] = useState(0)
  const [capturedSamples, setCapturedSamples] = useState(0)
  const [enrollmentMessage, setEnrollmentMessage] = useState<string | null>(null)
  const [enrollmentError, setEnrollmentError] = useState<string | null>(null)
  const [deletingUserId, setDeletingUserId] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  function loadEnrollments(): void {
    window.rex
      .getVoiceEnrollments()
      .then((result) => {
        if (!result.ok) return
        setActiveUserId(result.active_user_id || 'default')
        setEnrollments(result.enrollments ?? [])
      })
      .catch(() => {
        addToast('Failed to load voice enrollments', 'error')
      })
  }

  function loadUserNames(): void {
    window.rex
      .getSettings('users')
      .then((s: Settings) => {
        const names = s.names && typeof s.names === 'object' ? (s.names as Record<string, string>) : {}
        setUserNames(names)
      })
      .catch(() => {
        // Non-fatal; names fall back to user IDs
      })
  }

  useEffect(() => {
    Promise.all([
      new Promise<void>((resolve) => {
        loadEnrollments()
        resolve()
      }),
      new Promise<void>((resolve) => {
        loadUserNames()
        resolve()
      })
    ]).finally(() => setLoading(false))
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (selectedUserId === null) {
      setMemories([])
      return
    }
    setMemoriesLoading(true)
    window.rex
      .getMemories()
      .then((mems) => setMemories(mems))
      .catch(() => setMemories([]))
      .finally(() => setMemoriesLoading(false))
  }, [selectedUserId])

  function saveName(userId: string, name: string): void {
    const updated = { ...userNames, [userId]: name }
    setUserNames(updated)
    window.rex
      .setSettings('users', { names: updated } as Settings)
      .catch(() => {
        addToast('Failed to save user name', 'error')
      })
  }

  function handleStartEditName(userId: string): void {
    setEditingName(userId)
    setEditNameValue(userNames[userId] ?? userId)
  }

  function handleSaveName(userId: string): void {
    saveName(userId, editNameValue.trim() || userId)
    setEditingName(null)
  }

  async function handleEnroll(userId: string): Promise<void> {
    setEnrollingUserId(userId)
    setEnrollmentCountdown(0)
    setCapturedSamples(0)
    setEnrollmentMessage(null)
    setEnrollmentError(null)

    let stream: MediaStream | null = null
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const samples: number[][] = []
      for (let i = 0; i < ENROLLMENT_SAMPLE_TARGET; i++) {
        await runEnrollmentCountdown(setEnrollmentCountdown)
        const sample = await captureEnrollmentSample(stream)
        samples.push(sample)
        setCapturedSamples(i + 1)
        setEnrollmentMessage(`Captured sample ${i + 1} of ${ENROLLMENT_SAMPLE_TARGET}.`)
        await sleep(250)
      }
      const result = await window.rex.enrollVoice(userId, samples)
      if (!result.ok) throw new Error(result.error ?? 'Voice enrollment failed')
      setEnrollmentMessage(`Voice enrollment saved for ${userId}.`)
      addToast('Voice enrollment saved', 'success')
      loadEnrollments()
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Voice enrollment failed'
      setEnrollmentError(msg)
      addToast(msg, 'error')
    } finally {
      setEnrollingUserId(null)
      setEnrollmentCountdown(0)
      stream?.getTracks().forEach((t) => t.stop())
    }
  }

  function handleDelete(userId: string): void {
    setDeletingUserId(userId)
    window.rex
      .deleteVoiceEnrollment(userId)
      .then((result) => {
        if (!result.ok) throw new Error(result.error ?? 'Failed to delete')
        addToast('Voice enrollment deleted', 'success')
        if (selectedUserId === userId) setSelectedUserId(null)
        loadEnrollments()
      })
      .catch((err: unknown) => {
        addToast(err instanceof Error ? err.message : 'Failed to delete', 'error')
      })
      .finally(() => setDeletingUserId(null))
  }

  function handleAddUser(): void {
    const id = newUserId.trim()
    if (!id) return
    setAddingUser(false)
    setNewUserId('')
    void handleEnroll(id)
  }

  if (loading) return <PageLoadingFallback lines={4} />

  return (
    <div className="p-6 max-w-lg">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-text-primary">Users</h2>
        {!addingUser && (
          <button
            type="button"
            onClick={() => setAddingUser(true)}
            className="flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-accent/90 focus:outline-none focus:ring-2 focus:ring-accent"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            Add User
          </button>
        )}
      </div>

      {/* Add User form */}
      {addingUser && (
        <div className="mb-5 rounded-xl border border-accent/30 bg-accent/5 p-4">
          <p className="mb-3 text-sm font-medium text-text-primary">New User ID</p>
          <div className="flex gap-2">
            <input
              type="text"
              value={newUserId}
              placeholder="e.g. alice"
              onChange={(e) => setNewUserId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleAddUser()
                if (e.key === 'Escape') { setAddingUser(false); setNewUserId('') }
              }}
              autoFocus
              className="flex-1 bg-surface-raised border border-border rounded-lg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:outline-none focus:ring-2 focus:ring-accent"
            />
            <button
              type="button"
              onClick={handleAddUser}
              disabled={!newUserId.trim()}
              className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent/90 disabled:opacity-50"
            >
              Start Enrollment
            </button>
            <button
              type="button"
              onClick={() => { setAddingUser(false); setNewUserId('') }}
              className="rounded-lg border border-border px-3 py-2 text-sm text-text-secondary transition-colors hover:bg-surface-raised"
            >
              Cancel
            </button>
          </div>
          <p className="mt-2 text-xs text-text-secondary">
            You will be prompted to record {ENROLLMENT_SAMPLE_TARGET} voice samples.
          </p>
        </div>
      )}

      {/* Enrollment progress (shown during active enrollment) */}
      {enrollingUserId !== null && (
        <div className="mb-5 rounded-xl border border-border bg-surface-raised p-4">
          <p className="mb-3 text-sm font-medium text-text-primary">
            Enrolling: <span className="text-accent">{enrollingUserId}</span>
          </p>
          <div className="h-2 overflow-hidden rounded-full bg-surface mb-2">
            <div
              className="h-full rounded-full bg-accent transition-all duration-300"
              style={{ width: `${(capturedSamples / ENROLLMENT_SAMPLE_TARGET) * 100}%` }}
            />
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-text-secondary">
              {enrollmentCountdown > 0 ? `Sample ${capturedSamples + 1} starts in` : 'Recording now'}
            </span>
            <span className="text-2xl font-semibold tabular-nums text-text-primary">
              {enrollmentCountdown > 0 ? enrollmentCountdown : 'REC'}
            </span>
          </div>
          {enrollmentMessage && <p className="mt-2 text-sm text-success">{enrollmentMessage}</p>}
          {enrollmentError && <p className="mt-2 text-sm text-danger">{enrollmentError}</p>}
        </div>
      )}

      {/* Users list */}
      {enrollments.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border bg-surface-raised/40 px-4 py-8 text-center text-sm text-text-secondary">
          No enrolled users. Click "Add User" to get started.
        </div>
      ) : (
        <div className="space-y-3">
          {enrollments.map((enrollment) => {
            const displayName = userNames[enrollment.user_id] ?? enrollment.user_id
            const initial = displayName.charAt(0).toUpperCase()
            const isSelected = selectedUserId === enrollment.user_id
            const isActive = enrollment.user_id === activeUserId
            return (
              <div key={enrollment.user_id} className="rounded-xl border border-border bg-surface-raised">
                <button
                  type="button"
                  onClick={() => setSelectedUserId(isSelected ? null : enrollment.user_id)}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left focus:outline-none focus:ring-2 focus:ring-accent focus:ring-inset rounded-xl"
                >
                  <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent/20 text-accent font-semibold text-sm">
                    {initial}
                  </div>
                  <div className="flex-1 min-w-0">
                    {editingName === enrollment.user_id ? (
                      <input
                        type="text"
                        value={editNameValue}
                        onClick={(e) => e.stopPropagation()}
                        onChange={(e) => setEditNameValue(e.target.value)}
                        onBlur={() => handleSaveName(enrollment.user_id)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') handleSaveName(enrollment.user_id)
                          if (e.key === 'Escape') setEditingName(null)
                        }}
                        autoFocus
                        className="w-full bg-bg border border-accent rounded px-2 py-0.5 text-sm text-text-primary focus:outline-none"
                      />
                    ) : (
                      <div className="text-sm font-medium text-text-primary truncate">{displayName}</div>
                    )}
                    <div className="text-xs text-text-secondary">
                      {enrollment.sample_count} samples · {enrollment.model_id}
                      {isActive && <span className="ml-1.5 text-accent">· active</span>}
                    </div>
                  </div>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className={`shrink-0 text-text-secondary transition-transform ${isSelected ? 'rotate-180' : ''}`}
                  >
                    <polyline points="6 9 12 15 18 9" />
                  </svg>
                </button>

                {isSelected && (
                  <div className="border-t border-border px-4 pb-4 pt-3">
                    <div className="flex flex-wrap gap-2 mb-4">
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleStartEditName(enrollment.user_id) }}
                        className="rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised"
                      >
                        Edit Name
                      </button>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); void handleEnroll(enrollment.user_id) }}
                        disabled={enrollingUserId !== null}
                        className="rounded-lg border border-border bg-bg px-3 py-1.5 text-xs font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50"
                      >
                        Re-enroll Voice
                      </button>
                      <button
                        type="button"
                        onClick={(e) => { e.stopPropagation(); handleDelete(enrollment.user_id) }}
                        disabled={deletingUserId === enrollment.user_id}
                        className="rounded-lg border border-danger/30 bg-bg px-3 py-1.5 text-xs font-medium text-danger transition-colors hover:bg-danger/10 disabled:opacity-50"
                      >
                        {deletingUserId === enrollment.user_id ? 'Deleting…' : 'Delete User'}
                      </button>
                    </div>

                    {/* Memory viewer */}
                    <div>
                      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wider text-text-secondary">
                        Memory
                      </h4>
                      {memoriesLoading ? (
                        <div className="space-y-2">
                          <SkeletonLine width="100%" height="1rem" />
                          <SkeletonLine width="75%" height="1rem" />
                        </div>
                      ) : memories.length === 0 ? (
                        <p className="text-xs text-text-secondary">No stored memories.</p>
                      ) : (
                        <div className="max-h-48 overflow-y-auto space-y-2 rounded-lg border border-border bg-bg p-3">
                          {memories.map((mem) => (
                            <div key={mem.id} className="text-xs">
                              <span className="inline-block mr-1.5 rounded bg-surface-raised px-1.5 py-0.5 font-medium text-text-secondary">
                                {mem.category}
                              </span>
                              <span className="text-text-primary">{mem.text}</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function SystemPanel(): React.ReactElement {
  const addToast = useToast()
  const [settings, setSettings] = useState<SystemSettings>({
    autonomyMode: 'manual',
    toolTimeoutSeconds: 10,
    requireConfirmSystemChanges: true,
    allowedFileRoots: '',
    debugLogging: false
  })
  const [saved, setSaved] = useState(false)
  const [restarting, setRestarting] = useState(false)

  useEffect(() => {
    window.rex
      .getSettings('system')
      .then((s: Settings) => {
        setSettings({
          autonomyMode:
            s.autonomyMode === 'supervised' || s.autonomyMode === 'full-auto'
              ? (s.autonomyMode as 'supervised' | 'full-auto')
              : 'manual',
          toolTimeoutSeconds: typeof s.toolTimeoutSeconds === 'number' ? s.toolTimeoutSeconds : 10,
          requireConfirmSystemChanges:
            typeof s.requireConfirmSystemChanges === 'boolean' ? s.requireConfirmSystemChanges : true,
          allowedFileRoots: typeof s.allowedFileRoots === 'string' ? s.allowedFileRoots : '',
          debugLogging: typeof s.debugLogging === 'boolean' ? s.debugLogging : false
        })
      })
      .catch(() => {})
  }, [])

  function handleSave(): void {
    window.rex
      .setSettings('system', settings as unknown as Settings)
      .then(() => {
        setSaved(true)
        setTimeout(() => setSaved(false), 2000)
      })
      .catch(() => addToast('Failed to save system settings', 'error'))
  }

  function handleRestart(): void {
    setRestarting(true)
    window.rex
      .restartRex()
      .catch(() => addToast('Failed to restart Rex', 'error'))
      .finally(() => setRestarting(false))
  }

  return (
    <div className="p-6 max-w-lg">
      <h2 className="text-lg font-semibold text-text-primary mb-6">System &amp; Advanced</h2>

      {/* Autonomy mode */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-text-primary mb-2">Autonomy Mode</label>
        <select
          value={settings.autonomyMode}
          onChange={(e) =>
            setSettings((s) => ({
              ...s,
              autonomyMode: e.target.value as SystemSettings['autonomyMode']
            }))
          }
          className="w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text-primary focus:border-accent focus:outline-none"
        >
          <option value="manual">Manual — Rex only acts when explicitly asked</option>
          <option value="supervised">Supervised — Rex proposes actions, you confirm</option>
          <option value="full-auto">Full-Auto — Rex acts autonomously within budget</option>
        </select>
        <p className="mt-1 text-xs text-text-secondary">
          Controls how independently Rex takes actions on your behalf.
        </p>
      </div>

      {/* Tool timeout */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <label className="text-sm font-medium text-text-primary">
            Tool Timeout
            <span className="ml-2 text-xs text-text-secondary font-normal">
              {settings.toolTimeoutSeconds}s
            </span>
          </label>
        </div>
        <div className="flex items-center gap-2 text-xs text-text-secondary">
          <span>1s</span>
          <input
            type="range"
            min={1}
            max={60}
            step={1}
            value={settings.toolTimeoutSeconds}
            onChange={(e) =>
              setSettings((s) => ({ ...s, toolTimeoutSeconds: parseInt(e.target.value) }))
            }
            className="flex-1 accent-accent"
          />
          <span>60s</span>
        </div>
        <p className="mt-1 text-xs text-text-secondary">
          Maximum time Rex waits for a tool (email, calendar, search) before timing out.
        </p>
      </div>

      {/* Toggles */}
      <div className="mb-6 space-y-4">
        <div className="flex items-center justify-between rounded-xl border border-border bg-surface-raised p-4">
          <div>
            <div className="text-sm font-medium text-text-primary">Require confirmation for system changes</div>
            <div className="text-xs text-text-secondary mt-0.5">
              Ask before Rex modifies volume, brightness, or other system settings.
            </div>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={settings.requireConfirmSystemChanges}
            onClick={() =>
              setSettings((s) => ({ ...s, requireConfirmSystemChanges: !s.requireConfirmSystemChanges }))
            }
            className={[
              'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none',
              settings.requireConfirmSystemChanges ? 'bg-accent' : 'bg-border'
            ].join(' ')}
          >
            <span
              className={[
                'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform',
                settings.requireConfirmSystemChanges ? 'translate-x-5' : 'translate-x-0'
              ].join(' ')}
            />
          </button>
        </div>

        <div className="flex items-center justify-between rounded-xl border border-border bg-surface-raised p-4">
          <div>
            <div className="text-sm font-medium text-text-primary">Debug logging</div>
            <div className="text-xs text-text-secondary mt-0.5">
              Write verbose DEBUG-level logs. Useful for diagnosing issues.
            </div>
          </div>
          <button
            type="button"
            role="switch"
            aria-checked={settings.debugLogging}
            onClick={() => setSettings((s) => ({ ...s, debugLogging: !s.debugLogging }))}
            className={[
              'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors focus:outline-none',
              settings.debugLogging ? 'bg-accent' : 'bg-border'
            ].join(' ')}
          >
            <span
              className={[
                'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform',
                settings.debugLogging ? 'translate-x-5' : 'translate-x-0'
              ].join(' ')}
            />
          </button>
        </div>
      </div>

      {/* Allowed file roots */}
      <div className="mb-8">
        <label className="block text-sm font-medium text-text-primary mb-1">
          Allowed File Roots
        </label>
        <p className="text-xs text-text-secondary mb-2">
          Comma-separated directory paths Rex is allowed to read and write. Defaults to your home directory if left blank.
        </p>
        <input
          type="text"
          value={settings.allowedFileRoots}
          onChange={(e) => setSettings((s) => ({ ...s, allowedFileRoots: e.target.value }))}
          placeholder="C:\Users\you, D:\Documents"
          className="w-full rounded-lg border border-border bg-bg px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary focus:border-accent focus:outline-none"
        />
      </div>

      {/* Save */}
      <div className="flex items-center gap-3 mb-8">
        <button
          type="button"
          onClick={handleSave}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent/90 focus:outline-none"
        >
          Save
        </button>
        {saved && (
          <span className="text-xs text-success flex items-center gap-1">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Saved
          </span>
        )}
      </div>

      {/* Restart Rex */}
      <div className="border-t border-border pt-6">
        <h3 className="mb-1 text-sm font-semibold text-text-primary">Restart Rex</h3>
        <p className="mb-4 text-xs text-text-secondary">
          Gracefully restarts the Rex application. Use this after changing advanced settings.
        </p>
        <button
          type="button"
          onClick={handleRestart}
          disabled={restarting}
          className="flex items-center gap-2 rounded-lg border border-border bg-bg px-4 py-2 text-sm font-medium text-text-primary transition-colors hover:bg-surface-raised disabled:opacity-50 focus:outline-none"
        >
          {restarting ? (
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            </svg>
          ) : (
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="23 4 23 10 17 10" />
              <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" />
            </svg>
          )}
          Restart Rex
        </button>
      </div>
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
    case 'users':
      return <UsersPanel />
    case 'audio':
      return <AudioOutputPanel />
    case 'system':
      return <SystemPanel />
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
