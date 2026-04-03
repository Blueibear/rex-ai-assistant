import React, { useState, useCallback, useRef, useEffect } from 'react'
import { VoiceToggle } from '../components/voice/VoiceToggle'
import type { VoiceState } from '../components/voice/VoiceToggle'
import { WaveformVisualizer } from '../components/voice/WaveformVisualizer'
import { EmptyState } from '../components/ui/EmptyState'
import { Tooltip } from '../components/ui/Tooltip'
import type { VoiceTranscriptEntry } from '../types/ipc'

// ── Helpers ───────────────────────────────────────────────────────────────────

function formatTime(ms: number): string {
  const d = new Date(ms)
  const h = d.getHours().toString().padStart(2, '0')
  const m = d.getMinutes().toString().padStart(2, '0')
  const s = d.getSeconds().toString().padStart(2, '0')
  return `${h}:${m}:${s}`
}

// ── Wake word status indicator ────────────────────────────────────────────────

interface WakeWordStatusBadgeProps {
  isActive: boolean
  state: VoiceState
}

const statusConfig: Record<string, { label: string; color: string }> = {
  inactive: { label: 'Inactive', color: 'bg-surface-raised text-text-muted' },
  listening_wake: { label: 'Listening for wake word', color: 'bg-blue-600/20 text-blue-400' },
  detected: { label: 'Wake word detected', color: 'bg-red-600/20 text-red-400' },
  processing: { label: 'Processing', color: 'bg-accent/20 text-accent' },
  speaking: { label: 'Speaking', color: 'bg-green-600/20 text-green-400' },
}

const WakeWordStatusBadge: React.FC<WakeWordStatusBadgeProps> = ({ isActive, state }) => {
  let key: string
  if (!isActive) {
    key = 'inactive'
  } else if (state === 'idle') {
    key = 'listening_wake'
  } else if (state === 'listening') {
    key = 'detected'
  } else if (state === 'processing') {
    key = 'processing'
  } else {
    key = 'speaking'
  }
  const cfg = statusConfig[key] ?? statusConfig['inactive']!

  return (
    <div className="flex items-center gap-2">
      <span
        role="status"
        aria-live="polite"
        aria-label={`Voice status: ${cfg.label}`}
        className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium select-none ${cfg.color}`}
      >
        <span
          className={`w-1.5 h-1.5 rounded-full ${
            key === 'detected'
              ? 'bg-red-400 animate-pulse'
              : key === 'listening_wake'
                ? 'bg-blue-400 animate-pulse'
                : key === 'processing'
                  ? 'bg-accent animate-spin'
                  : key === 'speaking'
                    ? 'bg-green-400 animate-pulse'
                    : 'bg-text-muted'
          }`}
          aria-hidden="true"
        />
        {cfg.label}
      </span>
    </div>
  )
}

// ── Microphone device selector ────────────────────────────────────────────────

interface MicrophoneSelectorProps {
  devices: MediaDeviceInfo[]
  selectedId: string
  onChange: (id: string) => void
  onRequestPermission: () => void
}

const MicrophoneSelector: React.FC<MicrophoneSelectorProps> = ({
  devices,
  selectedId,
  onChange,
  onRequestPermission,
}) => {
  const hasLabels = devices.some((d) => d.label !== '')
  const hasDevices = devices.length > 0

  return (
    <div className="flex flex-col gap-1 w-full max-w-xs">
      <label
        htmlFor="mic-selector"
        className="text-xs text-text-muted uppercase tracking-wide"
      >
        Microphone
      </label>
      {!hasLabels && hasDevices ? (
        <div className="flex items-center gap-2">
          <select
            id="mic-selector"
            disabled
            className="flex-1 rounded-md bg-surface-raised border border-white/10 text-text-muted text-sm px-3 py-1.5 opacity-50"
          >
            <option>Permission required for device names</option>
          </select>
          <Tooltip text="Grant microphone access to see device names" position="right">
            <button
              type="button"
              onClick={onRequestPermission}
              className="text-xs px-2 py-1 rounded bg-accent/20 text-accent hover:bg-accent/30 transition-colors"
            >
              Allow
            </button>
          </Tooltip>
        </div>
      ) : (
        <select
          id="mic-selector"
          value={selectedId}
          onChange={(e) => onChange(e.target.value)}
          className="rounded-md bg-surface-raised border border-white/10 text-text-primary text-sm px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-accent"
        >
          <option value="">System default</option>
          {devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || `Microphone ${d.deviceId.slice(0, 8)}`}
            </option>
          ))}
        </select>
      )}
    </div>
  )
}

// ── Push-to-talk button ───────────────────────────────────────────────────────

const MicIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    className="w-5 h-5"
    aria-hidden="true"
  >
    <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm6.364 9.636a1 1 0 0 1 1 1A7.364 7.364 0 0 1 13 18.93V21h2a1 1 0 1 1 0 2H9a1 1 0 1 1 0-2h2v-2.07A7.364 7.364 0 0 1 4.636 11.636a1 1 0 0 1 2 0A5.364 5.364 0 0 0 12 17a5.364 5.364 0 0 0 5.364-5.364 1 1 0 0 1 1-1z" />
  </svg>
)

interface PushToTalkButtonProps {
  selectedMicId: string
  onTranscript: (entry: VoiceTranscriptEntry) => void
  onMicDevicesUpdated: (devices: MediaDeviceInfo[]) => void
}

const PushToTalkButton: React.FC<PushToTalkButtonProps> = ({
  selectedMicId,
  onTranscript,
  onMicDevicesUpdated,
}) => {
  const [recording, setRecording] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])

  const startRecording = useCallback(async () => {
    setError(null)
    try {
      const constraints: MediaStreamConstraints = {
        audio: selectedMicId ? { deviceId: { exact: selectedMicId } } : true,
        video: false,
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)

      // Re-enumerate now that permission is granted — labels will be populated.
      const all = await navigator.mediaDevices.enumerateDevices()
      onMicDevicesUpdated(all.filter((d) => d.kind === 'audioinput'))

      chunksRef.current = []
      const recorder = new MediaRecorder(stream)

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop())
        setRecording(false)
        if (chunksRef.current.length === 0) {
          setBusy(false)
          return
        }
        setBusy(true)
        try {
          const blob = new Blob(chunksRef.current, {
            type: recorder.mimeType || 'audio/webm',
          })
          const ab = await blob.arrayBuffer()
          const bytes = new Uint8Array(ab)
          // Convert binary to base64 in 32 KB chunks to avoid stack overflow.
          const CHUNK = 0x8000
          let binary = ''
          for (let i = 0; i < bytes.length; i += CHUNK) {
            binary += String.fromCharCode.apply(
              null,
              bytes.subarray(i, i + CHUNK) as unknown as number[],
            )
          }
          const b64 = btoa(binary)
          const result = await window.rex.sendChatAudio(b64)
          if (result.ok && result.transcript) {
            onTranscript({
              text: result.transcript,
              role: 'user',
              timestamp: Date.now(),
            })
          } else if (!result.ok) {
            setError(result.error ?? 'Transcription failed')
          }
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err))
        } finally {
          setBusy(false)
        }
      }

      recorder.start()
      recorderRef.current = recorder
      setRecording(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Microphone access denied')
    }
  }, [selectedMicId, onTranscript, onMicDevicesUpdated])

  const stopRecording = useCallback(() => {
    recorderRef.current?.stop()
    recorderRef.current = null
  }, [])

  const handlePointerDown = useCallback(
    (e: React.PointerEvent<HTMLButtonElement>) => {
      e.currentTarget.setPointerCapture(e.pointerId)
      if (!recording && !busy) void startRecording()
    },
    [recording, busy, startRecording],
  )

  const handlePointerUp = useCallback(() => {
    if (recording) stopRecording()
  }, [recording, stopRecording])

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLButtonElement>) => {
      if (e.key === ' ' && !e.repeat) {
        e.preventDefault()
        if (!recording && !busy) void startRecording()
      }
    },
    [recording, busy, startRecording],
  )

  const handleKeyUp = useCallback(
    (e: React.KeyboardEvent<HTMLButtonElement>) => {
      if (e.key === ' ' && recording) {
        e.preventDefault()
        stopRecording()
      }
    },
    [recording, stopRecording],
  )

  let label: string
  if (busy) label = 'Transcribing…'
  else if (recording) label = 'Release to send'
  else label = 'Hold to talk'

  return (
    <div className="flex flex-col items-center gap-1">
      <button
        type="button"
        aria-label={label}
        aria-pressed={recording}
        disabled={busy}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onKeyDown={handleKeyDown}
        onKeyUp={handleKeyUp}
        className={[
          'flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-medium transition-all duration-150 select-none focus:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-offset-surface',
          recording
            ? 'bg-red-600 text-white focus-visible:ring-red-400 scale-95'
            : busy
              ? 'bg-surface-raised text-text-muted cursor-not-allowed'
              : 'bg-surface-raised text-text-primary hover:bg-surface-raised/80 focus-visible:ring-accent',
        ].join(' ')}
      >
        <MicIcon />
        {label}
        {recording && (
          <span
            className="w-2 h-2 rounded-full bg-white animate-pulse"
            aria-hidden="true"
          />
        )}
      </button>
      {error && (
        <span className="text-xs text-danger mt-1">{error}</span>
      )}
    </div>
  )
}

// ── Clear button with two-click confirmation ──────────────────────────────────

interface ClearButtonProps {
  onClear: () => void
}

const ClearButton: React.FC<ClearButtonProps> = ({ onClear }) => {
  const [pendingClear, setPendingClear] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleClick = useCallback(() => {
    if (pendingClear) {
      // Second click — confirm clear
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current)
        timerRef.current = null
      }
      setPendingClear(false)
      onClear()
    } else {
      // First click — arm confirmation
      setPendingClear(true)
      timerRef.current = setTimeout(() => {
        setPendingClear(false)
        timerRef.current = null
      }, 2000)
    }
  }, [pendingClear, onClear])

  // Cleanup timer on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current !== null) clearTimeout(timerRef.current)
    }
  }, [])

  return (
    <Tooltip
      text={pendingClear ? 'Click again to confirm' : 'Clear history'}
      position="left"
    >
      <button
        onClick={handleClick}
        aria-label="Clear transcript history"
        className={[
          'text-xs px-2 py-1 rounded transition-colors',
          pendingClear
            ? 'bg-danger/20 text-danger'
            : 'text-text-muted hover:text-text-primary hover:bg-surface-raised'
        ].join(' ')}
      >
        {pendingClear ? 'Confirm?' : 'Clear'}
      </button>
    </Tooltip>
  )
}

// ── Transcript list ───────────────────────────────────────────────────────────

interface TranscriptListProps {
  entries: VoiceTranscriptEntry[]
  voiceState: VoiceState
  onClear: () => void
}

const TranscriptList: React.FC<TranscriptListProps> = ({
  entries,
  voiceState,
  onClear
}) => {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [entries, voiceState])

  const showPartial = voiceState === 'listening'

  return (
    <div className="w-full max-w-xl mt-6 flex flex-col gap-0">
      {/* Header with clear button */}
      <div className="flex items-center justify-between mb-2 px-1">
        <span className="text-xs text-text-muted uppercase tracking-wide">
          Transcript
        </span>
        <ClearButton onClear={onClear} />
      </div>

      {/* Scrollable entries */}
      <div className="flex flex-col gap-2 overflow-y-auto max-h-64 px-1">
        {entries.map((entry, idx) => {
          const isUser = entry.role === 'user'
          return (
            <div
              key={idx}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'}`}
            >
              <div
                className={`max-w-xs px-3 py-2 rounded-xl text-sm leading-relaxed ${
                  isUser
                    ? 'bg-accent text-white rounded-br-none'
                    : 'bg-surface-raised text-text-primary rounded-bl-none'
                }`}
              >
                {entry.text}
              </div>
              <span className="text-xs text-text-muted mt-0.5 px-1">
                {isUser ? 'You' : 'Rex'} · {formatTime(entry.timestamp)}
              </span>
            </div>
          )
        })}

        {/* Partial / in-progress row (italic + blinking cursor) */}
        {showPartial && (
          <div className="flex flex-col items-end">
            <div className="max-w-xs px-3 py-2 rounded-xl text-sm leading-relaxed bg-accent/40 text-white rounded-br-none italic opacity-80">
              Listening
              <span className="inline-block w-0.5 h-3.5 ml-0.5 align-middle bg-white animate-pulse rounded-sm" />
            </div>
            <span className="text-xs text-text-muted mt-0.5 px-1">You</span>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}

// ── MicOff icon for error EmptyState ─────────────────────────────────────────

const MicOffIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    className="w-10 h-10"
    aria-hidden="true"
  >
    <path d="M19 11a1 1 0 0 1 2 0 9 9 0 0 1-8 8.94V22h2a1 1 0 1 1 0 2H9a1 1 0 1 1 0-2h2v-2.06A9 9 0 0 1 3 11a1 1 0 1 1 2 0 7 7 0 0 0 11.43 5.41l1.42 1.41A8.96 8.96 0 0 1 13 19.94V22h-2v-2.06A9.02 9.02 0 0 1 3 11zm-5.16 3.42-1.42-1.42A2 2 0 0 1 10 11V5a2 2 0 0 1 4 0v5.59l-1.42-1.42A.5.5 0 0 0 12 9V5a1 1 0 0 0-2 0v5a1 1 0 0 0 .29.71l.55.55zM3.29 2.29l18 18-1.42 1.42-18-18 1.42-1.42z" />
  </svg>
)

// ── Main page ─────────────────────────────────────────────────────────────────

export function VoicePage(): React.ReactElement {
  const [voiceState, setVoiceState] = useState<VoiceState>('idle')
  const [isActive, setIsActive] = useState(false)
  const [transcripts, setTranscripts] = useState<VoiceTranscriptEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedMicId, setSelectedMicId] = useState<string>('')

  // Enumerate microphone devices on mount (labels may be empty until permission granted).
  useEffect(() => {
    const loadDevices = async (): Promise<void> => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices()
        setMicDevices(devices.filter((d) => d.kind === 'audioinput'))
      } catch {
        // mediaDevices not available (e.g. non-secure context) — silently skip
      }
    }
    void loadDevices()
    navigator.mediaDevices.addEventListener('devicechange', () => void loadDevices())
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', () => void loadDevices())
    }
  }, [])

  const handleMicDevicesUpdated = useCallback((devices: MediaDeviceInfo[]) => {
    setMicDevices(devices)
  }, [])

  const handleTranscript = useCallback((entry: VoiceTranscriptEntry) => {
    setTranscripts((prev) => [...prev, entry])
  }, [])

  const handleClearTranscripts = useCallback(() => {
    setTranscripts([])
  }, [])

  const handleToggle = useCallback(async () => {
    if (!isActive) {
      setError(null)
      try {
        await window.rex.startVoice(
          (state) => {
            // Only accept known VoiceState values.
            const known: VoiceState[] = ['idle', 'listening', 'processing', 'speaking']
            if (known.includes(state as VoiceState)) {
              setVoiceState(state as VoiceState)
            }
          },
          (entry) => {
            setTranscripts((prev) => [...prev, entry])
          },
          (err) => {
            setError(err)
            setVoiceState('idle')
            setIsActive(false)
          }
        )
        setIsActive(true)
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
        setVoiceState('idle')
      }
    } else {
      await window.rex.stopVoice()
      setIsActive(false)
      setVoiceState('idle')
    }
  }, [isActive])

  // Request mic permission (needed to populate device labels).
  const handleRequestMicPermission = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
      stream.getTracks().forEach((t) => t.stop())
      const devices = await navigator.mediaDevices.enumerateDevices()
      setMicDevices(devices.filter((d) => d.kind === 'audioinput'))
    } catch {
      // User denied — stay with current device list
    }
  }, [])

  useEffect(() => {
    const onToggle = (): void => {
      void handleToggle()
    }
    window.addEventListener('rex:toggle-voice', onToggle)
    return () => window.removeEventListener('rex:toggle-voice', onToggle)
  }, [handleToggle])

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4">
        <EmptyState
          icon={<MicOffIcon />}
          heading="Voice backend unavailable"
          subtext={error}
          action={{
            label: 'Try again',
            onClick: () => {
              setError(null)
            }
          }}
        />
      </div>
    )
  }

  const showTranscriptPanel = transcripts.length > 0 || voiceState === 'listening'

  return (
    <div className="flex flex-col items-center justify-start h-full pt-8 gap-6 overflow-y-auto pb-8">
      {/* Wake word status indicator */}
      <WakeWordStatusBadge isActive={isActive} state={voiceState} />

      {/* Audio waveform visualization */}
      <WaveformVisualizer state={voiceState} width={320} height={80} />

      {/* Main voice loop toggle (wake-word mode) */}
      <VoiceToggle state={voiceState} onToggle={() => void handleToggle()} />

      {/* Separator */}
      <div className="w-full max-w-xs border-t border-white/10 pt-4 flex flex-col items-center gap-4">
        {/* Microphone device selector */}
        <MicrophoneSelector
          devices={micDevices}
          selectedId={selectedMicId}
          onChange={setSelectedMicId}
          onRequestPermission={() => void handleRequestMicPermission()}
        />

        {/* Push-to-talk button (one-shot STT without wake word) */}
        <PushToTalkButton
          selectedMicId={selectedMicId}
          onTranscript={handleTranscript}
          onMicDevicesUpdated={handleMicDevicesUpdated}
        />
      </div>

      {/* Transcript panel */}
      {showTranscriptPanel && (
        <TranscriptList
          entries={transcripts}
          voiceState={voiceState}
          onClear={handleClearTranscripts}
        />
      )}
    </div>
  )
}
