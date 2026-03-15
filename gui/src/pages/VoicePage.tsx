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
    <div className="flex flex-col items-center justify-start h-full pt-12 gap-8">
      <WaveformVisualizer state={voiceState} width={320} height={80} />
      <VoiceToggle state={voiceState} onToggle={() => void handleToggle()} />
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
