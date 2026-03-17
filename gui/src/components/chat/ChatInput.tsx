import React, { useRef, useState } from 'react'
import { Textarea } from '../ui/Textarea'
import { Spinner } from '../ui/Spinner'
import { useToast } from '../ui/Toast'

export interface ChatInputProps {
  onSend: (message: string) => void
  sending?: boolean
  disabled?: boolean
}

const CHAR_COUNT_THRESHOLD = 200
const MAX_RECORD_MS = 5000

const SendIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    className="w-4 h-4"
    aria-hidden="true"
  >
    <path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" />
  </svg>
)

const MicIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    className="w-4 h-4"
    aria-hidden="true"
  >
    <path d="M8.25 4.5a3.75 3.75 0 117.5 0v8.25a3.75 3.75 0 11-7.5 0V4.5z" />
    <path d="M6 10.5a.75.75 0 01.75.75v1.5a5.25 5.25 0 1010.5 0v-1.5a.75.75 0 011.5 0v1.5a6.751 6.751 0 01-6 6.709v2.291h3a.75.75 0 010 1.5h-7.5a.75.75 0 010-1.5h3v-2.291a6.751 6.751 0 01-6-6.709v-1.5A.75.75 0 016 10.5z" />
  </svg>
)

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  sending = false,
  disabled = false
}) => {
  const [value, setValue] = useState('')
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const addToast = useToast()

  const isEmpty = value.trim().length === 0
  const isDisabled = disabled || sending || isEmpty
  const charCount = value.length
  const showCharCount = charCount > CHAR_COUNT_THRESHOLD

  const handleSend = (): void => {
    if (isDisabled) return
    onSend(value.trim())
    setValue('')
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const stopRecording = (): void => {
    if (stopTimerRef.current !== null) {
      clearTimeout(stopTimerRef.current)
      stopTimerRef.current = null
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop()
    }
  }

  const handleMicClick = async (): Promise<void> => {
    if (recording) {
      stopRecording()
      return
    }

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      addToast(
        'Microphone access denied. Please allow microphone permission in your browser settings.',
        'error'
      )
      return
    }

    chunksRef.current = []
    const recorder = new MediaRecorder(stream)
    mediaRecorderRef.current = recorder

    recorder.ondataavailable = (e): void => {
      if (e.data.size > 0) chunksRef.current.push(e.data)
    }

    recorder.onstop = async (): Promise<void> => {
      setRecording(false)
      stream.getTracks().forEach((t) => t.stop())

      const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
      chunksRef.current = []

      setTranscribing(true)
      try {
        const arrayBuffer = await blob.arrayBuffer()
        const uint8 = new Uint8Array(arrayBuffer)
        let binary = ''
        for (let i = 0; i < uint8.length; i++) {
          binary += String.fromCharCode(uint8[i])
        }
        const audioBase64 = btoa(binary)

        const result = await window.rex.sendChatAudio(audioBase64)
        if (result.ok && result.transcript) {
          setValue((prev) => (prev ? `${prev} ${result.transcript}` : result.transcript ?? ''))
        } else {
          addToast(result.error ?? 'Transcription failed', 'error')
        }
      } catch (err) {
        addToast('Failed to transcribe audio', 'error')
        console.error('sendChatAudio error:', err)
      } finally {
        setTranscribing(false)
      }
    }

    recorder.start()
    setRecording(true)

    stopTimerRef.current = setTimeout(() => {
      stopRecording()
    }, MAX_RECORD_MS)
  }

  const micBusy = recording || transcribing

  return (
    <div className="flex flex-col gap-1 p-4 border-t border-border bg-surface">
      <div className="flex items-end gap-2">
        <button
          onClick={handleMicClick}
          disabled={disabled || sending || transcribing}
          aria-label={recording ? 'Stop recording' : 'Start voice input'}
          title={recording ? 'Click to stop recording (or wait 5 s)' : 'Dictate a message'}
          className={[
            'flex items-center justify-center w-9 h-9 rounded-md transition-colors duration-150 mb-0.5 flex-shrink-0',
            'focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg',
            disabled || sending || transcribing
              ? 'bg-surface text-text-secondary/30 cursor-not-allowed'
              : recording
                ? 'bg-danger/20 text-danger hover:bg-danger/30'
                : 'bg-surface text-text-secondary hover:bg-surface-raised hover:text-text-primary'
          ].join(' ')}
        >
          {transcribing ? (
            <Spinner size="sm" />
          ) : recording ? (
            <span className="relative flex items-center justify-center">
              <span className="absolute inline-flex h-full w-full rounded-full bg-danger opacity-50 animate-ping" />
              <MicIcon />
            </span>
          ) : (
            <MicIcon />
          )}
        </button>
        <div className="flex-1">
          <Textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              micBusy
                ? transcribing
                  ? 'Transcribing…'
                  : 'Listening… (click mic or wait 5 s to stop)'
                : 'Message Rex… (Enter to send, Shift+Enter for newline)'
            }
            disabled={disabled || sending}
            aria-label="Chat message input"
          />
        </div>
        <button
          onClick={handleSend}
          disabled={isDisabled}
          aria-label="Send message"
          className={[
            'flex items-center justify-center w-9 h-9 rounded-md transition-colors duration-150 mb-0.5',
            'focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg',
            isDisabled
              ? 'bg-accent/30 text-white/30 cursor-not-allowed'
              : 'bg-accent text-white hover:bg-blue-600'
          ].join(' ')}
        >
          {sending ? <Spinner size="sm" className="text-white" /> : <SendIcon />}
        </button>
      </div>
      {showCharCount && (
        <div className="flex justify-end">
          <span
            className={`text-xs ${charCount > 1000 ? 'text-danger' : 'text-text-secondary'}`}
          >
            {charCount}
          </span>
        </div>
      )}
    </div>
  )
}

export default ChatInput
