import React, { useRef, useState, useCallback, useEffect } from 'react'
import { Textarea } from '../ui/Textarea'
import { Spinner } from '../ui/Spinner'
import { useToast } from '../ui/Toast'

export interface PendingAttachment {
  name: string
  mimeType: string
  sizeBytes: number
  dataBase64: string
  dataUrl: string // for image preview in chip
}

export interface ChatInputProps {
  onSend: (message: string, attachments: PendingAttachment[]) => void
  sending?: boolean
  disabled?: boolean
}

const CHAR_COUNT_THRESHOLD = 200
const MAX_RECORD_MS = 5000
const FILE_SIZE_LIMIT_BYTES = 10 * 1024 * 1024 // 10 MB
const SESSION_LIMIT_BYTES = 50 * 1024 * 1024 // 50 MB

const ACCEPTED_EXTENSIONS = ['.txt', '.md', '.pdf', '.png', '.jpg', '.jpeg', '.csv']
const ACCEPTED_MIME_TYPES = [
  'text/plain',
  'text/markdown',
  'text/csv',
  'application/csv',
  'application/pdf',
  'image/png',
  'image/jpeg'
].join(',')

function isAcceptedFile(file: File): boolean {
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
  return ACCEPTED_EXTENSIONS.includes(ext)
}

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

const PaperclipIcon: React.FC = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    viewBox="0 0 24 24"
    fill="currentColor"
    className="w-4 h-4"
    aria-hidden="true"
  >
    <path
      fillRule="evenodd"
      d="M18.97 3.659a2.25 2.25 0 00-3.182 0l-10.94 10.94a3.75 3.75 0 105.304 5.303l7.693-7.693a.75.75 0 011.06 1.06l-7.693 7.693a5.25 5.25 0 11-7.424-7.424l10.939-10.94a3.75 3.75 0 115.303 5.304L9.097 18.835l-.008.008-.007.007-.002.002-.003.002A2.25 2.25 0 015.91 15.66l7.81-7.81a.75.75 0 011.061 1.06l-7.81 7.81a.75.75 0 001.054 1.068L18.97 6.84a2.25 2.25 0 000-3.182z"
      clipRule="evenodd"
    />
  </svg>
)

function readFileAsBase64(file: File): Promise<{ dataBase64: string; dataUrl: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const dataUrl = reader.result as string
      const base64 = dataUrl.split(',')[1] ?? ''
      resolve({ dataBase64: base64, dataUrl })
    }
    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  sending = false,
  disabled = false
}) => {
  const [value, setValue] = useState('')
  const [recording, setRecording] = useState(false)
  const [transcribing, setTranscribing] = useState(false)
  const [attachments, setAttachments] = useState<PendingAttachment[]>([])
  const [sessionBytes, setSessionBytes] = useState(0)
  const [dragging, setDragging] = useState(false)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const stopTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dragCounterRef = useRef(0)

  const addToast = useToast()

  const isEmpty = value.trim().length === 0 && attachments.length === 0
  const isDisabled = disabled || sending || isEmpty
  const charCount = value.length
  const showCharCount = charCount > CHAR_COUNT_THRESHOLD

  const processFiles = useCallback(
    async (files: File[]): Promise<void> => {
      for (const file of files) {
        if (!isAcceptedFile(file)) {
          addToast(
            `Unsupported file type: ${file.name}. Accepted: ${ACCEPTED_EXTENSIONS.join(', ')}`,
            'error'
          )
          continue
        }

        if (file.size > FILE_SIZE_LIMIT_BYTES) {
          addToast(`File too large: ${file.name} (max 10 MB)`, 'error')
          continue
        }

        if (sessionBytes + file.size > SESSION_LIMIT_BYTES) {
          addToast('Session file limit reached (50 MB total)', 'error')
          return
        }

        try {
          const { dataBase64, dataUrl } = await readFileAsBase64(file)
          const attachment: PendingAttachment = {
            name: file.name,
            mimeType: file.type || 'application/octet-stream',
            sizeBytes: file.size,
            dataBase64,
            dataUrl
          }
          setAttachments((prev) => [...prev, attachment])
          setSessionBytes((prev) => prev + file.size)
        } catch {
          addToast(`Failed to read file: ${file.name}`, 'error')
        }
      }
    },
    [sessionBytes, addToast]
  )

  const handleFileInputChange = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>): Promise<void> => {
      const files = Array.from(e.target.files ?? [])
      if (files.length > 0) {
        await processFiles(files)
      }
      // Reset so the same file can be picked again
      if (fileInputRef.current) fileInputRef.current.value = ''
    },
    [processFiles]
  )

  const removeAttachment = useCallback((index: number): void => {
    setAttachments((prev) => {
      const removed = prev[index]
      setSessionBytes((b) => b - removed.sizeBytes)
      return prev.filter((_, i) => i !== index)
    })
  }, [])

  const handleSend = (): void => {
    if (isDisabled) return
    onSend(value.trim(), attachments)
    setValue('')
    setAttachments([])
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  // Drag-and-drop handlers
  const handleDragEnter = useCallback((e: React.DragEvent): void => {
    e.preventDefault()
    dragCounterRef.current++
    setDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent): void => {
    e.preventDefault()
    dragCounterRef.current--
    if (dragCounterRef.current === 0) setDragging(false)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent): void => {
    e.preventDefault()
  }, [])

  const handleDrop = useCallback(
    async (e: React.DragEvent): Promise<void> => {
      e.preventDefault()
      dragCounterRef.current = 0
      setDragging(false)
      const files = Array.from(e.dataTransfer.files)
      await processFiles(files)
    },
    [processFiles]
  )

  // Reset drag counter on component unmount or when dragging state is lost
  useEffect(() => {
    return () => {
      dragCounterRef.current = 0
    }
  }, [])

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
    <div
      className={[
        'flex flex-col gap-1 p-4 border-t border-border bg-surface transition-colors duration-150',
        dragging ? 'bg-accent/10 border-accent' : ''
      ].join(' ')}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_MIME_TYPES}
        multiple
        className="hidden"
        onChange={handleFileInputChange}
        aria-label="Attach files"
      />

      {/* Attachment chips */}
      {attachments.length > 0 && (
        <div className="flex flex-wrap gap-1.5 mb-1">
          {attachments.map((att, i) => {
            const isImage = att.mimeType.startsWith('image/')
            return (
              <div
                key={i}
                className="flex items-center gap-1 bg-surface-raised rounded-full px-2.5 py-1 text-xs text-text-primary max-w-[180px]"
              >
                {isImage ? (
                  <img
                    src={att.dataUrl}
                    alt=""
                    className="w-4 h-4 rounded object-cover flex-shrink-0"
                  />
                ) : (
                  <span aria-hidden="true" className="flex-shrink-0">
                    📎
                  </span>
                )}
                <span className="truncate">{att.name}</span>
                <button
                  onClick={() => removeAttachment(i)}
                  className="ml-0.5 flex-shrink-0 text-text-secondary hover:text-danger transition-colors"
                  aria-label={`Remove ${att.name}`}
                >
                  ×
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* Drag overlay hint */}
      {dragging && (
        <div className="text-center text-sm text-accent py-1 pointer-events-none">
          Drop files here
        </div>
      )}

      <div className="flex items-end gap-2">
        {/* Mic button */}
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

        {/* Paperclip / attach button */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || sending}
          aria-label="Attach file"
          title={`Attach file (${ACCEPTED_EXTENSIONS.join(', ')})`}
          className={[
            'flex items-center justify-center w-9 h-9 rounded-md transition-colors duration-150 mb-0.5 flex-shrink-0',
            'focus:outline-none focus:ring-2 focus:ring-accent focus:ring-offset-2 focus:ring-offset-bg',
            disabled || sending
              ? 'bg-surface text-text-secondary/30 cursor-not-allowed'
              : 'bg-surface text-text-secondary hover:bg-surface-raised hover:text-text-primary'
          ].join(' ')}
        >
          <PaperclipIcon />
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
                : dragging
                  ? 'Drop files to attach…'
                  : 'Message Rex… (Enter to send, Shift+Enter for newline)'
            }
            disabled={disabled || sending}
            aria-label="Chat message input"
          />
        </div>

        {/* Send button */}
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
          <span className={`text-xs ${charCount > 1000 ? 'text-danger' : 'text-text-secondary'}`}>
            {charCount}
          </span>
        </div>
      )}
    </div>
  )
}

export default ChatInput
