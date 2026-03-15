import React, { useState } from 'react'
import { Textarea } from '../ui/Textarea'
import { Spinner } from '../ui/Spinner'

export interface ChatInputProps {
  onSend: (message: string) => void
  sending?: boolean
  disabled?: boolean
}

const CHAR_COUNT_THRESHOLD = 200

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

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  sending = false,
  disabled = false
}) => {
  const [value, setValue] = useState('')

  const isEmpty = value.trim().length === 0
  const isDisabled = disabled || sending || isEmpty
  const charCount = value.length
  const showCharCount = charCount > CHAR_COUNT_THRESHOLD

  const handleSend = (): void => {
    if (isDisabled) return
    onSend(value.trim())
    setValue('')
    // Reset textarea height via value change (Textarea's useEffect handles it)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col gap-1 p-4 border-t border-border bg-surface">
      <div className="flex items-end gap-2">
        <div className="flex-1">
          <Textarea
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Message Rex… (Enter to send, Shift+Enter for newline)"
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
