import React, { useEffect, useRef } from 'react'

export interface MessageAttachment {
  name: string
  isImage: boolean
  dataUrl?: string // data URL for inline image display
}

export interface Message {
  id: string
  role: 'user' | 'rex'
  content: string
  timestamp: Date
  streaming?: boolean
  attachments?: MessageAttachment[]
}

export interface MessageListProps {
  messages: Message[]
}

function relativeTime(date: Date): string {
  const diffSec = Math.floor((Date.now() - date.getTime()) / 1000)
  if (diffSec < 10) return 'just now'
  if (diffSec < 60) return `${diffSec}s ago`
  const diffMin = Math.floor(diffSec / 60)
  if (diffMin < 60) return `${diffMin} min ago`
  const diffHr = Math.floor(diffMin / 60)
  if (diffHr < 24) return `${diffHr} hr ago`
  return `${Math.floor(diffHr / 24)}d ago`
}

function renderInline(text: string, keyPrefix: string): React.ReactNode[] {
  const parts: React.ReactNode[] = []
  const regex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g
  let last = 0
  let match: RegExpExecArray | null
  let idx = 0
  while ((match = regex.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(text.slice(last, match.index))
    }
    const token = match[0]
    const k = `${keyPrefix}-inline-${idx++}`
    if (token.startsWith('`')) {
      parts.push(
        <code key={k} className="bg-surface-raised text-accent font-mono text-sm px-1 rounded">
          {token.slice(1, -1)}
        </code>
      )
    } else if (token.startsWith('**')) {
      parts.push(<strong key={k}>{token.slice(2, -2)}</strong>)
    } else {
      parts.push(<em key={k}>{token.slice(1, -1)}</em>)
    }
    last = match.index + token.length
  }
  if (last < text.length) parts.push(text.slice(last))
  return parts
}

function renderMarkdown(content: string, msgId: string): React.ReactNode {
  const lines = content.split('\n')
  const nodes: React.ReactNode[] = []
  let i = 0
  let blockKey = 0

  while (i < lines.length) {
    const line = lines[i]
    const k = `${msgId}-${blockKey++}`

    if (line.trimStart().startsWith('```')) {
      const codeLines: string[] = []
      i++
      while (i < lines.length && !lines[i].trimStart().startsWith('```')) {
        codeLines.push(lines[i])
        i++
      }
      nodes.push(
        <pre
          key={k}
          className="bg-surface-raised rounded p-3 text-sm font-mono overflow-x-auto my-2 text-text-primary"
        >
          <code>{codeLines.join('\n')}</code>
        </pre>
      )
      i++
      continue
    }

    if (/^[-*] /.test(line)) {
      const listItems: string[] = []
      while (i < lines.length && /^[-*] /.test(lines[i])) {
        listItems.push(lines[i].slice(2))
        i++
      }
      nodes.push(
        <ul key={k} className="list-disc list-inside my-1 space-y-0.5">
          {listItems.map((item, j) => (
            <li key={j}>{renderInline(item, `${k}-li-${j}`)}</li>
          ))}
        </ul>
      )
      continue
    }

    if (line.trim() === '') {
      nodes.push(<br key={k} />)
      i++
      continue
    }

    nodes.push(
      <p key={k} className="my-0.5">
        {renderInline(line, k)}
      </p>
    )
    i++
  }

  return <>{nodes}</>
}

export const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-text-secondary text-sm">
        No messages yet. Say something!
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
      {messages.map((msg) => (
        <div
          key={msg.id}
          className={['flex flex-col', msg.role === 'user' ? 'items-end' : 'items-start'].join(' ')}
        >
          <div
            className={[
              'max-w-[75%] rounded-2xl px-4 py-2 text-sm',
              msg.role === 'user'
                ? 'bg-accent text-white rounded-br-sm'
                : 'bg-surface-raised text-text-primary rounded-bl-sm'
            ].join(' ')}
          >
            {msg.attachments && msg.attachments.length > 0 && (
              <div className="mb-2 flex flex-wrap gap-2">
                {msg.attachments.map((att, i) =>
                  att.isImage && att.dataUrl ? (
                    <img
                      key={i}
                      src={att.dataUrl}
                      alt={att.name}
                      className="max-w-[200px] max-h-[150px] rounded object-contain"
                    />
                  ) : (
                    <div
                      key={i}
                      className="flex items-center gap-1 bg-black/10 rounded px-2 py-1 text-xs"
                    >
                      <span aria-hidden="true">📎</span>
                      <span>{att.name}</span>
                    </div>
                  )
                )}
              </div>
            )}
            {msg.role === 'rex' ? (
              <>
                {renderMarkdown(msg.content, msg.id)}
                {msg.streaming && (
                  <span
                    className="inline-block w-0.5 h-3.5 bg-accent ml-0.5 align-middle animate-pulse"
                    aria-hidden="true"
                  />
                )}
              </>
            ) : (
              <p>{msg.content}</p>
            )}
          </div>
          <span className="text-[11px] text-text-secondary mt-1 px-1">
            {relativeTime(msg.timestamp)}
          </span>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}

export default MessageList
