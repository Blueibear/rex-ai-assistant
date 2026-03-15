import React, { useState, useCallback } from 'react'
import { MessageList } from '../components/chat/MessageList'
import { ChatInput } from '../components/chat/ChatInput'
import type { Message } from '../components/chat/MessageList'

let nextId = 1
function genId(): string {
  return `msg-${nextId++}`
}

export function ChatPage(): React.ReactElement {
  const [messages, setMessages] = useState<Message[]>([])
  const [sending, setSending] = useState(false)

  const handleSend = useCallback(async (text: string): Promise<void> => {
    const userMsg: Message = {
      id: genId(),
      role: 'user',
      content: text,
      timestamp: new Date()
    }

    setMessages((prev) => [...prev, userMsg])
    setSending(true)

    const rexMsgId = genId()
    const rexMsg: Message = {
      id: rexMsgId,
      role: 'rex',
      content: '',
      timestamp: new Date(),
      streaming: true
    }
    setMessages((prev) => [...prev, rexMsg])

    try {
      await window.rex.sendChatStream(text, (token) => {
        setMessages((prev) =>
          prev.map((m) => (m.id === rexMsgId ? { ...m, content: m.content + token } : m))
        )
      })
      // Finalize: remove streaming cursor
      setMessages((prev) =>
        prev.map((m) => (m.id === rexMsgId ? { ...m, streaming: false } : m))
      )
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === rexMsgId
            ? {
                ...m,
                content: `\`Error: ${err instanceof Error ? err.message : String(err)}\``,
                streaming: false
              }
            : m
        )
      )
    } finally {
      setSending(false)
    }
  }, [])

  return (
    <div className="flex flex-col h-full">
      <MessageList messages={messages} />
      <ChatInput onSend={handleSend} sending={sending} />
    </div>
  )
}
