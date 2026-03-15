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

    try {
      const reply = await window.rex.sendChat(text)
      const rexMsg: Message = {
        id: genId(),
        role: 'rex',
        content: reply,
        timestamp: new Date()
      }
      setMessages((prev) => [...prev, rexMsg])
    } catch (err) {
      const errorMsg: Message = {
        id: genId(),
        role: 'rex',
        content: `\`Error: ${err instanceof Error ? err.message : String(err)}\``,
        timestamp: new Date()
      }
      setMessages((prev) => [...prev, errorMsg])
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
