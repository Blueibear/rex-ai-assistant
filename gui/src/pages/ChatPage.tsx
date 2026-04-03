import React, { useState, useCallback, useEffect } from 'react'
import { MessageList } from '../components/chat/MessageList'
import { ChatInput } from '../components/chat/ChatInput'
import type { Message, MessageAttachment } from '../components/chat/MessageList'
import type { PendingAttachment } from '../components/chat/ChatInput'

let nextId = 1
function genId(): string {
  return `msg-${nextId++}`
}

/** Build the augmented message text that includes extracted file content. */
function buildAugmentedMessage(text: string, extractions: Map<string, string>): string {
  const parts: string[] = []
  for (const [name, content] of extractions) {
    parts.push(`[Attached file: ${name}]\n${content}\n---`)
  }
  if (text) parts.push(text)
  return parts.join('\n\n')
}

export function ChatPage(): React.ReactElement {
  const [messages, setMessages] = useState<Message[]>([])
  const [sending, setSending] = useState(false)

  useEffect(() => {
    const focusInput = (): void => {
      const el = document.querySelector<HTMLTextAreaElement>(
        'textarea[aria-label="Chat message input"]'
      )
      el?.focus()
    }
    window.addEventListener('rex:focus-chat', focusInput)
    return () => window.removeEventListener('rex:focus-chat', focusInput)
  }, [])

  const handleSend = useCallback(
    async (text: string, attachments: PendingAttachment[]): Promise<void> => {
      // Process attachments: extract text from documents, keep images as-is
      const displayAttachments: MessageAttachment[] = []
      const textExtractions = new Map<string, string>()

      for (const att of attachments) {
        const isImage = att.mimeType.startsWith('image/')

        if (isImage) {
          displayAttachments.push({ name: att.name, isImage: true, dataUrl: att.dataUrl })
        } else {
          // Extract text via IPC
          try {
            const result = await window.rex.extractFileForChat({
              filename: att.name,
              dataBase64: att.dataBase64,
              mimeType: att.mimeType,
              sizeBytes: att.sizeBytes
            })
            if (result.ok && result.extractedText) {
              textExtractions.set(att.name, result.extractedText)
            }
          } catch {
            // Extraction failed — still show the chip but don't inject content
          }
          displayAttachments.push({ name: att.name, isImage: false })
        }
      }

      const augmentedText = buildAugmentedMessage(text, textExtractions)
      const displayText = text || (attachments.length > 0 ? '(file attachment)' : '')

      const userMsg: Message = {
        id: genId(),
        role: 'user',
        content: displayText,
        timestamp: new Date(),
        attachments: displayAttachments.length > 0 ? displayAttachments : undefined
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
        await window.rex.sendChatStream(augmentedText, (token) => {
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
    },
    []
  )

  return (
    <div className="flex flex-col h-full">
      <MessageList messages={messages} />
      <ChatInput onSend={handleSend} sending={sending} />
    </div>
  )
}
