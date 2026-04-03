import React, { useState, useEffect, useRef } from 'react'

const API_BASE = ''

function formatTime(timestamp) {
  const d = new Date(timestamp * 1000)
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

const styles = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    height: 'calc(100vh - 60px)',
    maxWidth: '800px',
    margin: '0 auto',
    background: '#fff',
    borderRadius: '8px',
    boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
    overflow: 'hidden',
  },
  header: {
    padding: '1rem 1.25rem',
    borderBottom: '1px solid #e0e0e0',
    background: '#1a1a2e',
    color: '#fff',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  historyArea: {
    flex: 1,
    overflowY: 'auto',
    padding: '1rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '0.75rem',
    background: '#f9f9f9',
  },
  bubbleWrapper: (role) => ({
    display: 'flex',
    justifyContent: role === 'user' ? 'flex-end' : 'flex-start',
  }),
  bubble: (role) => ({
    maxWidth: '72%',
    padding: '0.6rem 0.9rem',
    borderRadius: role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
    background: role === 'user' ? '#4a90e2' : '#e8e8e8',
    color: role === 'user' ? '#fff' : '#1a1a1a',
    fontSize: '0.95rem',
    lineHeight: '1.45',
    wordBreak: 'break-word',
  }),
  bubbleMeta: (role) => ({
    fontSize: '0.75rem',
    color: role === 'user' ? 'rgba(255,255,255,0.75)' : '#888',
    marginTop: '0.25rem',
    textAlign: role === 'user' ? 'right' : 'left',
  }),
  attachment: {
    fontSize: '0.8rem',
    marginTop: '0.25rem',
    fontStyle: 'italic',
    opacity: 0.8,
  },
  thinkingDot: {
    display: 'inline-block',
    animation: 'blink 1.2s infinite',
    fontSize: '1.5rem',
    lineHeight: 1,
    color: '#888',
  },
  inputBar: {
    display: 'flex',
    gap: '0.5rem',
    padding: '0.75rem 1rem',
    borderTop: '1px solid #e0e0e0',
    background: '#fff',
    alignItems: 'flex-end',
  },
  textarea: {
    flex: 1,
    resize: 'none',
    border: '1px solid #ccc',
    borderRadius: '8px',
    padding: '0.55rem 0.75rem',
    fontFamily: 'system-ui, sans-serif',
    fontSize: '0.95rem',
    lineHeight: '1.4',
    minHeight: '40px',
    maxHeight: '120px',
    outline: 'none',
  },
  sendBtn: (disabled) => ({
    padding: '0.55rem 1.1rem',
    background: disabled ? '#aaa' : '#4a90e2',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontWeight: 600,
    fontSize: '0.9rem',
    flexShrink: 0,
  }),
  uploadBtn: {
    padding: '0.55rem 0.75rem',
    background: '#f0f0f0',
    border: '1px solid #ccc',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '1.1rem',
    flexShrink: 0,
    lineHeight: 1,
  },
  attachmentBadge: {
    fontSize: '0.8rem',
    color: '#4a90e2',
    padding: '0.1rem 0.4rem',
    background: '#e8f0fb',
    borderRadius: '4px',
    display: 'flex',
    alignItems: 'center',
    gap: '0.3rem',
    flexShrink: 0,
  },
  emptyHint: {
    textAlign: 'center',
    color: '#aaa',
    marginTop: '4rem',
    fontSize: '0.95rem',
  },
  clearBtn: {
    background: 'transparent',
    border: '1px solid rgba(255,255,255,0.4)',
    color: 'rgba(255,255,255,0.8)',
    borderRadius: '6px',
    padding: '0.3rem 0.7rem',
    cursor: 'pointer',
    fontSize: '0.8rem',
  },
}

export default function Chat() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [attachedFile, setAttachedFile] = useState(null) // { name, base64 }
  const bottomRef = useRef(null)
  const fileInputRef = useRef(null)
  const textareaRef = useRef(null)

  // Load history on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/chat/history`)
      .then((r) => r.json())
      .then((data) => {
        if (Array.isArray(data)) setMessages(data)
      })
      .catch(() => {
        // Backend not running — start with empty history
      })
  }, [])

  // Scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function handleFileChange(e) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      setAttachedFile({ name: file.name, base64: ev.target.result })
    }
    reader.readAsDataURL(file)
    // Reset input so the same file can be re-selected
    e.target.value = ''
  }

  function removeAttachment() {
    setAttachedFile(null)
  }

  async function sendMessage() {
    const text = input.trim()
    if (!text || sending) return

    const userMsg = {
      id: `local-${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: Date.now() / 1000,
      attachment_name: attachedFile?.name ?? null,
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setAttachedFile(null)
    setSending(true)

    // Placeholder assistant message while streaming
    const placeholderId = `pending-${Date.now()}`
    setMessages((prev) => [
      ...prev,
      { id: placeholderId, role: 'assistant', content: '', timestamp: Date.now() / 1000, attachment_name: null },
    ])

    try {
      const body = { message: text }
      if (attachedFile) {
        body.filename = attachedFile.name
        body.attachment_base64 = attachedFile.base64
      }

      const resp = await fetch(`${API_BASE}/api/chat/send`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`)

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let finalContent = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const parsed = JSON.parse(line.slice(6))
            if (parsed.content !== undefined) {
              finalContent = parsed.content
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === placeholderId ? { ...m, content: finalContent } : m,
                ),
              )
            }
          } catch {
            // ignore parse errors
          }
        }
      }
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === placeholderId
            ? { ...m, content: '(Error: could not reach Rex backend)' }
            : m,
        ),
      )
    } finally {
      // Reload history to get server-assigned IDs
      try {
        const hist = await fetch(`${API_BASE}/api/chat/history`).then((r) => r.json())
        if (Array.isArray(hist) && hist.length > 0) setMessages(hist)
      } catch {
        // keep local state
      }
      setSending(false)
    }
  }

  async function clearHistory() {
    try {
      await fetch(`${API_BASE}/api/chat/clear`, { method: 'POST' })
    } catch {
      // ignore
    }
    setMessages([])
  }

  return (
    <div style={styles.container}>
      <style>{`
        @keyframes blink {
          0%, 80%, 100% { opacity: 0; }
          40% { opacity: 1; }
        }
      `}</style>

      <div style={styles.header}>
        <span style={{ fontWeight: 600, fontSize: '1rem' }}>Chat with Rex</span>
        <button style={styles.clearBtn} onClick={clearHistory} title="Clear conversation">
          Clear
        </button>
      </div>

      <div style={styles.historyArea}>
        {messages.length === 0 && (
          <p style={styles.emptyHint}>Type a message below to start chatting with Rex.</p>
        )}

        {messages.map((msg) => (
          <div key={msg.id} style={styles.bubbleWrapper(msg.role)}>
            <div>
              <div style={styles.bubble(msg.role)}>
                {msg.content === '' && msg.role === 'assistant' ? (
                  <span style={styles.thinkingDot}>• • •</span>
                ) : (
                  msg.content
                )}
                {msg.attachment_name && (
                  <div style={styles.attachment}>📎 {msg.attachment_name}</div>
                )}
              </div>
              <div style={styles.bubbleMeta(msg.role)}>
                {msg.role === 'user' ? 'You' : 'Rex'} · {formatTime(msg.timestamp)}
              </div>
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      <div style={styles.inputBar}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,.pdf,.txt,.doc,.docx"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />

        <button
          style={styles.uploadBtn}
          onClick={() => fileInputRef.current?.click()}
          title="Attach file"
          aria-label="Attach file"
        >
          📎
        </button>

        {attachedFile && (
          <span style={styles.attachmentBadge}>
            {attachedFile.name}
            <span
              style={{ cursor: 'pointer', fontWeight: 700 }}
              onClick={removeAttachment}
              title="Remove attachment"
            >
              ×
            </span>
          </span>
        )}

        <textarea
          ref={textareaRef}
          style={styles.textarea}
          placeholder="Message Rex… (Enter to send, Shift+Enter for new line)"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={sending}
        />

        <button
          style={styles.sendBtn(!input.trim() || sending)}
          onClick={sendMessage}
          disabled={!input.trim() || sending}
          aria-label="Send message"
        >
          Send
        </button>
      </div>
    </div>
  )
}
