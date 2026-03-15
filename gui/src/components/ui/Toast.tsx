import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'

export type ToastType = 'info' | 'success' | 'warning' | 'error'

export interface ToastItem {
  id: string
  type: ToastType
  message: string
}

interface ToastContextValue {
  addToast: (message: string, type?: ToastType) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

const typeClasses: Record<ToastType, string> = {
  info: 'border-accent bg-surface text-text-primary',
  success: 'border-success bg-surface text-text-primary',
  warning: 'border-yellow-400 bg-surface text-text-primary',
  error: 'border-danger bg-surface text-text-primary'
}

const typeIcons: Record<ToastType, string> = {
  info: 'ℹ',
  success: '✓',
  warning: '⚠',
  error: '✕'
}

const iconColors: Record<ToastType, string> = {
  info: 'text-accent',
  success: 'text-success',
  warning: 'text-yellow-400',
  error: 'text-danger'
}

const AUTO_DISMISS_MS = 4000

const ToastEntry: React.FC<{ item: ToastItem; onDismiss: (id: string) => void }> = ({
  item,
  onDismiss
}) => {
  const [visible, setVisible] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    // Trigger slide-in on next frame
    const raf = requestAnimationFrame(() => setVisible(true))
    timerRef.current = setTimeout(() => {
      setVisible(false)
      setTimeout(() => onDismiss(item.id), 300)
    }, AUTO_DISMISS_MS)

    return () => {
      cancelAnimationFrame(raf)
      if (timerRef.current !== null) clearTimeout(timerRef.current)
    }
  }, [item.id, onDismiss])

  return (
    <div
      role="alert"
      aria-live="assertive"
      className={[
        'flex items-start gap-2 rounded-lg border px-4 py-3 shadow-lg text-sm',
        'transition-all duration-300',
        visible ? 'translate-x-0 opacity-100' : 'translate-x-8 opacity-0',
        typeClasses[item.type]
      ].join(' ')}
    >
      <span className={`font-bold mt-px ${iconColors[item.type]}`} aria-hidden="true">
        {typeIcons[item.type]}
      </span>
      <span className="flex-1">{item.message}</span>
      <button
        onClick={() => {
          setVisible(false)
          setTimeout(() => onDismiss(item.id), 300)
        }}
        className="text-text-secondary hover:text-text-primary transition-colors ml-1"
        aria-label="Dismiss notification"
      >
        ✕
      </button>
    </div>
  )
}

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([])

  const addToast = useCallback((message: string, type: ToastType = 'info') => {
    const id = `${Date.now()}-${Math.random()}`
    setToasts((prev) => [...prev, { id, type, message }])
  }, [])

  const dismiss = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  return (
    <ToastContext.Provider value={{ addToast }}>
      {children}
      <div
        className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 w-80"
        aria-label="Notifications"
      >
        {toasts.map((item) => (
          <ToastEntry key={item.id} item={item} onDismiss={dismiss} />
        ))}
      </div>
    </ToastContext.Provider>
  )
}

export const useToast = (): ((message: string, type?: ToastType) => void) => {
  const ctx = useContext(ToastContext)
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return ctx.addToast
}

export default ToastProvider
