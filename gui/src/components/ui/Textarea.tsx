import React, { useEffect, useRef } from 'react'

export interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  error?: string
  helperText?: string
}

const MAX_ROWS = 6

export const Textarea: React.FC<TextareaProps> = ({
  label,
  error,
  helperText,
  id,
  className = '',
  onChange,
  ...rest
}) => {
  const ref = useRef<HTMLTextAreaElement>(null)
  const inputId = id ?? label?.toLowerCase().replace(/\s+/g, '-')

  const resize = (): void => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    const lineHeight = parseInt(getComputedStyle(el).lineHeight || '20', 10)
    const maxHeight = lineHeight * MAX_ROWS + 16 // 16 = padding top + bottom
    el.style.height = `${Math.min(el.scrollHeight, maxHeight)}px`
    el.style.overflowY = el.scrollHeight > maxHeight ? 'auto' : 'hidden'
  }

  useEffect(() => {
    resize()
  }, [rest.value])

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>): void => {
    resize()
    onChange?.(e)
  }

  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label htmlFor={inputId} className="text-sm font-medium text-text-secondary">
          {label}
        </label>
      )}
      <textarea
        ref={ref}
        id={inputId}
        rows={1}
        className={[
          'bg-surface-raised border rounded-md px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary',
          'transition-colors duration-150 outline-none resize-none',
          'focus:ring-2 focus:ring-accent focus:border-accent',
          error ? 'border-danger focus:ring-danger' : 'border-border',
          className
        ].join(' ')}
        onChange={handleChange}
        {...rest}
      />
      {error && <span className="text-xs text-danger">{error}</span>}
      {!error && helperText && (
        <span className="text-xs text-text-secondary">{helperText}</span>
      )}
    </div>
  )
}

export default Textarea
