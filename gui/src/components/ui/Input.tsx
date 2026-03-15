import React from 'react'

export interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  helperText?: string
}

export const Input: React.FC<InputProps> = ({
  label,
  error,
  helperText,
  id,
  className = '',
  ...rest
}) => {
  const inputId = id ?? label?.toLowerCase().replace(/\s+/g, '-')

  return (
    <div className="flex flex-col gap-1">
      {label && (
        <label htmlFor={inputId} className="text-sm font-medium text-text-secondary">
          {label}
        </label>
      )}
      <input
        id={inputId}
        className={[
          'bg-surface-raised border rounded-md px-3 py-2 text-sm text-text-primary placeholder:text-text-secondary',
          'transition-colors duration-150 outline-none',
          'focus:ring-2 focus:ring-accent focus:border-accent',
          error ? 'border-danger focus:ring-danger' : 'border-border',
          className
        ].join(' ')}
        {...rest}
      />
      {error && <span className="text-xs text-danger">{error}</span>}
      {!error && helperText && (
        <span className="text-xs text-text-secondary">{helperText}</span>
      )}
    </div>
  )
}

export default Input
