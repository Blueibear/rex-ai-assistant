import React from 'react'

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  loading?: boolean
  children: React.ReactNode
}

const variantClasses: Record<NonNullable<ButtonProps['variant']>, string> = {
  primary:
    'bg-accent text-white hover:bg-blue-600 focus:ring-accent disabled:bg-accent/50',
  secondary:
    'bg-surface-raised text-text-primary border border-border hover:bg-surface-raised/80 focus:ring-accent disabled:opacity-50',
  ghost:
    'bg-transparent text-text-secondary hover:bg-surface-raised hover:text-text-primary focus:ring-accent disabled:opacity-50',
  danger:
    'bg-danger text-white hover:bg-red-600 focus:ring-danger disabled:bg-danger/50'
}

const sizeClasses: Record<NonNullable<ButtonProps['size']>, string> = {
  sm: 'px-3 py-1.5 text-sm rounded',
  md: 'px-4 py-2 text-sm rounded-md',
  lg: 'px-6 py-3 text-base rounded-lg'
}

const Spinner: React.FC<{ size: NonNullable<ButtonProps['size']> }> = ({ size }) => {
  const dim = size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'
  return (
    <svg
      className={`animate-spin ${dim} mr-2`}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      aria-hidden="true"
    >
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled,
  children,
  className = '',
  ...rest
}) => {
  return (
    <button
      disabled={loading || disabled}
      className={[
        'inline-flex items-center justify-center font-medium transition-colors duration-150',
        'focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-bg',
        'motion-safe:active:scale-[0.97] motion-safe:transition-transform',
        variantClasses[variant],
        sizeClasses[size],
        className
      ].join(' ')}
      {...rest}
    >
      {loading && <Spinner size={size} />}
      {children}
    </button>
  )
}

export default Button
