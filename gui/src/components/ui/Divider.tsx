import React from 'react'

export interface DividerProps {
  label?: string
  className?: string
}

export const Divider: React.FC<DividerProps> = ({ label, className = '' }) => {
  if (label) {
    return (
      <div className={`flex items-center gap-3 ${className}`}>
        <div className="flex-1 h-px bg-border" />
        <span className="text-xs text-text-secondary select-none">{label}</span>
        <div className="flex-1 h-px bg-border" />
      </div>
    )
  }

  return <div className={`h-px bg-border ${className}`} />
}

export default Divider
