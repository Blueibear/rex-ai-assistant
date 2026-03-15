import React from 'react'

export interface SkeletonLineProps {
  width?: string
  height?: string
  className?: string
}

export const SkeletonLine: React.FC<SkeletonLineProps> = ({
  width = '100%',
  height = '1rem',
  className = ''
}) => {
  return (
    <div
      className={`animate-pulse rounded bg-surface-raised ${className}`}
      style={{ width, height }}
      aria-hidden="true"
    />
  )
}

export default SkeletonLine
