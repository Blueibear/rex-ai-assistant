import React from 'react'
import { EmptyState } from './ui/EmptyState'

interface Props {
  children: React.ReactNode
}

interface State {
  hasError: boolean
}

export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  override componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error('ErrorBoundary caught an error:', error, info)
  }

  override render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-full py-12">
          <EmptyState
            icon="⚠"
            heading="Something went wrong"
            subtext="An unexpected error occurred on this page."
            action={{
              label: 'Retry',
              onClick: () => this.setState({ hasError: false })
            }}
          />
        </div>
      )
    }
    return this.props.children
  }
}
