import React, { useEffect, useState } from 'react'
import { Spinner } from '../../components/ui/Spinner'
import { SkeletonLine } from '../../components/ui/SkeletonLine'
import { EmptyState } from '../../components/ui/EmptyState'
import { useToast } from '../../components/ui/Toast'
import { AppLayout } from '../../layouts/AppLayout'

function App(): React.ReactElement {
  const [status, setStatus] = useState<string>('loading…')
  const addToast = useToast()

  useEffect(() => {
    window.rex
      .getStatus()
      .then((res) => {
        const s = res.status ?? 'unknown'
        setStatus(s)
        addToast(`Rex status: ${s}`, 'info')
      })
      .catch(() => {
        setStatus('error')
        addToast('Could not reach Rex backend', 'error')
      })
  }, [addToast])

  if (status === 'loading…') {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-bg gap-4">
        <Spinner size="lg" />
        <SkeletonLine width="200px" height="16px" />
      </div>
    )
  }

  if (status === 'error') {
    return (
      <div className="flex items-center justify-center h-screen bg-bg">
        <EmptyState
          icon="⚠"
          heading="Rex backend unavailable"
          subtext="Start the Rex Python process and relaunch."
        />
      </div>
    )
  }

  return (
    <AppLayout sectionName="Chat">
      <div className="flex items-center justify-center h-full text-text-primary">
        <p className="text-xl font-semibold">Rex is starting… ({status})</p>
      </div>
    </AppLayout>
  )
}

export default App
