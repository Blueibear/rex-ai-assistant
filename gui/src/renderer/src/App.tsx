import React, { useEffect, useState } from 'react'

function App(): React.ReactElement {
  const [status, setStatus] = useState<string>('loading…')

  useEffect(() => {
    window.rex
      .getStatus()
      .then((res) => setStatus(res.status ?? 'unknown'))
      .catch(() => setStatus('error'))
  }, [])

  return (
    <div className="flex items-center justify-center h-screen bg-[#0F1117] text-white">
      <p className="text-xl font-semibold">Rex is starting… ({status})</p>
    </div>
  )
}

export default App
