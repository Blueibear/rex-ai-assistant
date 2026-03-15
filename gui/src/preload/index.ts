import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type { Settings } from '../types/ipc'

function makeSendChatStream(
  message: string,
  onToken: (token: string) => void
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const streamId = `${Date.now()}-${Math.random()}`

    function tokenHandler(_e: unknown, data: { streamId: string; token: string }): void {
      if (data.streamId === streamId) onToken(data.token)
    }
    function doneHandler(_e: unknown, data: { streamId: string }): void {
      if (data.streamId === streamId) {
        cleanup()
        resolve()
      }
    }
    function errorHandler(_e: unknown, data: { streamId: string; error: string }): void {
      if (data.streamId === streamId) {
        cleanup()
        reject(new Error(data.error))
      }
    }

    function cleanup(): void {
      ipcRenderer.removeListener('rex:chatToken', tokenHandler)
      ipcRenderer.removeListener('rex:chatDone', doneHandler)
      ipcRenderer.removeListener('rex:chatError', errorHandler)
    }

    ipcRenderer.on('rex:chatToken', tokenHandler)
    ipcRenderer.on('rex:chatDone', doneHandler)
    ipcRenderer.on('rex:chatError', errorHandler)

    ipcRenderer.invoke('rex:startChatStream', { message, streamId }).catch((err: unknown) => {
      cleanup()
      reject(err)
    })
  })
}

const rexAPI = {
  sendChat: (message: string) => ipcRenderer.invoke('rex:sendChat', message),
  sendChatStream: makeSendChatStream,
  getStatus: () => ipcRenderer.invoke('rex:getStatus'),
  getSettings: () => ipcRenderer.invoke('rex:getSettings'),
  setSettings: (settings: Settings) => ipcRenderer.invoke('rex:setSettings', settings)
}

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('rex', rexAPI)
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.rex = rexAPI
}
