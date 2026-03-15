import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type { ChatRequest, Settings } from '../types/ipc'

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('rex', {
      sendChat: (req: ChatRequest) => ipcRenderer.invoke('rex:sendChat', req),
      getStatus: () => ipcRenderer.invoke('rex:getStatus'),
      getSettings: () => ipcRenderer.invoke('rex:getSettings'),
      setSettings: (settings: Settings) => ipcRenderer.invoke('rex:setSettings', settings)
    })
  } catch (error) {
    console.error(error)
  }
} else {
  // @ts-ignore (define in dts)
  window.electron = electronAPI
  // @ts-ignore (define in dts)
  window.rex = {
    sendChat: (req: ChatRequest) => ipcRenderer.invoke('rex:sendChat', req),
    getStatus: () => ipcRenderer.invoke('rex:getStatus'),
    getSettings: () => ipcRenderer.invoke('rex:getSettings'),
    setSettings: (settings: Settings) => ipcRenderer.invoke('rex:setSettings', settings)
  }
}
