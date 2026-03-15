import { contextBridge, ipcRenderer } from 'electron'
import { electronAPI } from '@electron-toolkit/preload'
import type { Settings } from '../types/ipc'

if (process.contextIsolated) {
  try {
    contextBridge.exposeInMainWorld('electron', electronAPI)
    contextBridge.exposeInMainWorld('rex', {
      sendChat: (message: string) => ipcRenderer.invoke('rex:sendChat', message),
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
    sendChat: (message: string) => ipcRenderer.invoke('rex:sendChat', message),
    getStatus: () => ipcRenderer.invoke('rex:getStatus'),
    getSettings: () => ipcRenderer.invoke('rex:getSettings'),
    setSettings: (settings: Settings) => ipcRenderer.invoke('rex:setSettings', settings)
  }
}
