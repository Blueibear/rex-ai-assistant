import { app, Menu, nativeImage, Tray, BrowserWindow } from 'electron'
import { join } from 'path'

let tray: Tray | null = null
let isQuitting = false

/** Resolve the path to a tray icon asset for the given pixel size. */
function getIconPath(size: 16 | 32): string {
  // Packed app: assets live in process.resourcesPath/assets/
  // Dev:        assets live at gui/assets/ (four levels up from compiled main/index.js)
  const assetsBase = app.isPackaged
    ? join(process.resourcesPath, 'assets')
    : join(__dirname, '../../../../assets')
  return join(assetsBase, `tray-icon-${size}.png`)
}

function buildContextMenu(mainWindow: BrowserWindow): Menu {
  return Menu.buildFromTemplate([
    {
      label: 'Show Rex',
      click: () => {
        mainWindow.show()
        mainWindow.focus()
      }
    },
    {
      label: 'New Chat',
      click: () => {
        mainWindow.show()
        mainWindow.focus()
        mainWindow.webContents.send('rex:navigate', '/chat')
        mainWindow.webContents.send('rex:focusChatInput')
      }
    },
    {
      label: 'Toggle Voice',
      click: () => {
        mainWindow.show()
        mainWindow.focus()
        mainWindow.webContents.send('rex:toggleVoice')
      }
    },
    { type: 'separator' },
    {
      label: 'Quit Rex',
      click: () => {
        isQuitting = true
        app.quit()
      }
    }
  ])
}

export function createTray(mainWindow: BrowserWindow): void {
  const icon = nativeImage.createFromPath(getIconPath(32))

  tray = new Tray(icon)
  tray.setToolTip('Rex AI Assistant')
  tray.setContextMenu(buildContextMenu(mainWindow))

  // Single-click on the tray icon restores the window
  tray.on('click', () => {
    mainWindow.show()
    mainWindow.focus()
  })

  // Mark that we are about to quit so the close handler below allows it
  app.on('before-quit', () => {
    isQuitting = true
  })

  // Intercept window close — hide to tray unless the app is quitting
  mainWindow.on('close', (event) => {
    if (!isQuitting) {
      event.preventDefault()
      mainWindow.hide()
    }
  })
}

export function destroyTray(): void {
  if (tray) {
    tray.destroy()
    tray = null
  }
}
