import { ipcMain } from 'electron'
import { spawn } from 'child_process'
import { join } from 'path'

const FILE_SIZE_LIMIT_BYTES = 10 * 1024 * 1024 // 10 MB

const TEXT_MIME_TYPES = new Set([
  'text/plain',
  'text/markdown',
  'text/csv',
  'application/csv',
  'text/x-markdown'
])

const IMAGE_MIME_TYPES = new Set(['image/png', 'image/jpeg', 'image/jpg'])

function callExtractBridge(
  filename: string,
  dataBase64: string,
  mimeType: string
): Promise<{ ok: boolean; isImage: boolean; extractedText?: string; error?: string }> {
  return new Promise((resolve) => {
    const scriptPath = join(__dirname, '../../../../rex_file_extract_bridge.py')
    const py = spawn('python', [scriptPath], { stdio: ['pipe', 'pipe', 'pipe'] })
    let stdout = ''

    py.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString()
    })

    py.on('close', () => {
      try {
        const result = JSON.parse(stdout.trim()) as {
          ok: boolean
          is_image?: boolean
          extracted_text?: string
          error?: string
        }
        resolve({
          ok: result.ok,
          isImage: result.is_image ?? false,
          extractedText: result.extracted_text,
          error: result.error
        })
      } catch {
        resolve({ ok: false, isImage: false, error: 'Failed to parse extraction result' })
      }
    })

    py.on('error', () => {
      resolve({ ok: false, isImage: false, error: 'Failed to start extraction bridge' })
    })

    py.stdin.write(
      JSON.stringify({ filename, data_base64: dataBase64, mime_type: mimeType })
    )
    py.stdin.end()
  })
}

export function registerFileHandlers(): void {
  ipcMain.handle(
    'rex:extractFileForChat',
    async (
      _event,
      {
        filename,
        dataBase64,
        mimeType,
        sizeBytes
      }: { filename: string; dataBase64: string; mimeType: string; sizeBytes: number }
    ): Promise<{ ok: boolean; isImage: boolean; extractedText?: string; error?: string }> => {
      if (sizeBytes > FILE_SIZE_LIMIT_BYTES) {
        return { ok: false, isImage: false, error: 'File too large (max 10 MB)' }
      }

      const normalizedMime = mimeType.toLowerCase().split(';')[0].trim()

      if (IMAGE_MIME_TYPES.has(normalizedMime)) {
        return { ok: true, isImage: true }
      }

      if (
        TEXT_MIME_TYPES.has(normalizedMime) ||
        /\.(txt|md|csv)$/i.test(filename)
      ) {
        try {
          const text = Buffer.from(dataBase64, 'base64').toString('utf-8')
          return { ok: true, isImage: false, extractedText: text }
        } catch {
          return { ok: false, isImage: false, error: 'Failed to decode text file' }
        }
      }

      // PDF or other — delegate to Python bridge
      return callExtractBridge(filename, dataBase64, normalizedMime)
    }
  )
}
