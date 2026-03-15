import type { Config } from 'tailwindcss'

const config: Config = {
  content: ['./src/**/*.{ts,tsx}', './src/renderer/index.html'],
  theme: {
    extend: {
      colors: {
        bg: '#0F1117',
        surface: '#1A1D27',
        'surface-raised': '#22263A',
        accent: '#3B82F6',
        danger: '#EF4444',
        success: '#22C55E'
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif']
      }
    }
  },
  plugins: []
}

export default config
