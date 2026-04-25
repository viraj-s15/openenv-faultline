import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { scenarioGenerator } from './vite-plugin-scenarios'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    scenarioGenerator(),
  ],
  server: {
    proxy: {
      '/reset': 'http://localhost:8000',
      '/step': 'http://localhost:8000',
      '/state': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})