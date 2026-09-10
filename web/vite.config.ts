import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// `autorl.web` serves the data; this serves the views. The proxy is what lets
// `pnpm dev` talk to it without CORS, and a built bundle is served against a
// same-origin API from wherever it is copied to.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8000' },
  },
})
